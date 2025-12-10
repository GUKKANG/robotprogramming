import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64MultiArray
from twolink_msgs.msg import CMD
import math

# 로봇 링크 길이 (mm)
L1 = 143.0
L2 = 102.0
D_MAINTAIN = 250.0  # 유지 거리 (mm)

# --- [튜닝 파라미터] ---
GAIN_LINEAR = 0.05  # 전후 이동 속도 (거리 조절)
GAIN_ANGULAR = 0.08 # 좌우 회전 속도 (중심 맞추기)
DEADZONE_DIST = 5.0 # 거리 오차 허용 범위
DEADZONE_LAT = 2.0  # 좌우 오차 허용 범위
MAX_STEP = 10.0     # 한 번에 이동할 수 있는 최대 거리 (mm)

class KinematicsNode(Node):
    def __init__(self):
        super().__init__('kinematics_node')
        
        self.curr_q1 = 0.0
        self.curr_q2 = 0.0

        self.create_subscription(Pose, '/aruco/pose', self.aruco_callback, 10)
        self.create_subscription(CMD, '/des_value', self.joint_state_callback, 10)
        self.angle_pub = self.create_publisher(Float64MultiArray, '/target_angles', 10)

    def joint_state_callback(self, msg):
        self.curr_q1 = -math.degrees(msg.q1des)
        self.curr_q2 = math.degrees(msg.q2des)

    def aruco_callback(self, msg):
        # 1. 카메라 데이터 (mm)
        x_cam = msg.position.x * 1000.0
        z_cam = msg.position.z * 1000.0

        # 2. 현재 상태 계산 (FK)
        t1 = math.radians(self.curr_q1)
        t2 = math.radians(self.curr_q2)
        
        curr_x_ee = L1 * math.cos(t1) + L2 * math.cos(t1 + t2)
        curr_y_ee = L1 * math.sin(t1) + L2 * math.sin(t1 + t2)
        curr_phi = t1 + t2 # 현재 로봇이 바라보는 절대 각도 (End-Effector Orientation)

        # 3. 오차 계산
        err_x = x_cam               # 좌우 오차
        err_z = z_cam - D_MAINTAIN  # 거리 오차

        if abs(err_x) < DEADZONE_LAT and abs(err_z) < DEADZONE_DIST:
            return

        # 4. [핵심 변경] "바라보는 방향(curr_phi)"을 기준으로 이동 벡터 생성
        # 기존에는 절대 좌표계(x, y)에서 오차를 더했지만, 
        # 이제는 로봇의 "현재 시선 방향"으로 전진/후진하고, "시선의 수직 방향"으로 좌우 이동합니다.
        # 이렇게 하면 로봇 팔 전체가 '몸통'처럼 움직여 링크 1, 2가 조화롭게 움직입니다.

        move_forward = err_z * GAIN_LINEAR  # 시선 방향 이동 (거리 조절)
        move_side = -err_x * GAIN_ANGULAR   # 시선 수직 방향 이동 (좌우 조절)

        # 급발진 방지 (벡터 크기 제한)
        total_move = math.sqrt(move_forward**2 + move_side**2)
        if total_move > MAX_STEP:
            scale = MAX_STEP / total_move
            move_forward *= scale
            move_side *= scale

        # 5. 로봇 기준 이동량 -> Global 좌표계 이동량 변환 (회전 행렬 적용)
        # delta_x = Forward * cos(phi) - Side * sin(phi)
        # delta_y = Forward * sin(phi) + Side * cos(phi)
        delta_global_x = move_forward * math.cos(curr_phi) - move_side * math.sin(curr_phi)
        delta_global_y = move_forward * math.sin(curr_phi) + move_side * math.cos(curr_phi)

        target_x = curr_x_ee + delta_global_x
        target_y = curr_y_ee + delta_global_y

        # 6. 역기구학 및 발행
        try:
            q1_target, q2_target = self.inverse_kinematics_optimized(target_x, target_y)
            
            out_msg = Float64MultiArray()
            out_msg.data = [q1_target, q2_target]
            self.angle_pub.publish(out_msg)
            
        except ValueError:
            pass

    def inverse_kinematics_optimized(self, x, y):
        dist_sq = x**2 + y**2
        max_reach = L1 + L2
        
        # 작업 영역 제한 (완전히 펴지기 직전에 멈춤 -> 특이점 방지)
        if dist_sq > (max_reach * 0.99)**2:
            scale = (max_reach * 0.99) / math.sqrt(dist_sq)
            x *= scale
            y *= scale
            dist_sq = x**2 + y**2
        
        # 너무 가까운 거리 제한 (몸체 충돌 방지)
        min_reach = 50.0 
        if dist_sq < min_reach**2:
             return self.curr_q1, self.curr_q2 # 너무 가까우면 이동 포기

        cos_q2 = (dist_sq - L1**2 - L2**2) / (2 * L1 * L2)
        
        if cos_q2 > 1.0: cos_q2 = 1.0
        elif cos_q2 < -1.0: cos_q2 = -1.0

        # 해 1 (Elbow Down)
        q2_sol1 = -math.acos(cos_q2)
        k1_1 = L1 + L2 * math.cos(q2_sol1)
        k2_1 = L2 * math.sin(q2_sol1)
        q1_sol1 = math.atan2(y, x) - math.atan2(k2_1, k1_1)

        # 해 2 (Elbow Up)
        q2_sol2 = math.acos(cos_q2)
        k1_2 = L1 + L2 * math.cos(q2_sol2)
        k2_2 = L2 * math.sin(q2_sol2)
        q1_sol2 = math.atan2(y, x) - math.atan2(k2_2, k1_2)

        d_q1_1, d_q2_1 = math.degrees(q1_sol1), math.degrees(q2_sol1)
        d_q1_2, d_q2_2 = math.degrees(q1_sol2), math.degrees(q2_sol2)

        # [중요] Cost Function: "균형 잡힌 이동"을 위한 가중치 조정
        # 링크 1과 링크 2의 움직임 비율을 비슷하게 맞추기 위해 가중치를 1:1로 둡니다.
        # 대신, 현재 각도(curr_phi)와 새로운 각도(new_phi)의 차이를 최소화하는 항을 추가할 수도 있지만,
        # 2-Link에서는 위치가 정해지면 각도가 고정되므로, 
        # "현재 자세에서 가장 적게 변하는 해"를 고르는 것이 최선입니다.
        
        cost1 = abs(d_q1_1 - self.curr_q1) + abs(d_q2_1 - self.curr_q2)
        cost2 = abs(d_q1_2 - self.curr_q1) + abs(d_q2_2 - self.curr_q2)

        if cost1 < cost2:
            return d_q1_1, d_q2_1
        else:
            return d_q1_2, d_q2_2

def main(args=None):
    rclpy.init(args=args)
    node = KinematicsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()