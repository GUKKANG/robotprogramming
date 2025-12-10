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
GAIN = 0.05         # 반응 속도 (0.01 ~ 0.1): 값이 클수록 빠르지만 진동할 수 있음
DEADZONE_DIST = 5.0 # 거리 오차 허용 범위 (mm): 이 안에서는 앞뒤로 움직이지 않음
DEADZONE_LAT = 2.0  # 좌우 오차 허용 범위 (mm): 이 안에서는 좌우로 움직이지 않음
MAX_STEP = 15.0     # 한 번에 이동할 수 있는 최대 거리 (mm) - 급발진 방지

class KinematicsNode(Node):
    def __init__(self):
        super().__init__('kinematics_node')
        
        # 현재 로봇의 각도 상태 (피드백)
        self.curr_q1 = 0.0
        self.curr_q2 = 0.0

        # Subscriber
        self.create_subscription(Pose, '/aruco/pose', self.aruco_callback, 10)
        self.create_subscription(CMD, '/des_value', self.joint_state_callback, 10)

        # Publisher
        self.angle_pub = self.create_publisher(Float64MultiArray, '/target_angles', 10)

    def joint_state_callback(self, msg):
        # 하드웨어/시뮬레이션 상태 업데이트
        self.curr_q1 = -math.degrees(msg.q1des)
        self.curr_q2 = math.degrees(msg.q2des)

    def aruco_callback(self, msg):
        # 1. 카메라 좌표계 상의 마커 위치 (m -> mm)
        # x_cam: 화면 좌우 (오른쪽이 +)
        # z_cam: 화면 깊이 (전방이 +)
        x_cam = msg.position.x * 1000.0
        z_cam = msg.position.z * 1000.0

        # 2. Forward Kinematics (현재 End-Effector 위치 및 각도 계산)
        t1 = math.radians(self.curr_q1)
        t2 = math.radians(self.curr_q2)
        
        curr_x_ee = L1 * math.cos(t1) + L2 * math.cos(t1 + t2)
        curr_y_ee = L1 * math.sin(t1) + L2 * math.sin(t1 + t2)
        phi = t1 + t2 # 현재 End-Effector의 절대 각도

        # 3. 오차 계산 (Visual Servoing Error)
        # 목표: x_cam -> 0 (화면 중앙), z_cam -> D_MAINTAIN (거리 유지)
        err_x = x_cam               # 좌우 오차
        err_z = z_cam - D_MAINTAIN  # 거리 오차

        # Deadzone 체크 (오차가 작으면 움직이지 않음 -> 떨림 방지)
        if abs(err_x) < DEADZONE_LAT and abs(err_z) < DEADZONE_DIST:
            return

        # 4. 제어 입력 생성 (P-Control)
        # 카메라 좌표계에서의 이동량 계산
        delta_cam_x = err_x * GAIN
        delta_cam_z = err_z * GAIN

        # 급격한 움직임 제한 (Safety Clamping)
        total_move = math.sqrt(delta_cam_x**2 + delta_cam_z**2)
        if total_move > MAX_STEP:
            scale = MAX_STEP / total_move
            delta_cam_x *= scale
            delta_cam_z *= scale

        # 5. 좌표계 변환 (Camera Frame -> Global Frame)
        # 로봇 팔 끝(End-Effector) 좌표계에서 Global 좌표계로 회전 변환
        # z_cam 방향은 로봇 팔이 뻗은 방향(phi), x_cam 방향은 그 수직 방향(-90도)
        delta_global_x = delta_cam_z * math.cos(phi) - delta_cam_x * math.sin(phi)
        delta_global_y = delta_cam_z * math.sin(phi) + delta_cam_x * math.cos(phi)

        # 6. 다음 목표 위치 설정 (Incremental Update)
        # 현재 위치에서 계산된 오차만큼만 더해줌
        target_x = curr_x_ee + delta_global_x
        target_y = curr_y_ee + delta_global_y

        # 7. 역기구학 (Inverse Kinematics) 및 발행
        try:
            q1_target, q2_target = self.inverse_kinematics_optimized(target_x, target_y)
            
            out_msg = Float64MultiArray()
            out_msg.data = [q1_target, q2_target]
            self.angle_pub.publish(out_msg)
            
        except ValueError:
            self.get_logger().warn("Target unreachable")

    def inverse_kinematics_optimized(self, x, y):
        dist_sq = x**2 + y**2
        
        # 작업 영역(Workspace) 체크 및 제한
        max_reach = L1 + L2
        if dist_sq > max_reach**2:
            scale = max_reach / math.sqrt(dist_sq)
            x *= scale * 0.99 # 특이점 회피를 위해 살짝 안쪽으로
            y *= scale * 0.99
            dist_sq = x**2 + y**2

        cos_q2 = (dist_sq - L1**2 - L2**2) / (2 * L1 * L2)

        # 수치 오차 보정
        if cos_q2 > 1.0: cos_q2 = 1.0
        elif cos_q2 < -1.0: cos_q2 = -1.0

        # 해 1: Elbow Down
        q2_sol1 = -math.acos(cos_q2)
        k1_1 = L1 + L2 * math.cos(q2_sol1)
        k2_1 = L2 * math.sin(q2_sol1)
        q1_sol1 = math.atan2(y, x) - math.atan2(k2_1, k1_1)

        # 해 2: Elbow Up
        q2_sol2 = math.acos(cos_q2)
        k1_2 = L1 + L2 * math.cos(q2_sol2)
        k2_2 = L2 * math.sin(q2_sol2)
        q1_sol2 = math.atan2(y, x) - math.atan2(k2_2, k1_2)

        deg_q1_1, deg_q2_1 = math.degrees(q1_sol1), math.degrees(q2_sol1)
        deg_q1_2, deg_q2_2 = math.degrees(q1_sol2), math.degrees(q2_sol2)

        # 현재 자세와 가까운 해 선택 (Cost Function)
        cost1 = abs(deg_q1_1 - self.curr_q1) + abs(deg_q2_1 - self.curr_q2)
        cost2 = abs(deg_q1_2 - self.curr_q1) + abs(deg_q2_2 - self.curr_q2)

        if cost1 < cost2:
            return deg_q1_1, deg_q2_1
        else:
            return deg_q1_2, deg_q2_2

def main(args=None):
    rclpy.init(args=args)
    node = KinematicsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()