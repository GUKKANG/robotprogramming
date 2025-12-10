import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64MultiArray
from twolink_msgs.msg import CMD
import math

# 로봇 링크 길이 (mm)
L1 = 143.0
L2 = 102.0
D_MAINTAIN = 250.0  # 유지 거리 (너무 멀면 펴지기만 하므로 약간 줄임, 필요시 조정)

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
        # 하드웨어 특성에 따른 부호 처리 (기존 유지)
        self.curr_q1 = -math.degrees(msg.q1des)
        self.curr_q2 = math.degrees(msg.q2des)

    def aruco_callback(self, msg):
        # 1. 카메라 좌표계 상의 마커 위치 (m -> mm)
        x_cam = msg.position.x * 1000.0
        z_cam = msg.position.z * 1000.0

        # 2. Forward Kinematics (현재 End-Effector 위치 및 각도)
        t1 = math.radians(self.curr_q1)
        t2 = math.radians(self.curr_q2)
        
        x_ee = L1 * math.cos(t1) + L2 * math.cos(t1 + t2)
        y_ee = L1 * math.sin(t1) + L2 * math.sin(t1 + t2)
        phi = t1 + t2

        # 3. 마커의 전역 좌표(Global Frame) 계산
        # 회전 변환
        marker_global_x = x_ee + (z_cam * math.cos(phi) - x_cam * math.sin(phi))
        marker_global_y = y_ee + (z_cam * math.sin(phi) + x_cam * math.cos(phi))

        # --- [개선 포인트 1] 목표 지점 생성 방식 변경 ---
        # 기존: 현재 바라보는 방향(phi)의 반대편으로 후퇴 -> 카메라 흔들림에 취약하여 링크1이 흔들림
        # 변경: 원점(0,0)에서 마커를 잇는 직선 상에서 거리를 유지하도록 설정
        #      이렇게 하면 로봇이 팔을 '뻗거나 당기는' 동작을 주력으로 하게 되어 q1, q2가 고루 움직임
        
        # 원점부터 마커까지의 거리와 각도
        dist_origin_to_marker = math.sqrt(marker_global_x**2 + marker_global_y**2)
        angle_origin_to_marker = math.atan2(marker_global_y, marker_global_x)

        # 목표 위치: 마커 방향을 바라보되, 마커보다 D_MAINTAIN만큼 덜 간 위치
        # (만약 마커가 너무 가까우면 뒤로 물러나야 하므로 로직은 그대로 유효)
        target_dist = dist_origin_to_marker - D_MAINTAIN
        
        target_x = target_dist * math.cos(angle_origin_to_marker)
        target_y = target_dist * math.sin(angle_origin_to_marker)

        # 4. 역기구학 (Inverse Kinematics) - 최적 해 선택
        try:
            q1_target, q2_target = self.inverse_kinematics_optimized(target_x, target_y)
            
            # 결과 발행
            out_msg = Float64MultiArray()
            out_msg.data = [q1_target, q2_target]
            self.angle_pub.publish(out_msg)
            
        except ValueError:
            self.get_logger().warn("Target unreachable")

    def inverse_kinematics_optimized(self, x, y):
        """
        가능한 두 가지 해(Elbow Up, Elbow Down)를 모두 계산한 뒤,
        현재 로봇 자세에서 움직임이 더 적은 쪽을 선택합니다.
        """
        dist_sq = x**2 + y**2
        cos_q2 = (dist_sq - L1**2 - L2**2) / (2 * L1 * L2)

        # 도달 불가능 예외 처리
        if cos_q2 > 1.0: cos_q2 = 1.0
        elif cos_q2 < -1.0: cos_q2 = -1.0

        # 해 1: Elbow Down (기존 방식)
        q2_sol1 = -math.acos(cos_q2)
        k1_1 = L1 + L2 * math.cos(q2_sol1)
        k2_1 = L2 * math.sin(q2_sol1)
        q1_sol1 = math.atan2(y, x) - math.atan2(k2_1, k1_1)

        # 해 2: Elbow Up (반대 방향)
        q2_sol2 = math.acos(cos_q2) # 부호 반대
        k1_2 = L1 + L2 * math.cos(q2_sol2)
        k2_2 = L2 * math.sin(q2_sol2)
        q1_sol2 = math.atan2(y, x) - math.atan2(k2_2, k1_2)

        # Radian -> Degree 변환
        deg_q1_1, deg_q2_1 = math.degrees(q1_sol1), math.degrees(q2_sol1)
        deg_q1_2, deg_q2_2 = math.degrees(q1_sol2), math.degrees(q2_sol2)

        # --- [개선 포인트 2] 비용 함수(Cost Function)를 통한 해 선택 ---
        # 현재 각도와의 차이(절대값 합)가 더 작은 해를 선택
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