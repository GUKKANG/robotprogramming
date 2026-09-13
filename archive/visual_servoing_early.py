import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64MultiArray
from twolink_msgs.msg import CMD
import math

# 로봇 링크 길이 (mm)
L1 = 143.0
L2 = 102.0
D_MAINTAIN = 250.0  # 유지 거리

class VisualServoingNode(Node):
    def __init__(self):
        super().__init__('visual_servoing_node')
        
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
        # 1. 카메라 데이터 수신 (m -> mm)
        # 카메라 좌표계: X(Right), Y(Down), Z(Forward)
        x_cam = msg.position.x * 1000.0
        z_cam = msg.position.z * 1000.0

        # 2. 현재 로봇 상태 계산 (Forward Kinematics)
        t1 = math.radians(self.curr_q1)
        t2 = math.radians(self.curr_q2)
        
        x_ee = L1 * math.cos(t1) + L2 * math.cos(t1 + t2)
        y_ee = L1 * math.sin(t1) + L2 * math.sin(t1 + t2)
        phi = t1 + t2 # 현재 End-Effector의 방향 (Global)

        # 3. 마커의 전역 좌표(Global Frame) 추정
        # 카메라가 End-Effector 끝에 로봇 진행 방향(phi)을 바라보고 있다고 가정
        # Robot Frame: X_local (Link direction), Y_local (Left +90deg)
        # Cam Frame: Z_cam (Forward) -> X_local
        #            X_cam (Right)    -> -Y_local (Right is -Left)
        
        # 변환 행렬 (Rotation matrix from Local to Global)
        # [ cos(phi)  -sin(phi) ]
        # [ sin(phi)   cos(phi) ]
        
        # Local Vector: 
        # local_x = z_cam (앞으로 가는 거리)
        # local_y = -x_cam (오른쪽으로 가는 거리 -> 로봇 기준 왼쪽의 반대)
        
        # 따라서:
        # global_dx = local_x * cos(phi) - local_y * sin(phi)
        #           = z_cam * cos(phi) - (-x_cam) * sin(phi)
        #           = z_cam * cos(phi) + x_cam * sin(phi)
        
        # 기존 코드는 -x_cam * sin(phi) 였으므로 좌우가 반전되어 있었을 가능성이 큼
        
        marker_global_x = x_ee + (z_cam * math.cos(phi) + x_cam * math.sin(phi))
        marker_global_y = y_ee + (z_cam * math.sin(phi) - x_cam * math.cos(phi))

        # 4. 제어 목표 생성 (Virtual Link Approach)
        # 목표: 로봇이 마커를 바라보면서(Orientation), 거리는 D_MAINTAIN 유지
        # 이를 위해 가상의 L2 링크 길이를 (L2 + D_MAINTAIN)으로 설정하고,
        # 마커의 좌표에 도달하도록 IK를 풉니다.
        # 이렇게 하면 실제 로봇은 마커를 향해 뻗으면서 D_MAINTAIN 만큼 떨어진 곳에 멈춥니다.
        
        L2_virtual = L2 + D_MAINTAIN
        
        try:
            q1_target, q2_target = self.inverse_kinematics_virtual(marker_global_x, marker_global_y, L1, L2_virtual)
            
            # 결과 발행
            out_msg = Float64MultiArray()
            out_msg.data = [q1_target, q2_target]
            self.angle_pub.publish(out_msg)
            
        except ValueError:
            self.get_logger().warn("Target unreachable (Too far or too close)")

    def inverse_kinematics_virtual(self, x, y, l1, l2_virt):
        """
        가상 링크 길이를 이용한 IK 계산
        """
        dist_sq = x**2 + y**2
        
        # 코사인 법칙
        cos_q2 = (dist_sq - l1**2 - l2_virt**2) / (2 * l1 * l2_virt)
        
        if cos_q2 > 1.0: 
            raise ValueError("Too Far")
        if cos_q2 < -1.0: 
            raise ValueError("Too Close") # 3각형 형성 불가

        # 해 선택: 현재 각도와 가까운 쪽 (Elbow Up/Down)
        # 일반적으로 마커를 바라보는 자연스러운 자세는 Elbow Down 인 경우가 많으나 상황에 따라 다름
        
        # Solution 1: Elbow Down
        q2_1 = -math.acos(cos_q2)
        k1_1 = l1 + l2_virt * math.cos(q2_1)
        k2_1 = l2_virt * math.sin(q2_1)
        q1_1 = math.atan2(y, x) - math.atan2(k2_1, k1_1)
        
        # Solution 2: Elbow Up
        q2_2 = math.acos(cos_q2)
        k1_2 = l1 + l2_virt * math.cos(q2_2)
        k2_2 = l2_virt * math.sin(q2_2)
        q1_2 = math.atan2(y, x) - math.atan2(k2_2, k1_2)

        deg_q1_1, deg_q2_1 = math.degrees(q1_1), math.degrees(q2_1)
        deg_q1_2, deg_q2_2 = math.degrees(q1_2), math.degrees(q2_2)

        # Cost 계산 (현재 위치에서 이동량 최소화)
        cost1 = abs(deg_q1_1 - self.curr_q1) + abs(deg_q2_1 - self.curr_q2)
        cost2 = abs(deg_q1_2 - self.curr_q1) + abs(deg_q2_2 - self.curr_q2)

        if cost1 < cost2:
            return deg_q1_1, deg_q2_1
        else:
            return deg_q1_2, deg_q2_2

def main(args=None):
    rclpy.init(args=args)
    node = VisualServoingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
