import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64MultiArray
from twolink_msgs.msg import CMD
import math

# 로봇 링크 길이 (mm 단위로 계산 후 미터 변환 등 주의, 여기선 mm 기준)
L1 = 143.0
L2 = 102.0
D_MAINTAIN = 150.0  # 마커와 유지할 거리 (mm)

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
        # 현재 로봇이 향하고 있는 각도 업데이트 (Radian -> Degree)
        self.curr_q1 = math.degrees(msg.q1des)
        self.curr_q2 = math.degrees(msg.q2des)

    def aruco_callback(self, msg):
        # 1. 카메라 좌표계 상의 마커 위치 (m -> mm)
        # 카메라는 End-Effector에 달려있고, x축이 오른쪽, z축이 정면이라고 가정
        x_cam = msg.position.x * 1000.0
        z_cam = msg.position.z * 1000.0

        # 2. 현재 로봇의 End-Effector 위치 및 각도 계산 (Forward Kinematics)
        t1 = math.radians(self.curr_q1)
        t2 = math.radians(self.curr_q2)
        
        # 현재 End-Effector의 전역 좌표
        x_ee = L1 * math.cos(t1) + L2 * math.cos(t1 + t2)
        y_ee = L1 * math.sin(t1) + L2 * math.sin(t1 + t2)
        phi = t1 + t2  # End-Effector의 현재 절대 각도

        # 3. 마커의 전역 좌표(Global Frame) 계산
        # 로봇 팔 끝에서 phi만큼 회전된 상태에서 카메라 좌표계 값을 더함
        # 카메라 좌표계: 전방이 z, 우측이 x라고 가정했을 때 회전 변환 적용
        # (2D 평면 상에서 카메라 z축이 로봇 팔 연장선, x축이 직각 방향이라 가정)
        
        # 회전 행렬을 통한 변환 (2D)
        # Global X = x_ee + (z_cam * cos(phi) - x_cam * sin(phi))
        # Global Y = y_ee + (z_cam * sin(phi) + x_cam * cos(phi))
        marker_global_x = x_ee + (z_cam * math.cos(phi) - x_cam * math.sin(phi))
        marker_global_y = y_ee + (z_cam * math.sin(phi) + x_cam * math.cos(phi))

        # 4. 목표 위치 설정 (마커로부터 D_MAINTAIN 만큼 떨어진 위치)
        # 마커를 바라보는 방향(phi)의 반대 방향으로 거리 유지
        target_x = marker_global_x - D_MAINTAIN * math.cos(phi)
        target_y = marker_global_y - D_MAINTAIN * math.sin(phi)

        # 5. 역기구학 (Inverse Kinematics)
        try:
            q1_target, q2_target = self.inverse_kinematics(target_x, target_y)
            
            # 결과 발행
            out_msg = Float64MultiArray()
            # 좌우 이동(q1)만 부호 반전
            out_msg.data = [-q1_target, q2_target]
            self.angle_pub.publish(out_msg)
            
        except ValueError:
            self.get_logger().warn("Target unreachable")

    def inverse_kinematics(self, x, y):
        # 코사인 법칙 이용
        dist_sq = x**2 + y**2
        cos_q2 = (dist_sq - L1**2 - L2**2) / (2 * L1 * L2)
        
        # 도달 불가능 예외 처리
        if cos_q2 > 1.0: cos_q2 = 1.0
        elif cos_q2 < -1.0: cos_q2 = -1.0
            
        q2 = -math.acos(cos_q2) # Elbow down (일반적인 형태)
        
        k1 = L1 + L2 * math.cos(q2)
        k2 = L2 * math.sin(q2)
        q1 = math.atan2(y, x) - math.atan2(k2, k1)
        
        return math.degrees(q1), math.degrees(q2)

def main(args=None):
    rclpy.init(args=args)
    node = KinematicsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()