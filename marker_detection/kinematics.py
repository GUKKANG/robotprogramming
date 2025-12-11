import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64MultiArray
from twolink_msgs.msg import CMD
import math
import time

# 로봇 링크 길이 (mm)
L1 = 143.0
L2 = 102.0
D_MAINTAIN = 400.0  # 유지 거리

class VisualServoingNode(Node):
    def __init__(self):
        super().__init__('visual_servoing_node')
        
        # 현재 로봇의 각도 상태 (피드백)
        self.curr_q1 = 0.0
        self.curr_q2 = 0.0
        
        # 마커 인식 상태 관리
        self.last_aruco_time = time.time()
        self.is_searching = False
        self.search_q2_angle = 0.0
        
        # 스무딩을 위한 이전 목표 각도 저장
        self.prev_q1_target = None
        self.prev_q2_target = None
        # 매 프레임 현재 위치를 기준으로 목표를 재설정하므로, alpha를 낮게 설정하여 부드럽게 이동
        self.alpha_q1 = 1.0
        self.alpha_q2 = 1.0
        
        # 로봇 상태 수신 여부 확인
        self.robot_state_received = False

        # Subscriber
        self.create_subscription(Pose, '/aruco/pose', self.aruco_callback, 10)
        self.create_subscription(CMD, '/des_value', self.joint_state_callback, 10)

        # Publisher
        self.angle_pub = self.create_publisher(Float64MultiArray, '/target_angles', 10)
        
        # 마커 미감지 확인 타이머 (0.1초 주기)
        self.create_timer(0.1, self.check_marker_timeout)

    def joint_state_callback(self, msg):
        # 하드웨어/시뮬레이션 상태 업데이트
        self.curr_q1 = math.degrees(msg.q1des)
        self.curr_q2 = math.degrees(msg.q2des)
        self.robot_state_received = True

    def check_marker_timeout(self):
        # 6.0초 이상 마커 인식이 안 되면 검색 모드 진입
        if time.time() - self.last_aruco_time > 10.0:
            current_time = time.time()
            
            if not self.is_searching:
                self.is_searching = True
                self.search_start_angle = self.curr_q2
                self.search_q2_target = self.curr_q2
                self.last_step_time = current_time - 2.5 # 바로 첫 스텝 시작하도록 설정
                self.step_count = 0
                self.get_logger().info("Marker lost for 5s. Starting search mode (Step 90 deg every 2.5s)...")
            # 3초마다 90도씩 회전
            if current_time - self.last_step_time > 3.0:
                # 4번(360도)까지만 회전하고 멈춤
                if self.step_count < 4:
                    self.step_count += 1
                    self.last_step_time = current_time
                    
                    # 90도씩 증가 (반시계 방향)
                    self.search_q2_target += 90.0
                    self.get_logger().info(f"Search step {self.step_count}/4: Rotating +90 deg")
                
            q2_target = self.search_q2_target
            
            # Link 1은 현재 상태 유지, Link 2만 회전
            q1_target = self.curr_q1
            
            # 결과 발행
            out_msg = Float64MultiArray()
            out_msg.data = [q1_target, q2_target]
            self.angle_pub.publish(out_msg)

    def aruco_callback(self, msg):
        # 로봇 상태를 아직 모르면 계산 스킵 (초기 0도 복귀 방지)
        if not self.robot_state_received:
            self.get_logger().warn("Waiting for robot state...")
            return

        # 마커 인식됨 -> 시간 갱신 및 검색 모드 해제
        self.last_aruco_time = time.time()
        
        # 항상 현재 위치를 기준으로 이동하도록 이전 목표값 초기화
        # 이렇게 하면 마커를 다시 찾았을 때 이전의 오래된 목표값(초기 위치 등)으로 튀는 것을 방지함
        self.prev_q1_target = None
        self.prev_q2_target = None

        if self.is_searching:
            self.is_searching = False
            self.get_logger().info("Marker found! Resuming visual servoing.")

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

        # 4. 제어 목표 생성 (단계별 제어: 거리 우선 -> 각도 보정)
        # 마커와의 거리 오차와 각도 오차를 분리하여 처리
        
        # 현재 마커까지의 거리 (카메라 기준 Z축)
        dist_error = z_cam - D_MAINTAIN
        
        # 현재 마커의 좌우 편차 (카메라 기준 X축)
        # x_cam이 0이 되도록 하는 것이 목표
        
        # 전략:
        # 1. 거리가 너무 멀거나 가까우면(오차 > 50mm) 거리 조절에 집중 (Virtual Link 사용)
        # 2. 거리가 어느 정도 맞으면(오차 < 50mm) 좌우 각도(Orientation) 정밀 보정
        
        # 기본적으로 Virtual Link 방식은 거리와 각도를 동시에 맞추려 함
        # 하지만 마지막에 확 트는 현상은 IK 해가 불안정하거나, 두 목표가 충돌하기 때문일 수 있음
        
        # 개선된 Virtual Link 길이 가변 적용
        # 목표 거리에 거의 도달했다면, Virtual Link 길이를 고정하지 않고 미세 조정
        
        L2_virtual = L2 + D_MAINTAIN
        
        # 거리 오차가 크면 거리 맞추기 우선 (기존 로직)
        if abs(dist_error) > 50.0:
             pass # L2_virtual 그대로 사용
             
        # 거리가 거의 맞았으면, 좌우 정렬에 집중
        # 이때는 Virtual Link 길이를 현재 거리(z_cam)에 맞춰서 IK를 풀면
        # 로봇이 앞뒤로 움직이지 않고 제자리에서 회전만 하여 좌우를 맞출 수 있음
        else:
             # 현재 거리를 유지하면서 회전만 하도록 가상 링크 길이 조정
             # 단, 너무 급격한 변화를 막기 위해 서서히 변경하거나, 
             # 단순히 현재 z_cam을 목표 거리로 간주
             L2_virtual = L2 + z_cam

        try:
            q1_target, q2_target = self.inverse_kinematics_virtual(marker_global_x, marker_global_y, L1, L2_virtual)
            
            # 초기화 (첫 실행 시)
            if self.prev_q1_target is None:
                self.prev_q1_target = self.curr_q1
                self.prev_q2_target = self.curr_q2

            # Low-Pass Filter (스무딩) 적용
            # 급격한 목표 각도 변화를 억제하여 부드럽게 추종하도록 함
            q1_target = self.alpha_q1 * q1_target + (1 - self.alpha_q1) * self.prev_q1_target
            q2_target = self.alpha_q2 * q2_target + (1 - self.alpha_q2) * self.prev_q2_target
            
            # 값 업데이트
            self.prev_q1_target = q1_target
            self.prev_q2_target = q2_target

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
            # raise ValueError("Too Far")
            # 목표가 너무 멀 경우: 팔을 마커 방향으로 쭉 뻗음 (q2=0)
            # 링크가 일자가 되도록 하여 마커를 향하게 함
            q1 = math.atan2(y, x)
            q2 = 0.0
            return math.degrees(q1), math.degrees(q2)

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

        # 현재 자세(Elbow Up/Down) 유지 선호 (급격한 자세 변경 방지)
        # Solution 1: q2 <= 0 (Elbow Down)
        # Solution 2: q2 >= 0 (Elbow Up)
        
        # 현재 q2가 음수(Elbow Down)인데 양수 솔루션(Elbow Up)을 선택하려 하면 페널티
        if self.curr_q2 < -1.0: # 0 근처가 아닐 때 확실한 자세 유지
            cost2 += 100.0
            
        # 현재 q2가 양수(Elbow Up)인데 음수 솔루션(Elbow Down)을 선택하려 하면 페널티
        elif self.curr_q2 > 1.0:
            cost1 += 100.0

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