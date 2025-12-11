import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

class ArucoDetectorNode_Realsense(Node):
    def __init__(self):
        super().__init__('aruco_detector_node_realsense')
        self.bridge = CvBridge()

        # 카메라 파라미터 저장
        self.camera_matrix = None
        self.dist_coeffs = None

        # 아루코 설정 (Original: DICT_6X6_50)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # 인식률 향상을 위한 파라미터 튜닝
        self.aruco_params.adaptiveThreshWinSizeMin = 3
        self.aruco_params.adaptiveThreshWinSizeMax = 23
        self.aruco_params.adaptiveThreshWinSizeStep = 10
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        
        self.marker_length = 0.06  # m
        
        # 광학 흐름(Optical Flow) 추적을 위한 변수
        self.prev_gray = None
        self.prev_corners = None
        self.tracking_active = False
        
        # 이미지 발행 주기 조절을 위한 카운터
        self.frame_count = 0
        self.pub_rate = 5 # 5프레임마다 1번씩 이미지 발행 (부하 감소)

        # Subscriber (QoS: Best Effort for low latency)
        self.image_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, qos_profile_sensor_data)
        self.cam_info_sub = self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10)

        # Publisher
        self.image_pub = self.create_publisher(Image, '/aruco/image', 10)
        self.pose_pub = self.create_publisher(Pose, '/aruco/pose', 10)        

        self.get_logger().info('ArucoDetectorNode start ~~ (Realsense Optimized)')

    def camera_info_callback(self, msg): # CameraInfo를 받으면 실행되는 callback 함수
        if self.camera_matrix is None:
            K = np.array(msg.k).reshape(3, 3)
            D = np.array(msg.d, dtype=np.float64)
            self.camera_matrix = K
            self.dist_coeffs = D
            self.get_logger().info('CameraInfo OK ~~')

    def image_callback(self, msg): # Image를 받는면 실행되는 callback 함수
        if self.camera_matrix is None or self.dist_coeffs is None:
            return

        # ROS Image -> OpenCV 이미지
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 아루코 마커 검출
        if hasattr(cv2.aruco, 'ArucoDetector'):
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            corners, ids, rejected = detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        detected = False
        current_corners = None
        current_rvec = None
        current_tvec = None

        # 1. 마커가 직접 인식된 경우
        if ids is not None and len(ids) > 0:
            detected = True
            self.tracking_active = True
            
            # 첫 번째 마커 기준
            current_corners = corners[0] # (1, 4, 2) shape
            
            # Pose 추정
            rvec, tvec, quaternion = self.my_estimatePoseSingleMarkers(corners, ids)
            
            if rvec is not None:
                current_rvec = rvec
                current_tvec = tvec
                self.publish_pose(tvec, quaternion)
                
                # 추적을 위해 현재 상태 저장
                self.prev_gray = gray
                self.prev_corners = current_corners.reshape(-1, 1, 2) # (4, 1, 2) for optical flow

        # 2. 마커 인식 실패 시 -> 광학 흐름(Optical Flow)으로 추적 시도
        elif self.tracking_active and self.prev_gray is not None and self.prev_corners is not None:
            # Lucas-Kanade Optical Flow 계산
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_corners, None, winSize=(21, 21), maxLevel=3)
            
            # 4개 코너가 모두 잘 추적되었는지 확인
            if p1 is not None and st is not None and np.sum(st) == 4:
                # 추적된 코너로 Pose 추정 (solvePnP)
                # p1 shape: (4, 1, 2) -> (1, 4, 2)로 변환하여 solvePnP에 전달
                tracked_corners = [p1.reshape(1, 4, 2)]
                
                # 가상의 id (0) 부여하여 Pose 계산
                rvec, tvec, quaternion = self.my_estimatePoseSingleMarkers(tracked_corners, np.array([[0]]))
                
                if rvec is not None:
                    detected = True
                    current_corners = tracked_corners[0]
                    current_rvec = rvec
                    current_tvec = tvec
                    
                    self.get_logger().info("Tracking marker via Optical Flow...")
                    self.publish_pose(tvec, quaternion)
                    
                    # 다음 프레임을 위해 업데이트
                    self.prev_gray = gray
                    self.prev_corners = p1
            else:
                self.tracking_active = False # 추적 실패

        # 이미지 발행 (부하 감소)
        if self.frame_count % self.pub_rate == 0:
            # 시각화
            if detected and current_rvec is not None:
                # 마커 테두리 그리기
                cv2.polylines(frame, [current_corners.astype(np.int32)], True, (0, 255, 0), 2)
                # 좌표축 그리기
                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, current_rvec, current_tvec, self.marker_length * 0.5)
                
                if not (ids is not None and len(ids) > 0):
                    cv2.putText(frame, "Tracking", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            out_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            out_msg.header = msg.header
            self.image_pub.publish(out_msg)
            
        self.frame_count += 1

    def publish_pose(self, tvec, quaternion):
        pose_msg = Pose()
        pose_msg.position.x = float(tvec[0])
        pose_msg.position.y = float(tvec[1])
        pose_msg.position.z = float(tvec[2])
        pose_msg.orientation.x = float(quaternion[0])
        pose_msg.orientation.y = float(quaternion[1])
        pose_msg.orientation.z = float(quaternion[2])
        pose_msg.orientation.w = float(quaternion[3])
        self.pose_pub.publish(pose_msg)

    def my_estimatePoseSingleMarkers(self, corners, ids):
        # 마커 좌표
        object_points = np.array([
            [-self.marker_length / 2, self.marker_length / 2, 0],   # 좌상단 좌표
            [self.marker_length / 2, self.marker_length / 2, 0],    # 우상단 좌표
            [self.marker_length / 2, -self.marker_length / 2, 0],   # 우하단 좌표
            [-self.marker_length / 2, -self.marker_length / 2, 0]   # 좌하단 좌표
        ], dtype=np.float32)

        if ids is not None:
            # 첫 번째 마커만 사용
            corner = corners[0][0]

            # solvePnP를 사용하여 3차원 위치 추정
            success, rvec, tvec = cv2.solvePnP(object_points, corner, self.camera_matrix , self.dist_coeffs)

            if success:
                R = Rotation.from_rotvec(rvec.flatten())
                quaternion = R.as_quat() # xyzw
                return rvec, tvec, quaternion
            
        return None, None, None

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectorNode_Realsense()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
