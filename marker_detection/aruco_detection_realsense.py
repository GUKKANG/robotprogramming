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

        # 아루코 설정 (Original: DICT_4X4_50)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.marker_length = 0.06  # m
        
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
        # OpenCV 버전에 따른 호환성 처리
        if hasattr(cv2.aruco, 'ArucoDetector'):
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            corners, ids, rejected = detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None and len(ids) > 0: # 마커 검출에 성공했다면?
            # 마커 3차원 위치 추정
            rvec, tvec, quaternion = self.my_estimatePoseSingleMarkers(corners, ids)

            if rvec is not None:
                # 결과 Publish (Pose는 매번 발행)
                pose_msg = Pose()
                pose_msg.position.x = float(tvec[0])
                pose_msg.position.y = float(tvec[1])
                pose_msg.position.z = float(tvec[2])
                pose_msg.orientation.x = float(quaternion[0])
                pose_msg.orientation.y = float(quaternion[1])
                pose_msg.orientation.z = float(quaternion[2])
                pose_msg.orientation.w = float(quaternion[3])
                self.pose_pub.publish(pose_msg)

                # self.get_logger().info(f'{pose_msg.position}, {pose_msg.orientation}')
                
                # 이미지 발행은 부하를 줄이기 위해 가끔씩만 수행
                if self.frame_count % self.pub_rate == 0:
                    # 마커 그리기 (디버깅용)
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    # 좌표축 그리기
                    cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_length * 0.5)

        # 이미지 발행 (부하 감소)
        if self.frame_count % self.pub_rate == 0:
            out_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            out_msg.header = msg.header
            self.image_pub.publish(out_msg)
            
        self.frame_count += 1

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

