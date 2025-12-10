import rclpy
from rclpy.node import Node

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

        # -----------------------------------------------------------
        # [수정] 6x6 마커 설정 (속도 최적화를 위해 불필요한 튜닝 제거)
        # -----------------------------------------------------------
        # 6x6 마커 딕셔너리 (마커 생성시 사용한 것과 일치해야 함. 보통 50 or 250)
        try:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
        except AttributeError:
            self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)

        # 파라미터 설정 (OpenCV 버전 호환성 처리: DetectorParameters_create vs DetectorParameters)
        try:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
        except AttributeError:
            try:
                self.aruco_params = cv2.aruco.DetectorParameters()
            except AttributeError:
                self.aruco_params = None
        
        self.marker_length = 0.06  # m

        # Subscriber
        self.image_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.cam_info_sub = self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10)

        # Publisher
        self.image_pub = self.create_publisher(Image, '/aruco/image', 10)
        self.pose_pub = self.create_publisher(Pose, '/aruco/pose', 10)        

        self.get_logger().info('ArucoDetectorNode start ~~ (Realsense 6x6 Optimized with Offset)')

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

        # 아루코 마커 검출 (버전 호환성 + 속도 유지)
        if hasattr(cv2.aruco, 'ArucoDetector') and self.aruco_params is not None:
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            corners, ids, rejected = detector.detectMarkers(gray)
        else:
            if self.aruco_params is not None:
                corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
            else:
                corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict)

        if ids is not None and len(ids) > 0: # 마커 검출에 성공했다면?
            # 마커 그리기
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # 마커 3차원 위치 추정
            rvec, tvec, quaternion = self.my_estimatePoseSingleMarkers(corners, ids)

            if rvec is not None and tvec is not None:
                # 좌표축 그리기 (x: 빨강, y: 초록, z: 파랑)
                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_length * 0.5)

                tvec = tvec.flatten()       

                # 결과 Publish
                pose_msg = Pose()
                pose_msg.position.x = float(tvec[0])
                pose_msg.position.y = float(tvec[1])
                
                # [보정] 측정값이 실제보다 2cm(0.02m) 더 크게 나오므로 뺍니다.
                pose_msg.position.z = float(tvec[2]) - 0.01

                pose_msg.orientation.x = float(quaternion[0])
                pose_msg.orientation.y = float(quaternion[1])
                pose_msg.orientation.z = float(quaternion[2])
                pose_msg.orientation.w = float(quaternion[3])
                self.pose_pub.publish(pose_msg)

                self.get_logger().info(f'{pose_msg.position}, {pose_msg.orientation}')

        # OpenCV -> ROS Image
        out_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        out_msg.header = msg.header
        # 결과 이미지 Publish
        self.image_pub.publish(out_msg)

    def my_estimatePoseSingleMarkers(self, corners, ids):
        # 마커 좌표
        object_points = np.array([
            [-self.marker_length / 2, self.marker_length / 2, 0],   # 좌상단 좌표
            [self.marker_length / 2, self.marker_length / 2, 0],    # 우상단 좌표
            [self.marker_length / 2, -self.marker_length / 2, 0],   # 우하단 좌표
            [-self.marker_length / 2, -self.marker_length / 2, 0]   # 좌하단 좌표
        ], dtype=np.float32)

        if ids is not None:
            # 첫 번째 마커만 사용 (오리지널 코드 로직 유지)
            corner = corners[0][0]

            # solvePnP를 사용하여 3차원 위치 추정
            try:
                success, rvec, tvec = cv2.solvePnP(object_points, corner, self.camera_matrix , self.dist_coeffs)
            except Exception:
                # 일부 버전에서 반환값이 다를 수 있음
                res = cv2.solvePnP(object_points, corner, self.camera_matrix , self.dist_coeffs)
                if isinstance(res, tuple) and len(res) >= 3:
                    success, rvec, tvec = res[0], res[1], res[2]
                else:
                    return None, None, None

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