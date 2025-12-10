import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__('aruco_detector_node')
        self.bridge = CvBridge()
        
        # 카메라 파라미터 (CameraInfo 콜백에서 업데이트)
        self.camera_matrix = None
        self.dist_coeffs = None

        # 마커 설정 (6x6, ID 0~49 사용 가정)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.marker_length = 0.06  # 마커 실제 크기 (미터 단위)

        # Subscriber
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.info_callback, 10)

        # Publisher
        self.pose_pub = self.create_publisher(Pose, '/aruco/pose', 10)
        self.image_pub = self.create_publisher(Image, '/aruco/image', 10)

    def info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)

    def image_callback(self, msg):
        if self.camera_matrix is None: return

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # 첫 번째 발견된 마커 기준 Pose 추정
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[0], self.marker_length, self.camera_matrix, self.dist_coeffs)
            
            # 좌표축 그리기
            cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.03)

            # Pose 메시지 생성 및 발행
            pose_msg = Pose()
            pose_msg.position.x = float(tvec[0][0])
            pose_msg.position.y = float(tvec[0][1])
            pose_msg.position.z = float(tvec[0][2])
            
            # 회전 정보 (Rodrigues vector -> Quaternion)
            rot_mat, _ = cv2.Rodrigues(rvec)
            quat = Rotation.from_matrix(rot_mat).as_quat()
            pose_msg.orientation.x = quat[0]
            pose_msg.orientation.y = quat[1]
            pose_msg.orientation.z = quat[2]
            pose_msg.orientation.w = quat[3]

            self.pose_pub.publish(pose_msg)

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, 'bgr8'))

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()