#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from twolink_msgs.msg import CMD
from std_msgs.msg import Float64MultiArray
import math

class RobotDriver(Node):
    def __init__(self):
        super().__init__('robot_driver')
       
        # Publisher: 로봇에게 명령 전달
        self.publisher = self.create_publisher(CMD, 'des_value', 10)
        
        # Subscriber: 계산된 목표 각도 수신 (단위: 도)
        self.create_subscription(Float64MultiArray, '/target_angles', self.listener_callback, 10)

        self.q1des = 0.0
        self.q2des = 0.0

        # 50Hz 주기로 명령 발행
        self.create_timer(0.02, self.publish_cmd)

    def listener_callback(self, msg):
        if len(msg.data) >= 2:
            # 들어오는 값은 Degree, 나가는 값은 Radian
            self.q1des = math.radians(msg.data[0])
            self.q2des = math.radians(msg.data[1])

    def publish_cmd(self):
        msg = CMD()
        msg.xdes = 0.0
        msg.ydes = 0.0
        msg.q1des = self.q1des
        msg.q2des = self.q2des
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = RobotDriver()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()