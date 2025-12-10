#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from twolink_msgs.msg import CMD
from std_msgs.msg import Float64MultiArray
import math

class PIDController:
    def __init__(self, kp, ki, kd, dt, max_vel):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.max_vel = max_vel  # 최대 속도 제한 (rad/s)
        
        self.prev_error = 0.0
        self.integral = 0.0

    
    def update(self, target, current):
        error = target - current
        
        # P term
        p_out = self.kp * error
        
        # I term
        self.integral += error * self.dt
        i_out = self.ki * self.integral
        
        # D term
        derivative = (error - self.prev_error) / self.dt
        d_out = self.kd * derivative
        
        # Total output (Velocity)
        output = p_out + i_out + d_out
        
        # 속도 제한 (Clamping)
        if output > self.max_vel:
            output = self.max_vel
        elif output < -self.max_vel:
            outpuselft = -self.max_vel
            
        self.prev_error = error
        return output


class RobotDriver(Node):
    def __init__(self):
        super().__init__('robot_driver')
       
        # Publisher
        self.publisher = self.create_publisher(CMD, 'des_value', 10)
        
        # Subscriber
        self.create_subscription(Float64MultiArray, '/target_angles', self.listener_callback, 10)

        # 목표 각도
        self.q1des = 0.0
        self.q2des = 0.0

        # 현재 로봇 각도 (시뮬레이션 상의 현재 위치)
        self.curr_q1 = 0.0
        self.curr_q2 = 0.0
        
        # 제어 주기
        self.dt = 0.02  # 50Hz

        # PID 제어기 설정 (Kp, Ki, Kd, dt, max_vel)
        # Kp: 반응 속도 (클수록 빠름)
        # Ki: 오차 누적 보정 (작게 설정)
        # Kd: 급격한 변화 억제 (진동 방지)
        # max_vel: 0.5 rad/s (약 28도/초)
        self.pid_q1 = PIDController(kp=1.5, ki=0.01, kd=0.05, dt=self.dt, max_vel=0.5)
        self.pid_q2 = PIDController(kp=1.5, ki=0.01, kd=0.05, dt=self.dt, max_vel=0.5)

        # 50Hz 주기로 명령 발행
        self.create_timer(self.dt, self.publish_cmd)

    def listener_callback(self, msg):
        if len(msg.data) >= 2:
            self.q1des = -math.radians(msg.data[0])
            self.q2des = math.radians(msg.data[1])

    def publish_cmd(self):
        # PID를 통해 목표 속도 계산
        vel_q1 = self.pid_q1.update(self.q1des, self.curr_q1)
        vel_q2 = self.pid_q2.update(self.q2des, self.curr_q2)
        
        # 현재 위치self 업데이트 (적분: 위치 += 속도 * 시간)
        self.curr_q1 += vel_q1 * self.dt
        self.curr_q2 += vel_q2 * self.dt

        msg = CMD()
        msg.xdes = 0.0
        msg.ydes = 0.0
        msg.q1des = self.curr_q1
        msg.q2des = self.curr_q2
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = RobotDriver()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()