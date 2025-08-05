import rclpy
import time
import os
import tf
from rclpy.node import Node

from sensor_msgs.msg import Range
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Range

class RealEnv(Node):
    def __init__(self,name,environment_dim):
        super().__init__(name)
        self.environment_dim = environment_dim
        self.done = False
        self.epoches = 0
        self.angle_limit = 0.5236   #30度
        self.pump_output = Float32MultiArray()
        self.pump_control = 0.0

        self.min_distance = 0.0
        self.max_distance = 0.0
        self.pitch = 0.0   #俯仰角

        #发布话题
        self.velocity_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pump_publisher = self.create_publisher(Float32MultiArray, '/water_pump', 10)

        #订阅话题
        #self.odom_subscriber = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.imu_subcriber = self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.sonar_subscriber = self.create_subscription(Range, '/range_sensor', self.sonar_callback, 10)


    def odom_callback(self, msg):
        pass

    def imu_callback(self, msg):
        #将四元数转成欧拉角
        (row,pitch,yaw) = tf.transformations.euler_from_quaternion((msg.orientation.x,msg.orientation.y,msg.orientation.z,msg.orientation.w))
        self.pitch = pitch

    def sonar_callback(self, msg):
        self.min_distance = msg.min_range
        self.max_distance = msg.max_range
    
    def is_overturn(self):
        if self.pitch > self.angle_limit:
            return True,True
        else:
            return False,False

    def get_state(self):
        self.epoches += 1
        overturn = False
        overturn,self.done = self.is_overturn()
        if self.epoches >500:
            self.done = True
        min_distance = self.min_distance
        max_distance = self.max_distance
        state = self.pitch
        return state,overturn,self.done
    
    def reset(self):
        self.done = False
        self.epoches = 0
        self.sonar_data = []
        self.pitch = 0.0
        return self.get_state()
    
    def action_publish(self,action):
        self.pump_output.data = [self.pump_control,action]
        self.pump_publisher.publish(self.pump_output)

