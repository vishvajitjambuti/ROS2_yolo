#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from example_interfaces.msg import String
from example_interfaces.msg import Int32


class SmartphoneNode(Node):
    def __init__(self):
        super().__init__("smartphone")
        self.subscriber_ = self.create_subscription(
            String, "robot_news", self.callback_robot_news, 1)
        
        self.subscriber2_ = self.create_subscription(
                String, "counter", self.counter_callback, 1)
        
        self.get_logger().info("Smartphone has been started.")

    def callback_robot_news(self, msg):
        print('callback_robot_news')
        self.get_logger().info(msg.data)

    def counter_callback(self, msg):
        print(f'counter_callback {msg.data}')
        

def main(args=None):
    rclpy.init(args=args)
    node = SmartphoneNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
