#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from example_interfaces.msg import String
from example_interfaces.msg import Int32


class RobotNewsStationNode(Node):
    def __init__(self):
        super().__init__("robot_news_station")

        self.robot_name_ = "C3PO"
        self.counter_ = 0
        self.publisher_ = self.create_publisher(String, "robot_news", 10)
        self.timer_ = self.create_timer(2, self.publish_news)
        self.get_logger().info("Robot News Station has been started")
        self.publisher2_ = self.create_publisher(Int32, "counter", 10)

    def publish_news(self):
        print('publish_news')
        msg = String()
        msg.data = "Hi, this is " + \
            str(self.robot_name_) + " from the robot news station."
        self.publisher_.publish(msg)
        
        count = Int32()
        count.data = self.counter_
        self.counter_ += 1
        self.get_logger().info(f"Counter: {self.counter_}")
        self.publisher2_.publish(count)


def main(args=None):
    rclpy.init(args=args)
    node = RobotNewsStationNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
