#include <chrono>
#include <memory>

#include "cv_bridge/cv_bridge.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/header.hpp"
#include <opencv2/opencv.hpp>
#include <stdio.h>

// for Size
#include <opencv2/core/types.hpp>
// for CV_8UC3
#include <opencv2/core/hal/interface.h>
// for compressing the image
#include <image_transport/image_transport.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std::chrono_literals;

class CameraPublisher : public rclcpp::Node 
{
  public:
    CameraPublisher() : Node("Camera_pub2"), count_(0) 
    {
        publisher_ =
            this->create_publisher<sensor_msgs::msg::Image>("topic", 10);
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100), std::bind(&CameraPublisher::timer_callback, this));
        RCLCPP_INFO(this->get_logger(), "camera publisher cpp node");
    }

  private:
    void timer_callback() {
        cv_bridge::CvImagePtr cv_ptr;

        cv::Mat img(cv::Size(1280, 720), CV_8UC3);
        cv::randu(img, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

        sensor_msgs::msg::Image::SharedPtr msg =
            cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", img)
                .toImageMsg();

        publisher_->publish(*msg.get());
        std::cout << "Published!" << std::endl;
    }

    rclcpp::TimerBase::SharedPtr timer_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;

    size_t count_;
};

int main(int argc, char *argv[]) {
    printf("Starting...");
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CameraPublisher>();
    //rclcpp::spin(std::make_shared<CameraPublisher>());
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}