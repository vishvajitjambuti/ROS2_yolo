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
#include "include/camera.h"

class UndistortImagePublisher :  public rclcpp::Node
{
    public:
        UndistortImagePublisher() : Node("UndistortImagePublisher")
        {
            this->declare_parameter("camera_id", "rtsp://192.168.73.11:8554/h264");
            camera_id_ = this->get_parameter("camera_id").as_string();       
            
            publisher_ = this->create_publisher<sensor_msgs::msg::Image>("image_raw", 10);
            timer_ = this->create_wall_timer(std::chrono::milliseconds(10), std::bind(&UndistortImagePublisher::timer_callback, this));
                
            RCLCPP_INFO(this->get_logger(), "camera publisher cpp node");
            
            cap.open(camera_id_);
                
    

        }
       

        private:      
            void timer_callback() 
            {
                // this-> capture >> this->img;
                cap >> this->img;
                cv_bridge::CvImagePtr cv_ptr;
                if (!this->cap.isOpened()) {
                    RCLCPP_ERROR(this->get_logger(), "Could not read image from camera");

                    }
                RCLCPP_INFO(this->get_logger(), "Image read from camera");

                sensor_msgs::msg::Image::SharedPtr msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8",  this->img).toImageMsg();

                publisher_->publish(*msg.get());
                std::cout << "Published!" << std::endl;
                   

                     
                
                }

        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
        rclcpp::TimerBase::SharedPtr timer_;
        cv::VideoCapture cap;
        cv::Mat img;
        std:: string camera_id_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<UndistortImagePublisher>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}