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


class UndistortImagePublisher :  public rclcpp::Node, public Camera
{
    public:
        UndistortImagePublisher() : Node("UndistortImagePublisher"),  Camera("rtsp://192.168.73.11:8554/jpeg", 1280, 720 )
        {
            publisher_ = this->create_publisher<sensor_msgs::msg::Image>("image_raw", 10);
            timer_ = this->create_wall_timer(std::chrono::milliseconds(10), std::bind(&UndistortImagePublisher::timer_callback, this));
            RCLCPP_INFO(this->get_logger(), "camera publisher cpp node");
            
            
        }

    private:
        void timer_callback(){

            // RCLCPP_INFO(this->get_logger(), "in timer callback");
            cv_bridge::CvImagePtr cv_ptr;
            
            
            // while(true){
                this->initialize = this-> getImage(true);
                this->img = this ->undistort();
                cv::resize(this->img, this->outImg, cv::Size(640, 480));
                // this->showView(this->rstpAddress, this->outImg);
                
                sensor_msgs::msg::Image::SharedPtr msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8",  this->outImg).toImageMsg();
                
                publisher_->publish(*msg.get());
                RCLCPP_INFO(this->get_logger(), "Published!");
            // }

            
        }
        
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
        rclcpp::TimerBase::SharedPtr timer_;
       
        cv::Mat img;
        cv::Mat outImg;
        cv::Mat initialize; 
        
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<UndistortImagePublisher>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
