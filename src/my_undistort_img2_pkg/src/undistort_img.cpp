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
        UndistortImagePublisher() : Node("UndistortImagePublisher1"),  Camera()
        {


            this->declare_parameter("camera_id", "rtsp://192.168.73.11:8554/h264");
            camera_id_ = this->get_parameter("camera_id").as_string();   

            publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/undistorded/image_raw_1", 1);
            timer_ = this->create_wall_timer(std::chrono::milliseconds(100), std::bind(&UndistortImagePublisher::timer_callback, this));
            RCLCPP_INFO(this->get_logger(), "camera publisher cpp node");
            
            this->isCamInitialized= this->initializeCamera("front", camera_id_,"/home/vishvajit/ros2_ws/src/my_undistort_img2_pkg/src", "../perspTransMatrix.yml");
        }

    private:
        void timer_callback(){

            
            cv_bridge::CvImagePtr cv_ptr;
            
            if (this->isCamInitialized){
                this->initialize = this-> getImage(true);
                this->img = this ->undistort();
                cv::resize(this->img, this->outImg, cv::Size( 640, 480));
                // this->showView(this->rstpAddress, this->outImg);
                
                sensor_msgs::msg::Image::SharedPtr msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8",  this->outImg).toImageMsg();
                
                publisher_->publish(*msg.get());
                // RCLCPP_INFO(this->get_logger(), "Published!");
            }
            else{
                RCLCPP_INFO(this->get_logger(), "Camera not initialized!");
            }

            // this->initialize = this-> getImage(true);
            // this->img = this ->undistort();
            // cv::resize(this->img, this->outImg, cv::Size( 640, 480));
            // // this->showView(this->rstpAddress, this->outImg);
            
            // sensor_msgs::msg::Image::SharedPtr msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8",  this->outImg).toImageMsg();
            
            // publisher_->publish(*msg.get());
            // RCLCPP_INFO(this->get_logger(), "Published!");
       

            
        }
        
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
        rclcpp::TimerBase::SharedPtr timer_;
       
        cv::Mat img;
        cv::Mat outImg;
        cv::Mat initialize; 
        bool isCamInitialized;
         std::string camera_id_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<UndistortImagePublisher>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
