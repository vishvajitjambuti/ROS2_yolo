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

class CameraSub: public rclcpp::Node
{
    public:
        CameraSub() : Node("Camera_sub")
        {
         subscription_ = this->create_subscription<sensor_msgs::msg::Image>("image_raw", 10, std::bind(&CameraSub::imageCallback, this, std::placeholders::_1));
         RCLCPP_INFO(this->get_logger(), "camera subscriber cpp node");

        }
    private:
        // void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
        // {
        //     cv_bridge::CvImagePtr cv_ptr;
        //     try
        //     {
        //         // cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        //         cv::imshow("view", cv_bridge::toCvShare(msg, "bgr8")->image);
        //         cv::waitKey(1);
        //     }
        //     catch (cv_bridge::Exception& e)
        //     {
        //         RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        //         return;
        //     }
           
        //     // cv::imshow("view", cv_ptr->image);
        //     // cv::waitKey(1);
            

           
        // }
        void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr  msg)
            {
            try {
                cv::imshow("view", cv_bridge::toCvShare(msg, "bgr8")->image);
                cv::waitKey(10);
            } catch (const cv_bridge::Exception & e) {
                auto logger = rclcpp::get_logger("my_subscriber");
                RCLCPP_ERROR(logger, "Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
            }
            }


        rclcpp::TimerBase::SharedPtr timer_;
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
        cv::Mat image;
};
int main(int argc, char **argv)
{
    
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CameraSub>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}