#include <opencv2/highgui.hpp>
#include <iostream>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"


class CameraPublisher : public rclcpp::Node
    {
    public:
        CameraPublisher() : Node("camera_pub")
        {
            this->declare_parameter("camera_id", 0);

            camera_id_ = this->get_parameter("camera_id").as_int();

            camera_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("image_raw", 10);


            
            // cv ::Mat frame;
            // cv::VideoCapture cap(camera_id_);
            
            camera_timer_ = this->create_wall_timer(std::chrono::milliseconds(100),
                                                std::bind(&CameraPublisher::timer_callback, this));
            RCLCPP_INFO(this->get_logger(), "camera publisher cpp node");
        }
    private:
    void timer_callback()
    {   
        cv::Mat frame;
        cv::VideoCapture cap(camera_id_);
        cap >> frame;
        cv::imshow("camera", frame);
        cv::waitKey(1);
        cap.release();
        sensor_msgs::msg::Image::SharedPtr msg = std::make_shared<sensor_msgs::msg::Image>();
        msg->header.stamp = this->now();
        msg->header.frame_id = "camera";
        msg->height = frame.rows;
        msg->width = frame.cols;
        msg->encoding = "bgr8";
        msg->is_bigendian = false;
        msg->step = frame.cols * frame.elemSize();
        size_t size = msg->step * frame.rows;
        msg->data.resize(size);
        memcpy(&msg->data[0], frame.data, size);
        camera_publisher_->publish(*msg.get());

    }
    int camera_id_;
    rclcpp::TimerBase::SharedPtr camera_timer_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr camera_publisher_;


};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CameraPublisher>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}