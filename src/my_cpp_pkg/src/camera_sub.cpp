#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <opencv2/opencv.hpp>

class ImageDisplayNode : public rclcpp::Node
{
public:
  ImageDisplayNode()
  : Node("image_display_node")
  {
    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
    "image_raw", rclcpp::QoS(10),
    std::bind(&ImageDisplayNode::image_callback, this, std::placeholders::_1));
  }

private:
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    // Convert ROS image message to a cv::Mat
    cv::Mat image = cv::Mat(msg->height, msg->width, CV_8UC3,
                            const_cast<unsigned char*>(msg->data.data()));
    // Display the image
    cv::imshow("Image", image);
    cv::waitKey(1);
  }
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ImageDisplayNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
