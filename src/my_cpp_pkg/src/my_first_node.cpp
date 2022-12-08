#include "rclcpp/rclcpp.hpp"



class MyFirstNode : public rclcpp::Node
{
public:
    MyFirstNode() : Node("my_cpp_node"), count_(0)
    {
        RCLCPP_INFO(this->get_logger(), "Hello cpp World!");
        timer_ = this->create_wall_timer(std::chrono::milliseconds(100),
                                            std::bind(&MyFirstNode::timer_callback, this));
    }
private:
    void timer_callback()
    {   
        count_++;
        RCLCPP_INFO(this->get_logger(), "Hello timer calback %d", count_);
    }
    rclcpp::TimerBase::SharedPtr timer_;
    int count_;


};




int main(int argc, char **argv)
{


    rclcpp::init(argc, argv);
    //auto node = std::make_shared<rclcpp::Node>("cpp_test");
    auto node = std::make_shared<MyFirstNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;

}