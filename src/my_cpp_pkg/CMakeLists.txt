cmake_minimum_required(VERSION 3.5)
project(my_cpp_pkg)



# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

find_package(image_transport REQUIRED)

find_package(OpenCV 4  REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)


pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)



add_executable(my_cpp_node src/my_first_node.cpp )
ament_target_dependencies(my_cpp_node rclcpp)

add_executable(camera_pub src/camera_pub.cpp)
ament_target_dependencies(camera_pub OpenCV rclcpp sensor_msgs) # OpenCV is the package name

# camera_pub 2 is the name of the executable
add_executable(camera_pub2 src/camera_pub2.cpp)
ament_target_dependencies(camera_pub2 OpenCV rclcpp cv_bridge image_transport sensor_msgs)

add_executable(camera_sub src/camera_sub.cpp)
ament_target_dependencies(camera_sub OpenCV rclcpp cv_bridge image_transport sensor_msgs)

# camera_pub 3 is the name of the executable

add_executable(camera_pub3 src/camera_pub3.cpp)
ament_target_dependencies(camera_pub3 OpenCV rclcpp cv_bridge image_transport sensor_msgs)


install(TARGETS
  my_cpp_node
  camera_pub2
  camera_pub
  camera_pub3
  camera_sub
  
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
