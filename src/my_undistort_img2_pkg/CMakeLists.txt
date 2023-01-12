cmake_minimum_required(VERSION 3.5)
project(my_undistort_img2_pkg)



# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

find_package(image_transport REQUIRED)

find_package(OpenCV 4  REQUIRED)
find_package(cv_bridge REQUIRED)
#find_package(OpenCV REQUIRED)

set(SOURCES "${CMAKE_CURRENT_LIST_DIR}/src/src/camera.cpp")

pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)


add_executable(undistort_img src/undistort_img.cpp ${SOURCES})
ament_target_dependencies(undistort_img rclcpp std_msgs sensor_msgs image_transport OpenCV cv_bridge)


install(TARGETS
     undistort_img 


    DESTINATION lib/${PROJECT_NAME}
)

ament_package()
