from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()
    
    remap_camera_topic1 = ('image_raw', 'cam_1')
    
    camera_publisher_node = Node(
        package='my_undistort_img_pkg',
        executable='undistort_img2',
        remappings=[remap_camera_topic1],)
    
    object_detection_node = Node(
        package='my_yolov5_pkg',
        executable='yolo_detect',
        name='object_detection_node', 
        remappings=[remap_camera_topic1],
    )
    
    ld.add_action(camera_publisher_node)
    ld.add_action(object_detection_node)
    
    return ld