from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()
    
    src_1 = "rtsp://192.168.73.10:8554/h264"
    src_2 = "rtsp://192.168.73.11:8554/h264"

    src_3 = "rtsp://192.168.73.12:8554/h264"
    src_4 = "rtsp://192.168.73.13:8554/h264"

    source_list = [src_1, src_2, src_3, src_4]
    
    for i in range(len(source_list)):
        camera_publisher_node = Node(
            package='my_camera_pkg',
            executable='camera_stride_pub',
            name='camera_publisher_node_' + str(i),
            remappings=[('image_raw', 'img_' + str(i))],
            parameters=[{'camera_id': source_list[i]}]
            )    
        camera_sub = Node (
            package='my_camera_pkg',
            executable='camera_sub',
            name='camera_sub_' + str(i),
            parameters=[{'window_name': 'camera_sub_' + str(i)}],
            remappings=[('image_raw', 'img_' + str(i))],
        )
        ld.add_action(camera_publisher_node)
        ld.add_action(camera_sub)
    
    
    
    
    
    
    
    
    # camera_publisher_node_1 = Node(
    #     package='my_camera_pkg',
    #     executable='camera_stride_pub',
    #     name='camera_publisher_node_1',
    #     remappings=[('image_raw', 'img_1')],
    #     parameters=[{'camera_id': src_1}]
    #     )    
    # camera_sub_1 = Node (
    #     package='my_camera_pkg',
    #     executable='camera_sub',
    #     name='camera_sub_1',
    #     parameters=[{'window_name': 'camera_sub_1'}],
    #     remappings=[('image_raw', 'img_1')],
    # )
    # ld.add_action(camera_publisher_node_1)
    # ld.add_action(camera_sub_1)   
    
    # camera_publisher_node_2 = Node(
    #     package='my_camera_pkg',
    #     executable='camera_stride_pub',
    #     name='camera_publisher_node_2',
    #     remappings=[('image_raw', 'img_2')],
    #     parameters=[{'camera_id': src_2}]
    # )
    # camera_sub_2 = Node (
    #     package='my_camera_pkg',
    #     executable='camera_sub',
    #     name='camera_sub_2',
    #     parameters=[{'window_name': 'camera_sub_2'}],
    #     remappings=[('image_raw', 'img_2')],
    # )
    # ld.add_action(camera_publisher_node_2)
    # ld.add_action(camera_sub_2)
    
    # camera_publisher_node_3 = Node(
    #     package='my_camera_pkg',
    #     executable='camera_stride_pub',
    #     name='camera_publisher_node_3',
    #     remappings=[('image_raw', 'img_3')],
    #     parameters=[{'camera_id': src_4}]
    # )
    # camera_sub_3 = Node (
    #     package='my_camera_pkg',
    #     executable='camera_sub',
    #     name='camera_sub_3',
    #     remappings=[('image_raw', 'img_3')],
    #     parameters=[{'window_name': 'camera_sub_3'}],
    # )
    # ld.add_action(camera_publisher_node_3)
    # ld.add_action(camera_sub_3)
    
    return ld


