import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_dir = get_package_share_directory('irdpl_toolpath_verifier')
    params_file = os.path.join(pkg_dir, 'config', 'process_params.yaml')

    verifier_node = Node(
        package='irdpl_toolpath_verifier',
        executable='verifier_service_node',
        name='irdpl_toolpath_verifier_node',
        output='screen',
        parameters=[params_file]
    )

    return LaunchDescription([
        verifier_node
    ])
