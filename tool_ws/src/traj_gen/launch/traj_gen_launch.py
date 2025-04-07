from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node
import os
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch.launch_service import LaunchService
from pathlib import Path

import sys
sys.executable = "/usr/bin/python3"  # 替换成系统默认的Python路径

def generate_launch_description():
    cur_path = Path(__file__).resolve()
    package_path = cur_path.parent.parent
    rviz_config = os.path.join(package_path, 'rviz', 'traj_gen.rviz')
    
    return LaunchDescription([
        DeclareLaunchArgument('env', default_value='env_airsim_16.yaml', description='环境配置文件'),
        
        # 修改RViz节点配置，设置日志级别为WARN
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz',
            output='screen',
            arguments=['-d', rviz_config, '--ros-args', '--log-level', 'WARN']  # 添加日志级别参数
        ),

        Node(
            package='traj_gen',
            executable='traj_gen_node',
            name='traj_gen_node',
            output='screen',
            parameters=[{'env': LaunchConfiguration('env')}],
        ),
    ])

def main():
    launch_description = generate_launch_description()
    launch_service = LaunchService()
    launch_service.include_launch_description(launch_description)
    print("启动ROS 2 launch文件...")
    launch_service.run()

if __name__ == '__main__':
    main()