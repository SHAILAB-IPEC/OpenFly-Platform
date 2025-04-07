from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node
import os
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch.launch_service import LaunchService
from pathlib import Path

def generate_launch_description():
    # 获取包的共享目录路径
    # planner_share_dir = get_package_share_directory('traj_gen')
    # cur_path = Path(__file__).resolve()
    # package_path = cur_path.parent.parent
    # RViz 配置文件路径
    # rviz_config = os.path.join(package_path, 'rviz', 'traj_gen.rviz')
    cur_path = Path(__file__).resolve()
    package_path = cur_path.parent.parent
    # RViz 配置文件路径
    rviz_config = os.path.join(package_path, 'rviz', 'seg_gen.rviz')

    return LaunchDescription([
        DeclareLaunchArgument('env', default_value='env_airsim_16', description='Environment configuration file'),


        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz',
            output='screen',
            arguments=['-d', rviz_config]
        ),

        # # 启动 RViz 节点
        # 延迟启动 Planner 节点，设置自动重启延迟为 2 秒
        Node(
            package='seg_gen',
            executable='manual_seg_node',
            name='manual_seg_node',
            output='screen',
            parameters=[{'env': LaunchConfiguration('env')}],
        ),
    ])


def main():
    launch_description = generate_launch_description()
    launch_service = LaunchService()
    launch_service.include_launch_description(launch_description)
    
    print("Launching the ROS 2 launch file...")
    launch_service.run()

if __name__ == '__main__':
    main()