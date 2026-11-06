"""Start Gazebo Harmonic with the obstacle course, spawn the robot car, and bridge topics to ROS 2.

Usage:
    ros2 launch robot_sim sim.launch.py
    ros2 launch robot_sim sim.launch.py robot:=custom world:=custom
    ros2 launch robot_sim sim.launch.py gui:=false
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import xacro


def launch_setup(context, *args, **kwargs):
    pkg_dir = get_package_share_directory("robot_sim")

    robot_arg = context.perform_substitution(LaunchConfiguration("robot"))
    world_arg = context.perform_substitution(LaunchConfiguration("world"))
    gui = LaunchConfiguration("gui")

    # Select robot model
    if robot_arg == "custom":
        robot_file = os.path.join(pkg_dir, "urdf", "cleanupbot.urdf.xacro")
        robot_name = "cleanupbot"
    else:
        robot_file = os.path.join(pkg_dir, "urdf", "gazebo_car.urdf.xacro")
        robot_name = "gazebo_car"

    # Select world file
    if world_arg in ["apartment", "custom"]:
        world_file = os.path.join(pkg_dir, "worlds", "apartment.sdf")
        world_name = "apartment"
    elif world_arg == "obstacle_course_custom":
        world_file = os.path.join(pkg_dir, "worlds", "obstacle_course_custom.sdf")
        world_name = "obstacle_course"
    else:
        world_file = os.path.join(pkg_dir, "worlds", "obstacle_course.sdf")
        world_name = "obstacle_course"

    bridge_file = os.path.join(pkg_dir, "config", "bridge.yaml")

    # Process Xacro
    doc = xacro.process_file(robot_file)
    robot_description = doc.toxml()

    # Resource paths for Gazebo to resolve model:// and package:// mesh URIs
    pkg_share_parent = os.path.dirname(pkg_dir)
    gz_res = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
    extra_gz_res = f"{pkg_share_parent}:{pkg_dir}:{os.path.join(pkg_dir, 'urdf', 'meshes')}:{os.path.join(pkg_dir, 'models')}"
    full_gz_res = f"{extra_gz_res}:{gz_res}" if gz_res else extra_gz_res

    extra_env = {
        "GZ_SIM_RESOURCE_PATH": full_gz_res,
        "IGN_GAZEBO_RESOURCE_PATH": full_gz_res,
        "SDF_PATH": full_gz_res,
        "GAZEBO_MODEL_PATH": full_gz_res,
    }

    # 1. Gazebo simulator (server). -r starts the simulation running.
    gazebo_server = ExecuteProcess(
        cmd=["gz", "sim", "-s", "-r", "-v", "2", world_file],
        output="screen",
        additional_env=extra_env,
    )

    # 2. Gazebo window (GUI)
    gazebo_gui = ExecuteProcess(
        cmd=["gz", "sim", "-g", "-v", "1"],
        output="screen",
        condition=IfCondition(gui),
        additional_env=extra_env,
    )

    # 3. Publishes robot model on /robot_description and TF
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{
            "robot_description": robot_description,
            "use_sim_time": True,
        }],
        output="screen",
    )

    # 4. Spawns the robot at (0, 0)
    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-world", world_name,
            "-topic", "robot_description",
            "-name", robot_name,
            "-x", "0.0", "-y", "0.0", "-z", "0.04",
        ],
        output="screen",
    )

    # 5. Parameter bridge for Gazebo <-> ROS 2 communication
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        parameters=[{"config_file": bridge_file}],
        output="screen",
    )

    # 6. Point the Gazebo window's camera at the arena once the window is open
    camera_request = (
        "pose: {position: {x: 1.5, y: -2.4, z: 1.8}, "
        "orientation: {x: -0.2236, y: 0.2236, z: 0.6708, w: 0.6708}}"
    )
    set_gui_camera = ExecuteProcess(
        cmd=["bash", "-c",
             "for i in $(seq 1 30); do "
             "gz service -s /gui/move_to/pose --reqtype gz.msgs.GUICamera "
             f"--reptype gz.msgs.Boolean --timeout 2000 --req '{camera_request}' "
             "> /dev/null 2>&1 && break; sleep 2; done"],
        condition=IfCondition(gui),
    )

    return [
        gazebo_server,
        gazebo_gui,
        robot_state_publisher,
        bridge,
        TimerAction(period=5.0, actions=[spawn_robot]),
        TimerAction(period=8.0, actions=[set_gui_camera]),
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="true",
                              description="Show the Gazebo window"),
        DeclareLaunchArgument("robot", default_value="custom",
                              description="Robot model: 'baseline' or 'custom'"),
        DeclareLaunchArgument("world", default_value="apartment",
                              description="World file: 'apartment', 'custom', or 'standard'"),
        OpaqueFunction(function=launch_setup),
    ])
