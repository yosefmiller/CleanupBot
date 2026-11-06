"""
ROS2 publisher helpers.
All hardware commands funnel through here so the rest of the code
stays decoupled from ROS2 message types.
"""

import atexit
import threading

import math

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from typing import Optional

from config import (
    TOPIC_CMD_VEL,
    TOPIC_ODOM,
    TOPIC_GRIPPER_CMD,
    TOPIC_AGENT_STATUS,
)


class RobotPublisher(Node):
    """
    Singleton-style ROS2 publisher node.
    Instantiate once via get_publisher(); reuse across the application lifetime.
    """

    def __init__(self) -> None:
        super().__init__("robot_publisher")
        self._cmd_vel_pub = self.create_publisher(Twist, TOPIC_CMD_VEL, 10)
        self._odom_pub = self.create_publisher(Odometry, TOPIC_ODOM, 10)
        self._gripper_pub = self.create_publisher(String, TOPIC_GRIPPER_CMD, 10)
        self._status_pub = self.create_publisher(String, TOPIC_AGENT_STATUS, 10)

        self._executor = SingleThreadedExecutor(context=self.context)
        self._executor.add_node(self)
        self._spin_thread = threading.Thread(
            target=self._executor.spin,
            daemon=True,
            name="publisher-spin",
        )
        self._spin_thread.start()
    def _shutdown(self) -> None:
        """Best-effort graceful shutdown on process exit."""
        try:
            self._executor.shutdown()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Drive
    # ------------------------------------------------------------------

    def publish_cmd_vel(self, linear_x: float, angular_z: float) -> None:
        """
        Publish a velocity command.

        Args:
            linear_x: Forward speed in m/s (negative = reverse).
            angular_z: Rotational speed in rad/s (positive = left turn).
        """
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self._cmd_vel_pub.publish(msg)

    def stop(self) -> None:
        """Publish zero velocity — immediate software stop."""
        self.publish_cmd_vel(0.0, 0.0)

    # ------------------------------------------------------------------
    # Odometry
    # ------------------------------------------------------------------

    def publish_odometry(self, x_m: float, y_m: float, yaw_deg: float) -> None:
        """
        Publish updated 2D robot odometry in the global world frame.

        Args:
            x_m: Global X position in meters.
            y_m: Global Y position in meters.
            yaw_deg: Robot heading in degrees (0 = along +X axis).
        """
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_link"
        msg.pose.pose.position.x = float(x_m)
        msg.pose.pose.position.y = float(y_m)
        msg.pose.pose.position.z = 0.0

        yaw_rad = math.radians(yaw_deg)
        msg.pose.pose.orientation.z = math.sin(yaw_rad / 2.0)
        msg.pose.pose.orientation.w = math.cos(yaw_rad / 2.0)
        self._odom_pub.publish(msg)

    # ------------------------------------------------------------------
    # Gripper
    # ------------------------------------------------------------------

    def publish_gripper_command(self, command: str) -> None:
        """
        Send a gripper command string to the gripper controller topic.

        Valid commands: 'open', 'stop', 'close_small', 'close_medium',
                        'close_large', 'tilt_down', 'tilt_up'.

        Args:
            command: One of the valid command strings above.
        """
        msg = String()
        msg.data = command
        self._gripper_pub.publish(msg)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def publish_status(self, status: str) -> None:
        """Broadcast a human-readable agent status string."""
        msg = String()
        msg.data = status
        self._status_pub.publish(msg)


# Module-level singleton
_publisher_instance: Optional[RobotPublisher] = None
_publisher_lock = threading.Lock()


def get_publisher() -> RobotPublisher:
    """
    Return the module-level publisher singleton, initialising ROS2 if needed.

    Thread-safe via a lock; safe to call from multiple threads at startup.

    Returns:
        The shared RobotPublisher node.
    """
    global _publisher_instance
    if _publisher_instance is None:
        with _publisher_lock:
            if _publisher_instance is None:  # double-checked locking
                if not rclpy.ok():
                    import os
                    os.environ.setdefault("ROS_LOG_DIR", "/tmp/ros_log")
                    try:
                        os.makedirs("/tmp/ros_log", exist_ok=True)
                    except Exception:
                        pass
                    rclpy.init()
                _publisher_instance = RobotPublisher()
    return _publisher_instance