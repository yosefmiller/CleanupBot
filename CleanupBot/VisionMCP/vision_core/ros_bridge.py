"""Thread-safe ROS2 subscriber for camera images and IMU pitch."""

from __future__ import annotations

import atexit
import logging
import math
import threading
import time

import numpy as np
import rclpy
import rclpy.node
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, Imu

log = logging.getLogger("vision_core.ros_bridge")


class RosBridge:
    """Subscribes to ROS2 camera, IMU, and odometry topics with thread-safe access."""

    def __init__(
        self,
        camera_topic: str = "/camera/image_raw",
        imu_topic: str = "/imu/data",
        odom_topic: str = "/odom",
        frame_timeout_s: float = 3.0,
        max_frame_age_s: float = 2.0,
    ) -> None:
        if not rclpy.ok():
            import os
            os.environ.setdefault("ROS_LOG_DIR", "/tmp/ros_log")
            try:
                os.makedirs("/tmp/ros_log", exist_ok=True)
            except Exception:
                pass
            rclpy.init()

        self._node: rclpy.node.Node = rclpy.node.Node("vision_mcp_bridge")
        self._bridge: CvBridge = CvBridge()
        self._frame_timeout_s: float = frame_timeout_s
        self._max_frame_age_s: float = max_frame_age_s

        # Image state
        self._frame_lock: threading.Lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._frame_time: float = 0.0
        self._first_frame_event: threading.Event = threading.Event()

        # IMU state
        self._imu_lock: threading.Lock = threading.Lock()
        self._pitch_deg: float = 0.0
        self._pitch_time: float = 0.0

        # Odometry state
        self._odom_lock: threading.Lock = threading.Lock()
        self._odom_x_m: float = 0.0
        self._odom_y_m: float = 0.0
        self._odom_yaw_rad: float = 0.0
        self._odom_time: float = 0.0

        # Subscriptions
        self._node.create_subscription(Image, camera_topic, self._image_callback, 1)
        self._node.create_subscription(Imu, imu_topic, self._imu_callback, 10)
        self._node.create_subscription(Odometry, odom_topic, self._odom_callback, 10)

        # Background spin thread
        self._spin_thread: threading.Thread = threading.Thread(
            target=rclpy.spin,
            args=(self._node,),
            daemon=True,
            name="ros2_vision_spin",
        )
        self._spin_thread.start()
        atexit.register(self.close)

    def _image_callback(self, msg: Image) -> None:
        """Convert ROS Image message to OpenCV BGR array."""
        try:
            frame: np.ndarray = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self._frame_lock:
                self._latest_frame = frame
                self._frame_time = time.monotonic()
            self._first_frame_event.set()
        except Exception as exc:
            log.warning("Image conversion error: %s", exc)

    def _imu_callback(self, msg: Imu) -> None:
        """Compute pitch angle in degrees from orientation quaternion."""
        q = msg.orientation
        sin_pitch: float = max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x)))
        with self._imu_lock:
            self._pitch_deg = math.degrees(math.asin(sin_pitch))
            self._pitch_time = time.monotonic()

    def _odom_callback(self, msg: Odometry) -> None:
        """Extract 2D position (meters) and yaw heading (radians) from Odometry."""
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        with self._odom_lock:
            self._odom_x_m = float(pos.x)
            self._odom_y_m = float(pos.y)
            self._odom_yaw_rad = float(yaw)
            self._odom_time = time.monotonic()

    def get_frame(self) -> np.ndarray | None:
        """Return the newest camera frame, or None if timed out or stale."""
        self._first_frame_event.wait(timeout=self._frame_timeout_s)
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            if (time.monotonic() - self._frame_time) > self._max_frame_age_s:
                return None
            return self._latest_frame.copy()

    def get_pitch_deg(self) -> float:
        """Return the current arm pitch in degrees (positive = nose down)."""
        with self._imu_lock:
            if (time.monotonic() - self._pitch_time) > 1.0:
                return 0.0
            return self._pitch_deg

    def get_odom_pose(self) -> tuple[float, float, float]:
        """Return current estimated robot pose (x_m, y_m, yaw_rad) in the world frame."""
        with self._odom_lock:
            return self._odom_x_m, self._odom_y_m, self._odom_yaw_rad

    def close(self) -> None:
        """Safely clean up the ROS2 node."""
        try:
            self._node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
