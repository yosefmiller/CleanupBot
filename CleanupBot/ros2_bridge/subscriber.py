"""
ROS2 subscriber that maintains the latest sensor readings in memory.
All hardware reads funnel through here.
"""

import copy
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Imu, Image
from std_msgs.msg import Float32

from config import (
    TOPIC_ENCODER_LEFT,
    TOPIC_ENCODER_RIGHT,
    TOPIC_IMU,
    TOPIC_ULTRASONIC_LEFT,
    TOPIC_ULTRASONIC_RIGHT,
    TOPIC_CAMERA_RAW,
)


@dataclass
class SensorSnapshot:
    """Latest fused sensor state — updated by subscriber callbacks."""
    # Fields without defaults must come first
    encoder_left_ticks: int = 0
    encoder_right_ticks: int = 0
    imu_heading_deg: float = 0.0
    imu_pitch_deg: float = 0.0
    imu_roll_deg: float = 0.0
    ultrasonic_left_m: float = 3.0
    ultrasonic_right_m: float = 3.0
    last_updated: float = field(default_factory=time.time)
    # Optional numpy array must be last (has a non-trivial default)
    camera_frame: Optional[np.ndarray] = field(default=None, repr=False)


class RobotSubscriber(Node):
    """
    Subscribes to all sensor topics and keeps a SensorSnapshot up to date.
    Thread-safe via a single RLock.
    """

    def __init__(self) -> None:
        super().__init__("robot_subscriber")
        self._lock = threading.RLock()
        self._state = SensorSnapshot()

        self.create_subscription(Float32, TOPIC_ENCODER_LEFT,
                                 self._cb_enc_left, 10)
        self.create_subscription(Float32, TOPIC_ENCODER_RIGHT,
                                 self._cb_enc_right, 10)
        self.create_subscription(Imu, TOPIC_IMU, self._cb_imu, 10)
        self.create_subscription(Float32, TOPIC_ULTRASONIC_LEFT,
                                 self._cb_us_left, 10)
        self.create_subscription(Float32, TOPIC_ULTRASONIC_RIGHT,
                                 self._cb_us_right, 10)
        self.create_subscription(Image, TOPIC_CAMERA_RAW,
                                 self._cb_camera, 10)

        self._executor = SingleThreadedExecutor(context=self.context)
        self._executor.add_node(self)
        self._spin_thread = threading.Thread(
            target=self._executor.spin,
            daemon=True,
            name="subscriber-spin",
        )
        self._spin_thread.start()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _cb_enc_left(self, msg: Float32) -> None:
        with self._lock:
            self._state.encoder_left_ticks = int(msg.data)
            self._state.last_updated = time.time()

    def _cb_enc_right(self, msg: Float32) -> None:
        with self._lock:
            self._state.encoder_right_ticks = int(msg.data)
            self._state.last_updated = time.time()

    def _cb_imu(self, msg: Imu) -> None:
        """Convert IMU quaternion to Euler angles (degrees) and store."""
        q = msg.orientation
        sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        roll = float(np.degrees(np.arctan2(sinr_cosp, cosr_cosp)))

        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        pitch = float(np.degrees(np.arcsin(np.clip(sinp, -1.0, 1.0))))

        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = float(np.degrees(np.arctan2(siny_cosp, cosy_cosp)))

        with self._lock:
            self._state.imu_heading_deg = yaw
            self._state.imu_pitch_deg = pitch
            self._state.imu_roll_deg = roll
            self._state.last_updated = time.time()

    def _cb_us_left(self, msg: Float32) -> None:
        with self._lock:
            self._state.ultrasonic_left_m = float(msg.data)
            self._state.last_updated = time.time()

    def _cb_us_right(self, msg: Float32) -> None:
        with self._lock:
            self._state.ultrasonic_right_m = float(msg.data)
            self._state.last_updated = time.time()

    def _cb_camera(self, msg: Image) -> None:
        """Convert ROS2 Image to a NumPy BGR array."""
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, -1
        )
        with self._lock:
            self._state.camera_frame = frame.copy()
            self._state.last_updated = time.time()

    # ------------------------------------------------------------------
    # Public read
    # ------------------------------------------------------------------

    def snapshot(self) -> SensorSnapshot:
        """Return a thread-safe shallow copy of the current sensor state."""
        with self._lock:
            return copy.copy(self._state)

    def shutdown(self) -> None:
        """Stop spinning and release the subscriber node."""
        self._executor.shutdown()
        self.destroy_node()


# Module-level singleton
_subscriber_instance: Optional[RobotSubscriber] = None
_subscriber_lock = threading.Lock()


def get_subscriber() -> RobotSubscriber:
    """
    Return the module-level subscriber singleton.

    Thread-safe via double-checked locking.

    Returns:
        The shared RobotSubscriber node.
    """
    global _subscriber_instance
    if _subscriber_instance is None:
        with _subscriber_lock:
            if _subscriber_instance is None:
                if not rclpy.ok():
                    rclpy.init()
                _subscriber_instance = RobotSubscriber()
    return _subscriber_instance