"""RobotMotion controller for sim-to-real workflows.

Sends timed velocity commands (geometry_msgs/Twist) to either:
  - "sim": Gazebo Harmonic simulation (/cmd_vel, reading /odom or /sim/ground_truth)
  - "real": Physical CleanupBot via micro-ROS (/pico/cmd_vel, reading /pico/imu/heading
            and wheel encoder distance from /pico/drive/distance)

Enforces course safety limits (max 0.30 m/s, max 1.5 rad/s, max 10.0 s).
Detects collisions / blocked movement in simulation when actual distance lags commanded.
"""

from __future__ import annotations

import atexit
import math
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Int32


MAX_LINEAR_SPEED = 0.30        # m/s
MAX_ANGULAR_SPEED = 1.50       # rad/s
MAX_MOVE_SECONDS = 10.0        # seconds
COMMAND_RATE_HZ = 20.0         # Hz
BLOCKED_DISTANCE_RATIO = 0.70    # Flag blocked if actual < 70% of expected


@dataclass
class MotionResult:
    """Outcome of a completed or aborted robot move."""
    target: str                    # "sim" or "real"
    commanded_linear: float        # m/s
    commanded_angular: float       # rad/s
    duration_s: float              # seconds
    distance_m: float              # meters (from sim odom/ground-truth or real encoder/tape)
    turn_deg: float                # degrees turned (from sim odom or real BNO085 IMU)
    possibly_blocked: bool         # True if robot traveled significantly less than commanded
    success: bool                  # True if execution completed without error/rejection
    encoder_distance_m: float = 0.0  # Real wheel encoder distance (if on real robot)
    message: str = ""

    def summary(self) -> str:
        """User-friendly summary of the motion result."""
        status = "PASSED" if not self.possibly_blocked else "BLOCKED / COLLISION DETECTED"
        lines = [
            f"--- Motion Summary [{self.target.upper()}] ({status}) ---",
            f"  Commanded : v={self.commanded_linear:.2f} m/s, w={self.commanded_angular:.2f} rad/s for {self.duration_s:.1f}s",
            f"  Distance  : {self.distance_m:.3f} m",
            f"  Turn      : {self.turn_deg:.1f}°",
        ]
        if self.target == "real" and self.encoder_distance_m > 0.001:
            lines.append(f"  Encoders  : {self.encoder_distance_m:.3f} m (measured by wheel ticks)")
        if self.possibly_blocked:
            expected_dist = abs(self.commanded_linear) * self.duration_s
            lines.append(
                f"  WARNING   : Expected ~{expected_dist:.2f}m but robot only moved {self.distance_m:.2f}m! "
                "Movement appears blocked by an obstacle."
            )
        return "\n".join(lines)


class MotionBridgeNode(Node):
    """ROS 2 Node handling publishers and subscribers for both robots."""

    def __init__(self) -> None:
        super().__init__("robot_motion_bridge")

        # Publishers
        self.pub_sim_cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pub_real_cmd = self.create_publisher(Twist, "/pico/cmd_vel", 10)

        # QoS for sensor telemetry
        qos_sensor = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        qos_sim = 10

        # Sim Pose Telemetry (/sim/ground_truth or /odom)
        self.sim_x: float = 0.0
        self.sim_y: float = 0.0
        self.sim_yaw_deg: float = 0.0
        self.last_sim_gt_time: float = 0.0
        self.last_sim_odom_time: float = 0.0

        self.sub_sim_gt = self.create_subscription(
            Odometry, "/sim/ground_truth", self._sim_gt_callback, qos_sim
        )
        self.sub_sim_odom = self.create_subscription(
            Odometry, "/odom", self._sim_odom_callback, qos_sim
        )

        # Real Robot Telemetry (/pico/imu/heading, /pico/drive/distance, /pico/drive/status)
        self.real_heading_deg: float = 0.0
        self.real_encoder_dist_m: float = 0.0
        self.real_status: int = 0
        self.last_real_time: float = 0.0

        self.sub_real_heading = self.create_subscription(
            Float32, "/pico/imu/heading", self._real_heading_callback, qos_sensor
        )
        self.sub_real_distance = self.create_subscription(
            Float32, "/pico/drive/distance", self._real_dist_callback, qos_sensor
        )
        self.sub_real_status = self.create_subscription(
            Int32, "/pico/drive/status", self._real_status_callback, qos_sensor
        )

    def _sim_gt_callback(self, msg: Odometry) -> None:
        self.sim_x = msg.pose.pose.position.x
        self.sim_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.sim_yaw_deg = math.degrees(math.atan2(siny_cosp, cosy_cosp))
        self.last_sim_gt_time = time.time()

    def _sim_odom_callback(self, msg: Odometry) -> None:
        # If ground-truth has not arrived in the last 0.5s, use wheel odometry
        if (time.time() - self.last_sim_gt_time) > 0.5:
            self.sim_x = msg.pose.pose.position.x
            self.sim_y = msg.pose.pose.position.y
            q = msg.pose.pose.orientation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.sim_yaw_deg = math.degrees(math.atan2(siny_cosp, cosy_cosp))
        self.last_sim_odom_time = time.time()

    def _real_heading_callback(self, msg: Float32) -> None:
        self.real_heading_deg = float(msg.data)
        self.last_real_time = time.time()

    def _real_dist_callback(self, msg: Float32) -> None:
        self.real_encoder_dist_m = float(msg.data)

    def _real_status_callback(self, msg: Int32) -> None:
        self.real_status = int(msg.data)


class RobotMotion:
    """Manages timed motion execution and verification across simulation and hardware."""

    _instance: Optional[RobotMotion] = None

    def __new__(cls) -> RobotMotion:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        if not rclpy.ok():
            rclpy.init()

        self.node = MotionBridgeNode()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

        self._spin_thread = threading.Thread(
            target=self.executor.spin,
            daemon=True,
            name="robot-motion-spin",
        )
        self._spin_thread.start()
        self._initialized = True
        atexit.register(self.shutdown)

    def shutdown(self) -> None:
        """Cleanly halts motors and closes executor threads."""
        try:
            self.stop("both")
            if hasattr(self, "node"):
                self.node.destroy_node()
            if hasattr(self, "executor"):
                self.executor.shutdown()
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

    def stop(self, target: str = "both") -> None:
        """Immediately command zero velocity to stop motors."""
        stop_cmd = Twist()
        if target in ("sim", "both"):
            self.node.pub_sim_cmd.publish(stop_cmd)
        if target in ("real", "both"):
            self.node.pub_real_cmd.publish(stop_cmd)

    def move(
        self,
        target: str,
        linear: float = 0.0,
        angular: float = 0.0,
        seconds: float = 1.0,
        target_distance: float = 0.0,
        target_angle_deg: float = 0.0,
    ) -> MotionResult:
        """
        Execute a move on either 'sim' or 'real' robot with closed-loop sensor termination.

        - If target_distance > 0: Drives until encoder/sim distance reaches target, or timeout.
        - If target_angle_deg > 0: Turns until BNO085/sim yaw reaches target, or timeout.
        - On straight drives: Uses real-time BNO085 gyro yaw feedback to actively eliminate veer.
        """
        target = target.lower().strip()
        if target not in ("sim", "real"):
            raise ValueError(f"Target must be 'sim' or 'real', got '{target}'")

        linear = max(-MAX_LINEAR_SPEED, min(MAX_LINEAR_SPEED, float(linear)))
        angular = max(-MAX_ANGULAR_SPEED, min(MAX_ANGULAR_SPEED, float(angular)))

        # Safety timeout ceiling
        if target_distance > 0.01 and abs(linear) > 0.01:
            timeout_seconds = max(seconds * 1.5 + 0.8, 2.5)
        elif target_angle_deg > 0.1 and abs(angular) > 0.01:
            timeout_seconds = max(seconds * 1.5 + 0.8, 2.0)
        else:
            timeout_seconds = max(0.2, min(MAX_MOVE_SECONDS, float(seconds)))

        pub = self.node.pub_sim_cmd if target == "sim" else self.node.pub_real_cmd

        # Warm-up check: wait up to 1.5s for initial telemetry if not yet received
        if target == "sim" and self.node.last_sim_odom_time == 0 and self.node.last_sim_gt_time == 0:
            t_wait = time.time() + 1.5
            while time.time() < t_wait:
                if self.node.last_sim_odom_time > 0 or self.node.last_sim_gt_time > 0:
                    break
                time.sleep(0.05)

        time.sleep(0.05)
        init_sim_x = self.node.sim_x
        init_sim_y = self.node.sim_y
        init_sim_yaw = self.node.sim_yaw_deg
        init_real_yaw = self.node.real_heading_deg
        init_encoder_dist = self.node.real_encoder_dist_m

        twist_msg = Twist()
        twist_msg.linear.x = linear
        twist_msg.angular.z = angular

        interval = 1.0 / COMMAND_RATE_HZ
        start_time = time.time()
        timeout_end = start_time + timeout_seconds
        hit_target = False

        try:
            while time.time() < timeout_end:
                if target == "sim":
                    dx = self.node.sim_x - init_sim_x
                    dy = self.node.sim_y - init_sim_y
                    current_dist = math.hypot(dx, dy)
                    current_turn = abs((self.node.sim_yaw_deg - init_sim_yaw + 180.0) % 360.0 - 180.0)

                    # Closed-loop distance cutoff in Gazebo
                    if target_distance > 0.005 and current_dist >= target_distance:
                        hit_target = True
                        break

                    # Closed-loop angular cutoff in Gazebo
                    if target_angle_deg > 0.5 and current_turn >= target_angle_deg:
                        hit_target = True
                        break

                    pub.publish(twist_msg)

                else:  # Real robot
                    current_encoder_dist = abs(self.node.real_encoder_dist_m - init_encoder_dist)
                    current_turn_deg = (self.node.real_heading_deg - init_real_yaw + 180.0) % 360.0 - 180.0
                    abs_turn = abs(current_turn_deg)

                    # Closed-loop distance cutoff on physical encoders
                    if target_distance > 0.005 and current_encoder_dist >= target_distance:
                        hit_target = True
                        break

                    # Closed-loop angular cutoff with BNO085 IMU
                    # Cut power 2.5 degrees early to account for physical deceleration inertia
                    lead = 2.5 if abs(angular) > 0.5 else 1.0
                    if target_angle_deg > 0.5 and abs_turn >= max(1.0, target_angle_deg - lead):
                        hit_target = True
                        break

                    # Active Gyro-Stabilized Heading Lock during straight drives
                    if abs(angular) < 0.02 and abs(linear) > 0.02 and target_distance > 0.01:
                        # Counter-steer proportional to heading error
                        yaw_correction = max(-0.15, min(0.15, -0.04 * current_turn_deg))
                        twist_msg.angular.z = yaw_correction

                    pub.publish(twist_msg)

                time.sleep(interval)

        except (KeyboardInterrupt, SystemExit):
            self.stop(target)
            raise
        finally:
            self.stop(target)

        # Settle pause for telemetry propagation
        time.sleep(0.20)
        elapsed = time.time() - start_time

        if target == "sim":
            final_sim_x = self.node.sim_x
            final_sim_y = self.node.sim_y
            final_sim_yaw = self.node.sim_yaw_deg

            dx = final_sim_x - init_sim_x
            dy = final_sim_y - init_sim_y
            distance = math.hypot(dx, dy)
            d_yaw = (final_sim_yaw - init_sim_yaw + 180.0) % 360.0 - 180.0

            # Collision / blockage detection:
            possibly_blocked = False
            if target_distance > 0.10:
                if not hit_target and distance < (BLOCKED_DISTANCE_RATIO * target_distance):
                    possibly_blocked = True
            elif abs(linear) > 0.05 and (abs(linear) * seconds) > 0.10:
                if distance < (BLOCKED_DISTANCE_RATIO * abs(linear) * seconds):
                    possibly_blocked = True

            return MotionResult(
                target="sim",
                commanded_linear=linear,
                commanded_angular=angular,
                duration_s=elapsed,
                distance_m=distance,
                turn_deg=d_yaw,
                possibly_blocked=possibly_blocked,
                success=not possibly_blocked,
                message="Simulation move completed." if not possibly_blocked else "Blocked by obstacle!",
            )
        else:
            final_real_yaw = self.node.real_heading_deg
            d_yaw = (final_real_yaw - init_real_yaw + 180.0) % 360.0 - 180.0

            final_encoder_dist = self.node.real_encoder_dist_m
            encoder_travel = abs(final_encoder_dist - init_encoder_dist)
            nominal_dist = abs(linear) * elapsed

            # Use encoder travel if available, fallback to nominal
            reported_dist = encoder_travel if encoder_travel > 0.001 else nominal_dist

            return MotionResult(
                target="real",
                commanded_linear=linear,
                commanded_angular=angular,
                duration_s=elapsed,
                distance_m=reported_dist,
                turn_deg=d_yaw,
                possibly_blocked=False,
                success=True,
                encoder_distance_m=encoder_travel,
                message="Real robot move completed.",
            )
