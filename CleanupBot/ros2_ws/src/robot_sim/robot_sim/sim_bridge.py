"""
Gazebo Harmonic simulation bridge for ROS2 Jazzy.

Changes from Humble version:
- SpawnEntity → gz.msgs.EntityFactory via /world/apartment/create
- Gripper commands → ForwardCommandController /gripper_controller/commands
- Encoder ticks derived from /joint_states (ros2_control) not /odom
"""

from __future__ import annotations

import math
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, String, Float64MultiArray
from ros_gz_interfaces.srv import SpawnEntity   # Jazzy: ros_gz_interfaces


# Gripper command → [tilt_position, left_gap, right_gap]
_GRIPPER_COMMANDS: dict[str, tuple[float, float, float]] = {
    "open":         ( 0.00,  0.0225, 0.0225),
    "tilt_down":    (-0.75,  0.0225, 0.0225),
    "tilt_up":      ( 0.00,  0.0225, 0.0225),
    "close_small":  (-0.75,  0.005,  0.005 ),
    "close_medium": (-0.75,  0.010,  0.010 ),
    "close_large":  (-0.75,  0.016,  0.016 ),
    "stop":         ( None,  None,   None  ),   # No change
}

_HAZARD_SDF_TEMPLATE = """<?xml version="1.0"?>
<sdf version="1.10">
  <model name="{name}">
    <static>false</static>
    <pose>{x} {y} {z} 0 0 0</pose>
    <link name="link">
      <inertial>
        <mass>0.05</mass>
        <inertia>
          <ixx>0.0001</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>0.0001</iyy><iyz>0</iyz><izz>0.0001</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
        <surface>
          <friction><ode><mu>0.6</mu><mu2>0.6</mu2></ode></friction>
        </surface>
      </collision>
      <visual name="visual">
        <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
        <material>
          <ambient>{r} {g} {b} 1</ambient>
          <diffuse>{r} {g} {b} 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""

# (x, y, z, sx, sy, sz, r, g, b)
_HAZARD_POSITIONS: list[tuple[float, ...]] = [
    (1.0,  0.3,  0.01, 0.030, 0.020, 0.020, 1.0, 0.2, 0.2),
    (0.8, -0.4,  0.01, 0.025, 0.025, 0.025, 0.2, 1.0, 0.2),
    (1.5,  0.0,  0.01, 0.040, 0.020, 0.015, 0.2, 0.2, 1.0),
]


class SimBridge(Node):
    """
    Bridges Gazebo Harmonic to the ROS2 topics expected by the robot code.

    Responsibilities:
    - /joint_states → synthetic /encoder/left and /encoder/right
    - /gripper/command String → /gripper_controller/commands Float64MultiArray
    - /odom → cached robot pose for ultrasonic simulation
    - Spawns hazard SDF models at startup via ros_gz_interfaces SpawnEntity
    - Publishes synthetic /ultrasonic/left and /ultrasonic/right
    """

    _TICKS_PER_M: float = (12 * 90) / (2.0 * math.pi * 0.033)

    def __init__(self) -> None:
        super().__init__("sim_bridge")

        # ── QoS for sensor data ───────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )

        # ── Publishers ───────────────────────────────────────────
        self._enc_left_pub  = self.create_publisher(Float32, "/encoder/left",  10)
        self._enc_right_pub = self.create_publisher(Float32, "/encoder/right", 10)
        self._us_left_pub   = self.create_publisher(Float32, "/ultrasonic/left",  10)
        self._us_right_pub  = self.create_publisher(Float32, "/ultrasonic/right", 10)

        # Forward gripper positions to ros2_control ForwardCommandController
        self._gripper_cmd_pub = self.create_publisher(
            Float64MultiArray, "/gripper_controller/commands", 10
        )

        # ── Subscribers ──────────────────────────────────────────
        # Joint states from ros2_control — gives us wheel encoder data
        self.create_subscription(
            JointState, "/joint_states", self._joint_states_cb, sensor_qos
        )
        # Odometry for robot position (used in ultrasonic simulation)
        self.create_subscription(
            Odometry, "/odom", self._odom_cb, sensor_qos
        )
        # Gripper string commands from cleanupbot_tools
        self.create_subscription(
            String, "/gripper/command", self._gripper_cb, 10
        )

        # ── State ────────────────────────────────────────────────
        self._prev_left_pos:  float = 0.0
        self._prev_right_pos: float = 0.0
        self._enc_left:       float = 0.0
        self._enc_right:      float = 0.0
        self._robot_x:        float = 0.0
        self._robot_y:        float = 0.0
        self._gripper_state:  list[float] = [0.0, 0.0225, 0.0225]
        self._hazard_xy: list[tuple[float, float]] = [
            (p[0], p[1]) for p in _HAZARD_POSITIONS
        ]
        self._hazards_spawned = False

        # ── Timers ───────────────────────────────────────────────
        self.create_timer(0.05,  self._publish_ultrasonic)   # 20 Hz
        self.create_timer(10.0,  self._spawn_hazards_once)   # One-shot after 10s

        self.get_logger().info("SimBridge (Jazzy/Harmonic) initialised")

    # ── Callbacks ────────────────────────────────────────────────

    def _joint_states_cb(self, msg: JointState) -> None:
        """
        Derive cumulative encoder ticks from wheel joint positions.

        ros2_control reports joint positions in radians (cumulative).
        Convert to ticks: ticks = radians * (CPR * gear_ratio) / (2π)
        """
        try:
            li = msg.name.index("left_wheel_joint")
            ri = msg.name.index("right_wheel_joint")
        except ValueError:
            return   # Joint not yet reported

        left_pos  = msg.position[li]   # radians
        right_pos = msg.position[ri]

        # Delta radians → delta metres → delta ticks
        d_left_rad  = left_pos  - self._prev_left_pos
        d_right_rad = right_pos - self._prev_right_pos

        wheel_circumference = 2.0 * math.pi * 0.033
        d_left_m  = (d_left_rad  / (2.0 * math.pi)) * wheel_circumference
        d_right_m = (d_right_rad / (2.0 * math.pi)) * wheel_circumference

        self._enc_left  += d_left_m  * self._TICKS_PER_M
        self._enc_right += d_right_m * self._TICKS_PER_M

        self._prev_left_pos  = left_pos
        self._prev_right_pos = right_pos

        msg_l, msg_r = Float32(), Float32()
        msg_l.data = float(self._enc_left)
        msg_r.data = float(self._enc_right)
        self._enc_left_pub.publish(msg_l)
        self._enc_right_pub.publish(msg_r)

    def _odom_cb(self, msg: Odometry) -> None:
        """Cache robot world position for ultrasonic proximity simulation."""
        self._robot_x = msg.pose.pose.position.x
        self._robot_y = msg.pose.pose.position.y

    def _gripper_cb(self, msg: String) -> None:
        """
        Translate string gripper command to ForwardCommandController message.

        Publishes [tilt_position, left_gap, right_gap] as Float64MultiArray
        to /gripper_controller/commands.
        """
        cmd = msg.data
        if cmd not in _GRIPPER_COMMANDS:
            self.get_logger().warning("Unknown gripper command: %s", cmd)
            return

        tilt, left, right = _GRIPPER_COMMANDS[cmd]

        # "stop" means keep current state
        if tilt is None:
            return

        self._gripper_state = [tilt, left, right]

        out = Float64MultiArray()
        out.data = self._gripper_state
        self._gripper_cmd_pub.publish(out)
        self.get_logger().info(
            "Gripper → tilt=%.3f left=%.4f right=%.4f", tilt, left, right
        )

    # ── Periodic publishers ──────────────────────────────────────

    def _publish_ultrasonic(self) -> None:
        """Simulate ultrasonic readings from hazard proximity."""
        left_dist  = 3.0
        right_dist = 3.0

        for hx, hy in self._hazard_xy:
            dist = math.hypot(hx - self._robot_x, hy - self._robot_y)
            if dist < left_dist:
                left_dist = dist
            if dist < right_dist:
                right_dist = dist

        msg_l, msg_r = Float32(), Float32()
        msg_l.data = float(max(0.02, left_dist))
        msg_r.data = float(max(0.02, right_dist))
        self._us_left_pub.publish(msg_l)
        self._us_right_pub.publish(msg_r)

    # ── Hazard spawning ──────────────────────────────────────────

    def _spawn_hazards_once(self) -> None:
        """
        Spawn hazard SDF models into Gazebo Harmonic.

        Uses ros_gz_interfaces/srv/SpawnEntity which wraps the
        /world/<name>/create gz-transport service.
        """
        if self._hazards_spawned:
            return
        self._hazards_spawned = True

        client = self.create_client(SpawnEntity, "/world/apartment/create")
        if not client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error(
                "SpawnEntity service not available — hazards not spawned. "
                "Is Gazebo running with the apartment world?"
            )
            return

        for i, (x, y, z, sx, sy, sz, r, g, b) in enumerate(_HAZARD_POSITIONS):
            sdf = _HAZARD_SDF_TEMPLATE.format(
                name=f"hazard_{i}",
                x=x, y=y, z=z,
                sx=sx, sy=sy, sz=sz,
                r=r, g=g, b=b,
            )
            req = SpawnEntity.Request()
            req.xml = sdf

            future = client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

            if future.result() is not None and future.result().success:
                self.get_logger().info(
                    "Spawned hazard_%d at (%.2f, %.2f)", i, x, y
                )
            else:
                self.get_logger().warning("Failed to spawn hazard_%d", i)


def main() -> None:
    """Entry point for the sim_bridge ROS2 node."""
    rclpy.init()
    node = SimBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()