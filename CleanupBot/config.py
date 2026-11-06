"""
Central configuration for all physical robot dimensions, tunable constants, and mission parameters.
Single source of truth across CleanupBot5:
- Hardware kinematics (axle, tilt joint, continuous arm, 3cm claws, camera, BNO085 IMU)
- Mission waypoints (40cm standoff, 16cm grasp contact)
- Gripper aperture & hazard thresholds (4.5cm max opening, 3.17cm choking threshold)
- Dynamic optical calibration loading (VisionMCP/camera_calibration.json)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent
CALIBRATION_FILE = ROOT_DIR / "VisionMCP" / "camera_calibration.json"


# ---------------------------------------------------------------------------
# 1. Physical Hardware Kinematics (Ground Truth Geometry)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RobotKinematicsConfig:
    """
    Fixed physical structural dimensions (CAD / physical build).
    Coordinate frame:
      Origin (0, 0, 0): Ground point directly beneath center of drive wheel axle.
      +X: Forward along robot heading
      +Y: Lateral left
      +Z: Vertical upward from floor
    """
    # Drive Base & Chassis
    wheel_diameter_cm: float = 6.6
    wheel_radius_cm: float = 3.3
    wheel_width_cm: float = 2.5
    wheel_base_cm: float = 14.0       # Center-to-center track width between wheels
    chassis_length_cm: float = 20.0
    chassis_width_cm: float = 10.0
    chassis_thickness_cm: float = 0.3
    chassis_bottom_z_cm: float = 4.4   # Floor clearance of chassis plate
    axle_to_front_cm: float = 8.0     # Front edge of chassis ahead of axle

    # Gripper Arm Tilt Joint (Top-front edge of chassis)
    joint_x_cm: float = 8.0           # Joint hinge pin forward from axle
    joint_z_cm: float = 7.0           # Joint hinge pin height from floor

    # Gripper Claws & Continuous Arm Structure
    arm_structure_length_cm: float = 12.0  # Total length from tilt hinge pin to jaw tips
    claw_finger_length_cm: float = 3.0    # Throat depth of moving claws
    claws_rel_joint_x_cm: float = 12.0    # Clamp tips forward from joint at 0 deg pitch
    claws_rel_joint_z_cm: float = 1.0     # Clamp tips above joint at 0 deg pitch

    # Camera Lens Mounting (Rides on tilting gripper arm)
    cam_rel_joint_x_cm: float = 2.0       # Lens forward from tilt hinge pin
    cam_rel_joint_z_cm: float = 5.0       # Lens above tilt hinge pin
    cam_pitch_bias_deg_nominal: float = 7.76  # Fixed downward tilt relative to arm (ArUco consensus)
    cam_roll_bias_deg_nominal: float = -0.87

    # IMU Sensor Mounting (BNO085 on tilting gripper arm)
    imu_model: str = "BNO085"
    imu_rel_joint_x_cm: float = 2.0       # Sensor forward from tilt hinge pin
    imu_rel_joint_z_cm: float = 3.0       # Sensor above tilt hinge pin

    # Ultrasonic Sensors (HC-SR04 on front corners)
    sonar_left_yaw_deg: float = 45.0      # Angled outward 45 deg left
    sonar_right_yaw_deg: float = -45.0    # Angled outward 45 deg right

    # --- Derived Level Geometry (Arm Pitch = 0 deg) ---
    @property
    def claw_tip_level_x_cm(self) -> float:
        """Clamp tip forward position when stowed level (8.0 + 12.0 = 20.0 cm)."""
        return self.joint_x_cm + self.claws_rel_joint_x_cm

    @property
    def claw_tip_level_z_cm(self) -> float:
        """Clamp tip height when stowed level (7.0 + 1.0 = 8.0 cm)."""
        return self.joint_z_cm + self.claws_rel_joint_z_cm

    @property
    def claw_throat_level_x_cm(self) -> float:
        """Base throat of gripper jaws when stowed level (20.0 - 3.0 = 17.0 cm)."""
        return self.claw_tip_level_x_cm - self.claw_finger_length_cm

    @property
    def cam_level_x_cm(self) -> float:
        """Camera lens forward position when stowed level (8.0 + 2.0 = 10.0 cm)."""
        return self.joint_x_cm + self.cam_rel_joint_x_cm

    @property
    def cam_level_z_cm(self) -> float:
        """Camera lens height when stowed level (7.0 + 5.0 = 12.0 cm)."""
        return self.joint_z_cm + self.cam_rel_joint_z_cm

    @property
    def imu_level_x_cm(self) -> float:
        """BNO085 IMU forward position when stowed level (8.0 + 2.0 = 10.0 cm)."""
        return self.joint_x_cm + self.imu_rel_joint_x_cm

    @property
    def imu_level_z_cm(self) -> float:
        """BNO085 IMU height when stowed level (7.0 + 3.0 = 10.0 cm)."""
        return self.joint_z_cm + self.imu_rel_joint_z_cm

    def get_claw_floor_reach_pitch(self) -> float:
        """
        Compute the arm pitch angle (deg) at which the claws touch the floor (Z_claws = 0).
        Calculates: alpha = asin(joint_z / r) + atan2(claws_rel_z, claws_rel_x).
        """
        r = math.hypot(self.claws_rel_joint_x_cm, self.claws_rel_joint_z_cm)
        phi = math.atan2(self.claws_rel_joint_z_cm, self.claws_rel_joint_x_cm)
        val = self.joint_z_cm / r
        if abs(val) > 1.0:
            return 40.3
        return math.degrees(math.asin(val) + phi)


ROBOT_KINEMATICS = RobotKinematicsConfig()


# ---------------------------------------------------------------------------
# 2. Mission Navigation Distances & Standoff Waypoints
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MissionNavigationConfig:
    """
    Waypoints and distance thresholds for autonomous hazard retrieval.
    """
    # Standoff Pose: Parking position for visual centering & TypeSafe Jev inspection
    # When stowed level, camera sees beyond gripper as close as 40cm (safe 20cm clearance to clamp tip)
    standoff_distance_m: float = 0.40
    standoff_distance_cm: float = 40.0

    # Final Grasp Pose: Axle position where hazard sits deep inside the 3cm claw throat
    grasp_contact_distance_m: float = 0.16
    grasp_contact_distance_cm: float = 16.0

    # Advance distance from standoff to grasp contact
    approach_advance_m: float = 0.24      # 0.40m - 0.16m = 0.24m (24.0cm)
    approach_advance_cm: float = 24.0

    # Minimum visual ground detection range (floor distance visible beyond level gripper)
    min_visible_floor_dist_cm: float = 38.0


MISSION_NAV = MissionNavigationConfig()


def calculate_standoff_pose(
    target_x_m: float,
    target_y_m: float,
    robot_x_m: float = 0.0,
    robot_y_m: float = 0.0,
    approach: str = "head_on",
    standoff_dist_m: float = MISSION_NAV.standoff_distance_m,
) -> dict[str, float]:
    """
    Calculate the exact (x_m, y_m, yaw_deg) standoff pose in the world frame
    at the recommended distance (default 0.40m) and approach angle facing the target object.

    Args:
        target_x_m: Object world X coordinate in meters.
        target_y_m: Object world Y coordinate in meters.
        robot_x_m: Current robot world X coordinate in meters.
        robot_y_m: Current robot world Y coordinate in meters.
        approach: Recommended approach vector: 'head_on', 'approach_from_left', or 'approach_from_right'.
        standoff_dist_m: Standoff distance in meters (default 0.40m).

    Returns:
        {'x_m': float, 'y_m': float, 'yaw_deg': float}
    """
    dx = target_x_m - robot_x_m
    dy = target_y_m - robot_y_m
    theta_los_rad = math.atan2(dy, dx)

    # Approach angle offsets relative to line-of-sight:
    # 'approach_from_left': robot positions at +Y (left of hazard), facing -35 deg toward hazard
    # 'approach_from_right': robot positions at -Y (right of hazard), facing +35 deg toward hazard
    approach_offsets_deg = {
        "head_on": 0.0,
        "approach_from_left": -35.0,
        "approach_from_right": +35.0,
    }
    offset_deg = approach_offsets_deg.get(approach, 0.0)
    theta_standoff_rad = theta_los_rad + math.radians(offset_deg)

    standoff_x_m = target_x_m - standoff_dist_m * math.cos(theta_standoff_rad)
    standoff_y_m = target_y_m - standoff_dist_m * math.sin(theta_standoff_rad)
    yaw_deg = math.degrees(theta_standoff_rad)
    yaw_deg = (yaw_deg + 180.0) % 360.0 - 180.0

    return {
        "x_m": round(standoff_x_m, 2),
        "y_m": round(standoff_y_m, 2),
        "yaw_deg": round(yaw_deg, 1),
    }


# ---------------------------------------------------------------------------
# 3. Gripper Aperture & Hazard Dimensions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GripperApertureConfig:
    """Gripper jaw stroke limits and choking hazard thresholds."""
    max_aperture_cm: float = 4.5           # Maximum physical jaw opening span
    max_hazard_width_cm: float = 4.5       # Upper physical grasp limit
    min_hazard_width_cm: float = 1.0       # Lower physical grasp limit
    choking_hazard_diameter_cm: float = 3.17  # U.S. CPSC 1.25-inch infant choking tube limit
    choking_hazard_length_cm: float = 5.71    # U.S. CPSC 2.25-inch infant choking length limit


GRIPPER_APERTURE = GripperApertureConfig()

# Backwards compatibility aliases
MAX_HAZARD_WIDTH_CM: float = GRIPPER_APERTURE.max_hazard_width_cm
WHEEL_BASE_M: float = ROBOT_KINEMATICS.wheel_base_cm / 100.0
WHEEL_RADIUS_M: float = ROBOT_KINEMATICS.wheel_radius_cm / 100.0


# ---------------------------------------------------------------------------
# 4. Drive & Motion Control
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PIDGains:
    kp: float
    ki: float
    kd: float


DRIVE_LINEAR_PID: PIDGains = PIDGains(kp=1.2, ki=0.05, kd=0.3)
DRIVE_ANGULAR_PID: PIDGains = PIDGains(kp=2.0, ki=0.02, kd=0.5)

ENCODER_CPR: int = 12
GEAR_RATIO: int = 90

MAX_LINEAR_SPEED_MS: float = 0.25
MAX_ANGULAR_SPEED_RS: float = 1.2
APPROACH_SLOW_ZONE_M: float = 0.12
APPROACH_TARGET_DIST_M: float = MISSION_NAV.grasp_contact_distance_m

OBSTACLE_STOP_DIST_M: float = 0.08
OBSTACLE_WARN_DIST_M: float = 0.20

CAMERA_RESOLUTION_W: int = 640
CAMERA_RESOLUTION_H: int = 480


# ---------------------------------------------------------------------------
# 5. Gripper Motor Drive Timings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GripperTimings:
    tilt_down_full: float = 0.85
    tilt_up_full: float = 0.85
    close_small: float = 0.40       # object < 2.0 cm
    close_medium: float = 0.60      # object 2.0–3.5 cm
    close_large: float = 0.80       # object 3.5–4.5 cm
    open_full: float = 0.50
    nudge_rotate_deg: float = 15.0


GRIPPER_TIMINGS: GripperTimings = GripperTimings()

OBJECT_SIZE_BINS: tuple[tuple[float, str], ...] = (
    (2.0, "close_small"),
    (3.5, "close_medium"),
    (4.5, "close_large"),
)


# ---------------------------------------------------------------------------
# 6. ROS2 Topic Names
# ---------------------------------------------------------------------------

TOPIC_CMD_VEL: str = "/cmd_vel"
TOPIC_ODOM: str = "/odom"
TOPIC_ENCODER_LEFT: str = "/encoder/left"
TOPIC_ENCODER_RIGHT: str = "/encoder/right"
TOPIC_IMU: str = "/imu/data"
TOPIC_ULTRASONIC_LEFT: str = "/ultrasonic/left"
TOPIC_ULTRASONIC_RIGHT: str = "/ultrasonic/right"
TOPIC_GRIPPER_CMD: str = "/gripper/command"
TOPIC_CAMERA_RAW: str = "/camera/image_raw"
TOPIC_AGENT_STATUS: str = "/agent/status"


# ---------------------------------------------------------------------------
# 7. Approach Weights
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ApproachWeights:
    clearance: float = 0.50
    alignment: float = 0.30
    distance: float = 0.20


APPROACH_WEIGHTS: ApproachWeights = ApproachWeights()


# ---------------------------------------------------------------------------
# 8. Dynamic Camera Calibration Loader (Hybrid Architecture)
# ---------------------------------------------------------------------------

def load_camera_calibration() -> dict[str, Any]:
    """
    Load dynamic optical calibration from camera_calibration.json,
    falling back to static ROBOT_KINEMATICS nominals for any missing values.
    """
    defaults: dict[str, Any] = {
        "pitch_bias_deg": ROBOT_KINEMATICS.cam_pitch_bias_deg_nominal,
        "roll_bias_deg": ROBOT_KINEMATICS.cam_roll_bias_deg_nominal,
        "camera_height_cm": ROBOT_KINEMATICS.cam_level_z_cm,
        "forward_offset_cm": ROBOT_KINEMATICS.cam_level_x_cm,
        "focal_length_px": 954.17,
        "reference_width_px": 1024,
        "reference_height_px": 768,
    }

    if not CALIBRATION_FILE.exists():
        log.warning("Calibration file %s not found; using nominal defaults.", CALIBRATION_FILE)
        return defaults

    try:
        with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Overlay dynamic calibration on top of nominal defaults
            defaults.update(data)
            return defaults
    except Exception as exc:
        log.warning("Error reading %s: %s; using nominal defaults.", CALIBRATION_FILE, exc)
        return defaults
