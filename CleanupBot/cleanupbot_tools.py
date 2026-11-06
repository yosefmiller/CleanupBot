"""
Local ROSA tools — hardware control and motor execution primitives.

Minimal macro architecture (3 core tools):
- navigate_to_pose  — Navigate robot to target world pose (meters & degrees) with human-in-the-loop pause
- execute_gripper   — Physical floor grasp reflex with parameterised motor timing from episodic memory
- get_robot_pose    — Return current estimated robot pose in world coordinates (x_m, y_m, yaw_deg)

Safety primitive:
- emergency_stop    — Immediate motion halt and gripper open

Vision perception (scan_for_hazards, verify_grasp) is provided by VisionMCP.
Episodic memory (query_grasp_timing, record_grasp_episode, query_spatial_map) is provided by MemoryMCP.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from langchain.tools import tool
from pydantic import BaseModel, Field

from ros2_bridge.publisher import get_publisher
from cleanupbot_gui import get_visualizer
from config import MISSION_NAV

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Robot Global Pose Tracking Singleton
# ---------------------------------------------------------------------------

class RobotPoseTracker:
    """Maintains estimated robot world position (X, Y in meters, heading in degrees)."""

    def __init__(self, x_m: float = 0.0, y_m: float = 0.0, yaw_deg: float = 0.0) -> None:
        self.x_m = float(x_m)
        self.y_m = float(y_m)
        self.yaw_deg = float(yaw_deg)

    def get_pose(self) -> dict[str, float]:
        return {
            "x_m": round(self.x_m, 3),
            "y_m": round(self.y_m, 3),
            "yaw_deg": round(self.yaw_deg, 1),
        }

    def update_pose(self, x_m: float, y_m: float, yaw_deg: float | None = None) -> None:
        self.x_m = float(x_m)
        self.y_m = float(y_m)
        if yaw_deg is not None:
            self.yaw_deg = float(yaw_deg)


_pose_tracker = RobotPoseTracker()


def get_robot_pose_tracker() -> RobotPoseTracker:
    return _pose_tracker


# ---------------------------------------------------------------------------
# Tool 1 — Navigate to Pose (Global Frame)
# ---------------------------------------------------------------------------

class NavigateToPoseArgs(BaseModel):
    x_m: float = Field(
        description="Target global X coordinate on the floor in meters (from scan_for_hazards world_x_m).",
    )
    y_m: float = Field(
        description="Target global Y coordinate on the floor in meters (from scan_for_hazards world_y_m).",
    )
    yaw_deg: float = Field(
        default=0.0,
        description="Target robot heading in degrees (0.0 = aligned along +X axis facing forward).",
    )


@tool(args_schema=NavigateToPoseArgs)
def navigate_to_pose(
    x_m: float,
    y_m: float,
    yaw_deg: float = 0.0,
) -> dict[str, Any]:
    """
    Navigate the robot to a target world pose (X, Y in meters, yaw in degrees).
    Computes translation distance and turn delta, pauses for operator repositioning
    during embodied development, broadcasts odometry to ROS2 /odom, and updates global pose.
    """
    curr = _pose_tracker.get_pose()
    dx = float(x_m) - curr["x_m"]
    dy = float(y_m) - curr["y_m"]
    dist_m = math.hypot(dx, dy)
    d_theta = float(yaw_deg) - curr["yaw_deg"]

    print("\n" + "═" * 72)
    print("  [HUMAN-IN-THE-LOOP NAVIGATION]")
    print(f"  Current Robot Pose : (X={curr['x_m']:.2f}m, Y={curr['y_m']:.2f}m, θ={curr['yaw_deg']:.1f}°)")
    print(f"  Target Goal Pose   : (X={x_m:.2f}m, Y={y_m:.2f}m, θ={yaw_deg:.1f}°)")
    print(f"  Trajectory Delta   : Drive {dist_m:.2f}m ({dist_m * 100:.1f}cm), Turn {d_theta:+.1f}°")
    print("  ➔ Please physically reposition the robot at the target stand-off pose.")

    vis = None
    try:
        vis = get_visualizer()
        vis.update_pose(curr["x_m"], curr["y_m"], curr["yaw_deg"])
        vis.set_pause(True, f"Reposition robot to (X={x_m:.2f}m, Y={y_m:.2f}m, theta={yaw_deg:.1f} deg) & press ENTER")
    except Exception:
        pass

    try:
        input("  Press [ENTER] to confirm robot has arrived at target pose... ")
    except (EOFError, KeyboardInterrupt):
        pass

    # Update pose tracker
    _pose_tracker.update_pose(x_m, y_m, yaw_deg)

    # Publish updated odometry to ROS2 /odom
    try:
        pub = get_publisher()
        pub.publish_odometry(x_m, y_m, yaw_deg)
        pub.stop()
    except Exception as exc:
        log.warning("Could not publish odometry to ROS2: %s", exc)

    if vis is not None:
        vis.set_pause(False)
        vis.update_pose(x_m, y_m, yaw_deg)

    print("  ✓ Arrival confirmed. Odometry published and world pose updated.")
    print("═" * 72 + "\n")

    return {
        "status": "arrived_at_pose",
        "reached_target": True,
        "distance_traveled_m": round(dist_m, 3),
        "target_pose": {"x_m": round(x_m, 3), "y_m": round(y_m, 3), "yaw_deg": round(yaw_deg, 1)},
        "current_pose": _pose_tracker.get_pose(),
    }


# ---------------------------------------------------------------------------
# Tool 2 — Execute Gripper (Physical Grasp Execution)
# ---------------------------------------------------------------------------

class ExecuteGripperArgs(BaseModel):
    action: str = Field(
        default="clamp",
        description="Gripper action to perform: 'clamp' (tilt down, close jaws, lift) or 'open' (open jaws, release).",
    )
    clamp_duration_s: float = Field(
        default=0.60,
        description=(
            "Motor close drive duration in seconds. "
            "IMPORTANT: Obtain this from query_grasp_timing (e.g. 1.20s for diagonal cylinders) "
            "to compensate for shape-specific physical slip."
        ),
    )


@tool(args_schema=ExecuteGripperArgs)
def execute_gripper(
    action: str = "clamp",
    clamp_duration_s: float = 0.60,
) -> dict[str, Any]:
    """
    Execute the physical gripper reflex:
    1. Tilts gripper down to floor contact and opens jaws.
    2. Clamps jaws closed for clamp_duration_s (grounded in episodic memory).
    3. Lifts arm back up to travel height.

    Always follow this tool call with verify_grasp(phase='post_lift').
    """
    safe_duration = min(max(float(clamp_duration_s), 0.20), 1.50)
    log.info("Executing gripper action '%s' with duration=%.2fs", action, safe_duration)

    print("\n" + "═" * 72)
    print("  [HUMAN-IN-THE-LOOP GRASP EXECUTION]")
    print(f"  Commanded Gripper Action : {action.upper()}")
    print(f"  Motor Drive Duration     : {safe_duration:.2f}s (grounded in episodic memory)")
    print(f"  ➔ Please roll robot forward from {int(MISSION_NAV.standoff_distance_cm)}cm standoff to {int(MISSION_NAV.grasp_contact_distance_cm)}cm grasp contact")
    print("    (hazard placed deep in the 3cm claw throat) / verify physical grasp.")

    vis = None
    try:
        vis = get_visualizer()
        vis.set_pause(
            True,
            f"Gripper {action} ({safe_duration:.2f}s): Roll forward to {int(MISSION_NAV.grasp_contact_distance_cm)}cm grasp contact & press ENTER",
        )
    except Exception:
        pass

    try:
        input("  Press [ENTER] to confirm physical grasp execution... ")
    except (EOFError, KeyboardInterrupt):
        pass

    # Publish hardware gripper command to ROS2
    try:
        pub = get_publisher()
        pub.publish_gripper_command(f"{action}:{safe_duration:.2f}")
    except Exception as exc:
        log.warning("Could not publish gripper command to ROS2: %s", exc)

    if vis is not None:
        vis.set_pause(False)

    print("  ✓ Grasp execution confirmed. Arm lifted to travel height.")
    print("═" * 72 + "\n")

    return {
        "action": action,
        "clamp_duration_used_s": round(safe_duration, 2),
        "arm_lifted": True,
        "status": (
            f"Physical gripper {action} completed with {safe_duration:.2f}s motor drive. "
            "Arm raised. Call verify_grasp(phase='post_lift') now to audit grasp hold."
        ),
    }


# ---------------------------------------------------------------------------
# Tool 3 — Get Robot Pose
# ---------------------------------------------------------------------------

class GetRobotPoseArgs(BaseModel):
    pass


@tool(args_schema=GetRobotPoseArgs)
def get_robot_pose() -> dict[str, float]:
    """
    Return current estimated robot pose in world coordinates (x_m, y_m, yaw_deg).
    Call this tool with NO arguments.
    """
    return _pose_tracker.get_pose()


# ---------------------------------------------------------------------------
# Tool 4 — Emergency Stop
# ---------------------------------------------------------------------------

class EmergencyStopArgs(BaseModel):
    reason: str = Field(
        default="unspecified",
        description="Human-readable reason for the emergency stop.",
    )


@tool(args_schema=EmergencyStopArgs)
def emergency_stop(reason: str = "unspecified") -> dict[str, Any]:
    """
    Immediately halt all motion and open the gripper.
    Call whenever any tool returns an error or when unsafe conditions are detected.
    """
    try:
        pub = get_publisher()
        pub.stop()
        pub.publish_gripper_command("open")
    except Exception as exc:
        log.warning("Emergency stop ROS2 publish error: %s", exc)

    log.critical("EMERGENCY STOP: %s", reason)
    return {"stopped": True, "reason": reason}


# ---------------------------------------------------------------------------
# Backward-Compatibility Aliases
# ---------------------------------------------------------------------------

class _NavigateToHazardArgs(BaseModel):
    target_x_cm: float | None = Field(default=None, description="Hazard X in cm.")
    target_y_cm: float | None = Field(default=None, description="Hazard Y in cm.")
    target_x_m: float | None = Field(default=None, description="Hazard X in meters.")
    target_y_m: float | None = Field(default=None, description="Hazard Y in meters.")


@tool(args_schema=_NavigateToHazardArgs)
def navigate_to_hazard(
    target_x_cm: float | None = None,
    target_y_cm: float | None = None,
    target_x_m: float | None = None,
    target_y_m: float | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper for navigate_to_pose."""
    x = target_x_m if target_x_m is not None else ((target_x_cm or 0.0) / 100.0)
    y = target_y_m if target_y_m is not None else ((target_y_cm or 0.0) / 100.0)
    return navigate_to_pose.invoke({"x_m": x, "y_m": y, "yaw_deg": 0.0})


class _ApproachArgs(BaseModel):
    pixel_offset: int = Field(default=0, description="Horizontal pixel offset.")
    distance_cm: float = Field(default=40.0, description="Distance in cm (default 40.0cm for camera floor visibility beyond level gripper).")


@tool(args_schema=_ApproachArgs)
def approach_hazard(pixel_offset: int = 0, distance_cm: float = 40.0) -> dict[str, Any]:
    """Compatibility wrapper for approaching."""
    curr = _pose_tracker.get_pose()
    rad = math.radians(curr["yaw_deg"])
    dist_m = distance_cm / 100.0
    target_x = curr["x_m"] + dist_m * math.cos(rad)
    target_y = curr["y_m"] + dist_m * math.sin(rad)
    return navigate_to_pose.invoke({"x_m": target_x, "y_m": target_y, "yaw_deg": curr["yaw_deg"]})


class _ExecuteMotorClampArgs(BaseModel):
    clamp_duration_s: float = Field(default=0.60, description="Close duration in seconds.")
    width_cm: float = Field(default=3.0, description="Object width in cm.")


@tool(args_schema=_ExecuteMotorClampArgs)
def execute_motor_clamp(clamp_duration_s: float = 0.60, width_cm: float = 3.0) -> dict[str, Any]:
    """Compatibility wrapper for execute_gripper."""
    return execute_gripper.invoke({"action": "clamp", "clamp_duration_s": clamp_duration_s})


class _GetRobotStatusArgs(BaseModel):
    pass


@tool(args_schema=_GetRobotStatusArgs)
def get_robot_status() -> dict[str, Any]:
    """Compatibility wrapper returning robot pose and status."""
    pose = _pose_tracker.get_pose()
    return {
        "global_x_m": pose["x_m"],
        "global_y_m": pose["y_m"],
        "yaw_deg": pose["yaw_deg"],
        "gripper_open": True,
        "status": "operational",
    }