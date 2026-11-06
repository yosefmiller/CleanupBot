"""
Vision FastMCP Server

Exposes two primary tools over stdio JSON-RPC optimized for small local LLMs:
- scan_for_hazards: Open-vocabulary YOLO hazard detection with ground projection geometry.
- verify_grasp: TypeSafe Jev visual audit of the parallel jaw tips.
"""

from __future__ import annotations

import logging
import os
import sys
import math
from typing import Any

os.environ.setdefault("ROS_LOG_DIR", "/tmp/ros_log")
try:
    os.makedirs("/tmp/ros_log", exist_ok=True)
except Exception:
    pass

from mcp.server import FastMCP

from vision_core import GraspVerifier, HazardDetector, RosBridge

# Logging setup (all logs go to stderr so stdout is reserved for JSON-RPC)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s [%(name)s]: %(message)s",
    stream=sys.stderr,
)
logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.WARNING)
log = logging.getLogger("mcp_vision_server")

# Initialize modular components
_ros: RosBridge = RosBridge()
_detector: HazardDetector = HazardDetector()
_verifier: GraspVerifier = GraspVerifier()

# FastMCP application instance
mcp: FastMCP = FastMCP("vision")


@mcp.tool()
def scan_for_hazards(query: str | None = None) -> dict[str, Any]:
    """
    Scan the floor for graspable hazards using class-agnostic YOLO.
    Automatically reads the robot's current pose from ROS /odom, computes 3D ground projection,
    and outputs hazards in world coordinates (meters).
    Targeted semantic queries automatically utilize TypeSafe Jev Box locator.

    Args:
        query: Optional specific object name to search for (e.g. 'battery', 'ball', 'candle').
               If None, scans for all general floor clutter and objects.

    Returns:
        hazards: A list of detected objects. Each hazard includes:
            - id (str): Unique hazard identifier (e.g. 'hazard_1').
            - label (str): Object category (e.g. 'toy', 'battery', 'container').
            - world_x_m (float): World X coordinate in meters.
            - world_y_m (float): World Y coordinate in meters.
            - shape (str): YOLO raw class (e.g. 'battery', 'toy block', 'cube').
            - width_cm (float): Physical head-on lateral width in centimetres.
            - is_diagonal (bool): True if target rests diagonally/rotated (from TypeSafe Jev).
            - needs_extended_clamp (bool): True if rotational torque requires extra motor duration.
            - approach (str): 'head_on' | 'approach_from_left' | 'approach_from_right'.
            - shape_symmetry (str): 'symmetric_round' | 'elongated' | 'irregular'.
            - is_graspable (bool): True if within gripper span (1.0cm to 4.5cm).
        robot_pose: Dict with current odom (x_m, y_m, yaw_deg).
        summary: Natural-language overview of the floor scene in world coordinates.
    """
    frame = _ros.get_frame()
    if frame is None:
        return {
            "hazards": [],
            "robot_pose": {"x_m": 0.0, "y_m": 0.0, "yaw_deg": 0.0},
            "summary": "No camera frame received from /camera/image_raw.",
            "error": "no_frame",
        }

    # Automatically pull current robot pose from ROS /odom
    rx_m, ry_m, ryaw_rad = _ros.get_odom_pose()
    imu_pitch_deg: float = _ros.get_pitch_deg()
    raw_hazards: list[dict[str, Any]] = _detector.detect(frame, imu_pitch_deg, query=query)

    cos_h = math.cos(ryaw_rad)
    sin_h = math.sin(ryaw_rad)

    # Perform atomic Jev 3D orientation inspection on the primary target hazard
    primary_orientation: dict[str, Any] = {
        "approach": "head_on",
        "shape_symmetry": "symmetric_round",
        "alignment": "standard_clamp",
        "is_diagonal": False,
        "needs_extended_clamp": False,
        "can_reorient": False,
    }

    if raw_hazards:
        target = raw_hazards[0]
        hx1, hy1, hx2, hy2 = target["bbox_px"]
        fh, fw = frame.shape[:2]
        pad_w = max(24, int((hx2 - hx1) * 0.35))
        pad_h = max(24, int((hy2 - hy1) * 0.35))
        cy1, cy2 = max(0, hy1 - pad_h), min(fh, hy2 + pad_h)
        cx1, cx2 = max(0, hx1 - pad_w), min(fw, hx2 + pad_w)
        crop = frame[cy1:cy2, cx1:cx2]

        try:
            primary_orientation = _verifier.inspect_orientation(
                crop,
                label=target.get("label"),
                width_cm=target.get("width_cm"),
            )
        except Exception as err:
            log.warning("TypeSafe Jev orientation analysis skipped: %s", err)

    compact_hazards: list[dict[str, Any]] = []

    for idx, h in enumerate(raw_hazards, 1):
        # Local metric coordinates relative to robot axle (convert cm to meters)
        x_loc_m = float(h.get("x_cm", 0.0)) / 100.0
        y_loc_m = float(h.get("y_cm", 0.0)) / 100.0

        # Transform to global world coordinates in meters
        glob_x_m = rx_m + x_loc_m * cos_h - y_loc_m * sin_h
        glob_y_m = ry_m + x_loc_m * sin_h + y_loc_m * cos_h

        is_target = (idx == 1)
        app_mode = str(primary_orientation.get("approach", "head_on")) if is_target else "head_on"

        # Compute exact standoff pose at recommended distance (0.40m) and approach angle
        dx = glob_x_m - rx_m
        dy = glob_y_m - ry_m
        theta_los = math.atan2(dy, dx)
        app_offsets_deg = {
            "head_on": 0.0,
            "approach_from_left": -35.0,
            "approach_from_right": +35.0,
        }
        offset_deg = app_offsets_deg.get(app_mode, 0.0)
        theta_standoff = theta_los + math.radians(offset_deg)
        so_x = glob_x_m - 0.40 * math.cos(theta_standoff)
        so_y = glob_y_m - 0.40 * math.sin(theta_standoff)
        so_yaw = (math.degrees(theta_standoff) + 180.0) % 360.0 - 180.0

        compact_hazards.append({
            "id": f"hazard_{idx}",
            "label": str(h.get("label", "hazard")),
            "color": str(h.get("color", "unknown")),
            "color_family": str(h.get("color_family", "unknown")),
            "is_multicolored": bool(h.get("is_multicolored", False)),
            "world_x_m": round(float(glob_x_m), 2),
            "world_y_m": round(float(glob_y_m), 2),
            "standoff_pose": {
                "x_m": round(float(so_x), 2),
                "y_m": round(float(so_y), 2),
                "yaw_deg": round(float(so_yaw), 1),
            },
            "shape": str(h.get("raw_class", "object")),
            "width_cm": round(float(h.get("width_cm", 0.0)), 1),
            "is_diagonal": bool(primary_orientation.get("is_diagonal", False)) if is_target else False,
            "needs_extended_clamp": bool(primary_orientation.get("needs_extended_clamp", False)) if is_target else False,
            "approach": app_mode,
            "shape_symmetry": str(primary_orientation.get("shape_symmetry", "symmetric_round")) if is_target else "unknown",
            "is_graspable": bool(h.get("is_graspable", False)),
        })

    if compact_hazards:
        desc_items = [
            f"#{item['id']} '{item['label']}' at world (X={item['world_x_m']}m, Y={item['world_y_m']}m), "
            f"standoff: (X={item['standoff_pose']['x_m']}m, Y={item['standoff_pose']['y_m']}m, theta={item['standoff_pose']['yaw_deg']} deg), "
            f"width: {item['width_cm']}cm, diagonal: {item['is_diagonal']}, approach: {item['approach']}"
            for item in compact_hazards
        ]
        summary = f"Detected {len(compact_hazards)} hazard(s) in world frame: " + "; ".join(desc_items)
    else:
        summary = "No hazards detected on floor in current camera view."

    return {
        "hazards": compact_hazards,
        "robot_pose": {
            "x_m": round(rx_m, 2),
            "y_m": round(ry_m, 2),
            "yaw_deg": round(math.degrees(ryaw_rad), 1),
        },
        "summary": summary,
    }


@mcp.tool()
def verify_grasp(phase: str = "post_lift") -> dict[str, Any]:
    """
    Verify whether the robot gripper successfully grasped an object.

    Args:
        phase: Grasp phase to verify: 'pre_lift' (audits closed jaws on floor before lifting)
               or 'post_lift' (audits jaws raised in the air).

    Returns:
        dict with:
            - grasped (bool): True if an object is securely clamped between the jaws.
            - outcome (str): 'SUCCESS' if holding object, 'SLIP' if dropped pre-lift, or 'GRAVITY_SLIP' if dropped post-lift.
            - confidence (float): Visual verification confidence score between 0.0 and 1.0.
            - note (str): Brief human-readable description of the visual inspection result.
    """
    frame = _ros.get_frame()
    if frame is None:
        return {
            "grasped": False,
            "outcome": "SLIP" if phase == "pre_lift" else "GRAVITY_SLIP",
            "confidence": 0.0,
            "note": "No camera frame received from ROS2 /camera/image_raw.",
            "error": "no_frame",
        }

    raw_verdict = _verifier.verify(frame, phase=phase)
    grasped = bool(raw_verdict.get("grasped", False))
    return {
        "grasped": grasped,
        "outcome": str(raw_verdict.get("outcome", "SUCCESS" if grasped else "SLIP")),
        "confidence": round(float(raw_verdict.get("confidence", 0.0)), 2),
        "note": str(raw_verdict.get("note", "")),
    }


if __name__ == "__main__":
    log.info("Starting Vision FastMCP Server on stdio...")
    mcp.run(transport="stdio")
