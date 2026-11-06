"""
Camera Extrinsics Calibration Tool using Floor-Plane 4x4 ArUco Markers (100mm, ID 0 & 1).

Solves:
1. Exact Camera Pitch Angle (downward tilt in degrees)
2. Exact Camera Roll Angle (mounting tilt left/right in degrees)
3. Exact Camera Height (cm above floor plane)
4. Exact Ground Coordinates (X_base, Y_base) of the marker(s) relative to the robot axle
5. Kinematic Cross-Check against CleanupBot5 dimensions (joint, chassis, axle, claws)

Usage:
    # 1. Single image (snapshot from robot):
    python tests/calibrate_extrinsics.py floor_markers.jpg

    # 2. Live ROS2 camera stream:
    python tests/calibrate_extrinsics.py --live

    # 3. Live USB camera:
    python tests/calibrate_extrinsics.py --camera 0

    # Optional: If robot arm was tilted during capture, specify arm pitch:
    python tests/calibrate_extrinsics.py floor_markers.jpg --arm-pitch 0.0

Controls in Live GUI:
    [S]     : Save calibrated extrinsics to MCP/camera_calibration.json
    [Q/ESC] : Exit
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

MCP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MCP_DIR))

CALIBRATION_FILE = MCP_DIR / "camera_calibration.json"

from vision_core import GroundProjector, RobotKinematics, RosBridge

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("extrinsics_calibrator")

MARKER_SIZE_CM = 10.6  # 106mm outer black square edge-to-edge
VALID_IDS = {0, 1}


def get_marker_3d_corners(size_cm: float = MARKER_SIZE_CM) -> np.ndarray:
    """
    3D object coordinates of the 4 corners of a planar ArUco marker centered at origin (Z=0).
    OpenCV ArUco corner order: [Top-Left, Top-Right, Bottom-Right, Bottom-Left].
    """
    half = size_cm / 2.0
    return np.array([
        [-half,  half, 0.0],
        [ half,  half, 0.0],
        [ half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float32)


def load_camera_intrinsics(frame_w: int, frame_h: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Load calibrated intrinsics (K, D) from camera_calibration.json,
    scaled to the current frame resolution.
    """
    k_default = np.array([
        [1600.0 * (frame_w / 1920.0), 0.0, frame_w / 2.0],
        [0.0, 1600.0 * (frame_w / 1920.0), frame_h / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    d_default = np.zeros(5, dtype=np.float64)

    if not CALIBRATION_FILE.exists():
        log.warning("camera_calibration.json not found; using default pinhole intrinsics.")
        return k_default, d_default

    try:
        data = json.loads(CALIBRATION_FILE.read_text())
        ref_w = float(data.get("reference_width_px", 1920))
        ref_h = float(data.get("reference_height_px", 1080))
        scale_x = frame_w / ref_w
        scale_y = frame_h / ref_h

        if "camera_matrix" in data and data.get("intrinsics_calibrated", False):
            m = np.array(data["camera_matrix"], dtype=np.float64)
            k = np.array([
                [m[0, 0] * scale_x, 0.0, m[0, 2] * scale_x],
                [0.0, m[1, 1] * scale_y, m[1, 2] * scale_y],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
            d = np.array(data.get("dist_coeffs", [0.0] * 5), dtype=np.float64)
            # log.info("Loaded calibrated intrinsics: focal=(%.1f, %.1f), principal=(%.1f, %.1f)", k[0, 0], k[1, 1], k[0, 2], k[1, 2])
            return k, d
        else:
            f = float(data.get("focal_length_px", 1600.0)) * scale_x
            k_scaled = np.array([
                [f, 0.0, frame_w / 2.0],
                [0.0, f, frame_h / 2.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
            return k_scaled, d_default
    except Exception as exc:
        log.warning("Could not parse camera_calibration.json (%s); using defaults.", exc)
        return k_default, d_default


def solve_marker_floor_pose(
    corners_2d: np.ndarray,
    k_mtx: np.ndarray,
    dist_coeffs: np.ndarray,
    size_cm: float = MARKER_SIZE_CM,
) -> dict[str, Any] | None:
    """
    Solve the 6DOF pose of a planar floor marker using cv2.solvePnP (IPPE_SQUARE).
    Derives:
    - Normal vector n_c in camera frame
    - Perpendicular camera height h above the floor plane
    - Downward pitch angle (deg)
    - Roll angle (deg)
    - Marker center (X_c, Y_c, Z_c) in camera frame
    """
    obj_3d = get_marker_3d_corners(size_cm)
    corners_reshaped = corners_2d.reshape(-1, 2).astype(np.float32)

    # Use SOLVEPNP_IPPE_SQUARE for optimal planar square solver
    success, rvec, tvec = cv2.solvePnP(
        obj_3d,
        corners_reshaped,
        k_mtx,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success:
        return None

    r_m_c, _ = cv2.Rodrigues(rvec)
    t_m_c = tvec.flatten()

    # Normal vector to marker surface in camera coordinates (+Z_m)
    n_c = r_m_c[:, 2]

    # Ensure normal points upwards towards ceiling (in camera optical frame, up is -Y_c and backwards is -Z_c)
    if n_c[2] > 0:
        n_c = -n_c

    # Perpendicular height from camera optical center to the floor plane (Z=0 in marker frame)
    h_plane = float(-np.dot(n_c, t_m_c))
    if h_plane <= 0:
        h_plane = abs(h_plane)

    # Downward pitch: angle of camera optical axis (+Z_c) below the floor plane
    # Derived from n_c[2] = -sin(pitch)
    sin_p = max(-1.0, min(1.0, -n_c[2]))
    pitch_deg = float(math.degrees(math.asin(sin_p)))

    # Roll angle: tilt of camera around optical axis
    # Derived from xc[2] / yc[2] = -sin(roll) / -cos(roll)
    roll_deg = float(math.degrees(math.atan2(-n_c[0], -n_c[1])))

    # Marker center in camera optical frame
    center_c = (t_m_c[0], t_m_c[1], t_m_c[2])

    return {
        "rvec": rvec,
        "tvec": tvec,
        "n_c": n_c,
        "height_cm": h_plane,
        "pitch_deg": pitch_deg,
        "roll_deg": roll_deg,
        "center_c": center_c,
    }


def analyze_floor_markers(
    frame: np.ndarray,
    arm_pitch_deg: float = 0.0,
    marker_size_cm: float = MARKER_SIZE_CM,
) -> tuple[dict[str, Any] | None, np.ndarray]:
    """
    Detect markers, solve floor poses, and cross-validate against robot kinematics.
    """
    h_px, w_px = frame.shape[:2]
    k_mtx, dist_coeffs = load_camera_intrinsics(w_px, h_px)

    # Detect ArUco markers
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    annotated = frame.copy()
    marker_results: dict[int, dict[str, Any]] = {}

    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
        for i, mid in enumerate(ids.flatten()):
            mid_int = int(mid)
            if mid_int in VALID_IDS:
                pose = solve_marker_floor_pose(corners[i], k_mtx, dist_coeffs, marker_size_cm)
                if pose is not None:
                    marker_results[mid_int] = pose
                    # Draw 3D axis at marker center
                    cv2.drawFrameAxes(annotated, k_mtx, dist_coeffs, pose["rvec"], pose["tvec"], 8.0, 2)

    if len(marker_results) == 0:
        return None, annotated

    # Aggregate estimates
    kinematics = RobotKinematics()
    expected_cam_x, expected_cam_z = kinematics.get_camera_pose(arm_pitch_deg)

    pitches = [p["pitch_deg"] for p in marker_results.values()]
    rolls = [p["roll_deg"] for p in marker_results.values()]
    heights = [p["height_cm"] for p in marker_results.values()]

    consensus_pitch = float(np.mean(pitches))
    consensus_roll = float(np.mean(rolls))
    consensus_height = float(np.mean(heights))

    # Static camera mounting pitch bias = total measured pitch - arm pitch
    camera_fixed_pitch_bias = consensus_pitch - arm_pitch_deg

    # Height discrepancy relative to kinematics model
    height_delta = consensus_height - expected_cam_z

    report: dict[str, Any] = {
        "markers_detected": list(marker_results.keys()),
        "consensus_pitch_deg": consensus_pitch,
        "camera_fixed_pitch_bias_deg": camera_fixed_pitch_bias,
        "consensus_roll_bias_deg": consensus_roll,
        "consensus_height_cm": consensus_height,
        "expected_height_cm": expected_cam_z,
        "height_delta_cm": height_delta,
        "expected_forward_offset_cm": expected_cam_x,
        "arm_pitch_deg": arm_pitch_deg,
        "marker_details": {},
    }

    # Project marker centers onto floor in robot base frame using consensus geometry
    projector = GroundProjector(
        camera_height_cm=consensus_height,
        forward_offset_cm=expected_cam_x,
        camera_fixed_pitch_deg=camera_fixed_pitch_bias,
        camera_roll_bias_deg=consensus_roll,
        use_kinematics=False,
    )

    for mid, p in marker_results.items():
        # Compute ground coordinates of marker center in robot base coordinates
        # Optical coordinates: (X_c, Y_c, Z_c)
        tc = p["center_c"]
        # Ground projection from optical coordinates
        total_p_rad = math.radians(consensus_pitch)
        roll_rad = math.radians(consensus_roll)
        R_c_b, _, _, _ = projector._build_camera_rotation(consensus_pitch)
        p_base = expected_cam_x + (R_c_b @ np.array(tc))[0]
        y_base = (R_c_b @ np.array(tc))[1]

        report["marker_details"][mid] = {
            "pitch_deg": round(p["pitch_deg"], 2),
            "roll_deg": round(p["roll_deg"], 2),
            "height_cm": round(p["height_cm"], 2),
            "x_base_cm": round(float(p_base), 1),
            "y_base_cm": round(float(y_base), 1),
            "dist_from_axle_cm": round(float(math.hypot(p_base, y_base)), 1),
        }

    # Draw HUD Badges
    hud = annotated.copy()
    cv2.rectangle(hud, (10, 10), (w_px - 10, 130), (15, 15, 15), -1)
    cv2.addWeighted(hud, 0.75, annotated, 0.25, 0, annotated)

    h_color = (0, 255, 0) if abs(height_delta) < 0.5 else (0, 200, 255)
    r_color = (0, 255, 0) if abs(consensus_roll) < 1.0 else (0, 200, 255)

    line1 = f"Pitch Bias: {camera_fixed_pitch_bias:+.2f} deg (Total: {consensus_pitch:.2f} deg at arm {arm_pitch_deg:.1f} deg)"
    line2 = f"Roll Bias:  {consensus_roll:+.2f} deg | Height: {consensus_height:.2f} cm (Model: {expected_cam_z:.1f}cm, delta: {height_delta:+.2f}cm)"
    cv2.putText(annotated, line1, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(annotated, line2, (25, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, h_color, 2, cv2.LINE_AA)

    details = " | ".join([f"ID {m}: R={d['dist_from_axle_cm']}cm, X={d['x_base_cm']}cm, Y={d['y_base_cm']}cm" for m, d in report["marker_details"].items()])
    cv2.putText(annotated, details, (25, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

    return report, annotated


def save_extrinsics_to_file(report: dict[str, Any]) -> None:
    """Save calibrated extrinsics into camera_calibration.json."""
    current_data: dict[str, Any] = {}
    if CALIBRATION_FILE.exists():
        try:
            current_data = json.loads(CALIBRATION_FILE.read_text())
        except Exception:
            pass

    current_data["extrinsics_calibrated"] = True
    current_data["pitch_bias_deg"] = round(report["camera_fixed_pitch_bias_deg"], 2)
    current_data["roll_bias_deg"] = round(report["consensus_roll_bias_deg"], 2)
    current_data["camera_height_cm"] = round(report["consensus_height_cm"], 2)
    current_data["forward_offset_cm"] = round(report["expected_forward_offset_cm"], 2)
    current_data["last_calibrated_pitch_deg"] = round(report["consensus_pitch_deg"], 2)

    CALIBRATION_FILE.write_text(json.dumps(current_data, indent=2))
    log.info("Saved extrinsics calibration to: %s", CALIBRATION_FILE)


def print_extrinsics_report(report: dict[str, Any]) -> None:
    """Print clean summary to console."""
    print("\n" + "=" * 68)
    print(" CAMERA EXTRINSICS CALIBRATION REPORT (Floor-Plane 4x4 ArUco)")
    print("=" * 68)
    print(f" Markers Detected:        {report['markers_detected']}")
    print(f" Measured Camera Height:  {report['consensus_height_cm']:.2f} cm (Nominal Model: {report['expected_height_cm']:.1f} cm)")
    print(f" Height Delta:            {report['height_delta_cm']:+.2f} cm")
    print(f" Camera Pitch Bias:       {report['camera_fixed_pitch_bias_deg']:+.2f} deg (Nose-down tilt)")
    print(f" Camera Roll Bias:        {report['consensus_roll_bias_deg']:+.2f} deg (Left/right tilt)")
    print(f" Forward Offset (Axle):   {report['expected_forward_offset_cm']:.1f} cm")
    print("-" * 68)
    for mid, d in report["marker_details"].items():
        print(f" Marker ID {mid}:")
        print(f"   Ground Location:       X = {d['x_base_cm']:+.1f} cm forward, Y = {d['y_base_cm']:+.1f} cm left")
        print(f"   Distance from Axle:    {d['dist_from_axle_cm']:.1f} cm")
        print(f"   Individual Pitch/Roll: pitch = {d['pitch_deg']:.2f} deg, roll = {d['roll_deg']:+.2f} deg")
    print("=" * 68 + "\n")


def live_mode(
    source: str | int = 0,
    is_ros: bool = False,
    arm_pitch_deg: float = 0.0,
    marker_size_cm: float = MARKER_SIZE_CM,
) -> None:
    """Run interactive extrinsics calibration on live camera feed."""
    cap = None
    ros_bridge = None

    if is_ros:
        try:
            ros_bridge = RosBridge()
            log.info("Subscribed to live ROS2 stream (/camera/image_raw + /imu/data)")
        except Exception as exc:
            log.error("Could not initialize ROS2 bridge: %s", exc)
            return
    else:
        cam_idx = int(source) if str(source).isdigit() else str(source)
        cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            log.error("Could not open video capture device: %s", source)
            return

    window_name = "Camera Extrinsics Calibration (ArUco Floor Plane)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    last_report: dict[str, Any] | None = None

    print("\n--- EXTRINSICS CALIBRATION CONTROLS ---")
    print(" [S]     : Save current calibration to camera_calibration.json")
    print(" [Q/ESC] : Quit")
    print("---------------------------------------\n")

    try:
        while True:
            if is_ros and ros_bridge is not None:
                frame = ros_bridge.get_frame()
                arm_pitch = ros_bridge.get_pitch_deg()
                if frame is None:
                    continue
            else:
                ret, frame = cap.read()
                arm_pitch = arm_pitch_deg
                if not ret or frame is None:
                    continue

            report, annotated = analyze_floor_markers(frame, arm_pitch_deg=arm_pitch, marker_size_cm=marker_size_cm)
            if report is not None:
                last_report = report

            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("s"), ord("S")):
                if last_report is not None:
                    save_extrinsics_to_file(last_report)
                    print_extrinsics_report(last_report)
                    print("[SUCCESS] Extrinsics calibration saved to camera_calibration.json!")
                else:
                    log.warning("No floor markers detected yet to save!")

    finally:
        if cap is not None:
            cap.release()
        if ros_bridge is not None:
            ros_bridge.close()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate Camera Extrinsics using Floor ArUco Markers")
    parser.add_argument("image_path", nargs="?", type=Path, default=None, help="Path to single image file")
    parser.add_argument("--camera", type=int, default=None, help="OpenCV VideoCapture index (e.g. 0)")
    parser.add_argument("--live", action="store_true", help="Capture live from ROS2 stream")
    parser.add_argument("--arm-pitch", type=float, default=0.0, help="Arm tilt angle during capture (default: 0.0 deg)")
    parser.add_argument("--marker-size", type=float, default=MARKER_SIZE_CM, help=f"Outer marker width in cm (default: {MARKER_SIZE_CM} cm)")
    parser.add_argument("--save", action="store_true", help="Automatically save results to camera_calibration.json")
    args = parser.parse_args()

    if args.live:
        live_mode(0, is_ros=True, arm_pitch_deg=args.arm_pitch, marker_size_cm=args.marker_size)
    elif args.camera is not None:
        live_mode(args.camera, is_ros=False, arm_pitch_deg=args.arm_pitch, marker_size_cm=args.marker_size)
    elif args.image_path is not None:
        if not args.image_path.exists():
            log.error("Image file not found: %s", args.image_path)
            sys.exit(1)
        frame = cv2.imread(str(args.image_path))
        if frame is None:
            log.error("Could not decode image: %s", args.image_path)
            sys.exit(1)
        report, annotated = analyze_floor_markers(frame, arm_pitch_deg=args.arm_pitch, marker_size_cm=args.marker_size)
        if report is None:
            log.error("No valid ArUco markers (ID 0 or 1) found in image.")
            sys.exit(1)

        out_vis = MCP_DIR / "images" / "extrinsics_calibration_overlay.jpg"
        cv2.imwrite(str(out_vis), annotated)
        print(f"Saved visualization overlay to: {out_vis}")

        if args.save:
            save_extrinsics_to_file(report)
            print("[SUCCESS] Saved to camera_calibration.json")
    else:
        print("Please provide an image file, or run with --live / --camera.")
        parser.print_help()


if __name__ == "__main__":
    main()
