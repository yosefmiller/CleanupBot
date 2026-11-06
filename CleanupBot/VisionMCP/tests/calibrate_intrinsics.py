"""
Camera Intrinsics Calibration Tool using 4x4 ArUco Markers (100mm, ID 0 & 1).

Calibrates:
- Camera Matrix K (focal lengths fx, fy; principal point cx, cy)
- Lens Distortion Coefficients D (k1, k2, p1, p2, k3)
- Root-Mean-Square (RMS) reprojection error

Usage Modes:
1. Live Camera (Interactive GUI):
   python tests/calibrate_intrinsics.py --camera 0
   python tests/calibrate_intrinsics.py --live            # via ROS2 /camera/image_raw
   - Move marker page around (various distances, tilts, corners of frame).
   - Press [SPACE] to capture calibration frames (aim for 15-25 diverse frames).
   - Press [C] to compute calibration and display results.
   - Press [S] to save to camera_calibration.json.
   - Press [Q] or [ESC] to exit.

2. Batch Image Directory:
   python tests/calibrate_intrinsics.py --images path/to/snapshots/
   python tests/calibrate_intrinsics.py img1.jpg img2.jpg img3.jpg
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path
import sys
import math
from typing import Any

import cv2
import numpy as np

MCP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MCP_DIR))

CALIBRATION_FILE = MCP_DIR / "camera_calibration.json"

from vision_core import RosBridge

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("intrinsics_calibrator")

MARKER_SIZE_CM = 10.6  # 106mm outer black border edge-to-edge
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


def create_aruco_detector() -> tuple[cv2.aruco.ArucoDetector, Any]:
    """Create OpenCV 4x4 ArUco detector with sub-pixel corner refinement."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    # Enable corner sub-pixel refinement for calibration precision
    if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    return detector, dictionary


def process_frame(
    frame: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], np.ndarray]:
    """
    Detect valid ArUco markers (IDs 0 and 1) in a frame.
    Returns: (list_of_3d_corners, list_of_2d_corners, detected_ids, annotated_frame)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    annotated = frame.copy()
    obj_pts_list: list[np.ndarray] = []
    img_pts_list: list[np.ndarray] = []
    found_ids: list[int] = []

    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
        base_3d = get_marker_3d_corners(MARKER_SIZE_CM)

        for i, mid in enumerate(ids.flatten()):
            if int(mid) in VALID_IDS:
                found_ids.append(int(mid))
                obj_pts_list.append(base_3d.copy())
                # corners[i] has shape (1, 4, 2)
                img_pts_list.append(corners[i].reshape(4, 2).astype(np.float32))

    return obj_pts_list, img_pts_list, found_ids, annotated


def run_calibration(
    all_obj_points: list[np.ndarray],
    all_img_points: list[np.ndarray],
    image_size: tuple[int, int],
) -> dict[str, Any]:
    """
    Run cv2.calibrateCamera on the collected 3D/2D correspondences.
    """
    w, h = image_size
    log.info("Running cv2.calibrateCamera on %d marker views...", len(all_img_points))

    # Initial guess for camera matrix
    f_init = max(w, h) * 0.8
    k_init = np.array([
        [f_init, 0.0, w / 2.0],
        [0.0, f_init, h / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    flags = cv2.CALIB_USE_INTRINSIC_GUESS
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        all_obj_points,
        all_img_points,
        (w, h),
        k_init,
        None,
        flags=flags,
    )

    fx = float(mtx[0, 0])
    fy = float(mtx[1, 1])
    cx = float(mtx[0, 2])
    cy = float(mtx[1, 2])
    fov_x = math.degrees(2.0 * math.atan2(w / 2.0, fx))
    fov_y = math.degrees(2.0 * math.atan2(h / 2.0, fy))
    d_list = [float(v) for v in dist.flatten()[:5]]

    result = {
        "rms_error_px": float(ret),
        "camera_matrix": mtx.tolist(),
        "dist_coeffs": d_list,
        "fx": fx,
        "fy": fy,
        "focal_length_px": (fx + fy) / 2.0,
        "cx": cx,
        "cy": cy,
        "fov_x_deg": fov_x,
        "fov_y_deg": fov_y,
        "image_width": w,
        "image_height": h,
    }
    return result


def save_calibration_to_file(calib: dict[str, Any]) -> None:
    """Update camera_calibration.json with newly calibrated intrinsics."""
    current_data: dict[str, Any] = {}
    if CALIBRATION_FILE.exists():
        try:
            current_data = json.loads(CALIBRATION_FILE.read_text())
        except Exception:
            pass

    current_data["intrinsics_calibrated"] = True
    current_data["focal_length_px"] = round(calib["focal_length_px"], 2)
    current_data["reference_width_px"] = calib["image_width"]
    current_data["reference_height_px"] = calib["image_height"]
    current_data["dist_coeffs"] = [round(v, 6) for v in calib["dist_coeffs"]]
    current_data["camera_matrix"] = [[round(v, 3) for v in row] for row in calib["camera_matrix"]]
    current_data["intrinsics_summary"] = {
        "rms_reprojection_error_px": round(calib["rms_error_px"], 3),
        "fx": round(calib["fx"], 2),
        "fy": round(calib["fy"], 2),
        "cx": round(calib["cx"], 2),
        "cy": round(calib["cy"], 2),
        "fov_x_deg": round(calib["fov_x_deg"], 1),
        "fov_y_deg": round(calib["fov_y_deg"], 1),
    }

    CALIBRATION_FILE.write_text(json.dumps(current_data, indent=2))
    log.info("Saved intrinsic calibration to: %s", CALIBRATION_FILE)


def print_calibration_report(calib: dict[str, Any]) -> None:
    """Print readable summary report to console."""
    print("\n" + "=" * 65)
    print(" CAMERA INTRINSICS CALIBRATION REPORT (OV3660 / 4x4 ArUco)")
    print("=" * 65)
    print(f" Resolution:           {calib['image_width']} x {calib['image_height']} px")
    print(f" RMS Reprojection Err: {calib['rms_error_px']:.3f} px (excellent if < 0.5 px)")
    print(f" Focal Length (fx, fy):{calib['fx']:.2f}, {calib['fy']:.2f} px (mean: {calib['focal_length_px']:.2f} px)")
    print(f" Principal Point (cx): {calib['cx']:.2f} px (center: {calib['image_width']/2:.1f} px)")
    print(f" Principal Point (cy): {calib['cy']:.2f} px (center: {calib['image_height']/2:.1f} px)")
    print(f" Field of View (H x V):{calib['fov_x_deg']:.1f} deg x {calib['fov_y_deg']:.1f} deg")
    d = calib["dist_coeffs"]
    print(f" Distortion (k1, k2):  k1={d[0]:+.5f}, k2={d[1]:+.5f}")
    print(f" Distortion (p1, p2):  p1={d[2]:+.5f}, p2={d[3]:+.5f}, k3={d[4]:+.5f}")
    print("=" * 65 + "\n")


def interactive_gui_mode(source: str | int, is_ros: bool = False) -> None:
    """Interactive capture GUI: live video stream with [SPACE] capture and [C] calibrate."""
    detector, _ = create_aruco_detector()
    cap: cv2.VideoCapture | None = None
    ros_bridge = None

    if is_ros:
        try:
            ros_bridge = RosBridge()
            log.info("Subscribed to live ROS2 camera stream (/camera/image_raw)")
        except Exception as exc:
            log.error("Could not initialize ROS2 bridge: %s", exc)
            return
    else:
        cam_idx = int(source) if str(source).isdigit() else str(source)
        cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            log.error("Could not open video capture device: %s", source)
            return

    all_obj_points: list[np.ndarray] = []
    all_img_points: list[np.ndarray] = []
    captured_frames: list[np.ndarray] = []
    calib_result: dict[str, Any] | None = None
    flash_timer = 0
    w, h = 1920, 1080

    window_name = "Camera Intrinsics Calibration (ArUco 4x4, 100mm)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n--- INTRINSICS CALIBRATION CONTROLS ---")
    print(" [SPACE] : Capture current frame for calibration")
    print(" [C]     : Calculate calibration from captured frames")
    print(" [S]     : Save calibration to camera_calibration.json")
    print(" [R]     : Reset / clear all captured frames")
    print(" [Q/ESC] : Quit")
    print("---------------------------------------\n")

    try:
        while True:
            if is_ros and ros_bridge is not None:
                frame = ros_bridge.get_frame()
                if frame is None:
                    continue
            else:
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

            h, w = frame.shape[:2]
            obj_pts, img_pts, found_ids, annotated = process_frame(frame, detector)

            # Draw HUD overlay
            hud_bg = annotated.copy()
            cv2.rectangle(hud_bg, (0, 0), (w, 65), (20, 20, 20), -1)
            cv2.addWeighted(hud_bg, 0.7, annotated, 0.3, 0, annotated)

            count_str = f"Frames Captured: {len(captured_frames)} / 15-25 needed"
            color_count = (0, 255, 0) if len(captured_frames) >= 12 else (0, 200, 255)
            cv2.putText(annotated, count_str, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color_count, 2, cv2.LINE_AA)

            ids_str = f"Visible IDs: {found_ids if found_ids else 'None detected'}"
            cv2.putText(annotated, ids_str, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

            if calib_result is not None:
                calib_str = f"RMS Error: {calib_result['rms_error_px']:.3f} px | Focal: {calib_result['focal_length_px']:.1f} px | [S] to Save"
                cv2.putText(annotated, calib_str, (w - 650, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                instr_str = "[SPACE]=Capture | [C]=Calibrate | [Q]=Quit"
                cv2.putText(annotated, instr_str, (w - 480, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

            if flash_timer > 0:
                cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), (0, 255, 0), 8)
                flash_timer -= 1

            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break
            elif key == 32:  # SPACE
                if len(obj_pts) > 0:
                    for op, ip in zip(obj_pts, img_pts):
                        all_obj_points.append(op)
                        all_img_points.append(ip)
                    captured_frames.append(frame.copy())
                    flash_timer = 5
                    log.info("Captured frame #%d (total marker observations: %d)", len(captured_frames), len(all_img_points))
                else:
                    log.warning("No valid ArUco markers detected in current frame!")
            elif key in (ord("c"), ord("C")):
                if len(all_img_points) < 6:
                    log.warning("Need at least 6 marker observations to calibrate (currently %d). Press SPACE when marker is visible.", len(all_img_points))
                else:
                    try:
                        calib_result = run_calibration(all_obj_points, all_img_points, (w, h))
                        print_calibration_report(calib_result)
                    except Exception as exc:
                        log.error("Calibration failed: %s", exc)
            elif key in (ord("s"), ord("S")):
                if calib_result is not None:
                    save_calibration_to_file(calib_result)
                    print("[SUCCESS] Calibration saved! You can now run calibrate_extrinsics.py.")
                else:
                    log.warning("Run calibration [C] before saving [S]!")
            elif key in (ord("r"), ord("R")):
                all_obj_points.clear()
                all_img_points.clear()
                captured_frames.clear()
                calib_result = None
                log.info("Reset all captured frames.")

    finally:
        if cap is not None:
            cap.release()
        if ros_bridge is not None:
            ros_bridge.close()
        cv2.destroyAllWindows()


def batch_mode(image_paths: list[str]) -> None:
    """Run calibration on a list of image files."""
    detector, _ = create_aruco_detector()
    all_obj_points: list[np.ndarray] = []
    all_img_points: list[np.ndarray] = []
    w, h = 0, 0
    valid_frames = 0

    log.info("Processing %d candidate calibration images...", len(image_paths))
    for p_str in image_paths:
        p = Path(p_str)
        if not p.exists():
            continue
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        obj_pts, img_pts, found_ids, _ = process_frame(frame, detector)
        if len(obj_pts) > 0:
            for op, ip in zip(obj_pts, img_pts):
                all_obj_points.append(op)
                all_img_points.append(ip)
            valid_frames += 1
            log.info("Loaded '%s': found IDs %s", p.name, found_ids)

    if len(all_img_points) < 6:
        log.error("Insufficient marker observations (%d found, >=6 required).", len(all_img_points))
        return

    calib = run_calibration(all_obj_points, all_img_points, (w, h))
    print_calibration_report(calib)
    save_calibration_to_file(calib)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate Camera Intrinsics using 4x4 ArUco Markers")
    parser.add_argument("images", nargs="*", help="Paths or glob pattern of images to calibrate from")
    parser.add_argument("--camera", type=int, default=None, help="OpenCV VideoCapture camera index (e.g. 0)")
    parser.add_argument("--live", action="store_true", help="Capture live from ROS2 (/camera/image_raw)")
    parser.add_argument("--images-dir", type=Path, default=None, help="Directory containing calibration snapshots")
    args = parser.parse_args()

    if args.live:
        interactive_gui_mode(0, is_ros=True)
    elif args.camera is not None:
        interactive_gui_mode(args.camera, is_ros=False)
    elif args.images_dir is not None:
        files = sorted(glob.glob(str(args.images_dir / "*.jpg")) + glob.glob(str(args.images_dir / "*.png")))
        batch_mode(files)
    elif len(args.images) > 0:
        batch_mode(args.images)
    else:
        # Default fallback: check if camera 0 is available, else prompt usage
        print("No image paths provided. Defaulting to live webcam mode (device 0)...")
        print("Use --live for ROS2, or provide a list of images.")
        interactive_gui_mode(0, is_ros=False)


if __name__ == "__main__":
    main()
