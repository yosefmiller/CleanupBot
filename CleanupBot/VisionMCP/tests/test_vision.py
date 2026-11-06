"""
Live Vision & Perception Test for CleanupBot5.

Runs directly on the live ROS2 camera stream (/camera/image_raw + /imu/data) to test:
1. Real-time open-vocabulary hazard detection with YOLO-World.
2. Perspective floor grid projection (10 cm increments) with pitch compensation.
3. Metric 3D ground distance from axle (dist in cm), horizontal bearing, and grasp width.
4. On-demand TypeSafe Jev System One VLM audits (grasp verification & orientation analysis).

Terminology:
    dist  : Ground distance (cm) from the robot's front drive wheel axle center to the object footprint.
    bearing: Angular offset (degrees) from forward centerline (+ = right, - = left).
    w     : Estimated head-on physical width (cm) projected across the parallel gripper jaws.

Controls in Live GUI:
    [j]     : Trigger TypeSafe Jev grasp audit on live gripper jaws
    [o]     : Trigger TypeSafe Jev orientation, approach & shape symmetry check on nearest hazard
    [s]     : Save current live snapshot & crops to disk
    [q/ESC] : Quit

Usage:
    MCP/.pixi/envs/default/bin/python MCP/tests/test_vision.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

MCP_DIR = Path(__file__).resolve().parent.parent
IMAGES_DIR = MCP_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(MCP_DIR))

from vision_core import GraspVerifier, HazardDetector, RosBridge

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("test_vision")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Perception & Jev VLM Test")
    parser.add_argument("--pitch", type=float, default=0.0, help="IMU pitch override in degrees (0.0 uses live IMU)")
    parser.add_argument("--image", type=Path, default=None, help="Path to snapshot image file (runs offline without ROS2)")
    parser.add_argument("--stream", type=str, default=None, help="HTTP/MJPEG stream URL (e.g. http://127.0.0.1:8081/stream)")
    parser.add_argument("--live", action="store_true", help="Connect directly to livestream at http://127.0.0.1:8081/stream")
    parser.add_argument("--query", type=str, default=None, help="Target object query (e.g. 'yellow cube', 'battery', 'candle')")
    args = parser.parse_args()

    print("=" * 68)
    print("  CleanupBot5: Live Perception & TypeSafe Jev Test")
    print("=" * 68)

    ros = None
    stream_cap = None
    static_frame = None

    stream_url = args.stream or ("http://127.0.0.1:8081/stream" if args.live else None)

    if args.image is not None:
        if not args.image.exists():
            log.error("Image file not found: %s", args.image)
            sys.exit(1)
        static_frame = cv2.imread(str(args.image))
        if static_frame is None:
            log.error("Could not decode image: %s", args.image)
            sys.exit(1)
        log.info("Loaded snapshot image: %s (%dx%d)", args.image.name, static_frame.shape[1], static_frame.shape[0])
    elif stream_url:
        log.info("Connecting to HTTP stream: %s ...", stream_url)
        stream_cap = cv2.VideoCapture(stream_url)
        if not stream_cap.isOpened():
            log.error("Could not open video stream at %s", stream_url)
            sys.exit(1)
        log.info("Successfully connected to livestream: %s", stream_url)
    else:
        ros = RosBridge(frame_timeout_s=5.0)
        log.info("Connecting to live ROS2 stream (/camera/image_raw + /imu/data)...")

    detector = HazardDetector()
    verifier = GraspVerifier()
    current_query: str | None = args.query
    is_paused: bool = False
    last_frame: np.ndarray | None = None
    target_index: int = 0
    active_target_bbox: tuple[int, int, int, int] | None = None

    log.info("Controls: [g] Grasp Audit | [o] Orient Target | [t] Cycle Target | [f] Find Query | [space] Pause | [s] Save | [q] Exit")

    gui_available = bool(os.environ.get("DISPLAY"))
    if not gui_available:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        log.info("No DISPLAY detected. Operating in headless mode; snapshots will be saved to disk.")
    frame_count = 0
    last_jev_status = ""

    try:
        while True:
            frame_count += 1
            if is_paused and last_frame is not None:
                frame = last_frame.copy()
            elif static_frame is not None:
                frame = static_frame.copy()
                pitch_deg = args.pitch
                last_frame = frame.copy()
            elif stream_cap is not None:
                ret, frame = stream_cap.read()
                if not ret or frame is None:
                    time.sleep(0.03)
                    continue
                pitch_deg = args.pitch
                last_frame = frame.copy()
            elif ros is not None:
                frame = ros.get_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue
                pitch_deg = ros.get_pitch_deg() if args.pitch == 0.0 else args.pitch
                last_frame = frame.copy()
            else:
                break

            # 1. Run open-vocabulary hazard detection
            hazards: list[dict[str, Any]] = detector.detect(frame, imu_pitch_deg=pitch_deg, query=current_query)

            # 2. Draw perspective floor grid (10cm lines)
            canvas = detector.projector.draw_floor_grid(frame, imu_pitch_deg=pitch_deg)
            h_can, w_can = canvas.shape[:2]

            # 3. Draw bounding boxes & metric overlays (with bounds clamping)
            active_target_idx = (target_index % len(hazards)) if hazards else -1

            for h_idx, h in enumerate(hazards):
                x1, y1, x2, y2 = h["bbox_px"]
                is_target = (h_idx == active_target_idx)
                box_color = (0, 255, 0) if h["is_graspable"] else (0, 140, 255)

                # Distinct floor contact point marker (cyan ring with center dot)
                if h.get("contact_point_px"):
                    cpx, cpy = h["contact_point_px"]
                    cpx = int(np.clip(cpx, 2, w_can - 3))
                    cpy = int(np.clip(cpy, 2, h_can - 3))
                    cv2.circle(canvas, (cpx, cpy), 4, (255, 255, 0), 1, cv2.LINE_AA)
                    cv2.circle(canvas, (cpx, cpy), 1, (255, 255, 0), -1)

                # Draw bounding box (magenta border with yellow corner brackets for active Jev target)
                if is_target:
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    c_len = min(14, max(6, (x2 - x1) // 4))
                    # Top-left corner
                    cv2.line(canvas, (x1, y1), (x1 + c_len, y1), (0, 255, 255), 2)
                    cv2.line(canvas, (x1, y1), (x1, y1 + c_len), (0, 255, 255), 2)
                    # Top-right corner
                    cv2.line(canvas, (x2, y1), (x2 - c_len, y1), (0, 255, 255), 2)
                    cv2.line(canvas, (x2, y1), (x2, y1 + c_len), (0, 255, 255), 2)
                    # Bottom-left corner
                    cv2.line(canvas, (x1, y2), (x1 + c_len, y2), (0, 255, 255), 2)
                    cv2.line(canvas, (x1, y2), (x1, y2 - c_len), (0, 255, 255), 2)
                    # Bottom-right corner
                    cv2.line(canvas, (x2, y2), (x2 - c_len, y2), (0, 255, 255), 2)
                    cv2.line(canvas, (x2, y2), (x2, y2 - c_len), (0, 255, 255), 2)
                else:
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 1)

                diag_tag = " [diag]" if h.get("is_diagonal", False) else ""
                conf_pct = int(round(h["confidence"] * 100))
                prefix = "[TARGET] " if is_target else ""
                info = f"{prefix}{h['label']} ({conf_pct}%){diag_tag} | {h['distance_cm']:.0f}cm w={h['width_cm']:.1f}cm"

                # Measure text to clamp inside frame boundaries
                (tw, th), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                tx = max(4, min(x1, w_can - tw - 6))
                ty = y1 - 4
                top_margin = 62 if last_jev_status else 46
                if ty < top_margin:
                    ty = min(h_can - 6, y2 + th + 4)

                # Background badge behind label for readability
                badge_bg = (40, 0, 40) if is_target else (15, 15, 15)
                text_color = (255, 100, 255) if is_target else box_color
                cv2.rectangle(canvas, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2), badge_bg, cv2.FILLED)
                if is_target:
                    cv2.rectangle(canvas, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2), (255, 0, 255), 1)
                cv2.putText(canvas, info, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.38, text_color, 1, cv2.LINE_AA)

            # 4. Status HUD Banner (two lines + optional Jev status)
            banner_h = 58 if last_jev_status else 42
            overlay = canvas.copy()
            cv2.rectangle(overlay, (0, 0), (w_can, banner_h), (12, 12, 12), cv2.FILLED)
            cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

            # Line 1: State & Hazard summary
            status_tag = "  [PAUSED]" if is_paused else ""
            q_tag = f"Target: '{current_query}'" if current_query else "Target: [ALL CLUTTER]"
            hud_line1 = f"Hazards: {len(hazards)} | {q_tag} | Pitch: {pitch_deg:+.1f} deg{status_tag}"
            cv2.putText(canvas, hud_line1, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)

            # Line 2: Keyboard shortcuts
            hud_line2 = "Shortcuts: [g] Grasp  [o] Orient  [t] Cycle Target  [f] Find  [space] Pause  [s] Save  [q] Exit"
            cv2.putText(canvas, hud_line2, (8, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA)

            # Line 3: Optional Jev Status
            if last_jev_status:
                cv2.putText(canvas, f"Jev: {last_jev_status}", (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)

            # 5. Render GUI window
            if gui_available:
                try:
                    cv2.imshow("CleanupBot5 Live Perception Feed", canvas)
                    key = cv2.waitKey(1) & 0xFF

                    if key in (ord('q'), ord('Q'), 27):
                        break

                    # [space] - Pause / Resume feed
                    elif key in (ord(' '), 32):
                        is_paused = not is_paused
                        log.info("Feed %s", "PAUSED (press space to resume)" if is_paused else "RESUMED")

                    # [f] - Target Search Query
                    elif key in (ord('f'), ord('F')):
                        print("\n" + "=" * 60)
                        new_q = input("Enter target object query (or press Enter for all clutter): ").strip()
                        current_query = new_q if new_q else None
                        log.info("Active query set to: %s", current_query or "ALL CLUTTER")
                        print("=" * 60 + "\n")

                    # [g] - Trigger TypeSafe Jev Grasp Audit (also accepts 'j')
                    elif key in (ord('g'), ord('G'), ord('j'), ord('J')):
                        log.info("Triggering TypeSafe Jev grasp audit on live frame...")
                        try:
                            jaw_crop = verifier.crop_jaw_roi(frame)
                            cv2.imwrite(str(IMAGES_DIR / "jaw_crop_test.jpg"), jaw_crop)

                            verdict = verifier.verify(frame, phase="post_lift")
                            tok = verdict.get("tokens", {})
                            # GUI display: concise status without token bloat
                            last_jev_status = f"Grasped={verdict['grasped']} ({verdict.get('grip_state', 'unknown')}, Conf={verdict['confidence']:.2f})"
                            # Console log: full details and token counts
                            print("\n" + "-" * 62)
                            print("[Live TypeSafe Jev Grasp Audit Response]:")
                            print(json.dumps(verdict, indent=2))
                            print(f"Token Usage: Input: {tok.get('input', 0)} | Output: {tok.get('output', 0)} | Total: {tok.get('total', 0)}")
                            print("-" * 62 + "\n")
                        except Exception as err:
                            log.error("Jev grasp audit error: %s", err)
                            last_jev_status = f"Grasp Error: {err}"

                    # [t] - Cycle active target hazard for Jev orientation
                    elif key in (ord('t'), ord('T'), 9):
                        if hazards:
                            target_index = (target_index + 1) % len(hazards)
                            cur_t = hazards[target_index]
                            log.info("Selected active target #%d: '%s' (dist=%.1fcm, w=%.1fcm)",
                                     target_index + 1, cur_t['label'], cur_t['distance_cm'], cur_t['width_cm'])

                    # [o] - Trigger TypeSafe Jev Orientation & Approach Check
                    elif key in (ord('o'), ord('O')):
                        if hazards:
                            active_idx = target_index % len(hazards)
                            target = hazards[active_idx]
                            hx1, hy1, hx2, hy2 = target["bbox_px"]
                            # Contextual padding (35% margin, min 24px) retains floor plane perspective lines for Jev
                            pad_w = max(24, int((hx2 - hx1) * 0.35))
                            pad_h = max(24, int((hy2 - hy1) * 0.35))
                            cy1, cy2 = max(0, hy1 - pad_h), min(h_can, hy2 + pad_h)
                            cx1, cx2 = max(0, hx1 - pad_w), min(w_can, hx2 + pad_w)
                            crop = frame[cy1:cy2, cx1:cx2]
                            cv2.imwrite(str(IMAGES_DIR / "hazard_crop_test.jpg"), crop)

                            log.info("Querying Jev orientation analysis on target #%d '%s' (dist=%.1fcm, w=%.1fcm)...",
                                     active_idx + 1, target['label'], target['distance_cm'], target['width_cm'])
                            try:
                                o_verdict = verifier.inspect_orientation(
                                    crop,
                                    label=target.get("label"),
                                    width_cm=target.get("width_cm"),
                                )
                                tok = o_verdict.get("tokens", {})
                                last_jev_status = (
                                    f"[{target['label']} @ {target['distance_cm']:.0f}cm] "
                                    f"Approach={o_verdict['approach']} | Align={o_verdict['alignment']} | Shape={o_verdict['shape_symmetry']}"
                                )
                                print("\n" + "=" * 64)
                                print(f"  LIVE JEV ORIENTATION COMPUTATION FOR TARGET:")
                                print(f"  Target Object: '{target['label']}' (#{active_idx + 1} of {len(hazards)})")
                                print(f"  Distance:      {target['distance_cm']:.1f} cm")
                                print(f"  Width:         {target['width_cm']:.1f} cm")
                                print(f"  Bounding Box:  {target['bbox_px']}")
                                print("-" * 64)
                                print(f"  Approach:      {o_verdict['approach']}")
                                print(f"  Alignment:     {o_verdict['alignment']} (needs_extended_clamp={o_verdict['needs_extended_clamp']})")
                                print(f"  Shape:         {o_verdict['shape_symmetry']} (can_reorient={o_verdict['can_reorient']})")
                                print(f"  Confidence:    {o_verdict.get('confidence', 0.0):.2f}")
                                print(f"  Token Usage:   Input={tok.get('input', 0)}, Output={tok.get('output', 0)}, Total={tok.get('total', 0)}")
                                print("=" * 64 + "\n")
                            except Exception as err:
                                log.error("Jev orientation error: %s", err)
                                last_jev_status = f"Orient Error: {err}"
                        else:
                            log.info("No hazards currently in view to inspect orientation.")
                            last_jev_status = "No hazard visible"

                    # [s] - Save current frame snapshot
                    elif key in (ord('s'), ord('S')):
                        out_path = IMAGES_DIR / "vision_live_feed.jpg"
                        cv2.imwrite(str(out_path), canvas)
                        log.info("Saved current annotated live snapshot to: %s", out_path.name)
                        last_jev_status = f"Snapshot saved ({out_path.name})"

                except cv2.error:
                    gui_available = False
                    log.info("No GUI display detected. Operating in headless mode; saving snapshots...")

            # Headless periodic snapshot or static image export
            if not gui_available:
                if static_frame is not None or frame_count % 30 == 0:
                    out_path = IMAGES_DIR / "vision_live_feed.jpg"
                    cv2.imwrite(str(out_path), canvas)
                    log.info("Saved snapshot (%d hazard(s) visible) to %s", len(hazards), out_path.name)
                if static_frame is not None:
                    print(f"\n[Detection Results on {args.image.name}]:")
                    for idx, h in enumerate(hazards, 1):
                        g_str = "GRASPABLE" if h["is_graspable"] else "NOT GRASPABLE"
                        print(f"  {idx}. {h['label']:20s} | {g_str:13s} | dist={h['distance_cm']:4.1f}cm | w={h['width_cm']:3.1f}cm | bbox={h['bbox_px']}")
                    break

    except KeyboardInterrupt:
        log.info("Live perception interrupted by user.")
    finally:
        if stream_cap is not None:
            stream_cap.release()
        if ros is not None:
            ros.close()
        if gui_available:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        log.info("Live perception terminated.")


if __name__ == "__main__":
    main()
