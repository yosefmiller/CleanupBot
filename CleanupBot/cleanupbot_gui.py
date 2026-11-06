"""
Live OpenCV Visual Interface & Mission Monitor for CleanupBot.

Renders the robot's real-time camera feed alongside:
1. Ground plane projection grid and open-vocabulary YOLO hazard detections.
2. Real-time agent tool execution HUD (color-coded by tool with dark palettes).
3. ChromaDB Episodic Memory adaptation callouts (0.60s baseline -> 1.20s adapted).
4. Robot global frame pose tracking (X, Y, theta).
5. Human-in-the-loop pause prompts for physical repositioning and grasp placement.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ros2_bridge.subscriber import get_subscriber

log = logging.getLogger("cleanupbot_gui")

# Paths
BASE_DIR = Path(__file__).resolve().parent
VISION_MCP_DIR = BASE_DIR / "VisionMCP"
SNAPSHOTS_DIR = VISION_MCP_DIR / "images" / "mission_snapshots"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# Dark BGR Colors for OpenCV HUD (matching terminal dark palette)
TOOL_BGR_COLORS: dict[str, tuple[int, int, int]] = {
    # Vision MCP
    "scan_for_hazards": (120, 120, 20),      # Dark Teal / Cyan
    "verify_grasp": (30, 120, 30),           # Dark Forest Green
    # Memory MCP
    "query_spatial_map": (30, 100, 130),     # Dark Slate / Olive
    "query_grasp_timing": (20, 140, 200),    # Dark Warm Amber
    "record_grasp_episode": (120, 20, 100),  # Dark Purple
    # Local Hardware
    "navigate_to_pose": (130, 60, 20),       # Dark Slate Blue
    "execute_gripper": (90, 25, 115),        # Dark Wine / Plum
    "get_robot_pose": (90, 90, 90),          # Dark Steel Gray
    "emergency_stop": (20, 20, 150),         # Dark Crimson
    # Aliases
    "navigate_to_hazard": (130, 60, 20),
    "approach_hazard": (130, 60, 20),
    "execute_motor_clamp": (90, 25, 115),
    "get_robot_status": (90, 90, 90),
}
DEFAULT_TOOL_BGR = (60, 60, 60)


def _can_display_gui() -> bool:
    """Check if an active graphical display socket is actually reachable."""
    if os.environ.get("HEADLESS", "0") == "1":
        return False

    wayland_display = os.environ.get("WAYLAND_DISPLAY")
    if wayland_display:
        xdg_runtime = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
        socket_path = Path(xdg_runtime) / wayland_display
        if socket_path.exists():
            return True

    display = os.environ.get("DISPLAY")
    if display:
        if display.startswith(":"):
            disp_num = display.split(":")[1].split(".")[0]
            if Path(f"/tmp/.X11-unix/X{disp_num}").exists():
                return True
        elif Path(display).exists():
            return True

    return False


def _sanitize_ascii(text: str) -> str:
    """Replace Greek letters and unicode symbols with clean ASCII equivalents for OpenCV Hershey font."""
    if not text:
        return ""
    replacements = {
        "θ": "theta",
        "Θ": "Theta",
        "α": "alpha",
        "β": "beta",
        "γ": "gamma",
        "δ": "delta",
        "Δ": "Delta",
        "μ": "u",
        "π": "pi",
        "°": " deg",
        "±": "+/-",
        "★": "*",
        "➔": "->",
        "→": "->",
        "✓": "[OK]",
        "•": "*",
        "═": "=",
        "—": "-",
        "–": "-",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode("ascii", "ignore").decode("ascii")


def _fit_text(text: str, max_w_px: int, font_scale: float = 0.40, thickness: int = 1) -> str:
    """Truncate text with an ellipsis so it never overflows max_w_px."""
    text = _sanitize_ascii(text)
    if not text or max_w_px <= 20:
        return ""
    w_px = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0]
    if w_px <= max_w_px:
        return text
    ellipsis = "..."
    el_w = cv2.getTextSize(ellipsis, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0]
    avail_w = max_w_px - el_w
    cur = ""
    for ch in text:
        next_w = cv2.getTextSize(cur + ch, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0]
        if next_w > avail_w:
            break
        cur += ch
    return cur + ellipsis


def _wrap_text(text: str, max_w_px: int, font_scale: float = 0.38, thickness: int = 1) -> list[str]:
    """Wrap text into multiple lines such that each line fits within max_w_px."""
    text = _sanitize_ascii(text)
    words = text.split()
    lines: list[str] = []
    cur = ""
    for word in words:
        cand = f"{cur} {word}".strip()
        w_px = cv2.getTextSize(cand, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0]
        if w_px <= max_w_px:
            cur = cand
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


class MissionVisualizer:
    """
    Thread-safe OpenCV visual monitor that overlays live LLM tool calls,
    memory adaptation state, and robot pose over the camera stream.
    """

    def __init__(self, window_name: str = "CleanupBot5 - Mission Monitor") -> None:
        self.window_name = window_name
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

        # State
        self.active_tool: str = "idle"
        self.active_tool_args: dict[str, Any] = {}
        self.last_tool: str = ""
        self.last_tool_result_preview: str = ""
        self.memory_badge: str = ""
        self.memory_adapted: bool = False
        self.audit_badge: str = ""
        self.is_paused: bool = False
        self.pause_message: str = ""
        self.robot_pose: dict[str, float] = {"x_m": 0.0, "y_m": 0.0, "yaw_deg": 0.0}

        # Offline fallback image
        self._fallback_image_path = VISION_MCP_DIR / "images" / "vision_live_feed.jpg"
        self._static_frame: np.ndarray | None = None
        if self._fallback_image_path.is_file():
            self._static_frame = cv2.imread(str(self._fallback_image_path))

        self._subscriber = None

    def start(self) -> None:
        """Start the monitor loop in a background thread."""
        if self._running:
            return
        # Lazy subscriber connect
        if self._subscriber is None:
            try:
                self._subscriber = get_subscriber()
            except Exception as exc:
                log.warning("Could not connect ROS2 subscriber: %s", exc)
                self._subscriber = None

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        log.info("MissionVisualizer window started in background thread.")

    def stop(self) -> None:
        """Stop the monitor and close GUI windows."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    def update_pose(self, x_m: float, y_m: float, yaw_deg: float) -> None:
        with self._lock:
            self.robot_pose = {"x_m": float(x_m), "y_m": float(y_m), "yaw_deg": float(yaw_deg)}

    def set_tool_call(self, name: str, args: Any) -> None:
        with self._lock:
            self.active_tool = name
            self.active_tool_args = args if isinstance(args, dict) else {"arg": args}
            log.debug("Visualizer tool_call: %s", name)

    def set_tool_result(self, name: str, output: Any) -> None:
        with self._lock:
            self.last_tool = name
            parsed = output
            if isinstance(output, str):
                try:
                    parsed = json.loads(output)
                except Exception:
                    parsed = output

            # Format concise preview
            if isinstance(parsed, dict):
                if name == "query_grasp_timing":
                    rec = parsed.get("recommended_duration_s", 0.60)
                    base = parsed.get("baseline_duration_s", 0.60)
                    adapted = parsed.get("adapted_from_memory", False)
                    self.memory_adapted = adapted
                    self.memory_badge = f"ChromaDB Adaptation: {base:.2f}s -> {rec:.2f}s (SLIP History Compensated)"
                    self.last_tool_result_preview = f"Recommended Duration: {rec:.2f}s | Adapted: {adapted}"

                elif name == "verify_grasp":
                    outcome = parsed.get("outcome", "UNKNOWN")
                    grasped = parsed.get("grasped", False)
                    conf = parsed.get("confidence", 0.0)
                    self.audit_badge = f"Grasp Audit: {outcome} (Holding={grasped}, Conf={conf:.2f})"
                    self.last_tool_result_preview = self.audit_badge

                elif name == "scan_for_hazards":
                    hazards = parsed.get("hazards", [])
                    if hazards:
                        t = hazards[0]
                        diag_str = "DIAG" if t.get("is_diagonal") else "STRAIGHT"
                        app_str = t.get("approach", "head_on")
                        w_cm = t.get("width_cm", 0.0)
                        lbl = t.get("label", "hazard")
                        self.last_tool_result_preview = (
                            f"Perception: {len(hazards)} hazards | Target '{lbl}' ({w_cm}cm, {diag_str}, {app_str})"
                        )
                    else:
                        self.last_tool_result_preview = "Perception: 0 hazards detected in camera view"

                elif name in ("navigate_to_pose", "navigate_to_hazard"):
                    curr = parsed.get("current_pose", {})
                    self.last_tool_result_preview = f"Arrived at X={curr.get('x_m', 0.0):.2f}m, Y={curr.get('y_m', 0.0):.2f}m"

                elif name in ("execute_gripper", "execute_motor_clamp"):
                    dur = parsed.get("clamp_duration_used_s", 0.60)
                    act = parsed.get("action", "clamp")
                    self.last_tool_result_preview = f"Gripper {act} completed ({dur:.2f}s motor drive)"

                elif name == "query_spatial_map":
                    count = parsed.get("count", len(parsed.get("hazards", [])))
                    self.last_tool_result_preview = f"Spatial Map: {count} hazards remembered in ChromaDB"

                elif name == "record_grasp_episode":
                    self.last_tool_result_preview = "Persisted trial to ChromaDB episodic_memory"

                else:
                    self.last_tool_result_preview = str(parsed)[:80]
            else:
                self.last_tool_result_preview = str(parsed)[:80]

    def set_pause(self, active: bool, message: str = "") -> None:
        with self._lock:
            self.is_paused = active
            self.pause_message = message

    def save_screenshot(self, label: str = "step") -> str:
        """Save an annotated snapshot of current frame to disk."""
        frame = self._get_current_frame()
        if frame is None:
            return ""
        annotated = self._draw_overlay(frame.copy())
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        target_path = SNAPSHOTS_DIR / f"{label}_{timestamp}.jpg"
        cv2.imwrite(str(target_path), annotated)
        log.info("Saved visual monitor screenshot: %s", target_path)
        return str(target_path)

    def _get_current_frame(self) -> np.ndarray | None:
        # Check subscriber for live camera frame
        if self._subscriber is not None:
            try:
                snap = self._subscriber.snapshot()
                if snap.camera_frame is not None:
                    return snap.camera_frame
            except Exception:
                pass

        # Fallback to static snapshot
        if self._static_frame is not None:
            return self._static_frame.copy()

        # Blank default
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "Connecting to /camera/image_raw...", (80, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        return blank

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        with self._lock:
            tool_name = self.active_tool
            tool_args = dict(self.active_tool_args)
            last_t = self.last_tool
            last_res = self.last_tool_result_preview
            mem_badge = self.memory_badge
            mem_adapted = self.memory_adapted
            aud_badge = self.audit_badge
            paused = self.is_paused
            pause_msg = self.pause_message
            pose = dict(self.robot_pose)

        # 1. Top Header Banner (Global Pose & Mission Status)
        header_h = 48
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, header_h), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Header Text
        title_str = _fit_text("CLEANUPBOT5 - MISSION MONITOR", w - 28, font_scale=0.46, thickness=1)
        cv2.putText(frame, title_str, (14, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (230, 230, 230), 1, cv2.LINE_AA)

        pose_str = _fit_text(
            f"World Pose: X={pose.get('x_m', 0.0):.2f}m  Y={pose.get('y_m', 0.0):.2f}m  theta={pose.get('yaw_deg', 0.0):.1f} deg",
            w - 28,
            font_scale=0.38,
            thickness=1,
        )
        cv2.putText(frame, pose_str, (14, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 200, 240), 1, cv2.LINE_AA)

        # 2. Bottom Tool Execution Banner
        footer_h = 66
        footer_y = h - footer_h
        overlay_foot = frame.copy()
        bar_color = TOOL_BGR_COLORS.get(tool_name, DEFAULT_TOOL_BGR)
        cv2.rectangle(overlay_foot, (0, footer_y), (w, h), (20, 20, 20), -1)
        # Left colored accent bar
        cv2.rectangle(overlay_foot, (0, footer_y), (6, h), bar_color, -1)
        cv2.addWeighted(overlay_foot, 0.88, frame, 0.12, 0, frame)

        # Tool text
        arg_str = json.dumps(tool_args) if tool_args else "{}"
        tool_call_text = _fit_text(f"ACTIVE: {tool_name}  {arg_str}", w - 30, font_scale=0.40, thickness=1)
        cv2.putText(frame, tool_call_text, (16, footer_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        result_text = _fit_text(f"RESULT: {last_res if last_res else 'Awaiting response...'}", w - 30, font_scale=0.36, thickness=1)
        cv2.putText(frame, result_text, (16, footer_y + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 220, 180), 1, cv2.LINE_AA)

        # 3. Memory Adaptation Badge (Callout)
        if mem_badge:
            badge_y = header_h + 24
            badge_color = (20, 140, 200) if mem_adapted else (80, 80, 80)
            cv2.rectangle(frame, (12, badge_y - 14), (w - 12, badge_y + 8), (30, 30, 30), -1)
            cv2.rectangle(frame, (12, badge_y - 14), (w - 12, badge_y + 8), badge_color, 1)
            b_text = _fit_text(f"* {mem_badge}", w - 40, font_scale=0.36, thickness=1)
            cv2.putText(frame, b_text, (18, badge_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 220, 140), 1, cv2.LINE_AA)

        # 4. Visual Grasp Audit Badge
        if aud_badge:
            aud_y = header_h + 50
            cv2.rectangle(frame, (12, aud_y - 14), (w - 12, aud_y + 8), (25, 40, 25), -1)
            cv2.rectangle(frame, (12, aud_y - 14), (w - 12, aud_y + 8), (30, 160, 30), 1)
            a_text = _fit_text(f"* {aud_badge}", w - 40, font_scale=0.36, thickness=1)
            cv2.putText(frame, a_text, (18, aud_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 255, 160), 1, cv2.LINE_AA)

        # 5. Prominent Human-in-the-Loop Pause Banner
        if paused:
            banner_w = min(w - 30, 580)
            bx = (w - banner_w) // 2
            wrapped_msg = _wrap_text(
                pause_msg or "Perform physical action and press [ENTER] in terminal...",
                banner_w - 32,
                font_scale=0.38,
                thickness=1,
            )
            line_count = max(1, len(wrapped_msg))
            banner_h = 44 + line_count * 20
            by = (h - banner_h) // 2

            overlay_pause = frame.copy()
            cv2.rectangle(overlay_pause, (bx, by), (bx + banner_w, by + banner_h), (10, 10, 70), -1)
            cv2.addWeighted(overlay_pause, 0.85, frame, 0.15, 0, frame)
            cv2.rectangle(frame, (bx, by), (bx + banner_w, by + banner_h), (0, 165, 255), 2)

            cv2.putText(frame, "[PAUSED: OPERATOR ACTION REQUIRED]", (bx + 16, by + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 230, 255), 1, cv2.LINE_AA)
            for idx, line_text in enumerate(wrapped_msg):
                cv2.putText(frame, line_text, (bx + 16, by + 46 + idx * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1, cv2.LINE_AA)

        return frame

    def _run_loop(self) -> None:
        """Continuous GUI rendering loop (or headless snapshot loop)."""
        has_display = _can_display_gui()
        if not has_display:
            log.info("No active display socket detected; visualizer operating in headless snapshot mode.")
        else:
            try:
                cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            except Exception as exc:
                log.warning("Could not initialize OpenCV window (%s) - operating in headless mode.", exc)
                has_display = False

        while self._running:
            if has_display:
                raw = self._get_current_frame()
                if raw is not None:
                    display = self._draw_overlay(raw.copy())
                    try:
                        cv2.imshow(self.window_name, display)
                        key = cv2.waitKey(30) & 0xFF
                        if key == ord("s"):
                            self.save_screenshot("manual_capture")
                        elif key in (27, ord("q")):  # ESC or q
                            break
                    except Exception:
                        pass
                else:
                    time.sleep(0.05)
            else:
                time.sleep(0.1)

        if has_display:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass


# Singleton instance
_visualizer: MissionVisualizer | None = None


def get_visualizer() -> MissionVisualizer:
    global _visualizer
    if _visualizer is None:
        _visualizer = MissionVisualizer()
    return _visualizer
