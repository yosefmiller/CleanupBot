"""Class-Agnostic YOLO Hazard Detector with CSS3 Color Identification for CleanupBot5."""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")

import numpy as np
from ultralytics import YOLO

from .color_analyzer import ColorAnalyzer
from .geometry import GroundProjector
from .verifier import GraspVerifier

os.environ["YOLO_VERBOSE"] = "False"
log = logging.getLogger("vision_core.detector")

WEIGHTS_DIR: Path = Path(__file__).resolve().parent.parent / "weights"
MODEL_PATH: Path = WEIGHTS_DIR / "yolov8s-worldv2.pt"

# High-recall floor clutter & physical object categories
DEFAULT_CLASSES: list[str] = [
    "toy",
    "toy block",
    "cube",
    "triangle",
    "ball",
    "box",
    "container",
    "bottle",
    "can",
    "cup",
    "remote",
    "book",
    "battery",
    "candle",
    "tool",
    "cable",
    "object",
    "item",
]


class HazardDetector:
    """High-recall floor hazard detector using lightweight YOLO-World and CIELAB color matching."""

    def __init__(
        self,
        min_grasp_width_cm: float = 1.0,
        max_grasp_width_cm: float = 4.5,
        max_range_cm: float = 120.0,
        conf_thresh: float = 0.015,
        iou_thresh: float = 0.40,
    ) -> None:
        self.min_grasp_width_cm: float = min_grasp_width_cm
        self.max_grasp_width_cm: float = max_grasp_width_cm
        self.max_range_cm: float = max_range_cm
        self.conf_thresh: float = conf_thresh
        self.iou_thresh: float = iou_thresh

        self.projector: GroundProjector = GroundProjector()
        self.color_analyzer: ColorAnalyzer = ColorAnalyzer()
        self.verifier: GraspVerifier = GraspVerifier()

        # Lightweight temporal smoothing state (zero-dependency jitter filter)
        self._tracks: dict[int, dict[str, Any]] = {}
        self._next_track_id: int = 1

        log.info("Loading YOLO detector from %s...", MODEL_PATH)
        self.yolo: YOLO = YOLO(str(MODEL_PATH))
        self._set_model_classes(DEFAULT_CLASSES)

    def _set_model_classes(self, classes: list[str]) -> None:
        if hasattr(self.yolo, "set_classes"):
            try:
                self.yolo.set_classes(classes)
            except Exception as e:
                log.warning("Could not set classes on YOLO: %s", e)

    def set_classes(self, classes: list[str]) -> None:
        """Update detection vocabulary."""
        self._set_model_classes(classes)

    def detect(
        self,
        frame: np.ndarray,
        imu_pitch_deg: float = 0.0,
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Detect floor hazards and label them by their physical color (e.g. 'orange item', 'multicolored item').
        Optimized for high recall across all floor objects.
        If query is specified, filters by matching color/class or delegates to TypeSafe Jev Box locator.
        """
        frame_h, frame_w = frame.shape[:2]

        results = self.yolo.predict(
            frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            agnostic_nms=True,
            imgsz=640,
            verbose=False,
        )[0]

        hazards: list[dict[str, Any]] = []

        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                conf_val = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                cx = (x1 + x2) // 2

                # Exclude robot's own gripper arms visible at bottom-center of frame
                if y2 >= frame_h - 25 and (0.25 * frame_w) <= cx <= (0.75 * frame_w):
                    continue

                # Must be in floor region and reasonably sized
                if y2 <= 165 or (x2 - x1) >= (frame_w * 0.65) or (y2 - y1) >= (frame_h * 0.65):
                    continue
                if (x2 - x1) < 8 or (y2 - y1) < 8:
                    continue

                # Project bottom contact baseline to ground coordinates (Z = 0)
                ground_bounds = self.projector.bbox_to_metric_bounds(
                    (x1, y1, x2, y2), frame_w, frame_h, imu_pitch_deg
                )
                if ground_bounds is None:
                    continue

                x_cm, y_cm, width_cm = ground_bounds
                dist_cm = math.hypot(x_cm, y_cm)
                if dist_cm > self.max_range_cm:
                    continue

                is_graspable = (
                    (self.min_grasp_width_cm <= width_cm <= self.max_grasp_width_cm)
                    and (dist_cm <= self.max_range_cm)
                )

                # Raw shape primitive detected by YOLO
                cls_id = int(box.cls[0])
                raw_class = results.names.get(cls_id, "object")

                # Extract BGR bounding box crop and perform CIELAB color matching
                crop = frame[y1:y2, x1:x2]
                color_info = self.color_analyzer.analyze_crop(crop)

                # Color-first label (e.g. 'orange item', 'yellow item', 'multicolored item')
                label = color_info["label"]

                hazards.append({
                    "label": label,
                    "raw_class": raw_class,
                    "color": color_info["color_name"],
                    "color_family": color_info["color_family"],
                    "is_multicolored": color_info["is_multicolored"],
                    "confidence": round(conf_val, 2),
                    "x_cm": round(float(x_cm), 1),
                    "y_cm": round(float(y_cm), 1),
                    "distance_cm": round(float(dist_cm), 1),
                    "width_cm": round(float(width_cm), 1),
                    "is_graspable": bool(is_graspable),
                    "contact_point_px": (cx, y2),
                    "bbox_px": (x1, y1, x2, y2),
                })

        # Targeted query resolution
        if query:
            q = query.strip().lower()
            q_words = [w for w in q.split() if len(w) > 2]
            matched: list[dict[str, Any]] = []

            for h in hazards:
                hl = h["label"].lower()
                hc = h["color"].lower()
                hf = h["color_family"].lower()
                hr = h["raw_class"].lower()

                # Direct match against label, color family, specific CSS3 color, or YOLO class
                if q in hl or q in hf or q in hc or q in hr:
                    matched.append(h)
                # Multi-word match (e.g. "yellow cube" -> matches "yellow" color family or "cube" raw class)
                elif q_words and any(w in hl or w in hf or w in hc or w in hr for w in q_words):
                    matched.append(h)

            if matched:
                matched.sort(key=lambda h: h["distance_cm"])
                return matched

            # Semantic query fallback: delegate to TypeSafe Jev Box locator
            jev_target = self.verifier.locate_target(frame, query)
            if jev_target is not None:
                jx1, jy1, jx2, jy2, jconf = jev_target
                jcx = (jx1 + jx2) // 2
                j_bounds = self.projector.bbox_to_metric_bounds(
                    (jx1, jy1, jx2, jy2), frame_w, frame_h, imu_pitch_deg
                )
                if j_bounds is not None:
                    jx_cm, jy_cm, jw_cm = j_bounds
                    jdist_cm = math.hypot(jx_cm, jy_cm)
                    j_crop = frame[jy1:jy2, jx1:jx2]
                    j_color = self.color_analyzer.analyze_crop(j_crop)
                    j_graspable = (
                        (self.min_grasp_width_cm <= jw_cm <= self.max_grasp_width_cm)
                        and (jdist_cm <= self.max_range_cm)
                    )
                    return [{
                        "label": f"{query.strip()} ({j_color['label']})",
                        "raw_class": "jev_target",
                        "color": j_color["color_name"],
                        "color_family": j_color["color_family"],
                        "is_multicolored": j_color["is_multicolored"],
                        "confidence": round(jconf, 2),
                        "x_cm": round(float(jx_cm), 1),
                        "y_cm": round(float(jy_cm), 1),
                        "distance_cm": round(float(jdist_cm), 1),
                        "width_cm": round(float(jw_cm), 1),
                        "is_graspable": bool(j_graspable),
                        "contact_point_px": (jcx, jy2),
                        "bbox_px": (jx1, jy1, jx2, jy2),
                    }]
            return []

        # Apply lightweight temporal smoothing to eliminate bounding box jitter
        hazards = self._smooth_tracks(hazards)

        # Sort nearest to robot first
        hazards.sort(key=lambda h: h["distance_cm"])
        return hazards

    def _smooth_tracks(self, hazards: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Lightweight temporal smoothing (zero-dependency EMA).
        Stabilizes bounding box jitter and suppresses single-frame flicker when camera is still.
        """
        matched_track_ids: set[int] = set()
        smoothed_hazards: list[dict[str, Any]] = []

        for h in hazards:
            bx1, by1, bx2, by2 = h["bbox_px"]
            best_tid, best_iou = None, 0.35

            for tid, track in self._tracks.items():
                tx1, ty1, tx2, ty2 = track["bbox_px"]
                ix1, iy1 = max(bx1, tx1), max(by1, ty1)
                ix2, iy2 = min(bx2, tx2), min(by2, ty2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = (bx2 - bx1) * (by2 - by1) + (tx2 - tx1) * (ty2 - ty1) - inter
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None:
                # Smooth coordinates with Exponential Moving Average (EMA: 70% prev, 30% current)
                px1, py1, px2, py2 = self._tracks[best_tid]["bbox_px"]
                sx1 = int(round(0.70 * px1 + 0.30 * bx1))
                sy1 = int(round(0.70 * py1 + 0.30 * by1))
                sx2 = int(round(0.70 * px2 + 0.30 * bx2))
                sy2 = int(round(0.70 * py2 + 0.30 * by2))

                h["bbox_px"] = (sx1, sy1, sx2, sy2)
                h["contact_point_px"] = ((sx1 + sx2) // 2, sy2)

                self._tracks[best_tid]["bbox_px"] = (sx1, sy1, sx2, sy2)
                self._tracks[best_tid]["label"] = h["label"]
                self._tracks[best_tid]["ttl"] = 2
                matched_track_ids.add(best_tid)
                smoothed_hazards.append(h)
            else:
                tid = self._next_track_id
                self._next_track_id += 1
                self._tracks[tid] = {
                    "bbox_px": (bx1, by1, bx2, by2),
                    "label": h["label"],
                    "ttl": 2,
                }
                matched_track_ids.add(tid)
                smoothed_hazards.append(h)

        # Decay unmatched tracks
        for tid in list(self._tracks.keys()):
            if tid not in matched_track_ids:
                self._tracks[tid]["ttl"] -= 1
                if self._tracks[tid]["ttl"] <= 0:
                    del self._tracks[tid]

        return smoothed_hazards
