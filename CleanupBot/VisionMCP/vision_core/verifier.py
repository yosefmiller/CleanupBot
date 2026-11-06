"""Grasp outcome verification and visual reasoning using TypeSafe Jev System One model."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import cv2
import dotenv
import numpy as np

# Load environment variables (including TYPESAFE_API_KEY) from VisionMCP/.env
_MCP_DIR = Path(__file__).resolve().parent.parent
dotenv.load_dotenv(_MCP_DIR / ".env")

from typesafe_sdk import Box, Choice, Image as JevImage, Noul, TypeSafeClient

log = logging.getLogger("vision_core.verifier")


def prepare_image_for_jev(img: np.ndarray, max_dim: int = 1024, min_dim: int = 320) -> bytes:
    """
    Ensure the image dimensions satisfy TypeSafe Jev limits (280px to 1568px per side).
    Scales the image proportionally and encodes to high-quality JPEG bytes.
    """
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
    elif min(h, w) < min_dim:
        scale = min_dim / min(h, w)

    if scale != 1.0:
        new_w = max(280, min(1568, int(round(w * scale))))
        new_h = max(280, min(1568, int(round(h * scale))))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        img = cv2.resize(img, (new_w, new_h), interpolation=interp)

    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return buf.tobytes()


class GraspVerifier:
    """Evaluates whether an object is held between gripper jaws using TypeSafe Jev."""

    def __init__(
        self,
        typesafe_api_key: str | None = None,
        timeout_s: float = 15.0,
        roi_top_fraction: float | None = None,
        roi_margin_x_fraction: float | None = None,
    ) -> None:
        self.typesafe_api_key: str | None = typesafe_api_key or os.environ.get("TYPESAFE_API_KEY")
        self.timeout_s: float = timeout_s

        # The black jaw tips are located at the very bottom of the frame, close to camera (7-10 cm).
        # Default to lower 35% of frame (y: 0.65 -> 1.0) and central 60% of width (x: 0.20 -> 0.80).
        # Configurable via environment variables JAW_ROI_TOP and JAW_ROI_MARGIN_X.
        self.roi_top_fraction: float = (
            roi_top_fraction
            if roi_top_fraction is not None
            else float(os.environ.get("JAW_ROI_TOP", "0.65"))
        )
        self.roi_margin_x_fraction: float = (
            roi_margin_x_fraction
            if roi_margin_x_fraction is not None
            else float(os.environ.get("JAW_ROI_MARGIN_X", "0.20"))
        )

    def crop_jaw_roi(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract the targeted region covering the black gripper jaw tips.
        By default crops the bottom 35% across the central 60% width where the jaws meet.
        """
        h, w = frame.shape[:2]
        y1: int = max(0, int(h * self.roi_top_fraction))
        y2: int = h
        x1: int = max(0, int(w * self.roi_margin_x_fraction))
        x2: int = min(w, int(w * (1.0 - self.roi_margin_x_fraction)))
        return frame[y1:y2, x1:x2]

    def verify(self, frame: np.ndarray, phase: str = "post_lift") -> dict[str, Any]:
        """
        Determine if the gripper jaws are holding an object using TypeSafe Jev.

        Incorporates domain context:
        - The jaw tips and object are in extreme close-up (7-10 cm from camera lens).
        - Fixed-focus lens causes optical blur and soft edges on close foreground objects.
        - Objects are small (1.5-4.5 cm) but appear magnified and large between the jaws.

        Returns:
            {
                'grasped': bool,
                'confidence': float,
                'grip_state': str,
                'note': str,
                'outcome': 'SUCCESS' | 'SLIP' | 'GRAVITY_SLIP',
                'tokens': {'input': int, 'output': int, 'total': int},
                'backend': 'jev'
            }
        """
        if not self.typesafe_api_key:
            raise RuntimeError(
                "TYPESAFE_API_KEY is not set. Please ensure TYPESAFE_API_KEY is defined in VisionMCP/.env or environment."
            )

        crop: np.ndarray = self.crop_jaw_roi(frame)
        img_bytes: bytes = prepare_image_for_jev(crop)

        phase_desc = (
            "pre_lift: Gripper jaws have driven closed on the floor surface around target."
            if phase == "pre_lift"
            else "post_lift: Gripper has raised into the air against gravity."
        )

        state = {
            "camera_view": JevImage(img_bytes),
            "scene_context": {
                "view_geometry": (
                    "Extreme macro view. The two black mechanical jaw tips are positioned at the bottom "
                    "center of the frame, only 7 to 10 cm from the camera lens."
                ),
                "optical_blur_notice": (
                    "CRITICAL: The camera lens is focused on the distant floor (30-80 cm away). Any object "
                    "held between the jaw tips (7-10 cm away) is inside the minimum focal distance and will "
                    "appear BLURRY, soft-edged, and out of focus."
                ),
                "scale_notice": (
                    "The objects being grasped (plastic toys, batteries, bottle caps) are small (1.5 to 4.5 cm), "
                    "but due to the 7-10 cm proximity, they appear relatively LARGE and magnified in this crop."
                ),
                "inspection_phase": phase_desc,
            },
        }

        questions = {
            "is_holding_object": Noul(
                instructions=(
                    "Look directly at the space between the two black gripper jaw tips at the bottom of `camera_view`. "
                    "Given that the camera is only 7-10 cm away, any held object will look BLURRY, soft, and magnified. "
                    "Is there a physical object, toy, or material clamped or wedged between the black jaw tips?"
                ),
                criteria={
                    "true": (
                        "A blurry, soft-edged, magnified, or colored mass (e.g. plastic toy, battery, cylinder, cap) "
                        "is visible between or contacted by the two black jaw tips, partially occluding the floor."
                    ),
                    "false": (
                        "The space between the black jaw tips is empty, showing only the plain, continuous floor "
                        "surface with no object held between them."
                    ),
                },
            ),
            "jaw_grip_state": Choice(
                instructions="What is the physical grip state of the two black jaw tips in `camera_view`?",
                criteria={
                    "holding_object": "The black jaw tips are clamped onto a blurry, magnified object between them.",
                    "empty_closed": "The black jaw tips are closed together with nothing held between them (bare floor only).",
                    "empty_open": "The black jaw tips are spread open with bare floor clearly visible between them.",
                },
            ),
        }

        with TypeSafeClient(api_key=self.typesafe_api_key, timeout=self.timeout_s) as client:
            res = client.system_one(
                state=state,
                questions=questions,
                extra_headers={"TypeSafe-Danger-Zone-Images": "true"},
            )

        is_holding_prob = float(res.nouls["is_holding_object"].noul)
        grip_choice = str(res.choices["jaw_grip_state"].choice)
        choice_probs = res.choices["jaw_grip_state"].probabilities or {}
        choice_holding_prob = float(choice_probs.get("holding_object", 0.0))

        # Combined decision: fusion of calibrated Noul probability and Choice categorical outcome
        combined_prob = 0.5 * is_holding_prob + 0.5 * choice_holding_prob
        grasped = (combined_prob >= 0.50) or (grip_choice == "holding_object")

        input_toks = getattr(res.usage, "input_tokens", 0)
        output_toks = getattr(res.usage, "output_tokens", 0)
        total_toks = input_toks + output_toks

        log.info(
            "TypeSafe Jev grasp audit: holding_prob=%.2f, state=%s, grasped=%s (tokens: in=%d, out=%d, tot=%d)",
            combined_prob,
            grip_choice,
            grasped,
            input_toks,
            output_toks,
            total_toks,
        )

        outcome = "SUCCESS" if grasped else ("SLIP" if phase == "pre_lift" else "GRAVITY_SLIP")
        return {
            "grasped": grasped,
            "confidence": round(combined_prob, 3),
            "grip_state": grip_choice,
            "noul_prob": round(is_holding_prob, 3),
            "choice_prob": round(choice_holding_prob, 3),
            "note": f"TypeSafe Jev {phase} check: state={grip_choice}, holding={grasped} (P={combined_prob:.2f})",
            "outcome": outcome,
            "tokens": {
                "input": input_toks,
                "output": output_toks,
                "total": total_toks,
            },
            "backend": "jev",
        }

    def locate_target(self, frame: np.ndarray, query: str) -> tuple[int, int, int, int, float] | None:
        """
        Use TypeSafe Jev Box question to locate a specific queried object in the full camera frame,
        then refines the bounding box to enclose the entire item boundaries.
        Returns (x1, y1, x2, y2, confidence) in pixel coordinates, or None.
        """
        if not self.typesafe_api_key:
            return None

        h, w = frame.shape[:2]
        img_bytes = prepare_image_for_jev(frame)
        try:
            with TypeSafeClient(api_key=self.typesafe_api_key, timeout=self.timeout_s) as client:
                res = client.system_one(
                    state={"camera_view": JevImage(img_bytes)},
                    questions={
                        "target_box": Box(
                            instructions=(
                                f"Tight 2D bounding box enclosing the ENTIRE {query.strip()} on the floor, "
                                "including all its corners, outer edges, and visible physical boundaries."
                            )
                        )
                    },
                    extra_headers={"TypeSafe-Danger-Zone-Images": "true"},
                    model="jev-preview",
                )
            if "target_box" in res.boxes:
                b = res.boxes["target_box"]
                conf = float(b.confidence if b.confidence is not None else 0.5)
                top = max(0.0, min(1.0, float(b.box.top)))
                bottom = max(0.0, min(1.0, float(b.box.bottom)))
                left = max(0.0, min(1.0, float(b.box.left)))
                right = max(0.0, min(1.0, float(b.box.right)))
                x1 = int(round(left * w))
                y1 = int(round(top * h))
                x2 = int(round(right * w))
                y2 = int(round(bottom * h))
                if (x2 - x1) >= 6 and (y2 - y1) >= 6:
                    # Refine coarse VLM bounding box to tightly enclose full visible object extent
                    rx1, ry1, rx2, ry2 = self.refine_target_box(frame, (x1, y1, x2, y2), query=query)
                    log.info("Jev located target '%s' at [%d, %d, %d, %d] -> refined to [%d, %d, %d, %d] (conf=%.2f)",
                             query, x1, y1, x2, y2, rx1, ry1, rx2, ry2, conf)
                    return (rx1, ry1, rx2, ry2, round(conf, 2))
        except Exception as e:
            log.warning("Jev locate_target error for '%s': %s", query, e)
        return None

    @staticmethod
    def refine_target_box(
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        query: str = "",
    ) -> tuple[int, int, int, int]:
        """
        Refine a bounding box to tightly enclose the entire visible object boundaries.
        Leverages local color-adaptive segmentation and edge contours within the ROI.
        """
        x1, y1, x2, y2 = bbox
        fh, fw = frame.shape[:2]
        pad_w = max(10, int((x2 - x1) * 0.25))
        pad_h = max(10, int((y2 - y1) * 0.25))
        rx1, ry1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
        rx2, ry2 = min(fw, x2 + pad_w), min(fh, y2 + pad_h)

        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
            return bbox

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        q_lower = query.lower()
        mask = None
        if "yellow" in q_lower or "gold" in q_lower:
            mask = (hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 42) & (hsv[:, :, 1] > 40) & (hsv[:, :, 2] > 40)
        elif "orange" in q_lower:
            mask = (hsv[:, :, 0] >= 5) & (hsv[:, :, 0] < 20) & (hsv[:, :, 1] > 40) & (hsv[:, :, 2] > 40)
        elif "red" in q_lower:
            mask = ((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 170)) & (hsv[:, :, 1] > 40) & (hsv[:, :, 2] > 40)
        elif "green" in q_lower:
            mask = (hsv[:, :, 0] >= 40) & (hsv[:, :, 0] <= 85) & (hsv[:, :, 1] > 40) & (hsv[:, :, 2] > 40)
        elif "blue" in q_lower:
            mask = (hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 135) & (hsv[:, :, 1] > 40) & (hsv[:, :, 2] > 40)

        if mask is not None and np.count_nonzero(mask) > 20:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 25, 90)
            dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return bbox

        cx_roi, cy_roi = (x1 + x2) // 2 - rx1, (y1 + y2) // 2 - ry1
        valid_contours = []
        for c in contours:
            if cv2.contourArea(c) > 20:
                bx, by, bw, bh = cv2.boundingRect(c)
                if (bx <= cx_roi <= bx + bw) or (by <= cy_roi <= by + bh) or cv2.pointPolygonTest(c, (float(cx_roi), float(cy_roi)), True) > -20:
                    valid_contours.append(c)

        if not valid_contours:
            valid_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

        all_pts = np.vstack(valid_contours)
        fx, fy, fw, fh = cv2.boundingRect(all_pts)
        refined = (max(0, rx1 + fx), max(0, ry1 + fy), min(fw, rx1 + fx + fw), min(fh, ry1 + fy + fh))
        return refined

    def inspect_orientation(
        self,
        object_crop: np.ndarray,
        label: str | None = None,
        width_cm: float | None = None,
    ) -> dict[str, Any]:
        """
        Ask TypeSafe Jev for robotics grasp orientation, approach vector, and geometric symmetry.

        Accounts for:
        - Downward-tilted camera (~15°-25° pitch) mounted ~15 cm above floor.
        - Robot parallel gripper opening limit (4.5 cm maximum stroke).
        - Approach recommendation (head-on vs drive around to left/right for elongated items).
        - Alignment state (aligned square vs rotated diagonal requiring extended clamp time).
        - Geometric symmetry (symmetric/round vs elongated).

        Args:
            object_crop: BGR crop containing the target hazard.
            label: Optional detected object label (e.g. 'orange item', 'yellow cube').
            width_cm: Optional true head-on lateral width in cm from floor projection.

        Returns:
            {
                'approach': 'head_on' | 'approach_from_left' | 'approach_from_right',
                'shape_symmetry': 'symmetric_round' | 'elongated' | 'irregular',
                'alignment': 'standard_clamp' | 'significant_extra_clamp',
                'is_diagonal': bool,
                'needs_extended_clamp': bool,
                'can_reorient': bool,
                'orientation': str,
                'confidence': float,
                'tokens': {'input': int, 'output': int, 'total': int},
                'backend': 'jev'
            }
        """
        if not self.typesafe_api_key:
            raise RuntimeError("TYPESAFE_API_KEY is not set in VisionMCP/.env or environment.")

        img_bytes: bytes = prepare_image_for_jev(object_crop)

        state = {
            "hazard_crop": JevImage(img_bytes),
            "camera_geometry": {
                "mounting": "Camera mounted 12 cm above floor on the mobile robot, tilted downward at ~8° pitch.",
                "view": "Looking forward and down at an object resting on the floor plane.",
                "image_axes": (
                    "Top of crop is forward (farther from robot); bottom is rear (closer to robot); "
                    "left/right are the robot's left/right."
                ),
                "robot_gripper": (
                    "Two-finger parallel jaw gripper with 4.5 cm maximum opening stroke. Closes horizontally."
                ),
            },
            "physical_measurements": {
                "target_label": label or "floor object",
                "observed_lateral_width_cm": f"{width_cm:.1f} cm" if width_cm is not None else "unknown",
                "gripper_stroke_limit_cm": "4.5 cm",
            },
        }

        questions = {
            "shape_symmetry": Choice(
                instructions="What is the geometric shape and horizontal symmetry of this floor object in `hazard_crop`?",
                criteria={
                    "symmetric_round": (
                        "Cube, sphere, upright cylinder, or compact item where horizontal dimensions are roughly equal. "
                        "Approaching from another horizontal angle provides NO narrower profile."
                    ),
                    "elongated": (
                        "Distinctly elongated item with an obvious long axis and narrow axis "
                        "(e.g. fallen battery, marker, pen, rectangular toy block, remote, bottle). "
                        "Approaching along its long axis presents a significantly narrower grasp profile."
                    ),
                    "irregular": "Multi-part, complex, or irregular shape.",
                },
            ),
            "approach_recommendation": Choice(
                instructions="How should the robot approach this object to achieve a graspable profile facing its parallel jaws?",
                criteria={
                    "head_on": (
                        "Select when: (1) the object is visually symmetric/compact (ball, cube), OR "
                        "(2) the object is elongated, but its long axis already points primarily forward toward the camera "
                        "(within +-25° of vertical), presenting its narrow end directly to the jaws, OR "
                        "(3) its observed lateral width is <= 4.0 cm and fits easily within the 4.5 cm gripper opening."
                    ),
                    "approach_from_left": (
                        "Select when the object is elongated AND lies crosswise or diagonal across the robot's heading "
                        "(> 25° tilt from vertical, or observed lateral width > 4.0 cm), AND its long axis tilts with its "
                        "narrow end or leading edge angled toward the robot's left side (driving around in a ~30°-45° left "
                        "arc directly faces its narrow end)."
                    ),
                    "approach_from_right": (
                        "Select when the object is elongated AND lies crosswise or diagonal across the robot's heading "
                        "(> 25° tilt from vertical, or observed lateral width > 4.0 cm), AND its long axis tilts with its "
                        "narrow end or leading edge angled toward the robot's right side (driving around in a ~30°-45° right "
                        "arc directly faces its narrow end)."
                    ),
                },
            ),
            "alignment_state": Choice(
                instructions="When parallel gripper jaws close on this object, will it require significant extra clamp travel/time beyond its observed width?",
                criteria={
                    "standard_clamp": (
                        "The object is aligned square with the approaching gripper jaws or is round/symmetric. "
                        "Jaws will make flush contact on parallel faces; standard clamp travel based on observed width is sufficient."
                    ),
                    "significant_extra_clamp": (
                        "The object is resting at an angle/diagonal relative to the closing jaws (> 20° tilt). "
                        "Closing parallel jaws will make point/asymmetric contact, exerting torque that twists or straightens "
                        "the item before clamping. Requires extended motor clamp time (+30% to +50%) to ensure a secure grip."
                    ),
                },
            ),
            "needs_extended_clamp": Noul(
                instructions=(
                    "Does this object require significant extra clamp travel/duration beyond its observed width to overcome rotational twisting?"
                ),
                criteria={
                    "true": "Object is rotated/diagonal; closing parallel jaws will exert asymmetric torque that twists or straightens it, requiring extra motor clamp duration.",
                    "false": "Object is aligned square, round, or compact; standard clamp travel based on observed width is sufficient.",
                },
            ),
        }

        with TypeSafeClient(api_key=self.typesafe_api_key, timeout=self.timeout_s) as client:
            res = client.system_one(
                state=state,
                questions=questions,
                extra_headers={"TypeSafe-Danger-Zone-Images": "true"},
            )

        shape_choice = str(res.choices["shape_symmetry"].choice)
        approach_choice = str(res.choices["approach_recommendation"].choice)
        alignment_choice = str(res.choices["alignment_state"].choice)
        is_rotated_prob = float(res.nouls["needs_extended_clamp"].noul)

        is_diagonal = (is_rotated_prob >= 0.50) or (alignment_choice == "significant_extra_clamp")
        can_reorient = (shape_choice == "elongated") and (approach_choice != "head_on")

        input_toks = getattr(res.usage, "input_tokens", 0)
        output_toks = getattr(res.usage, "output_tokens", 0)
        total_toks = input_toks + output_toks

        log.info(
            "TypeSafe Jev orientation analysis: shape=%s, approach=%s, align=%s, is_diag=%s (P=%.2f, tokens=%d)",
            shape_choice,
            approach_choice,
            alignment_choice,
            is_diagonal,
            is_rotated_prob,
            total_toks,
        )

        return {
            "approach": approach_choice,
            "shape_symmetry": shape_choice,
            "alignment": alignment_choice,
            "is_diagonal": is_diagonal,
            "needs_extended_clamp": is_diagonal,
            "can_reorient": can_reorient,
            "orientation": "diagonal" if is_diagonal else "aligned",
            "confidence": round(is_rotated_prob if is_diagonal else (1.0 - is_rotated_prob), 3),
            "tokens": {
                "input": input_toks,
                "output": output_toks,
                "total": total_toks,
            },
            "backend": "jev",
        }
