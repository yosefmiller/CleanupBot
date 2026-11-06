"""
Ground Projection & Metric Geometry Pipeline.

Transforms 2D camera pixels to 2D metric ground coordinates on the floor plane (Z = 0),
and projects 3D floor objects and reference grids back into camera pixel coordinates.
Includes the full kinematic model of CleanupBot5 (chassis, gripper arm joint, camera, IMU, claws).

Coordinate Frames:
- Robot Axle Base Frame (base_link):
    +X_base: Forward along robot heading (cm)
    +Y_base: Left lateral offset (cm)
    +Z_base: Upwards normal from floor (cm)
    Origin: Ground point directly beneath center of drive wheel axle.
- Camera Nadir Frame:
    +X_cam = X_base - X_cam(alpha) (forward from camera lens nadir)
    +Y_cam = Y_base (left lateral offset)
    Origin: Ground point directly beneath camera lens.
- Camera Optical Frame (Standard OpenCV convention):
    +Z_c: Forward along camera optical axis
    +X_c: Image right
    +Y_c: Image down
    Pitch downward by theta rotates around camera X_c axis; roll rotates around Z_c axis.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

CALIBRATION_FILE: Path = Path(__file__).resolve().parent.parent / "camera_calibration.json"


class RobotKinematics:
    """
    Physical dimensions and kinematic relationships of CleanupBot5.
    Self-contained geometric relationships for axle, arm tilt joint, camera, and gripper claws.
    """

    def __init__(
        self,
        joint_x_cm: float = 8.0,
        joint_z_cm: float = 7.0,
        cam_rel_joint_x_cm: float = 2.0,
        cam_rel_joint_z_cm: float = 5.0,
        claws_rel_joint_x_cm: float = 12.0,
        claws_rel_joint_z_cm: float = 1.0,
    ) -> None:
        # Gripper arm joint (pivot pin relative to wheel axle center)
        self.joint_x_cm: float = joint_x_cm
        self.joint_z_cm: float = joint_z_cm

        # Arm relative offsets (camera lens and gripper claws relative to joint)
        self.cam_rel_joint_x_cm: float = cam_rel_joint_x_cm
        self.cam_rel_joint_z_cm: float = cam_rel_joint_z_cm
        self.claws_rel_joint_x_cm: float = claws_rel_joint_x_cm
        self.claws_rel_joint_z_cm: float = claws_rel_joint_z_cm

    def get_camera_pose(self, arm_pitch_deg: float = 0.0) -> tuple[float, float]:
        """
        Calculate instantaneous 2D camera lens position (x_forward_from_axle_cm, z_height_from_floor_cm).
        Arm pitch is positive nose-down.
        """
        rad = math.radians(arm_pitch_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        dx = self.cam_rel_joint_x_cm * cos_a + self.cam_rel_joint_z_cm * sin_a
        dz = -self.cam_rel_joint_x_cm * sin_a + self.cam_rel_joint_z_cm * cos_a
        return self.joint_x_cm + dx, self.joint_z_cm + dz

    def get_claw_pose(self, arm_pitch_deg: float = 0.0) -> tuple[float, float]:
        """
        Calculate instantaneous 2D claw tip position (x_forward_from_axle_cm, z_height_from_floor_cm).
        """
        rad = math.radians(arm_pitch_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        dx = self.claws_rel_joint_x_cm * cos_a + self.claws_rel_joint_z_cm * sin_a
        dz = -self.claws_rel_joint_x_cm * sin_a + self.claws_rel_joint_z_cm * cos_a
        return self.joint_x_cm + dx, self.joint_z_cm + dz


class GroundProjector:
    """Projects 2D image coordinates to 2D robot base coordinates on the floor plane."""

    def __init__(
        self,
        camera_height_cm: float | None = None,
        forward_offset_cm: float | None = None,
        focal_length_px: float | None = None,
        reference_width_px: int = 1920,
        reference_height_px: int = 1080,
        camera_fixed_pitch_deg: float | None = None,
        camera_roll_bias_deg: float | None = None,
        use_kinematics: bool = True,
    ) -> None:
        self.kinematics: RobotKinematics = RobotKinematics()
        self.use_kinematics: bool = use_kinematics

        # Base hardware defaults (overlaid by camera_calibration.json)
        cfg_pitch: float = 8.39
        cfg_roll: float = -1.01
        cfg_height: float = 11.91
        cfg_offset: float = 10.0
        cfg_focal: float = 954.17
        self.dist_coeffs: np.ndarray | None = None
        self.camera_matrix: np.ndarray | None = None

        # Load saved calibration if present
        if CALIBRATION_FILE.exists():
            try:
                data = json.loads(CALIBRATION_FILE.read_text())
                cfg_pitch = float(data.get("pitch_bias_deg", cfg_pitch))
                cfg_roll = float(data.get("roll_bias_deg", cfg_roll))
                cfg_height = float(data.get("camera_height_cm", cfg_height))
                cfg_offset = float(data.get("forward_offset_cm", cfg_offset))
                cfg_focal = float(data.get("focal_length_px", cfg_focal))
                reference_width_px = int(data.get("reference_width_px", reference_width_px))
                reference_height_px = int(data.get("reference_height_px", reference_height_px))
                if "joint_x_cm" in data:
                    self.kinematics.joint_x_cm = float(data["joint_x_cm"])
                if "joint_z_cm" in data:
                    self.kinematics.joint_z_cm = float(data["joint_z_cm"])
                if "dist_coeffs" in data:
                    self.dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64).reshape(-1)
                if "camera_matrix" in data:
                    self.camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
            except Exception:
                pass

        # Explicit caller arguments take top priority; otherwise use calibrated/defaults
        self.camera_fixed_pitch_deg: float = camera_fixed_pitch_deg if camera_fixed_pitch_deg is not None else cfg_pitch
        self.camera_roll_bias_deg: float = camera_roll_bias_deg if camera_roll_bias_deg is not None else cfg_roll
        self.camera_height_cm: float = camera_height_cm if camera_height_cm is not None else cfg_height
        self.forward_offset_cm: float = forward_offset_cm if forward_offset_cm is not None else cfg_offset
        self.focal_length_px: float = focal_length_px if focal_length_px is not None else cfg_focal
        self.reference_width_px: int = reference_width_px
        self.reference_height_px: int = reference_height_px

        # Synchronize kinematic arm joint offsets with calibrated camera pose at 0 pitch
        if self.camera_height_cm is not None:
            self.kinematics.cam_rel_joint_z_cm = self.camera_height_cm - self.kinematics.joint_z_cm
        if self.forward_offset_cm is not None:
            self.kinematics.cam_rel_joint_x_cm = self.forward_offset_cm - self.kinematics.joint_x_cm

    def reload_calibration(self) -> bool:
        """
        Reload calibration parameters from camera_calibration.json if present.
        Returns True if successfully reloaded, False otherwise.
        """
        if CALIBRATION_FILE.exists():
            try:
                data = json.loads(CALIBRATION_FILE.read_text())
                self.camera_fixed_pitch_deg = float(data.get("pitch_bias_deg", self.camera_fixed_pitch_deg))
                self.camera_roll_bias_deg = float(data.get("roll_bias_deg", self.camera_roll_bias_deg))
                self.camera_height_cm = float(data.get("camera_height_cm", self.camera_height_cm))
                self.forward_offset_cm = float(data.get("forward_offset_cm", self.forward_offset_cm))
                self.focal_length_px = float(data.get("focal_length_px", self.focal_length_px))
                self.reference_width_px = int(data.get("reference_width_px", self.reference_width_px))
                self.reference_height_px = int(data.get("reference_height_px", self.reference_height_px))
                if "joint_x_cm" in data:
                    self.kinematics.joint_x_cm = float(data["joint_x_cm"])
                if "joint_z_cm" in data:
                    self.kinematics.joint_z_cm = float(data["joint_z_cm"])
                if "dist_coeffs" in data:
                    self.dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64).reshape(-1)
                if "camera_matrix" in data:
                    self.camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)

                if self.camera_height_cm is not None:
                    self.kinematics.cam_rel_joint_z_cm = self.camera_height_cm - self.kinematics.joint_z_cm
                if self.forward_offset_cm is not None:
                    self.kinematics.cam_rel_joint_x_cm = self.forward_offset_cm - self.kinematics.joint_x_cm
                return True
            except Exception:
                pass
        return False

    def get_camera_effective_pose(self, imu_pitch_deg: float = 0.0) -> tuple[float, float]:
        """
        Get the effective (x_cam_forward_cm, z_cam_height_cm) of the camera lens,
        accounting for arm tilt kinematics if enabled.
        """
        if self.use_kinematics:
            return self.kinematics.get_camera_pose(imu_pitch_deg)
        return self.forward_offset_cm, self.camera_height_cm

    def _build_camera_rotation(self, total_pitch_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build 3D rotation from Camera Optical frame (+X right, +Y down, +Z forward)
        to Robot Base frame (+X forward, +Y left, +Z up) incorporating pitch and roll.
        """
        pitch_rad = math.radians(total_pitch_deg)
        roll_rad = math.radians(self.camera_roll_bias_deg)

        sin_p = math.sin(pitch_rad)
        cos_p = math.cos(pitch_rad)
        sin_r = math.sin(roll_rad)
        cos_r = math.cos(roll_rad)

        # Optical axes in base frame at roll = 0:
        xc_0 = np.array([0.0, -1.0, 0.0])
        yc_0 = np.array([-sin_p, 0.0, -cos_p])
        zc = np.array([cos_p, 0.0, -sin_p])

        # Apply roll around optical axis +zc:
        xc = cos_r * xc_0 + sin_r * yc_0
        yc = -sin_r * xc_0 + cos_r * yc_0

        R_c_b = np.column_stack([xc, yc, zc])
        return R_c_b, xc, yc, zc

    def get_camera_matrix(self, frame_w: int, frame_h: int) -> np.ndarray:
        """
        Get the 3x3 intrinsic camera matrix K scaled to the target frame dimensions.
        """
        scale_x: float = frame_w / self.reference_width_px
        scale_y: float = frame_h / self.reference_height_px

        if self.camera_matrix is not None:
            return np.array([
                [self.camera_matrix[0, 0] * scale_x, 0.0, self.camera_matrix[0, 2] * scale_x],
                [0.0, self.camera_matrix[1, 1] * scale_y, self.camera_matrix[1, 2] * scale_y],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
        else:
            focal = self.focal_length_px * scale_x
            cx = frame_w / 2.0
            cy = frame_h / 2.0
            return np.array([
                [focal, 0.0, cx],
                [0.0, focal, cy],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)

    def pixel_to_ground(
        self,
        u: float,
        v: float,
        frame_w: int,
        frame_h: int,
        imu_pitch_deg: float = 0.0,
    ) -> tuple[float, float] | None:
        """
        Intersect a pixel ray with the floor plane at Z=0.

        Returns (x_forward_cm, y_left_cm) relative to the axle base center,
        or None if pointing at or above the horizon.
        """
        k_scaled = self.get_camera_matrix(frame_w, frame_h)

        # Undistort point if non-zero distortion coefficients exist
        if self.dist_coeffs is not None and np.any(np.abs(self.dist_coeffs) > 1e-6):
            pt_in = np.array([[[u, v]]], dtype=np.float64)
            pt_undist = cv2.undistortPoints(pt_in, k_scaled, self.dist_coeffs)
            xn = float(pt_undist[0, 0, 0])
            yn = float(pt_undist[0, 0, 1])
        else:
            fx = k_scaled[0, 0]
            fy = k_scaled[1, 1]
            cx = k_scaled[0, 2]
            cy = k_scaled[1, 2]
            xn = (u - cx) / fx
            yn = (v - cy) / fy

        cam_x, cam_z = self.get_camera_effective_pose(imu_pitch_deg)
        total_pitch_deg = self.camera_fixed_pitch_deg + imu_pitch_deg
        R_c_b, _, _, _ = self._build_camera_rotation(total_pitch_deg)

        ray_c = np.array([xn, yn, 1.0], dtype=np.float64)
        ray_b = R_c_b @ ray_c

        # Downward component in base coordinates is -ray_b[2]
        downward = -ray_b[2]
        if downward <= 1e-4:
            return None

        # Distance along ray to hit Z=0 floor
        lambda_val = cam_z / downward
        x_base_cm = cam_x + lambda_val * ray_b[0]
        y_base_cm = lambda_val * ray_b[1]
        return float(x_base_cm), float(y_base_cm)

    def pixel_to_camera_nadir_ground(
        self,
        u: float,
        v: float,
        frame_w: int,
        frame_h: int,
        imu_pitch_deg: float = 0.0,
    ) -> tuple[float, float] | None:
        """
        Intersect a pixel ray with the floor plane at Z=0.
        Returns (x_cam_cm, y_cam_cm) relative to Camera Nadir (directly beneath lens).
        """
        ground = self.pixel_to_ground(u, v, frame_w, frame_h, imu_pitch_deg)
        if ground is None:
            return None
        cam_x, _ = self.get_camera_effective_pose(imu_pitch_deg)
        return ground[0] - cam_x, ground[1]

    def get_camera_radial_distance(
        self,
        u: float,
        v: float,
        frame_w: int,
        frame_h: int,
        imu_pitch_deg: float = 0.0,
    ) -> float | None:
        """
        Radial ground distance R_cam (cm) from Camera Nadir to the floor contact point.
        """
        cam_pt = self.pixel_to_camera_nadir_ground(u, v, frame_w, frame_h, imu_pitch_deg)
        if cam_pt is None:
            return None
        return math.hypot(cam_pt[0], cam_pt[1])

    def get_optical_depth(
        self,
        x_base_cm: float,
        y_base_cm: float,
        imu_pitch_deg: float = 0.0,
    ) -> float:
        """
        Compute optical depth Z_c along the camera optical axis for a ground point.
        """
        cam_x, cam_z = self.get_camera_effective_pose(imu_pitch_deg)
        total_pitch_deg = self.camera_fixed_pitch_deg + imu_pitch_deg
        R_c_b, _, _, _ = self._build_camera_rotation(total_pitch_deg)
        R_b_c = R_c_b.T

        delta_b = np.array([x_base_cm - cam_x, y_base_cm, -cam_z], dtype=np.float64)
        p_c = R_b_c @ delta_b
        return max(1.0, float(p_c[2]))

    def get_3d_distance(
        self,
        x_base_cm: float,
        y_base_cm: float,
        imu_pitch_deg: float = 0.0,
    ) -> float:
        """
        Compute 3D Euclidean slant range (cm) from camera lens to a ground point.
        """
        cam_x, cam_z = self.get_camera_effective_pose(imu_pitch_deg)
        dx = x_base_cm - cam_x
        dy = y_base_cm
        dz = cam_z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def ground_to_pixel(
        self,
        x_base_cm: float,
        y_base_cm: float,
        frame_w: int,
        frame_h: int,
        imu_pitch_deg: float = 0.0,
    ) -> tuple[int, int] | None:
        """
        Project a floor point (x_base_cm forward, y_base_cm left) in robot base coordinates
        back into 2D camera pixel coordinates (u, v).

        Returns (u, v) or None if the point lies behind the camera optical plane.
        """
        cam_x, cam_z = self.get_camera_effective_pose(imu_pitch_deg)
        total_pitch_deg = self.camera_fixed_pitch_deg + imu_pitch_deg
        R_c_b, _, _, _ = self._build_camera_rotation(total_pitch_deg)
        R_b_c = R_c_b.T

        delta_b = np.array([x_base_cm - cam_x, y_base_cm, -cam_z], dtype=np.float64)
        p_c = R_b_c @ delta_b

        z_optical = p_c[2]
        if z_optical <= 1e-4:
            return None

        xn = p_c[0] / z_optical
        yn = p_c[1] / z_optical
        r_sq = xn * xn + yn * yn

        # Guard against polynomial distortion runaway (Runge's phenomenon):
        # Sensor corners reach at most r^2 ~ 0.52. Beyond r^2 > 0.55, the 6th-order
        # polynomial distortion inverts and diverges, projecting points backwards into the sky.
        if r_sq > 0.55:
            return None

        k_scaled = self.get_camera_matrix(frame_w, frame_h)

        if self.dist_coeffs is not None and np.any(np.abs(self.dist_coeffs) > 1e-6):
            obj_pt = np.array([[[p_c[0], p_c[1], p_c[2]]]], dtype=np.float64)
            rvec = np.zeros((3, 1), dtype=np.float64)
            tvec = np.zeros((3, 1), dtype=np.float64)
            img_pts, _ = cv2.projectPoints(obj_pt, rvec, tvec, k_scaled, self.dist_coeffs)
            u = int(round(img_pts[0, 0, 0]))
            v = int(round(img_pts[0, 0, 1]))
        else:
            fx = k_scaled[0, 0]
            fy = k_scaled[1, 1]
            cx = k_scaled[0, 2]
            cy = k_scaled[1, 2]
            u = int(round(cx + fx * xn))
            v = int(round(cy + fy * yn))

        if not (-60 <= u < frame_w + 60 and -40 <= v < frame_h + 40):
            return None

        return u, v

    def pixels_to_ground(
        self,
        points_px: np.ndarray,
        frame_w: int,
        frame_h: int,
        imu_pitch_deg: float = 0.0,
    ) -> np.ndarray:
        """
        Vectorized intersection of an (N, 2) array of pixel coordinates (u, v)
        with the floor plane at Z=0.

        Returns an (N, 2) array of (x_forward_cm, y_left_cm) relative to the axle base center.
        Points at or above the horizon return NaN.
        """
        pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
        n_pts = len(pts)
        result = np.full((n_pts, 2), np.nan, dtype=np.float64)
        if n_pts == 0:
            return result

        k_scaled = self.get_camera_matrix(frame_w, frame_h)

        # Undistort if needed
        if self.dist_coeffs is not None and np.any(np.abs(self.dist_coeffs) > 1e-6):
            undist = cv2.undistortPoints(pts.reshape(-1, 1, 2), k_scaled, self.dist_coeffs).reshape(-1, 2)
            xn = undist[:, 0]
            yn = undist[:, 1]
        else:
            fx = k_scaled[0, 0]
            fy = k_scaled[1, 1]
            cx = k_scaled[0, 2]
            cy = k_scaled[1, 2]
            xn = (pts[:, 0] - cx) / fx
            yn = (pts[:, 1] - cy) / fy

        cam_x, cam_z = self.get_camera_effective_pose(imu_pitch_deg)
        total_pitch_deg = self.camera_fixed_pitch_deg + imu_pitch_deg
        R_c_b, _, _, _ = self._build_camera_rotation(total_pitch_deg)

        # Ray directions in base frame: ray_b = R_c_b @ [xn, yn, 1]
        # (N, 3) = [xn, yn, 1] @ R_c_b.T
        rays_c = np.column_stack([xn, yn, np.ones(n_pts, dtype=np.float64)])
        rays_b = rays_c @ R_c_b.T

        downward = -rays_b[:, 2]
        valid = downward > 1e-4

        if not np.any(valid):
            return result

        lambda_vals = np.zeros(n_pts, dtype=np.float64)
        lambda_vals[valid] = cam_z / downward[valid]

        result[valid, 0] = cam_x + lambda_vals[valid] * rays_b[valid, 0]
        result[valid, 1] = lambda_vals[valid] * rays_b[valid, 1]
        return result

    def ground_to_pixels(
        self,
        points_base_cm: np.ndarray,
        frame_w: int,
        frame_h: int,
        imu_pitch_deg: float = 0.0,
    ) -> np.ndarray:
        """
        Vectorized projection of an (N, 2) array of floor coordinates (x_forward_cm, y_left_cm)
        in robot base coordinates to 2D camera pixel coordinates (u, v).

        Returns an (N, 2) array of rounded pixel coordinates (or NaN for points behind camera).
        """
        pts = np.asarray(points_base_cm, dtype=np.float64).reshape(-1, 2)
        n_pts = len(pts)
        result = np.full((n_pts, 2), np.nan, dtype=np.float64)
        if n_pts == 0:
            return result

        cam_x, cam_z = self.get_camera_effective_pose(imu_pitch_deg)
        total_pitch_deg = self.camera_fixed_pitch_deg + imu_pitch_deg
        R_c_b, _, _, _ = self._build_camera_rotation(total_pitch_deg)
        R_b_c = R_c_b.T

        delta_b = np.column_stack([pts[:, 0] - cam_x, pts[:, 1], np.full(n_pts, -cam_z, dtype=np.float64)])
        pts_c = delta_b @ R_b_c.T

        z_optical = pts_c[:, 2]
        pos_z = z_optical > 1e-4

        xn = np.zeros(n_pts, dtype=np.float64)
        yn = np.zeros(n_pts, dtype=np.float64)
        xn[pos_z] = pts_c[pos_z, 0] / z_optical[pos_z]
        yn[pos_z] = pts_c[pos_z, 1] / z_optical[pos_z]
        r_sq = xn * xn + yn * yn

        valid = pos_z & (r_sq <= 0.55)
        if not np.any(valid):
            return result

        k_scaled = self.get_camera_matrix(frame_w, frame_h)

        if self.dist_coeffs is not None and np.any(np.abs(self.dist_coeffs) > 1e-6):
            obj_pts = pts_c[valid].reshape(-1, 1, 3)
            rvec = np.zeros((3, 1), dtype=np.float64)
            tvec = np.zeros((3, 1), dtype=np.float64)
            img_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, k_scaled, self.dist_coeffs)
            result[valid, :] = np.round(img_pts.reshape(-1, 2))
        else:
            fx = k_scaled[0, 0]
            fy = k_scaled[1, 1]
            cx = k_scaled[0, 2]
            cy = k_scaled[1, 2]
            u = np.round(cx + fx * xn[valid])
            v = np.round(cy + fy * yn[valid])
            result[valid, 0] = u
            result[valid, 1] = v

        return result

    def get_claw_position(self, imu_pitch_deg: float = 0.0) -> tuple[float, float]:
        """
        Get instantaneous 2D claw tip coordinates (x_forward_cm, z_height_cm) relative to axle base.
        """
        return self.kinematics.get_claw_pose(imu_pitch_deg)

    def get_claw_capture_zone_ground(self, imu_pitch_deg: float = 0.0) -> list[tuple[float, float]]:
        """
        Return 4 bounding ground corners (x_cm, y_cm) of the gripper capture zone
        projected on the floor for the current arm pitch.
        """
        claw_x, claw_z = self.get_claw_position(imu_pitch_deg)
        x_min = max(0.0, claw_x - 3.5)
        x_max = claw_x + 3.5
        y_span = 3.5  # half-width of claws (cm)
        return [(x_min, -y_span), (x_max, -y_span), (x_max, y_span), (x_min, y_span)]

    def polygon_to_metric_bounds(
        self,
        polygon_px: np.ndarray,
        frame_w: int,
        frame_h: int,
        imu_pitch_deg: float = 0.0,
    ) -> tuple[float, float, float, float, float] | None:
        """
        Convert a pixel polygon mask to ground metric dimensions.
        Uses the lowest contour point (max y) as the true floor contact (Z=0)
        and computes the true lateral width facing the gripper jaws to avoid
        the 3D perspective height-smearing artifact.

        Returns (x_base_cm, y_base_cm, length_cm, width_cm, angle_deg) or None.
        """
        pts = np.asarray(polygon_px, dtype=np.float32).reshape(-1, 2)
        if len(pts) < 3:
            return None

        x_min = float(pts[:, 0].min())
        x_max = float(pts[:, 0].max())
        y_min = float(pts[:, 1].min())
        y_max = float(pts[:, 1].max())

        cx = (x_min + x_max) / 2.0
        cy = y_max  # Lowest point of object contour is on the floor (Z = 0)

        ground_contact = self.pixel_to_ground(cx, cy, frame_w, frame_h, imu_pitch_deg)
        if ground_contact is None:
            return None

        x_base_cm, y_base_cm = ground_contact
        z_optical = max(1.0, self.get_optical_depth(x_base_cm, y_base_cm, imu_pitch_deg))

        k_scaled = self.get_camera_matrix(frame_w, frame_h)
        fx = k_scaled[0, 0]
        fy = k_scaled[1, 1]

        # True head-on lateral width facing the gripper jaws
        w_lateral_cm: float = ((x_max - x_min) / fx) * z_optical
        # Optical height in the camera plane
        h_optical_cm: float = ((y_max - y_min) / fy) * z_optical
        length_cm: float = max(w_lateral_cm, h_optical_cm)

        # Approximate 2D image plane orientation angle
        _, _, angle_deg = cv2.minAreaRect(pts)

        return float(x_base_cm), float(y_base_cm), float(length_cm), float(w_lateral_cm), float(angle_deg)

    def bbox_to_metric_bounds(
        self,
        bbox: tuple[int, int, int, int],
        frame_w: int,
        frame_h: int,
        imu_pitch_deg: float = 0.0,
    ) -> tuple[float, float, float] | None:
        """
        Estimate ground metric dimensions from a bounding box (x1, y1, x2, y2).
        Projects the bottom contact edge to the floor (Z = 0) and computes lateral head-on width.

        Returns:
            (x_base_cm, y_base_cm, width_cm) or None
        """
        x1, y1, x2, y2 = bbox
        cx: float = (x1 + x2) / 2.0
        cy: float = float(y2)

        ground_contact = self.pixel_to_ground(cx, cy, frame_w, frame_h, imu_pitch_deg)
        if ground_contact is None:
            return None

        x_base_cm, y_base_cm = ground_contact
        z_optical = max(1.0, self.get_optical_depth(x_base_cm, y_base_cm, imu_pitch_deg))

        k_scaled = self.get_camera_matrix(frame_w, frame_h)
        fx = k_scaled[0, 0]

        w_px: float = float(x2 - x1)
        width_cm: float = (w_px / fx) * z_optical

        return float(x_base_cm), float(y_base_cm), float(width_cm)

    def draw_floor_grid(
        self,
        image: np.ndarray,
        imu_pitch_deg: float = 0.0,
        dist_markers_cm: Sequence[int] | None = None,
        lateral_lines_cm: Sequence[int] = (-30, -20, -10, 0, 10, 20, 30),
    ) -> np.ndarray:
        """
        Render an overlay grid of ground distance lines, lateral offsets, and the
        projected gripper capture zone onto the image. Dynamically clips to the camera's
        true visible ground frustum to avoid distortion fold-over artifacts.
        """
        canvas = image.copy()
        h, w = canvas.shape[:2]
        cx = w / 2.0

        # Determine closest visible ground distance at bottom center of frame
        bottom_pt = self.pixel_to_ground(cx, float(h - 8), w, h, imu_pitch_deg)
        min_x = bottom_pt[0] if bottom_pt is not None else 30.0
        start_dist = int(math.ceil(min_x / 5.0)) * 5

        # 1. Lateral lines (running forward along the floor parallel to heading)
        x_min_run = max(15, int(min_x) - 2)
        for y_cm in lateral_lines_cm:
            pts = []
            for x_cm in range(x_min_run, 185, 2):
                uv = self.ground_to_pixel(float(x_cm), float(y_cm), w, h, imu_pitch_deg)
                if uv and -60 <= uv[0] < w + 60 and -40 <= uv[1] < h + 40:
                    pts.append(uv)
            if len(pts) >= 2:
                color = (0, 255, 255) if y_cm == 0 else (160, 160, 160)
                thickness = 2 if y_cm == 0 else 1
                cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)
                if y_cm == 0 and len(pts) > 0:
                    first_pt = pts[0]
                    lbl_x = max(10, min(w - 60, first_pt[0] - 25))
                    lbl_y = min(h - 10, first_pt[1] + 20)
                    cv2.putText(canvas, "CENTER", (lbl_x + 1, lbl_y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(canvas, "CENTER", (lbl_x, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        # 2. Distance lines (perpendicular to robot heading)
        if dist_markers_cm is None:
            candidate_dists = [start_dist, start_dist + 5, start_dist + 10, start_dist + 20, 50, 60, 80, 100, 120, 150]
            dists_to_draw = sorted(list(set([d for d in candidate_dists if d >= min_x - 1.0])))
        else:
            dists_to_draw = sorted([d for d in dist_markers_cm if d >= min_x - 1.0])

        for d_cm in dists_to_draw:
            pts = []
            for y_cm in np.linspace(-45, 45, 91):
                uv = self.ground_to_pixel(float(d_cm), float(y_cm), w, h, imu_pitch_deg)
                if uv and -60 <= uv[0] < w + 60 and -40 <= uv[1] < h + 40:
                    pts.append(uv)
            if len(pts) >= 2:
                cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, (0, 200, 255), 1, cv2.LINE_AA)
                center_pts = [p for p in pts if 0 <= p[0] < w and 0 <= p[1] < h]
                if center_pts:
                    mid_pt = center_pts[len(center_pts) // 2]
                    cv2.putText(canvas, f"{d_cm}cm", (mid_pt[0] + 7, mid_pt[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(canvas, f"{d_cm}cm", (mid_pt[0] + 6, mid_pt[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)

        # 3. Gripper Capture Zone (only if physically visible in the camera frame)
        zone_corners = self.get_claw_capture_zone_ground(imu_pitch_deg)
        zone_px = []
        for xg, yg in zone_corners:
            uv = self.ground_to_pixel(xg, yg, w, h, imu_pitch_deg)
            if uv and -20 <= uv[0] < w + 20 and -20 <= uv[1] < h + 20:
                zone_px.append(uv)
        if len(zone_px) == 4:
            cv2.polylines(canvas, [np.array(zone_px, dtype=np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)
            claw_x, claw_z = self.get_claw_position(imu_pitch_deg)
            tag = f"CLAWS ({claw_x:.1f}cm, Z={claw_z:.1f}cm)"
            cv2.putText(canvas, tag, (zone_px[0][0] + 1, zone_px[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, tag, (zone_px[0][0], zone_px[0][1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        return canvas
