from __future__ import annotations

from typing import Sequence

import numpy as np

from config import (
    DEADBAND,
    MIN_PICKUP_DISPLACEMENT,
    MIN_PUTBACK_DISPLACEMENT,
    MOTION_MIN_DEPTH_RATIO,
)
from utils.roi_utils import nearest_edge_normal
from utils.types import GlobalTrack


class MotionAnalyzer:
    """
    Analyzes motion in the canonical warped shelf plane.

    Uses the actual outward vector from the ROI config (not a fixed axis).
    Applies a configurable depth ratio and a hybrid displacement estimator.
    """

    def __init__(
        self,
        *,
        min_pickup_displacement: float = MIN_PICKUP_DISPLACEMENT,
        min_putback_displacement: float = MIN_PUTBACK_DISPLACEMENT,
        deadband: float = DEADBAND,
        min_depth_ratio: float = MOTION_MIN_DEPTH_RATIO,
    ) -> None:
        self.min_pickup_displacement = min_pickup_displacement
        self.min_putback_displacement = min_putback_displacement
        self.deadband = deadband
        self.min_depth_ratio = min_depth_ratio

    def analyze(
        self,
        track: GlobalTrack,
        roi_polygon: Sequence[Sequence[float]],
        edge_normals: Sequence[Sequence[float]],
        outward_vector: Sequence[float],
    ) -> dict:
        """
        Returns motion classification.
        """
        # ---- 1. Collect centroid history (per camera if possible) ----
        motion_points = None
        if track.current_update is not None:
            camera_history = getattr(track, "centroid_history_by_camera", {}).get(
                track.current_update.camera_id
            )
            if camera_history is not None and len(camera_history) >= 2:
                motion_points = camera_history
        if motion_points is None:
            motion_points = (
                track.centroid_history
                if len(track.centroid_history) >= 2
                else track.motion_centroid_history
            )

        if len(motion_points) < 2:
            return self._zero_result()

        points = np.asarray(list(motion_points), dtype=np.float32)

        # ---- 2. Compute displacement (hybrid: median + raw fallback) ----
        # Median over last 3‑frame windows (reduces jitter)
        if len(points) >= 6:
            start_median = np.median(points[-6:-3], axis=0)
            end_median = np.median(points[-3:], axis=0)
            disp_median = end_median - start_median
            mag_median = float(np.linalg.norm(disp_median))
        else:
            disp_median = points[-1] - points[0]
            mag_median = float(np.linalg.norm(disp_median))

        # Raw 1‑frame displacement (for fast pickups)
        if len(points) >= 2:
            disp_raw = points[-1] - points[-2]
            mag_raw = float(np.linalg.norm(disp_raw))
        else:
            disp_raw = disp_median
            mag_raw = mag_median

        # Take the larger magnitude (preserves fast motion)
        if mag_raw > mag_median:
            displacement = disp_raw
            magnitude = mag_raw
        else:
            displacement = disp_median
            magnitude = mag_median

        # ---- 3. Outward direction from ROI config (normalised) ----
        outward_dir = np.asarray(outward_vector, dtype=np.float32)
        norm = np.linalg.norm(outward_dir)
        if norm > 1e-6:
            outward_dir /= norm
        else:
            outward_dir = np.array([0.0, 1.0], dtype=np.float32)

        # ---- 4. Compute depth component and ratio ----
        depth_delta = float(np.dot(displacement, outward_dir))
        axis_magnitude = abs(depth_delta)
        depth_ratio = axis_magnitude / max(magnitude, 1e-6)

        directional_enough = (
            magnitude >= self.deadband
            and axis_magnitude >= self.deadband
            and depth_ratio >= self.min_depth_ratio
        )

        # ---- 5. Debug geometry (edge normal, not used for decision) ----
        current_point = points[-1]
        try:
            edge_index, normal = nearest_edge_normal(current_point, roi_polygon, edge_normals)
        except Exception:
            edge_index = 0
            normal = outward_dir.copy()

        # Deadband: ignore tiny movements
        if magnitude < self.deadband:
            return {
                "outward_motion": False,
                "inward_motion": False,
                "displacement_vector": displacement.astype(np.float32),
                "displacement_magnitude": magnitude,
                "motion_dot": depth_delta,
                "nearest_edge_index": edge_index,
                "edge_normal": normal.astype(np.float32),
            }

        # ---- 6. Final classification ----
        outward = directional_enough and depth_delta >= self.min_pickup_displacement
        inward = directional_enough and depth_delta <= -self.min_putback_displacement

        return {
            "outward_motion": outward,
            "inward_motion": inward,
            "displacement_vector": displacement.astype(np.float32),
            "displacement_magnitude": magnitude,
            "motion_dot": depth_delta,
            "nearest_edge_index": edge_index,
            "edge_normal": normal.astype(np.float32),
        }

    def _zero_result(self) -> dict:
        return {
            "outward_motion": False,
            "inward_motion": False,
            "displacement_vector": np.zeros(2, dtype=np.float32),
            "displacement_magnitude": 0.0,
            "motion_dot": 0.0,
            "nearest_edge_index": 0,
            "edge_normal": np.zeros(2, dtype=np.float32),
        }