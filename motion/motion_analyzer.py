# from __future__ import annotations

# from typing import Sequence

# import numpy as np

# from config import DEADBAND, MIN_PICKUP_DISPLACEMENT, MIN_PUTBACK_DISPLACEMENT
# from utils.roi_utils import nearest_edge_normal
# from utils.types import GlobalTrack


# class MotionAnalyzer:
#     """
#     Analyzes motion of a tracked object relative to the region of interest (ROI),
#     determining whether the motion is outward (pickup) or inward (putback).
#     """

#     def __init__(
#         self,
#         *,
#         min_pickup_displacement: float = MIN_PICKUP_DISPLACEMENT,
#         min_putback_displacement: float = MIN_PUTBACK_DISPLACEMENT,
#         deadband: float = DEADBAND,
#     ) -> None:
#         """
#         Initialize motion thresholds.

#         Args:
#             min_pickup_displacement: Minimum movement magnitude to classify as pickup.
#             min_putback_displacement: Minimum movement magnitude to classify as putback.
#             deadband: Ignore movements smaller than this (jitter filter).
#         """
#         self.min_pickup_displacement = min_pickup_displacement
#         self.min_putback_displacement = min_putback_displacement
#         self.deadband = deadband

#     def analyze(
#         self,
#         track: GlobalTrack,
#         roi_polygon: Sequence[Sequence[float]],
#         edge_normals: Sequence[Sequence[float]],
#         outward_vector: Sequence[float],
#     ) -> dict:
#         """
#         Analyze motion of the track's centroid.

#         Args:
#             track: The tracked object (contains centroid history).
#             roi_polygon: ROI polygon as list of (x,y) points.
#             edge_normals: Outward normals for each edge of the ROI polygon.
#             outward_vector: Global outward direction (e.g., from shelf to customer).

#         Returns:
#             Dictionary with keys:
#                 - outward_motion (bool): True if moving outward (pickup)
#                 - inward_motion (bool): True if moving inward (putback)
#                 - displacement_vector (np.ndarray): (dx, dy)
#                 - displacement_magnitude (float): Euclidean distance
#                 - motion_dot (float): Dot product used for direction decision
#                 - nearest_edge_index (int): Index of the closest ROI edge
#                 - edge_normal (np.ndarray): Outward normal of that edge
#         """
#         # Choose which centroid history to use (prefer filtered motion history)
#         motion_points = track.motion_centroid_history if len(track.motion_centroid_history) >= 2 else track.centroid_history

#         # Not enough points to compute displacement
#         if len(motion_points) < 2:
#             return {
#                 "outward_motion": False,
#                 "inward_motion": False,
#                 "displacement_vector": np.zeros(2, dtype=np.float32),
#                 "displacement_magnitude": 0.0,
#                 "motion_dot": 0.0,
#                 "nearest_edge_index": 0,
#                 "edge_normal": np.zeros(2, dtype=np.float32),
#             }

#         # Compute displacement over a short temporal gap to reduce jitter
#         # Use 5‑frame gap if enough history, otherwise from first to last point
#         if len(motion_points) >= 6:
#             displacement = motion_points[-1] - motion_points[-6]
#         else:
#             displacement = motion_points[-1] - motion_points[0]

#         magnitude = float(np.linalg.norm(displacement))

#         # Geometric references
#         current_point = motion_points[-1]
#         roi_centroid = np.mean(np.asarray(roi_polygon, dtype=np.float32), axis=0)
#         radial_vector = current_point - roi_centroid                     # from ROI center to object

#         # Nearest edge and its outward normal
#         edge_index, normal = nearest_edge_normal(current_point, roi_polygon, edge_normals)

#         # Dot products with different direction references
#         radial_dot = float(np.dot(displacement, radial_vector))          # motion relative to ROI center
#         edge_dot = float(np.dot(displacement, normal))                   # motion relative to nearest edge normal
#         customer_dot = float(np.dot(displacement, np.asarray(outward_vector, dtype=np.float32)))  # fixed outward

#         # Primary signal: radial dot (away from center = positive)
#         dot = radial_dot

#         # If radial signal is weak, fall back to the stronger of edge or customer dot
#         if abs(radial_dot) < max(20.0, magnitude * 0.35):
#             dot = edge_dot if abs(edge_dot) >= abs(customer_dot) else customer_dot

#         # Deadband: ignore tiny movements (jitter)
#         if magnitude < self.deadband:
#             return {
#                 "outward_motion": False,
#                 "inward_motion": False,
#                 "displacement_vector": displacement.astype(np.float32),
#                 "displacement_magnitude": magnitude,
#                 "motion_dot": dot,
#                 "nearest_edge_index": edge_index,
#                 "edge_normal": normal.astype(np.float32),
#             }

#         # Classify direction based on dot sign and magnitude thresholds
#         outward = dot > 0 and magnitude > self.min_pickup_displacement
#         inward = dot < 0 and magnitude > self.min_putback_displacement

#         return {
#             "outward_motion": outward,
#             "inward_motion": inward,
#             "displacement_vector": displacement.astype(np.float32),
#             "displacement_magnitude": magnitude,
#             "motion_dot": dot,
#             "nearest_edge_index": edge_index,
#             "edge_normal": normal.astype(np.float32),
#         }

# from __future__ import annotations

# from typing import Sequence

# import numpy as np

# from config import DEADBAND, MIN_PICKUP_DISPLACEMENT, MIN_PUTBACK_DISPLACEMENT
# from utils.roi_utils import nearest_edge_normal
# from utils.types import GlobalTrack


# class MotionAnalyzer:
#     """
#     Analyzes motion in the canonical warped shelf plane.

#     The production direction reference is fixed:
#     +Y is outward toward the customer, -Y is inward toward the shelf.
#     ROI geometry is still returned for debugging, but it is not used as the
#     source of truth for pickup/putback direction.
#     """

#     OUTWARD_AXIS = np.array([0.0, 1.0], dtype=np.float32)

#     def __init__(
#         self,
#         *,
#         min_pickup_displacement: float = MIN_PICKUP_DISPLACEMENT,
#         min_putback_displacement: float = MIN_PUTBACK_DISPLACEMENT,
#         deadband: float = DEADBAND,
#     ) -> None:
#         """
#         Initialize motion thresholds.

#         Args:
#             min_pickup_displacement: Minimum movement magnitude to classify as pickup.
#             min_putback_displacement: Minimum movement magnitude to classify as putback.
#             deadband: Ignore movements smaller than this (jitter filter).
#         """
#         self.min_pickup_displacement = min_pickup_displacement
#         self.min_putback_displacement = min_putback_displacement
#         self.deadband = deadband

#     def analyze(
#         self,
#         track: GlobalTrack,
#         roi_polygon: Sequence[Sequence[float]],
#         edge_normals: Sequence[Sequence[float]],
#         outward_vector: Sequence[float],
#     ) -> dict:
#         """
#         Analyze motion of the track's centroid.

#         Args:
#             track: The tracked object (contains centroid history).
#             roi_polygon: Canonical warped ROI polygon as list of (x,y) points.
#             edge_normals: Outward normals for each edge of the canonical ROI.
#             outward_vector: Kept for API compatibility; direction uses +Y.

#         Returns:
#             Dictionary with keys:
#                 - outward_motion (bool): True if moving outward (pickup)
#                 - inward_motion (bool): True if moving inward (putback)
#                 - displacement_vector (np.ndarray): (dx, dy)
#                 - displacement_magnitude (float): Euclidean distance
#                 - motion_dot (float): Dot product used for direction decision
#                 - nearest_edge_index (int): Index of the closest ROI edge
#                 - edge_normal (np.ndarray): Outward normal of that edge
#         """
#         motion_points = None
#         if track.current_update is not None:
#             camera_history = getattr(track, "centroid_history_by_camera", {}).get(
#                 track.current_update.camera_id
#             )
#             if camera_history is not None and len(camera_history) >= 2:
#                 motion_points = camera_history
#         if motion_points is None:
#             motion_points = (
#                 track.centroid_history
#                 if len(track.centroid_history) >= 2
#                 else track.motion_centroid_history
#             )

#         # Not enough points to compute displacement
#         if len(motion_points) < 2:
#             return {
#                 "outward_motion": False,
#                 "inward_motion": False,
#                 "displacement_vector": np.zeros(2, dtype=np.float32),
#                 "displacement_magnitude": 0.0,
#                 "motion_dot": 0.0,
#                 "nearest_edge_index": 0,
#                 "edge_normal": np.zeros(2, dtype=np.float32),
#             }

#         points = np.asarray(list(motion_points), dtype=np.float32)

#         # Use short median windows to reduce bbox-centroid jitter from product
#         # rotation, hand occlusion, and detector box shape changes.
#         if len(points) >= 6:
#             start_point = np.median(points[-6:-3], axis=0)
#             end_point = np.median(points[-3:], axis=0)
#         else:
#             start_point = points[0]
#             end_point = points[-1]

#         displacement = end_point - start_point

#         magnitude = float(np.linalg.norm(displacement))

#         # Debug geometry only. Direction classification below never depends on
#         # ROI shape, edge choice, or a user-drawn polygon being perfect.
#         current_point = points[-1]
#         try:
#             edge_index, normal = nearest_edge_normal(current_point, roi_polygon, edge_normals)
#         except Exception:
#             edge_index = 0
#             normal = self.OUTWARD_AXIS.copy()

#         dot = float(np.dot(displacement, self.OUTWARD_AXIS))
#         axis_magnitude = abs(dot)
#         directional_enough = (
#             magnitude >= self.deadband
#             and axis_magnitude >= self.deadband
#             and axis_magnitude >= magnitude * 0.20
#         )

#         # Deadband: ignore tiny movements (jitter)
#         if magnitude < self.deadband:
#             return {
#                 "outward_motion": False,
#                 "inward_motion": False,
#                 "displacement_vector": displacement.astype(np.float32),
#                 "displacement_magnitude": magnitude,
#                 "motion_dot": dot,
#                 "nearest_edge_index": edge_index,
#                 "edge_normal": normal.astype(np.float32),
#             }

#         # Classify direction from canonical shelf depth only.
#         outward = directional_enough and dot > self.min_pickup_displacement
#         inward = directional_enough and dot < -self.min_putback_displacement

#         return {
#             "outward_motion": outward,
#             "inward_motion": inward,
#             "displacement_vector": displacement.astype(np.float32),
#             "displacement_magnitude": magnitude,
#             "motion_dot": dot,
#             "nearest_edge_index": edge_index,
#             "edge_normal": normal.astype(np.float32),
#         }

# from __future__ import annotations

# from typing import Sequence

# import numpy as np

# from config import (
#     DEADBAND,
#     MIN_PICKUP_DISPLACEMENT,
#     MIN_PUTBACK_DISPLACEMENT,
#     MOTION_MIN_DEPTH_RATIO,
# )
# from utils.roi_utils import nearest_edge_normal
# from utils.types import GlobalTrack


# class MotionAnalyzer:
#     """
#     Analyzes motion in the canonical warped shelf plane.

#     The production direction reference is fixed:
#     +Y is outward toward the customer, -Y is inward toward the shelf.
#     ROI geometry is still returned for debugging, but it is not used as the
#     source of truth for pickup/putback direction.
#     """

#     OUTWARD_AXIS = np.array([0.0, 1.0], dtype=np.float32)

#     def __init__(
#         self,
#         *,
#         min_pickup_displacement: float = MIN_PICKUP_DISPLACEMENT,
#         min_putback_displacement: float = MIN_PUTBACK_DISPLACEMENT,
#         deadband: float = DEADBAND,
#         min_depth_ratio: float = MOTION_MIN_DEPTH_RATIO,
#     ) -> None:
#         """
#         Initialize motion thresholds.

#         Args:
#             min_pickup_displacement: Minimum movement magnitude to classify as pickup.
#             min_putback_displacement: Minimum movement magnitude to classify as putback.
#             deadband: Ignore movements smaller than this (jitter filter).
#             min_depth_ratio: Minimum y-axis share of motion required for a
#                 directional pickup/putback decision.
#         """
#         self.min_pickup_displacement = min_pickup_displacement
#         self.min_putback_displacement = min_putback_displacement
#         self.deadband = deadband
#         self.min_depth_ratio = min_depth_ratio

#     def analyze(
#         self,
#         track: GlobalTrack,
#         roi_polygon: Sequence[Sequence[float]],
#         edge_normals: Sequence[Sequence[float]],
#         outward_vector: Sequence[float],
#     ) -> dict:
#         """
#         Analyze motion of the track's centroid.

#         Args:
#             track: The tracked object (contains centroid history).
#             roi_polygon: Canonical warped ROI polygon as list of (x,y) points.
#             edge_normals: Outward normals for each edge of the canonical ROI.
#             outward_vector: Kept for API compatibility; direction uses +Y.

#         Returns:
#             Dictionary with keys:
#                 - outward_motion (bool): True if moving outward (pickup)
#                 - inward_motion (bool): True if moving inward (putback)
#                 - displacement_vector (np.ndarray): (dx, dy)
#                 - displacement_magnitude (float): Euclidean distance
#                 - motion_dot (float): Dot product used for direction decision
#                 - nearest_edge_index (int): Index of the closest ROI edge
#                 - edge_normal (np.ndarray): Outward normal of that edge
#         """
#         motion_points = None
#         if track.current_update is not None:
#             camera_history = getattr(track, "centroid_history_by_camera", {}).get(
#                 track.current_update.camera_id
#             )
#             if camera_history is not None and len(camera_history) >= 2:
#                 motion_points = camera_history
#         if motion_points is None:
#             motion_points = (
#                 track.centroid_history
#                 if len(track.centroid_history) >= 2
#                 else track.motion_centroid_history
#             )

#         # Not enough points to compute displacement
#         if len(motion_points) < 2:
#             return {
#                 "outward_motion": False,
#                 "inward_motion": False,
#                 "displacement_vector": np.zeros(2, dtype=np.float32),
#                 "displacement_magnitude": 0.0,
#                 "motion_dot": 0.0,
#                 "nearest_edge_index": 0,
#                 "edge_normal": np.zeros(2, dtype=np.float32),
#             }

#         points = np.asarray(list(motion_points), dtype=np.float32)

#         # Use short median windows to reduce bbox-centroid jitter from product
#         # rotation, hand occlusion, and detector box shape changes.
#         if len(points) >= 6:
#             start_point = np.median(points[-6:-3], axis=0)
#             end_point = np.median(points[-3:], axis=0)
#         else:
#             start_point = points[0]
#             end_point = points[-1]

#         displacement = end_point - start_point

#         magnitude = float(np.linalg.norm(displacement))

#         # Debug geometry only. Direction classification below never depends on
#         # ROI shape, edge choice, or a user-drawn polygon being perfect.
#         current_point = points[-1]
#         try:
#             edge_index, normal = nearest_edge_normal(current_point, roi_polygon, edge_normals)
#         except Exception:
#             edge_index = 0
#             normal = self.OUTWARD_AXIS.copy()

#         depth_delta = float(np.dot(displacement, self.OUTWARD_AXIS))
#         axis_magnitude = abs(depth_delta)
#         depth_ratio = axis_magnitude / max(magnitude, 1e-6)
#         directional_enough = (
#             magnitude >= self.deadband
#             and axis_magnitude >= self.deadband
#             and depth_ratio >= self.min_depth_ratio
#         )

#         # Deadband: ignore tiny movements (jitter)
#         if magnitude < self.deadband:
#             return {
#                 "outward_motion": False,
#                 "inward_motion": False,
#                 "displacement_vector": displacement.astype(np.float32),
#                 "displacement_magnitude": magnitude,
#                 "motion_dot": depth_delta,
#                 "nearest_edge_index": edge_index,
#                 "edge_normal": normal.astype(np.float32),
#             }

#         # Classify direction from canonical shelf depth only.
#         outward = directional_enough and depth_delta >= self.min_pickup_displacement
#         inward = directional_enough and depth_delta <= -self.min_putback_displacement

#         return {
#             "outward_motion": outward,
#             "inward_motion": inward,
#             "displacement_vector": displacement.astype(np.float32),
#             "displacement_magnitude": magnitude,
#             "motion_dot": depth_delta,
#             "nearest_edge_index": edge_index,
#             "edge_normal": normal.astype(np.float32),
#         }




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