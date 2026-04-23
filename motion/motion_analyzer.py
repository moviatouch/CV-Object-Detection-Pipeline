from __future__ import annotations

from typing import Sequence

import numpy as np

from config import DEADBAND, MIN_PICKUP_DISPLACEMENT, MIN_PUTBACK_DISPLACEMENT
from utils.roi_utils import nearest_edge_normal
from utils.types import GlobalTrack


class MotionAnalyzer:
    """
    Analyzes motion of a tracked object relative to the region of interest (ROI),
    determining whether the motion is outward (pickup) or inward (putback).
    """

    def __init__(
        self,
        *,
        min_pickup_displacement: float = MIN_PICKUP_DISPLACEMENT,
        min_putback_displacement: float = MIN_PUTBACK_DISPLACEMENT,
        deadband: float = DEADBAND,
    ) -> None:
        """
        Initialize motion thresholds.

        Args:
            min_pickup_displacement: Minimum movement magnitude to classify as pickup.
            min_putback_displacement: Minimum movement magnitude to classify as putback.
            deadband: Ignore movements smaller than this (jitter filter).
        """
        self.min_pickup_displacement = min_pickup_displacement
        self.min_putback_displacement = min_putback_displacement
        self.deadband = deadband

    def analyze(
        self,
        track: GlobalTrack,
        roi_polygon: Sequence[Sequence[float]],
        edge_normals: Sequence[Sequence[float]],
        outward_vector: Sequence[float],
    ) -> dict:
        """
        Analyze motion of the track's centroid.

        Args:
            track: The tracked object (contains centroid history).
            roi_polygon: ROI polygon as list of (x,y) points.
            edge_normals: Outward normals for each edge of the ROI polygon.
            outward_vector: Global outward direction (e.g., from shelf to customer).

        Returns:
            Dictionary with keys:
                - outward_motion (bool): True if moving outward (pickup)
                - inward_motion (bool): True if moving inward (putback)
                - displacement_vector (np.ndarray): (dx, dy)
                - displacement_magnitude (float): Euclidean distance
                - motion_dot (float): Dot product used for direction decision
                - nearest_edge_index (int): Index of the closest ROI edge
                - edge_normal (np.ndarray): Outward normal of that edge
        """
        # Choose which centroid history to use (prefer filtered motion history)
        motion_points = track.motion_centroid_history if len(track.motion_centroid_history) >= 2 else track.centroid_history

        # Not enough points to compute displacement
        if len(motion_points) < 2:
            return {
                "outward_motion": False,
                "inward_motion": False,
                "displacement_vector": np.zeros(2, dtype=np.float32),
                "displacement_magnitude": 0.0,
                "motion_dot": 0.0,
                "nearest_edge_index": 0,
                "edge_normal": np.zeros(2, dtype=np.float32),
            }

        # Compute displacement over a short temporal gap to reduce jitter
        # Use 5‑frame gap if enough history, otherwise from first to last point
        if len(motion_points) >= 6:
            displacement = motion_points[-1] - motion_points[-6]
        else:
            displacement = motion_points[-1] - motion_points[0]

        magnitude = float(np.linalg.norm(displacement))

        # Geometric references
        current_point = motion_points[-1]
        roi_centroid = np.mean(np.asarray(roi_polygon, dtype=np.float32), axis=0)
        radial_vector = current_point - roi_centroid                     # from ROI center to object

        # Nearest edge and its outward normal
        edge_index, normal = nearest_edge_normal(current_point, roi_polygon, edge_normals)

        # Dot products with different direction references
        radial_dot = float(np.dot(displacement, radial_vector))          # motion relative to ROI center
        edge_dot = float(np.dot(displacement, normal))                   # motion relative to nearest edge normal
        customer_dot = float(np.dot(displacement, np.asarray(outward_vector, dtype=np.float32)))  # fixed outward

        # Primary signal: radial dot (away from center = positive)
        dot = radial_dot

        # If radial signal is weak, fall back to the stronger of edge or customer dot
        if abs(radial_dot) < max(20.0, magnitude * 0.35):
            dot = edge_dot if abs(edge_dot) >= abs(customer_dot) else customer_dot

        # Deadband: ignore tiny movements (jitter)
        if magnitude < self.deadband:
            return {
                "outward_motion": False,
                "inward_motion": False,
                "displacement_vector": displacement.astype(np.float32),
                "displacement_magnitude": magnitude,
                "motion_dot": dot,
                "nearest_edge_index": edge_index,
                "edge_normal": normal.astype(np.float32),
            }

        # Classify direction based on dot sign and magnitude thresholds
        outward = dot > 0 and magnitude > self.min_pickup_displacement
        inward = dot < 0 and magnitude > self.min_putback_displacement

        return {
            "outward_motion": outward,
            "inward_motion": inward,
            "displacement_vector": displacement.astype(np.float32),
            "displacement_magnitude": magnitude,
            "motion_dot": dot,
            "nearest_edge_index": edge_index,
            "edge_normal": normal.astype(np.float32),
        }