from __future__ import annotations  # Allow forward references in type hints

import itertools
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from config import (
    LOCAL_TRACK_MAX_DISTANCE_NORM,
    LOCAL_TRACK_MAX_MISSES,
    LOCAL_TRACK_MIN_SIMILARITY,
    TEMPORAL_IOU_THRESHOLD,
    TRACK_EMBEDDING_BANK,
)
from motion.optical_flow import propagate_bbox_lk
from utils.roi_utils import bbox_iou, cosine_similarity, point_in_polygon
from utils.types import Detection, TrackObservation


@dataclass
class LocalTrack:
    """
    A short‑term track for a single camera.

    Stores the latest state and history of an object seen in one camera view.
    """
    track_id: int
    class_id: int
    class_name: str
    bbox: tuple[float, float, float, float]          # Current bounding box
    centroid: np.ndarray                              # Current centroid (x,y)
    last_confidence: float
    last_embedding: np.ndarray                        # Histogram embedding
    last_frame_index: int
    last_timestamp_ms: float
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    missed_frames: int = 0                            # Consecutive frames without a detection
    age: int = 1                                      # Number of frames this track has existed
    embedding_history: deque = field(default_factory=lambda: deque(maxlen=TRACK_EMBEDDING_BANK))
    prev_bbox: Optional[tuple[float, float, float, float]] = None
    temporal_iou: float = 1.0                        # IoU between prev_bbox and current bbox
    in_safe_roi_override: Optional[bool] = None      # Manual override for ROI membership
    in_outer_roi_override: Optional[bool] = None
    source_view: str = "warped"                      # "warped" (homography) or "original"
    motion_centroid: Optional[np.ndarray] = None     # Centroid from motion‑compensated tracking
    motion_velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))

    def update(self, detection: Detection) -> None:
        """Update the track with a new detection."""
        self.prev_bbox = self.bbox
        self.bbox = detection.bbox
        self.velocity = detection.centroid - self.centroid

        # Update motion centroid (used for more stable motion analysis)
        next_motion_centroid = (
            detection.original_centroid.copy() if detection.original_centroid is not None
            else detection.centroid.copy()
        )
        if self.motion_centroid is not None:
            self.motion_velocity = next_motion_centroid - self.motion_centroid

        self.centroid = detection.centroid
        self.last_confidence = detection.confidence
        self.last_embedding = detection.embedding
        self.last_frame_index = detection.frame_index
        self.last_timestamp_ms = detection.timestamp_ms
        self.missed_frames = 0
        self.age += 1
        self.embedding_history.append(detection.embedding)

        # Compute temporal IoU (continuity with previous frame)
        self.temporal_iou = 1.0 if self.prev_bbox is None else bbox_iou(self.prev_bbox, self.bbox)

        self.in_safe_roi_override = detection.in_safe_roi_override
        self.in_outer_roi_override = detection.in_outer_roi_override
        self.source_view = detection.source_view
        self.motion_centroid = next_motion_centroid


class SingleCameraTracker:
    """
    Tracks objects in one camera view.

    Matches detections to existing tracks, creates new tracks, propagates tracks
    through missing detections using optical flow, and outputs TrackObservations.
    """

    def __init__(self, camera_id: int) -> None:
        self.camera_id = camera_id
        self.next_track_id = itertools.count(1)          # Generates new local track IDs
        self.tracks: Dict[int, LocalTrack] = {}          # Maps local ID -> LocalTrack
        self.prev_gray = None                            # Previous grayscale frame for optical flow

    # -------------------------------------------------------------------------
    # Cost function for matching a detection to a track
    # -------------------------------------------------------------------------
    def _cost(self, track: LocalTrack, detection: Detection) -> float:
        """Return a matching cost (lower is better). Very high cost means impossible match."""
        # Different classes cannot match
        if track.class_id != detection.class_id:
            return 1e6

        # Normalised centroid distance (divided by 200 pixels)
        centroid_distance = np.linalg.norm(track.centroid - detection.centroid) / 200.0

        # Bounding box IoU
        iou = bbox_iou(track.bbox, detection.bbox)

        # Appearance similarity (cosine of embeddings)
        similarity = cosine_similarity(track.last_embedding, detection.embedding)

        # Conservative rejection: if IoU, spatial, and appearance are all weak, forbid match
        if (iou < TEMPORAL_IOU_THRESHOLD
                and centroid_distance > LOCAL_TRACK_MAX_DISTANCE_NORM
                and similarity < LOCAL_TRACK_MIN_SIMILARITY):
            return 1e6

        # Older tracks get a small bias to stay alive
        age_bias = min(track.age / 100.0, 0.2)

        # Weighted sum: appearance (45%), spatial (35%), IoU (20%), minus age bias
        return 0.45 * (1.0 - similarity) + 0.35 * centroid_distance + 0.2 * (1.0 - iou) - age_bias

    # -------------------------------------------------------------------------
    # Convert a LocalTrack to a TrackObservation (standard output format)
    # -------------------------------------------------------------------------
    def _make_observation(
        self,
        track: LocalTrack,
        *,
        safe_roi_polygon: np.ndarray,
        full_roi_polygon: np.ndarray,
        shaky: bool,
    ) -> TrackObservation:
        """Create a TrackObservation from a LocalTrack, evaluating ROI membership."""
        # Determine ROI membership (override if provided, else point‑in‑polygon test)
        in_safe_roi = (
            track.in_safe_roi_override
            if track.in_safe_roi_override is not None
            else point_in_polygon(track.centroid, safe_roi_polygon)
        )
        in_outer_roi = (
            track.in_outer_roi_override
            if track.in_outer_roi_override is not None
            else point_in_polygon(track.centroid, full_roi_polygon)
        )

        return TrackObservation(
            camera_id=self.camera_id,
            local_track_id=track.track_id,
            frame_index=track.last_frame_index,
            timestamp_ms=track.last_timestamp_ms,
            class_id=track.class_id,
            class_name=track.class_name,
            bbox=track.bbox,
            centroid=track.centroid.copy(),
            confidence=track.last_confidence,
            embedding=track.last_embedding.copy(),
            velocity=track.velocity.copy(),
            motion_centroid=(
                track.motion_centroid.copy() if track.motion_centroid is not None
                else track.centroid.copy()
            ),
            in_safe_roi=in_safe_roi,
            in_outer_roi=in_outer_roi,
            shaky=shaky,
            temporal_iou=track.temporal_iou,
        )

    # -------------------------------------------------------------------------
    # Main update method: process detections for one frame
    # -------------------------------------------------------------------------
    def update(
        self,
        detections: List[Detection],
        *,
        gray: np.ndarray,
        safe_roi_polygon: np.ndarray,
        full_roi_polygon: np.ndarray,
        shaky: bool,
    ) -> List[TrackObservation]:
        """
        Update the tracker with new detections.

        Args:
            detections: List of detections for this frame.
            gray: Current grayscale frame (for optical flow propagation).
            safe_roi_polygon: Polygon of the safe inner ROI.
            full_roi_polygon: Polygon of the full outer ROI.
            shaky: Whether the current frame is globally shaky.

        Returns:
            List of TrackObservation objects for all active local tracks.
        """
        track_ids = list(self.tracks.keys())

        # ----- Hungarian matching between existing tracks and detections -----
        if track_ids and detections:
            cost_matrix = np.full((len(track_ids), len(detections)), 1e6, dtype=np.float32)
            for row, track_id in enumerate(track_ids):
                for col, detection in enumerate(detections):
                    cost_matrix[row, col] = self._cost(self.tracks[track_id], detection)

            rows, cols = linear_sum_assignment(cost_matrix)

            assigned_tracks = set()
            assigned_detections = set()
            for row, col in zip(rows, cols):
                if cost_matrix[row, col] >= 1e5:   # Effectively infinite cost → ignore
                    continue
                track = self.tracks[track_ids[row]]
                track.update(detections[col])       # Update matched track with detection
                assigned_tracks.add(track.track_id)
                assigned_detections.add(col)
        else:
            assigned_tracks = set()
            assigned_detections = set()

        # ----- Create new tracks for unmatched detections -----
        for idx, detection in enumerate(detections):
            if idx in assigned_detections:
                continue
            track_id = next(self.next_track_id)
            track = LocalTrack(
                track_id=track_id,
                class_id=detection.class_id,
                class_name=detection.class_name,
                bbox=detection.bbox,
                centroid=detection.centroid.copy(),
                last_confidence=detection.confidence,
                last_embedding=detection.embedding.copy(),
                last_frame_index=detection.frame_index,
                last_timestamp_ms=detection.timestamp_ms,
                in_safe_roi_override=detection.in_safe_roi_override,
                in_outer_roi_override=detection.in_outer_roi_override,
                source_view=detection.source_view,
                motion_centroid=(
                    detection.original_centroid.copy() if detection.original_centroid is not None
                    else detection.centroid.copy()
                ),
            )
            track.embedding_history.append(detection.embedding.copy())
            self.tracks[track_id] = track

        # ----- Handle tracks that missed a detection (propagate or delete) -----
        stale_ids = []
        for track_id, track in list(self.tracks.items()):
            if track_id in assigned_tracks:
                continue

            # Increment missed frame counter
            if detections and track_id not in assigned_tracks:
                track.missed_frames += 1
            elif not detections:
                track.missed_frames += 1

            # Try to propagate the bounding box using optical flow (if within max misses)
            propagated = None
            if track.missed_frames <= LOCAL_TRACK_MAX_MISSES:
                propagated = propagate_bbox_lk(self.prev_gray, gray, track.bbox)

            if propagated is not None:
                # Successfully propagated: update track state with propagated bbox
                px1, py1, px2, py2 = propagated
                new_centroid = np.array([(px1 + px2) / 2.0, (py1 + py2) / 2.0], dtype=np.float32)
                track.velocity = new_centroid - track.centroid
                track.prev_bbox = track.bbox
                track.bbox = propagated
                track.centroid = new_centroid
                if track.motion_centroid is not None:
                    track.motion_centroid = track.motion_centroid + track.motion_velocity
                track.temporal_iou = bbox_iou(track.prev_bbox, track.bbox)
            elif track.missed_frames > LOCAL_TRACK_MAX_MISSES:
                # Too many missed frames → delete the track
                stale_ids.append(track_id)

        # Remove stale tracks
        for track_id in stale_ids:
            self.tracks.pop(track_id, None)

        # Store current grayscale frame for next iteration's optical flow
        self.prev_gray = gray.copy()

        # Generate TrackObservation for every remaining track
        observations = [
            self._make_observation(
                track,
                safe_roi_polygon=safe_roi_polygon,
                full_roi_polygon=full_roi_polygon,
                shaky=shaky,
            )
            for track in self.tracks.values()
        ]
        return observations