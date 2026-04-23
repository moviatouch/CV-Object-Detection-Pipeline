from __future__ import annotations  # Allow forward references in type hints

import itertools
import logging
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from config import (
    BASE_BUFFER_MS,
    EMBEDDING_SIM_THRESHOLD,
    ENFORCE_CLASS_CONSISTENCY,
    GLOBAL_EMBEDDING_BANK,
    HOMOGRAPHY_MAX_ERROR,
    MAX_LOST_SECONDS,
    RESERVED_SECONDS,
    REVIVAL_ANGLE_THRESHOLD,
    REVIVAL_EMBEDDING_THRESHOLD,
    REVIVAL_SPATIAL_THRESHOLD,
    TRACK_CONFIDENCE_HISTORY,
    TEMPORAL_IOU_THRESHOLD,
    VELOCITY_BUFFER_FACTOR,
)
from utils.roi_utils import angle_between, bbox_iou, cosine_similarity
from utils.types import GlobalTrack, TrackObservation, TrackUpdate

LOGGER = logging.getLogger("vending_pipeline.registry")


@dataclass
class PendingObservation:
    """A detection that is temporarily stored before becoming a global track."""
    observation: TrackObservation
    expires_at_ms: float  # timestamp when this pending entry expires


class GlobalRegistry:
    """
    Maintains global object tracks across multiple cameras.

    Responsibilities:
    - Assign unique global IDs to objects.
    - Match per‑camera observations to existing tracks.
    - Create new tracks when no match is found (with cross‑camera buffering).
    - Handle lost objects and eventual deletion.
    - Provide updated tracks for each frame.
    """

    def __init__(self) -> None:
        self.next_global_id = itertools.count(1)          # Generates new global IDs
        self.global_tracks: Dict[int, GlobalTrack] = {}   # Maps global_id -> track object
        self.pending: List[PendingObservation] = []       # Observations waiting for cross‑camera match
        self.source_to_global: Dict[tuple[int, int], int] = {}  # (camera_id, local_track_id) -> global_id
        self.cross_camera_frozen = False                  # Disables cross‑camera matching if sync is bad

    # -------------------------------------------------------------------------
    # Helper: buffer duration for pending observations (depends on object speed)
    # -------------------------------------------------------------------------
    def _buffer_ms(self, observation: TrackObservation) -> float:
        """How long to keep a pending observation (longer for faster objects)."""
        speed = float(np.linalg.norm(observation.velocity))
        return BASE_BUFFER_MS + VELOCITY_BUFFER_FACTOR * speed

    # -------------------------------------------------------------------------
    # Create a new global track from a single observation
    # -------------------------------------------------------------------------
    def _new_global_track(self, observation: TrackObservation) -> GlobalTrack:
        global_id = next(self.next_global_id)
        track = GlobalTrack(
            global_id=global_id,
            class_id=observation.class_id,
            class_name=observation.class_name,
            created_ms=observation.timestamp_ms,
            last_seen_ms=observation.timestamp_ms,
        )
        # Initialise history buffers with max lengths from config
        track.embedding_bank = track.embedding_bank.__class__(maxlen=GLOBAL_EMBEDDING_BANK)
        track.confidence_history = track.confidence_history.__class__(maxlen=TRACK_CONFIDENCE_HISTORY)
        self.global_tracks[global_id] = track
        return track

    # -------------------------------------------------------------------------
    # Convert an observation + motion analysis into a TrackUpdate object
    # -------------------------------------------------------------------------
    def _observation_to_update(self, track: GlobalTrack, observation: TrackObservation, motion: dict) -> TrackUpdate:
        return TrackUpdate(
            global_id=track.global_id,
            class_id=track.class_id,
            class_name=track.class_name,
            camera_id=observation.camera_id,
            frame_index=observation.frame_index,
            timestamp_ms=observation.timestamp_ms,
            bbox=observation.bbox,
            centroid=observation.centroid.copy(),
            motion_centroid=observation.motion_centroid.copy(),
            confidence=observation.confidence,
            in_safe_roi=observation.in_safe_roi,
            in_outer_roi=observation.in_outer_roi,
            outward_motion=motion["outward_motion"],
            inward_motion=motion["inward_motion"],
            displacement_vector=np.asarray(motion["displacement_vector"], dtype=np.float32),
            displacement_magnitude=motion["displacement_magnitude"],
            motion_dot=motion["motion_dot"],
            nearest_edge_index=motion["nearest_edge_index"],
            edge_normal=np.asarray(motion["edge_normal"], dtype=np.float32),
            shaky=observation.shaky,
        )

    # -------------------------------------------------------------------------
    # Apply an observation to a global track (update history, state, current_update)
    # -------------------------------------------------------------------------
    def _apply_observation(self, track: GlobalTrack, observation: TrackObservation, motion: dict) -> None:
        track.last_seen_ms = observation.timestamp_ms
        track.last_confidence = observation.confidence
        track.last_camera_id = observation.camera_id
        track.last_frame_index = observation.frame_index
        track.bbox_by_camera[observation.camera_id] = observation.bbox
        track.source_local_ids[observation.camera_id] = observation.local_track_id
        track.last_frame_by_camera[observation.camera_id] = observation.frame_index
        track.centroid_history.append(observation.centroid.copy())
        track.motion_centroid_history.append(observation.motion_centroid.copy())
        if observation.camera_id not in track.motion_history_by_camera:
            track.motion_history_by_camera[observation.camera_id] = deque(maxlen=20)
        track.motion_history_by_camera[observation.camera_id].append(observation.motion_centroid.copy())
        track.velocity_history.append(observation.velocity.copy())
        track.embedding_bank.append(observation.embedding.copy())
        track.confidence_history.append(float(observation.confidence))
        track.current_update = self._observation_to_update(track, observation, motion)
        track.missing_since_ms = None               # Reset missing timer
        track.lost_reserved_since_ms = None
        track.last_seen_in_safe_roi = observation.in_safe_roi
        track.last_seen_in_outer_roi = observation.in_outer_roi
        track.lifecycle_state = "ACTIVE"

    # -------------------------------------------------------------------------
    # Look up a global track directly from the source mapping (camera, local_id)
    # -------------------------------------------------------------------------
    def _match_known_source(self, observation: TrackObservation) -> Optional[GlobalTrack]:
        global_id = self.source_to_global.get((observation.camera_id, observation.local_track_id))
        if global_id is None:
            return None
        return self.global_tracks.get(global_id)

    # -------------------------------------------------------------------------
    # Cost function for matching a candidate global track to an observation
    # -------------------------------------------------------------------------
    def _candidate_cost(self, track: GlobalTrack, observation: TrackObservation, homography_active: bool) -> float:
        # Class mismatch → impossible match (very high cost)
        if ENFORCE_CLASS_CONSISTENCY and track.class_id != observation.class_id:
            return 1e6

        # Same‑camera temporal IoU check – if both the track's last update and the observation
        # have low IoU with the previous bbox, reject the match.
        if track.current_update is not None and track.current_update.camera_id == observation.camera_id:
            temporal_iou = bbox_iou(track.current_update.bbox, observation.bbox)
            if temporal_iou < TEMPORAL_IOU_THRESHOLD and observation.temporal_iou < TEMPORAL_IOU_THRESHOLD:
                return 1e6

        # Appearance similarity: compute mean embedding of the track and cosine similarity
        embedding_sim = cosine_similarity(track.mean_embedding(), observation.embedding)
        if embedding_sim < EMBEDDING_SIM_THRESHOLD and track.lifecycle_state != "LOST_RESERVED":
            return 1e6

        # Spatial distance (normalised by 200 pixels)
        latest_centroid = track.centroid_history[-1] if track.centroid_history else observation.centroid
        spatial_norm = np.linalg.norm(latest_centroid - observation.centroid) / 200.0

        # Temporal IoU (if same camera, use track's last bbox; otherwise 0)
        iou = 0.0
        if track.current_update is not None:
            iou = bbox_iou(track.current_update.bbox, observation.bbox)

        # Weighted cost: appearance dominates, then spatial, then IoU
        cost = 0.5 * (1.0 - embedding_sim) + 0.3 * spatial_norm + 0.2 * (1.0 - iou)

        # If homography is not active (e.g., cross‑camera sync bad), rely more on appearance
        if not homography_active:
            cost = 0.7 * (1.0 - embedding_sim) + 0.3 * spatial_norm

        # Special handling for LOST_RESERVED tracks (revival candidate)
        if track.lifecycle_state == "LOST_RESERVED":
            velocity_angle = angle_between(track.last_velocity(), observation.velocity)
            spatial = np.linalg.norm(latest_centroid - observation.centroid)
            if (embedding_sim < REVIVAL_EMBEDDING_THRESHOLD
                    or spatial > REVIVAL_SPATIAL_THRESHOLD
                    or velocity_angle > REVIVAL_ANGLE_THRESHOLD):
                return 1e6
            cost *= 0.5   # Lower cost encourages revival
        return cost

    # -------------------------------------------------------------------------
    # Hungarian algorithm matching between candidate tracks and unmapped observations
    # -------------------------------------------------------------------------
    def _associate_unmapped(
        self,
        observations: List[TrackObservation],
        *,
        sync_ok: bool,
        homography_error_px: float,
    ) -> tuple[list[tuple[GlobalTrack, TrackObservation]], list[TrackObservation]]:
        # Only consider active, lost, or lost_reserved tracks
        candidates = [
            track for track in self.global_tracks.values()
            if not track.deleted and track.lifecycle_state in {"ACTIVE", "LOST", "LOST_RESERVED"}
        ]
        if not candidates or not observations:
            return [], observations

        # Homography is considered active if reprojection error is within limit
        homography_active = homography_error_px <= HOMOGRAPHY_MAX_ERROR
        if not sync_ok:
            homography_active = False
            self.cross_camera_frozen = True
        else:
            self.cross_camera_frozen = False

        # Build cost matrix
        cost_matrix = np.full((len(candidates), len(observations)), 1e6, dtype=np.float32)
        for row, track in enumerate(candidates):
            for col, observation in enumerate(observations):
                # If cross‑camera matching is frozen, only allow same‑camera matches
                if self.cross_camera_frozen and observation.camera_id not in track.source_local_ids:
                    continue
                cost_matrix[row, col] = self._candidate_cost(track, observation, homography_active)

        # Solve assignment problem
        rows, cols = linear_sum_assignment(cost_matrix)
        matches: list[tuple[GlobalTrack, TrackObservation]] = []
        used_cols = set()
        for row, col in zip(rows, cols):
            if cost_matrix[row, col] >= 1e5:   # Effectively infinite cost → ignore
                continue
            matches.append((candidates[row], observations[col]))
            used_cols.add(col)
        leftovers = [obs for idx, obs in enumerate(observations) if idx not in used_cols]
        return matches, leftovers

    # -------------------------------------------------------------------------
    # Try to match a new observation against pending observations from other cameras
    # -------------------------------------------------------------------------
    def _pending_match(self, observation: TrackObservation) -> Optional[TrackObservation]:
        best_idx = None
        best_score = -1.0
        now = observation.timestamp_ms
        for idx, pending in enumerate(self.pending):
            other = pending.observation
            if pending.expires_at_ms < now or other.camera_id == observation.camera_id:
                continue
            similarity = cosine_similarity(other.embedding, observation.embedding)
            if similarity > max(best_score, EMBEDDING_SIM_THRESHOLD):
                best_idx = idx
                best_score = similarity
        if best_idx is None:
            return None
        matched = self.pending.pop(best_idx).observation
        return matched

    # -------------------------------------------------------------------------
    # Move expired pending observations to a list so they can become new tracks
    # -------------------------------------------------------------------------
    def _promote_pending_expired(self, now_ms: float) -> List[TrackObservation]:
        promotable = [p.observation for p in self.pending if p.expires_at_ms <= now_ms]
        self.pending = [p for p in self.pending if p.expires_at_ms > now_ms]
        return promotable

    # -------------------------------------------------------------------------
    # Mark tracks as missing, lost, lost_reserved, or deleted based on time
    # -------------------------------------------------------------------------
    def _mark_missing(self, now_ms: float) -> None:
        max_lost_ms = MAX_LOST_SECONDS * 1000.0
        reserved_ms = RESERVED_SECONDS * 1000.0
        for track in self.global_tracks.values():
            if track.deleted:
                continue
            # If the track was updated this frame, skip
            if track.current_update is not None and track.current_update.timestamp_ms == now_ms:
                continue
            # Start missing timer
            if track.missing_since_ms is None:
                track.missing_since_ms = now_ms
            missing_duration = now_ms - track.missing_since_ms

            # Transition from ACTIVE to LOST after MAX_LOST_SECONDS
            if missing_duration >= max_lost_ms and track.lifecycle_state == "ACTIVE":
                track.lifecycle_state = "LOST"

            # After additional RESERVED_SECONDS, become LOST_RESERVED (still kept but eligible for revival)
            if missing_duration >= (max_lost_ms + reserved_ms):
                track.lifecycle_state = "LOST_RESERVED"
                if track.lost_reserved_since_ms is None:
                    track.lost_reserved_since_ms = now_ms

            # If a track stays in LOST_RESERVED for reserved_ms, delete it permanently
            if track.lost_reserved_since_ms is not None and (now_ms - track.lost_reserved_since_ms) >= reserved_ms:
                track.deleted = True

    # -------------------------------------------------------------------------
    # Main update method – called each frame with a list of observations
    # -------------------------------------------------------------------------
    def update(
        self,
        observations: List[TrackObservation],
        *,
        timestamp_ms: float,
        homography_error_px: float,
        sync_ok: bool,
        motion_lookup: Dict[tuple[int, int], dict],
    ) -> List[GlobalTrack]:
        # Clear current_update for all tracks (will be re‑filled for those matched)
        for track in self.global_tracks.values():
            track.current_update = None

        # Step 1: Direct source mapping (camera, local_id)
        matched_globals: List[tuple[GlobalTrack, TrackObservation]] = []
        unmapped: List[TrackObservation] = []
        for observation in observations:
            track = self._match_known_source(observation)
            if track is not None and not track.deleted:
                matched_globals.append((track, observation))
            else:
                unmapped.append(observation)

        # Step 2: Hungarian matching for the remaining observations
        registry_matches, leftovers = self._associate_unmapped(
            unmapped,
            sync_ok=sync_ok,
            homography_error_px=homography_error_px,
        )
        matched_globals.extend(registry_matches)

        # Apply all matched observations to their tracks
        for track, observation in matched_globals:
            # Update source mapping
            self.source_to_global[(observation.camera_id, observation.local_track_id)] = track.global_id
            # Retrieve motion analysis for this observation (default if missing)
            motion = motion_lookup.get(
                (observation.camera_id, observation.local_track_id),
                {
                    "outward_motion": False,
                    "inward_motion": False,
                    "displacement_vector": np.zeros(2, dtype=np.float32),
                    "displacement_magnitude": 0.0,
                    "motion_dot": 0.0,
                    "nearest_edge_index": 0,
                    "edge_normal": np.zeros(2, dtype=np.float32),
                },
            )
            self._apply_observation(track, observation, motion)

        # Step 3: Handle observations that were not matched to any existing track
        for observation in leftovers:
            # Try to match cross‑camera with a pending observation (if cross‑camera sync is OK)
            paired = None if self.cross_camera_frozen else self._pending_match(observation)
            if paired is not None:
                # Two observations from different cameras match → create a single global track
                track = self._new_global_track(paired)
                paired_motion = motion_lookup.get((paired.camera_id, paired.local_track_id), {})
                current_motion = motion_lookup.get((observation.camera_id, observation.local_track_id), {})
                self._apply_observation(track, paired, paired_motion or {
                    "outward_motion": False,
                    "inward_motion": False,
                    "displacement_vector": np.zeros(2, dtype=np.float32),
                    "displacement_magnitude": 0.0,
                    "motion_dot": 0.0,
                    "nearest_edge_index": 0,
                    "edge_normal": np.zeros(2, dtype=np.float32),
                })
                self._apply_observation(track, observation, current_motion or {
                    "outward_motion": False,
                    "inward_motion": False,
                    "displacement_vector": np.zeros(2, dtype=np.float32),
                    "displacement_magnitude": 0.0,
                    "motion_dot": 0.0,
                    "nearest_edge_index": 0,
                    "edge_normal": np.zeros(2, dtype=np.float32),
                })
                # Update source mapping for both cameras
                self.source_to_global[(paired.camera_id, paired.local_track_id)] = track.global_id
                self.source_to_global[(observation.camera_id, observation.local_track_id)] = track.global_id
            else:
                # No cross‑camera match yet – buffer this observation as pending
                expires_at_ms = observation.timestamp_ms + self._buffer_ms(observation)
                self.pending.append(PendingObservation(observation=observation, expires_at_ms=expires_at_ms))

        # Step 4: Promote expired pending observations to new global tracks
        for observation in self._promote_pending_expired(timestamp_ms):
            # Skip if this observation was already mapped (should not happen)
            if self.source_to_global.get((observation.camera_id, observation.local_track_id)) is not None:
                continue
            track = self._new_global_track(observation)
            motion = motion_lookup.get((observation.camera_id, observation.local_track_id), {
                "outward_motion": False,
                "inward_motion": False,
                "displacement_vector": np.zeros(2, dtype=np.float32),
                "displacement_magnitude": 0.0,
                "motion_dot": 0.0,
                "nearest_edge_index": 0,
                "edge_normal": np.zeros(2, dtype=np.float32),
            })
            self._apply_observation(track, observation, motion)
            self.source_to_global[(observation.camera_id, observation.local_track_id)] = track.global_id

        # Step 5: Update missing/lost/deleted status for all tracks
        self._mark_missing(timestamp_ms)

        # Return only non‑deleted tracks
        return [track for track in self.global_tracks.values() if not track.deleted]