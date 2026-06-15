from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

# Type aliases for readability
BBox = Tuple[float, float, float, float]   # (x1, y1, x2, y2)
Point = Tuple[float, float]                # (x, y)


@dataclass
class Detection:
    """
    Raw detection from an object detector (YOLO/RT-DETR).

    Contains bounding box, class, confidence, histogram embedding, and metadata.
    """
    bbox: BBox
    class_id: int
    class_name: str
    confidence: float
    embedding: np.ndarray                    # 96-dim histogram embedding
    camera_id: int
    frame_index: int
    timestamp_ms: float
    source_view: str = "warped"              # "warped" (after homography) or "original"
    original_centroid: Optional[np.ndarray] = None
    display_bbox: Optional[BBox] = None
    in_safe_roi_override: Optional[bool] = None
    in_outer_roi_override: Optional[bool] = None

    @property
    def centroid(self) -> np.ndarray:
        """Compute centroid from bounding box."""
        x1, y1, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


@dataclass
class TrackObservation:
    """
    Output from a single-camera tracker (SingleCameraTracker).

    Represents a tracked object in one camera after matching/filtering.
    """
    camera_id: int
    local_track_id: int
    frame_index: int
    timestamp_ms: float
    class_id: int
    class_name: str
    bbox: BBox
    centroid: np.ndarray
    confidence: float
    embedding: np.ndarray
    velocity: np.ndarray
    motion_centroid: np.ndarray
    in_safe_roi: bool
    in_outer_roi: bool
    in_stable_roi: bool
    safe_roi_distance: float
    display_bbox: Optional[BBox] = None
    temporal_iou: float = 1.0


@dataclass
class TrackUpdate:
    """
    A single frame's update to a global track, after motion analysis (EventManager).

    Contains all information needed by the event state machine.
    """
    global_id: int
    class_id: int
    class_name: str
    camera_id: int
    frame_index: int
    timestamp_ms: float
    bbox: BBox
    centroid: np.ndarray
    motion_centroid: np.ndarray
    confidence: float
    in_safe_roi: bool
    in_outer_roi: bool
    in_stable_roi: bool
    safe_roi_distance: float
    outward_motion: bool
    inward_motion: bool
    displacement_vector: np.ndarray
    displacement_magnitude: float
    motion_dot: float
    nearest_edge_index: int
    edge_normal: np.ndarray
    merged_suppressed: bool = False



@dataclass
class GlobalTrack:
    """
    Persistent long-term track across cameras and time.

    Maintains history, state machine variables (event_state, lifecycle_state),
    merge locking info, and all data needed by the EventManager.
    """
    global_id: int
    class_id: int
    class_name: str
    created_ms: float
    last_seen_ms: float
    event_state: str = "INSIDE"
    lifecycle_state: str = "ACTIVE"
    inside_counter: int = 0
    confirmation_confidences: Deque[float] = field(default_factory=lambda: deque(maxlen=10))
    # -------------------------------------------------------------------------
    # Class identity locking / drift protection
    # -------------------------------------------------------------------------
    # Temporal class vote history. Each item is a dict:
    # {
    #   "class": str,
    #   "confidence": float,
    #   "timestamp_ms": float,
    #   "roi_weight": float,
    #   "in_stable_roi": bool,
    #   "in_safe_roi": bool,
    #   "in_outer_roi": bool,
    # }
    class_votes: Deque[dict] = field(default_factory=lambda: deque(maxlen=30))

    # Class votes specifically collected during pickup/putback confirmation.
    confirmation_class_votes: Deque[dict] = field(default_factory=lambda: deque(maxlen=10))

    # Product identity locked after stable shelf observation.
    locked_class_name: Optional[str] = None

    # Final class used for billing/inventory event.
    resolved_class_name: Optional[str] = None

    # True when latest detected class differs from locked/resolved identity.
    class_drift_flagged: bool = False

    # Optional appearance embedding for later stronger ledger matching.
    visual_embedding: Optional[Any] = None
    centroid_history: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=10))
    motion_centroid_history: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=10))
    velocity_history: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=10))
    embedding_bank: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=5))
    bbox_by_camera: Dict[int, BBox] = field(default_factory=dict)
    source_local_ids: Dict[int, int] = field(default_factory=dict)
    last_frame_by_camera: Dict[int, int] = field(default_factory=dict)
    current_update: Optional[TrackUpdate] = None
    last_confidence: float = 0.0
    confidence_history: Deque[float] = field(default_factory=lambda: deque(maxlen=10))
    last_camera_id: Optional[int] = None
    last_frame_index: int = -1
    missing_since_ms: Optional[float] = None
    lost_reserved_since_ms: Optional[float] = None
    deleted: bool = False
    last_event_ms: float = -1.0
    pending_since_ms: Optional[float] = None
    last_seen_in_safe_roi: Optional[bool] = None
    last_seen_in_outer_roi: Optional[bool] = None
    last_seen_in_stable_roi: Optional[bool] = None
    last_debug_reason: str = ""
    boundary_linger_start_ms: Optional[float] = None
    boundary_linger_flagged: bool = False
    merge_counter: int = 0
    merge_locked: bool = False
    merge_owner_id: Optional[int] = None
    overlap_history: Dict[int, int] = field(default_factory=dict)
    centroid_history_by_camera: Dict[int, Deque[np.ndarray]] = field(default_factory=dict)
    motion_history_by_camera: Dict[int, Deque[np.ndarray]] = field(default_factory=dict)
    motion_direction_history: Deque[Optional[bool]] = field(
        default_factory=lambda: deque(maxlen=5)
    )
    # True = outward, False = inward, None = stationary/ambiguous

    # counts how many times the PICKUP_PENDING timeout has been
    # extended for this track due to product hesitation in the OUTER ROI.
    # Reset to 0 whenever the track leaves a PENDING state.
    hesitation_resets: int = 0

    # True once the track has reached STABLE_INSIDE at least once.
    # Used to distinguish occlusion-during-pickup (confirm) from a track that
    # never stabilised (cancel silently).  Never reset after being set.
    was_stable: bool = False

    # True while this track is in the long-hold putback fallback
    # path (elapsed time > CLASS_RETURN_MAX_GAP_MS).  Cleared on any non-pending
    # transition.
    long_hold_pending: bool = False

    # DEDUPLICATION: True once this track has confirmed ONE pickup event.
    # Prevents the same track from entering PICKUP_PENDING multiple times,
    # which was causing duplicate pickup events for a single physical action.
    # pickup_confirmed: bool = False

    #timestamp (ms) at which this track's pickup was confirmed.
    # Stamped by _record_pickup so PICKED_UP state can compute per-track elapsed
    # time instead of reading the shared class-level deque (which may belong to
    # a different track of the same product class).
    pickup_confirmed_ms: Optional[float] = None

    # Timestamp of the latest confirmed putback. Used to suppress immediate
    # same-track pickup echoes caused by small post-return detector shifts.
    last_putback_confirmed_ms: Optional[float] = None

    # True once this track has received a neighbour-stability boost.
    stability_boosted: bool = False

    consecutive_absent_frames = 0
    quick_return_min_outside_frames = 0


    def mean_embedding(self) -> Optional[np.ndarray]:
        """Average of all embeddings in the embedding bank."""
        if not self.embedding_bank:
            return None
        return np.mean(np.stack(list(self.embedding_bank)), axis=0)

    def last_velocity(self) -> np.ndarray:
        """Most recent velocity vector (zero if no history)."""
        if not self.velocity_history:
            return np.zeros(2, dtype=np.float32)
        return np.array(self.velocity_history[-1], dtype=np.float32)

    def latest_bbox(self, camera_id: Optional[int] = None) -> Optional[BBox]:
        """
        Return the most recent bounding box for a specific camera,
        or the most recent overall if camera_id is None.
        """
        if camera_id is not None and camera_id in self.bbox_by_camera:
            return self.bbox_by_camera[camera_id]
        if not self.bbox_by_camera:
            return None
        latest_cam = next(reversed(self.bbox_by_camera))
        return self.bbox_by_camera[latest_cam]


@dataclass
class FramePacket:
    """
    Bundles all data associated with a single frame from one camera.
    """
    camera_id: int
    frame_index: int
    timestamp_ms: float
    frame: np.ndarray
    warped_frame: np.ndarray
    warped_display_frame: np.ndarray
    warp_matrix: np.ndarray
    full_roi_polygon: np.ndarray
    safe_roi_polygon: np.ndarray


@dataclass
class SessionSummary:
    """
    Final summary of a processing session (e.g., one video or one run).

    Contains all events (pickups/putbacks), per-class counts, net changes, and warnings.
    """
    session_id: str
    events: List[dict] = field(default_factory=list)
    pickup_records: List[dict] = field(default_factory=list)
    putback_records: List[dict] = field(default_factory=list)
    pickup_count: Dict[str, int] = field(default_factory=dict)
    putback_count: Dict[str, int] = field(default_factory=dict)
    net_change: Dict[str, int] = field(default_factory=dict)
    total_pickups: int = 0
    total_putbacks: int = 0
    warnings: List[str] = field(default_factory=list)
    pending_pickups: List[dict] = field(default_factory=list)
