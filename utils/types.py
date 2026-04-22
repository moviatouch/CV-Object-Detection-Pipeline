from __future__ import annotations  # Allow forward references in type hints

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


# Type aliases for readability
BBox = Tuple[float, float, float, float]   # (x1, y1, x2, y2)
Point = Tuple[float, float]                # (x, y)


@dataclass
class Detection:
    """
    Raw detection from an object detector (YOLO/RT‑DETR).

    Contains bounding box, class, confidence, histogram embedding, and metadata.
    """
    bbox: BBox
    class_id: int
    class_name: str
    confidence: float
    embedding: np.ndarray                    # 96‑dim histogram embedding
    camera_id: int
    frame_index: int
    timestamp_ms: float
    source_view: str = "warped"              # "warped" (after homography) or "original"
    original_centroid: Optional[np.ndarray] = None   # Centroid in original (unwarped) view
    in_safe_roi_override: Optional[bool] = None      # Manual override for ROI membership
    in_outer_roi_override: Optional[bool] = None

    @property
    def centroid(self) -> np.ndarray:
        """Compute centroid from bounding box."""
        x1, y1, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


@dataclass
class TrackObservation:
    """
    Output from a single‑camera tracker (SingleCameraTracker).

    Represents a tracked object in one camera after matching/filtering.
    """
    camera_id: int
    local_track_id: int                     # Unique ID within this camera
    frame_index: int
    timestamp_ms: float
    class_id: int
    class_name: str
    bbox: BBox
    centroid: np.ndarray                    # Current centroid
    confidence: float
    embedding: np.ndarray                   # Latest embedding
    velocity: np.ndarray                    # Displacement from previous centroid
    motion_centroid: np.ndarray             # Centroid after motion compensation (more stable)
    in_safe_roi: bool                       # Inside the safe inner ROI?
    in_outer_roi: bool                      # Inside the full outer ROI?
    shaky: bool = False                     # Was the frame globally shaky?
    temporal_iou: float = 1.0               # IoU between consecutive bboxes


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
    outward_motion: bool                    # Moving away from ROI (pickup)
    inward_motion: bool                     # Moving toward ROI (putback)
    displacement_vector: np.ndarray         # (dx, dy) over the temporal gap
    displacement_magnitude: float           # Euclidean norm
    motion_dot: float                       # Dot product used for direction decision
    nearest_edge_index: int                 # Which edge of the ROI is closest
    edge_normal: np.ndarray                 # Outward normal of that edge
    shaky: bool = False
    merged_suppressed: bool = False         # Set if this update was suppressed due to merge lock


@dataclass
class GlobalTrack:
    """
    Persistent long‑term track across cameras and time.

    Maintains history, state machine variables (event_state, lifecycle_state),
    merge locking info, and all data needed by the EventManager.
    """
    global_id: int
    class_id: int
    class_name: str
    created_ms: float                       # First appearance timestamp
    last_seen_ms: float                     # Most recent observation timestamp
    event_state: str = "INSIDE"             # INSIDE / STABLE_INSIDE / PICKUP_PENDING / PICKED_UP / PUTBACK_PENDING
    lifecycle_state: str = "ACTIVE"         # ACTIVE / LOST / LOST_RESERVED
    inside_counter: int = 0                 # Consecutive stable‑inside frames
    confirmation_confidences: Deque[float] = field(default_factory=lambda: deque(maxlen=10))
    centroid_history: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=10))
    motion_centroid_history: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=10))
    velocity_history: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=10))
    embedding_bank: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=5))
    bbox_by_camera: Dict[int, BBox] = field(default_factory=dict)          # Latest bbox per camera
    source_local_ids: Dict[int, int] = field(default_factory=dict)         # (camera_id → local_track_id)
    last_frame_by_camera: Dict[int, int] = field(default_factory=dict)
    current_update: Optional[TrackUpdate] = None                           # Most recent update
    last_confidence: float = 0.0
    confidence_history: Deque[float] = field(default_factory=lambda: deque(maxlen=10))
    last_camera_id: Optional[int] = None
    last_frame_index: int = -1
    missing_since_ms: Optional[float] = None                               # When object was last seen
    lost_reserved_since_ms: Optional[float] = None                         # When entered LOST_RESERVED
    deleted: bool = False
    last_event_ms: float = -1.0
    pending_since_ms: Optional[float] = None                               # When PENDING state started
    last_seen_in_safe_roi: Optional[bool] = None
    last_seen_in_outer_roi: Optional[bool] = None
    last_debug_reason: str = ""
    boundary_linger_start_ms: Optional[float] = None
    boundary_linger_flagged: bool = False
    merge_counter: int = 0
    merge_locked: bool = False                                              # Part of a merge group?
    merge_owner_id: Optional[int] = None                                    # Which track owns the group
    overlap_history: Dict[int, int] = field(default_factory=dict)           # Consecutive overlap frames per other track
    motion_history_by_camera: Dict[int, Deque[np.ndarray]] = field(default_factory=dict)
    motion_direction_history: Deque[Optional[bool]] = field(default_factory=lambda: deque(maxlen=5))
    # True = outward, False = inward, None = stationary/ambiguous

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

    Used to pass information between pipeline stages (preprocessing, detection,
    tracking, event management).
    """
    camera_id: int
    frame_index: int
    timestamp_ms: float
    frame: np.ndarray                       # Original BGR frame
    gray: np.ndarray                        # Grayscale version
    enhanced_frame: np.ndarray              # Enhanced (CLAHE etc.) for detection
    warped_frame: np.ndarray                # Perspective‑warped version (top‑down view)
    warped_display_frame: np.ndarray        # Warped frame for visualisation
    warped_gray: np.ndarray                 # Grayscale of warped frame
    warp_matrix: np.ndarray                 # Homography matrix from original to warped
    shaky: bool                             # Global camera shake flag
    full_roi_polygon: np.ndarray            # Outer ROI polygon in warped coordinates
    safe_roi_polygon: np.ndarray            # Inner (shrunk) ROI polygon


@dataclass
class SessionSummary:
    """
    Final summary of a processing session (e.g., one video or one run).

    Contains all events (pickups/putbacks), per‑class counts, net changes, and warnings.
    """
    session_id: str
    events: List[dict] = field(default_factory=list)           # All pickup+putback events
    pickup_records: List[dict] = field(default_factory=list)   # Only pickup events
    putback_records: List[dict] = field(default_factory=list)  # Only putback events
    pickup_count: Dict[str, int] = field(default_factory=dict) # Class → number of pickups
    putback_count: Dict[str, int] = field(default_factory=dict)# Class → number of putbacks
    net_change: Dict[str, int] = field(default_factory=dict)   # pickup_count - putback_count
    warnings: List[str] = field(default_factory=list)          # Non‑fatal issues (e.g., ignored putback)