from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# -----------------------------------------------------------------------------
# Project root – directory containing this config file
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# -----------------------------------------------------------------------------
# File paths
# -----------------------------------------------------------------------------
MODEL_PATH = str(PROJECT_ROOT / "models" / "best.pt")          # YOLO/RT‑DETR model weights
ROI_CONFIG_DIR = str(PROJECT_ROOT / "config" / "roi")          # JSON ROI definitions

# -----------------------------------------------------------------------------
# Stability & confirmation (event detection)
# -----------------------------------------------------------------------------
STABILITY_FRAMES = 2                     # Frames needed inside safe ROI to become STABLE_INSIDE
LONG_BOUNDARY_STABILITY_FRAMES = 5       # Extra frames if object lingered near boundary
CONFIRM_FRAMES = 2                       # Length of confirmation window for pickups/putbacks
CONF_THRESHOLD = 0.4                     # Average confidence required over window
CONF_MIN_FRAME = 0.4                     # Minimum confidence in any single frame of window

# -----------------------------------------------------------------------------
# ROI margin (shrinking outer polygon to create safe inner polygon)
# -----------------------------------------------------------------------------
ROI_MARGIN_PX = 10

# -----------------------------------------------------------------------------
# Motion analysis
# -----------------------------------------------------------------------------
MIN_PICKUP_DISPLACEMENT = 10             # Minimum displacement (pixels) to classify as pickup
MIN_PUTBACK_DISPLACEMENT = 10            # Minimum displacement to classify as putback
DEADBAND = 5                             # Ignore movements smaller than this (jitter)
MOTION_BUFFER_SIZE = 10                  # History size for motion smoothing

# -----------------------------------------------------------------------------
# Ghost track (lost object revival)
# -----------------------------------------------------------------------------
MAX_LOST_SECONDS = 5.0                   # After this time, ACTIVE → LOST
RESERVED_SECONDS = 10.0                   # After LOST, keep in LOST_RESERVED for this long
REVIVAL_EMBEDDING_THRESHOLD = 0.7       # Min embedding similarity to revive a LOST_RESERVED track
REVIVAL_SPATIAL_THRESHOLD = 100          # Max centroid distance for revival (pixels)
REVIVAL_ANGLE_THRESHOLD = 45            # Max velocity angle difference (degrees) for revival

# -----------------------------------------------------------------------------
# Cross‑camera matching (homography and buffering)
# -----------------------------------------------------------------------------
HOMOGRAPHY_AVAILABLE = True              # Whether homography between cameras is used
HOMOGRAPHY_THRESHOLD = 40                # Unused? (legacy)
HOMOGRAPHY_MAX_ERROR = 50                # Max reprojection error (pixels) to consider homography active
EMBEDDING_SIM_THRESHOLD = 0.6            # Min cosine similarity to consider a match
BASE_BUFFER_MS = 200                     # Base time to keep pending observations (ms)
VELOCITY_BUFFER_FACTOR = 0.5             # Additional buffer = speed * factor (ms per pixel/frame)
DESYNC_TOLERANCE_FRAMES = 2              # Allowed frame count difference between cameras

# -----------------------------------------------------------------------------
# Identity protection (avoiding wrong associations)
# -----------------------------------------------------------------------------
ENFORCE_CLASS_CONSISTENCY = True         # Only match observations of the same class
TEMPORAL_IOU_THRESHOLD = 0.1             # Minimum IoU between consecutive bboxes to consider same object
LOCAL_TRACK_MAX_DISTANCE_NORM = 2.5      # Max normalised centroid distance (divided by 200)
LOCAL_TRACK_MIN_SIMILARITY = 0.45        # Min embedding similarity for a match

# -----------------------------------------------------------------------------
# Merging overlapping tracks (identical products)
# -----------------------------------------------------------------------------
MERGE_IOU_THRESHOLD = 0.8                # IoU above which tracks are considered overlapping
MERGE_FRAMES = 5                         # Consecutive frames of high IoU to lock merge

# -----------------------------------------------------------------------------
# Rate limiting (prevent false pickup storms)
# -----------------------------------------------------------------------------
MAX_PICKUPS_PER_CLASS_PER_SECOND = 2     # Max pickups per product class per second

# -----------------------------------------------------------------------------
# Low‑light adaptation
# -----------------------------------------------------------------------------
LOW_CONF_AVG_THRESHOLD = 0.5             # Frame‑average confidence below this → low‑light mode
LOW_CONF_FRAMES = 30                     # Number of frames to average for low‑light detection
CRITICAL_CONF_THRESHOLD = 0.3            # Below this → pause events entirely

# -----------------------------------------------------------------------------
# Outputs (saving, preview, debugging)
# -----------------------------------------------------------------------------
SAVE_ANNOTATED_VIDEO = True              # Save output video with overlays
SAVE_SNAPSHOTS = True                    # Save snapshot images on events
SHOW_PREVIEW = True                      # Display live preview window
PREVIEW_WINDOW_NAME = "Vending Pipeline Preview"
PREVIEW_SCALE = 0.6                      # Scale factor for preview window
DETECTION_MODEL_CONFIDENCE = 0.80        # Confidence threshold for detector (YOLO/RT‑DETR)
ENABLE_ORIGINAL_VIEW_FUSION = True       # Combine detections from original and warped views
ORIGINAL_VIEW_ROI_PAD_PX = 400           # Padding around ROI when fusing original view detections
DETECTION_FUSION_IOU = 0.5               # IoU threshold for fusing duplicate detections
DETECTION_DEBUG_LOG_INTERVAL = 10        # Log detection summary every N frames
TRACK_CONFIDENCE_HISTORY = 10            # Length of confidence history per track
MISSING_PENDING_CONFIRM_MS = 450         # Timeout for pending pickup/putback (ms) 0.45sec
CLASS_RETURN_MAX_GAP_MS = 12000          # Max time between pickup and class‑level putback (ms) 12sec
PRINT_DEBUG_EVENTS = True                # Print event debug messages to console
PRINT_DETECTION_SUMMARY = True           # Print detection counts periodically
PRINT_TRACK_STATUS = True                # Print track status summaries
TRACK_STATUS_INTERVAL = 1                # Seconds between track status logs
OVERLAY_EVENT_TTL_MS = 3000              # How long events stay in overlay (ms)
OVERLAY_MAX_EVENT_LINES = 4              # Max lines in event overlay

# -----------------------------------------------------------------------------
# Processing defaults
# -----------------------------------------------------------------------------
WARP_SIZE = (1920, 1080)                  # Size of warped (top‑down) frame
CLAHE_CLIP_LIMIT = 2.0                   # Contrast limiting for CLAHE
CLAHE_TILE_GRID = (8, 8)                 # Tile grid size for CLAHE
SHAKE_FLOW_THRESHOLD = 2.0               # Median flow magnitude to consider frame shaky
LOCAL_TRACK_MAX_MISSES = 45             # Max consecutive missed frames before deleting local track
TRACK_EMBEDDING_BANK = 3                 # Number of embeddings to keep per local track
GLOBAL_EMBEDDING_BANK = 5                # Number of embeddings to keep per global track
ANNOTATION_FPS_FALLBACK = 30.0           # Fallback FPS if video cannot provide it

# -----------------------------------------------------------------------------
# Adaptive thresholds container (used by EventManager)
# -----------------------------------------------------------------------------
@dataclass
class AdaptiveThresholds:
    """Holds dynamically adjustable thresholds (e.g., during low‑light mode)."""
    conf_threshold: float = CONF_THRESHOLD
    conf_min_frame: float = CONF_MIN_FRAME
    confirm_frames: int = CONFIRM_FRAMES
    stability_frames: int = STABILITY_FRAMES
    paused: bool = False                  # Events completely paused?
    low_light_mode: bool = False          # Currently in low‑light adaptation?

# -----------------------------------------------------------------------------
# Output paths management
# -----------------------------------------------------------------------------
@dataclass
class OutputPaths:
    """Creates and holds paths to output subdirectories."""
    root: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs")
    logs: Path = field(init=False)
    snapshots: Path = field(init=False)
    sessions: Path = field(init=False)
    annotated_videos: Path = field(init=False)

    def __post_init__(self) -> None:
        """Initialise subdirectory paths after root is set."""
        self.logs = self.root / "logs"
        self.snapshots = self.root / "snapshots"
        self.sessions = self.root / "sessions"
        self.annotated_videos = self.root / "annotated_videos"

    def ensure(self) -> "OutputPaths":
        """Create all directories if they do not exist, then return self."""
        self.logs.mkdir(parents=True, exist_ok=True)
        self.snapshots.mkdir(parents=True, exist_ok=True)
        self.sessions.mkdir(parents=True, exist_ok=True)
        self.annotated_videos.mkdir(parents=True, exist_ok=True)
        return self

# -----------------------------------------------------------------------------
# Global instance of OutputPaths (ensured at import)
# -----------------------------------------------------------------------------
OUTPUT_PATHS = OutputPaths().ensure()