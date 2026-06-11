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


# REID_MODEL_TYPE = "osnet"
# OSNET_MODEL_PATH = str(PROJECT_ROOT / "models" / "osnet_x1_0_checkpoint.pth")
MOBILENET_MODEL_PATH = str(PROJECT_ROOT / "models" / "mobilenet_v3_small-047dcff4.pth")

# -----------------------------------------------------------------------------
# Detection source
# -----------------------------------------------------------------------------
# Detection is performed on original (unwarped) frames, then projected to warped coordinates.
# No warped-frame detection, no original-view fusion.
DETECT_ON_WARPED = False
ENABLE_ORIGINAL_VIEW_FUSION = False          # Only one detection pass (original frames)

POST_PUTBACK_COOLDOWN_MS: float = 1500.0


# -----------------------------------------------------------------------------
# Low‑light adaptation
# -----------------------------------------------------------------------------
LOW_CONF_AVG_THRESHOLD = 0.0             # Frame‑average confidence below this → low‑light mode
LOW_CONF_FRAMES = 30                     # Number of frames to average for low‑light detection
CRITICAL_CONF_THRESHOLD = 0.0            # Below this → pause events entirely


# -----------------------------------------------------------------------------
# Stability & confirmation (event detection)
# -----------------------------------------------------------------------------
STABILITY_FRAMES = 2                     # Frames needed inside safe ROI to become STABLE_INSIDE
LONG_BOUNDARY_STABILITY_FRAMES = 5       # Extra frames if object lingered near boundary
CONFIRM_FRAMES = 4                       # Length of confirmation window for pickups/putbacks
CONF_THRESHOLD = 0.4                     # Average confidence required over window
CONF_MIN_FRAME = 0.4                     # Minimum confidence in any single frame of window

# Threshold for quick pick-and-return detection (product leaves & re-enters SAFE ROI < this ms)
QUICK_RETURN_THRESHOLD_MS = 120.0        # If re-entry happens within 800ms, record both pickup + putback

STABLE_LOST_PICKUP_MS = 100.0            # Time threshold for stable lost pickup detection

# -----------------------------------------------------------------------------
# Motion analysis
# -----------------------------------------------------------------------------
MIN_PICKUP_DISPLACEMENT = 12             # Minimum displacement (pixels) to classify as pickup
MIN_PUTBACK_DISPLACEMENT = 12            # Minimum displacement to classify as putback
DEADBAND = 9                             # Ignore movements smaller than this (jitter)
MOTION_BUFFER_SIZE = 10                  # History size for motion smoothing
MOTION_MIN_DEPTH_RATIO = 0.45            # Reject mostly-sideways translations
ROI_STABLE_EDGE_MARGIN_PX = 140.0        # Stable band around SAFE ROI for top/bottom edge items

# -----------------------------------------------------------------------------
# Ghost track (lost object revival)
# -----------------------------------------------------------------------------
MAX_LOST_SECONDS = 5.0                   # After this time, ACTIVE → LOST
RESERVED_SECONDS = 10.0                  # After LOST, keep in LOST_RESERVED for this long
REVIVAL_EMBEDDING_THRESHOLD = 0.6        # Min embedding similarity to revive a LOST_RESERVED track
REVIVAL_SPATIAL_THRESHOLD = 100          # Max centroid distance for revival (pixels)
REVIVAL_ANGLE_THRESHOLD = 45             # Max velocity angle difference (degrees) for revival

# -----------------------------------------------------------------------------
# Cross‑camera matching (homography and buffering)
# -----------------------------------------------------------------------------
HOMOGRAPHY_AVAILABLE = True              # Whether homography between cameras is used
HOMOGRAPHY_THRESHOLD = 40                # Unused? (legacy)
HOMOGRAPHY_MAX_ERROR = 50               # Max reprojection error (pixels) to consider homography active
EMBEDDING_SIM_THRESHOLD = 0.45            # Min cosine similarity to consider a match
BASE_BUFFER_MS = 1000                     # Base time to keep pending observations (ms)
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
MAX_PICKUPS_PER_CLASS_PER_SECOND = 6     # Max pickups per product class per second

# -----------------------------------------------------------------------------
# Outputs (saving, preview, debugging)
# -----------------------------------------------------------------------------
SAVE_ANNOTATED_VIDEO = True              # Save output video with overlays
SAVE_SNAPSHOTS = True                    # Save snapshot images on events
SHOW_PREVIEW = False                     # Display live preview window
PREVIEW_WINDOW_NAME = "Vending Pipeline Preview"
PREVIEW_SCALE = 0.6                      # Scale factor for preview window
DETECTION_MODEL_CONFIDENCE = 0.7       # Confidence threshold for detector (YOLO/RT‑DETR)
DETECTION_BATCH_SIZE = 30                # Number of synchronized frame-pairs to batch for detector inference
DETECTION_DEBUG_LOG_INTERVAL = 10        # Log detection summary every N frames
TRACK_CONFIDENCE_HISTORY = 10            # Length of confidence history per track
MISSING_PENDING_CONFIRM_MS = 800         # Timeout for pending pickup/putback (ms)
CLASS_RETURN_MAX_GAP_MS = 120000          # Max time between pickup and class‑level putback (ms)
PRINT_DEBUG_EVENTS = True                # Print event debug messages to console
PRINT_DETECTION_SUMMARY = True           # Print detection counts periodically
PRINT_TRACK_STATUS = True                # Print track status summaries
TRACK_STATUS_INTERVAL = 1                # Seconds between track status logs
OVERLAY_EVENT_TTL_MS = 3000              # How long events stay in overlay (ms)
OVERLAY_MAX_EVENT_LINES = 4              # Max lines in event overlay

# -----------------------------------------------------------------------------
# Processing defaults (no image enhancement, no shake detection)
# -----------------------------------------------------------------------------
WARP_SIZE = (1920, 1080) 
SHAKE_FLOW_THRESHOLD = 2.0                      # Size of warped (top‑down) frame (used for projection & visualisation)
LOCAL_TRACK_MAX_MISSES = 45              # Max consecutive missed frames before deleting local track
TRACK_EMBEDDING_BANK = 3                 # Number of embeddings to keep per local track
GLOBAL_EMBEDDING_BANK = 5                # Number of embeddings to keep per global track
ANNOTATION_FPS_FALLBACK = 30.0           # Fallback FPS if video cannot provide it

# -----------------------------------------------------------------------------
# Class identity locking / class drift protection
# -----------------------------------------------------------------------------
CLASS_VOTE_HISTORY = 30

# Product class is locked only if top weighted class has at least this ratio.
CLASS_LOCK_RATIO = 0.65

# Minimum total score needed before locking class.
CLASS_LOCK_MIN_TOTAL_SCORE = 0.80

# Minimum detector confidence for a frame to participate in class voting.
CLASS_VOTE_MIN_CONFIDENCE = 0.20

# Class voting weights by ROI state.
CLASS_VOTE_WEIGHT_STABLE = 1.00
CLASS_VOTE_WEIGHT_SAFE = 0.70
CLASS_VOTE_WEIGHT_OUTER = 0.40
CLASS_VOTE_WEIGHT_OUTSIDE = 0.20

# Recency decay for votes. 1.0 means no decay.
CLASS_VOTE_RECENCY_DECAY = 0.92

# Confirmation class consistency ratio.
CLASS_CONSISTENCY_RATIO = 0.60

UNKNOWN_UNSTABLE_CLASS = "UNKNOWN_UNSTABLE_CLASS"

ENABLE_CLASS_DRIFT_WARNINGS = True


# -----------------------------------------------------------------------------
# Pending pickup ledger / cross-class putback reconciliation
# -----------------------------------------------------------------------------
PENDING_LEDGER_MAX_AGE_MS = 120000.0

PUTBACK_LEDGER_MATCH_THRESHOLD = 0.40

LEDGER_MATCH_WEIGHT_CLASS = 0.20
LEDGER_MATCH_WEIGHT_TIME = 0.15
LEDGER_MATCH_WEIGHT_TRACK = 0.15
LEDGER_MATCH_WEIGHT_CLASS_CANDIDATE = 0.15
LEDGER_MATCH_WEIGHT_MOTION = 0.10
LEDGER_MATCH_WEIGHT_EMBEDDING = 0.20
LEDGER_MATCH_WEIGHT_SPATIAL = 0.05

LEDGER_SAME_CLASS_SCORE = 1.0
LEDGER_DIFFERENT_CLASS_SCORE = 0.0
LEDGER_SAME_GROUP_SCORE = 0.65

LEDGER_SPATIAL_MAX_DISTANCE_PX = 350.0
LEDGER_EMBEDDING_MIN_SIMILARITY = 0.35

# -----------------------------------------------------------------------------
# Adaptive thresholds container (unused because low‑light mode removed)
# -----------------------------------------------------------------------------
@dataclass
class AdaptiveThresholds:
    """Holds dynamically adjustable thresholds (for compatibility, unused)."""
    conf_threshold: float = CONF_THRESHOLD
    conf_min_frame: float = CONF_MIN_FRAME
    confirm_frames: int = CONFIRM_FRAMES
    stability_frames: int = STABILITY_FRAMES
    paused: bool = False
    low_light_mode: bool = False

# -----------------------------------------------------------------------------
# Output paths management
# -----------------------------------------------------------------------------
@dataclass
class OutputPaths:
    """Creates and holds paths to output subdirectories."""
    root: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs")
    logs: Path = field(init=False)
    sessions: Path = field(init=False)
    annotated_videos: Path = field(init=False)

    def __post_init__(self) -> None:
        """Initialise subdirectory paths after root is set."""
        self.logs = self.root / "logs"
        self.sessions = self.root / "sessions"
        self.annotated_videos = self.root / "annotated_videos"

    def ensure(self) -> "OutputPaths":
        """Create all directories if they do not exist, then return self."""
        self.logs.mkdir(parents=True, exist_ok=True)
        self.sessions.mkdir(parents=True, exist_ok=True)
        self.annotated_videos.mkdir(parents=True, exist_ok=True)
        return self

# -----------------------------------------------------------------------------
# Global instance of OutputPaths (ensured at import)
# -----------------------------------------------------------------------------
OUTPUT_PATHS = OutputPaths().ensure()