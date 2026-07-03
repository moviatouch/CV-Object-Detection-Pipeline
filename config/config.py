from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# -----------------------------------------------------------------------------
# Project root – directory containing this config file
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_PATH = str(PROJECT_ROOT)

# -----------------------------------------------------------------------------
# File paths
# -----------------------------------------------------------------------------
MODEL_PATH = str(PROJECT_ROOT / "models" / "best.pt")          # YOLO/RT‑DETR model weights
ROI_CONFIG_DIR = str(PROJECT_ROOT / "config" / "roi")          # JSON ROI definitions
VICKI_APP   = "http://192.168.1.140:8085/tsv/flashapi"
RESIZE_VIDEO = (640, 480)

# REID_BACKEND = "dinov2"   # "mobilenet" or "dinov2"
# REID_DINOV2_MODEL = "facebook/dinov2-large"


# Map model detection names to dashboard (planogram) product names
MODEL_TO_DASHBOARD_MAPPING = {
    "Aquafina": "Aquafina",
    "Barebells Caramel Cashew": "Barebells - Caramel Cashew",
    "Barebells Cookies and Cream": "Barebells - Cookies & Cream",
    "Legendary Tasty Pastry - Strawberry": "Legendary Tasty Pastry - Strawberry",
    "One Bar - Hersheys Cookies N Cream": "One Bar - Hershey's Cookies N Cream",
    "One Bar - Reess PB Lovers": "One Bar - Reese's PB Lovers",
    "Quest Chips Chili Lime": "Quest Chips Chili Lime",
    "Quest Chips Hot and Spicy": "Quest Chips - Hot & Spicy",
    "Quest Chips Nacho Cheese": "Quest Chips - Nacho Cheese",
    "Skullcandy DIME 3": "Skullcandy DIME 3",
}


# -----------------------------------------------------------------------------
# Detection source
# -----------------------------------------------------------------------------
DETECT_ON_WARPED = False
ENABLE_ORIGINAL_VIEW_FUSION = False

POST_PUTBACK_COOLDOWN_MS: float = 1500.0


LOW_CONF_FRAMES = 30

# -----------------------------------------------------------------------------
# Stability & confirmation (event detection)
# -----------------------------------------------------------------------------
STABILITY_FRAMES = 2
LONG_BOUNDARY_STABILITY_FRAMES = 3
CONFIRM_FRAMES = 3
CONF_THRESHOLD = 0.4
CONF_MIN_FRAME = 0.4

# Quick return: product must stay outside stable ROI for at least this many frames
QUICK_RETURN_MIN_OUTSIDE_FRAMES = 0
QUICK_RETURN_THRESHOLD_MS = 0

STABLE_LOST_PICKUP_MS = 100.0

PUTBACK_STABLE_FRAMES = 5

# -----------------------------------------------------------------------------
# Motion analysis
# -----------------------------------------------------------------------------
MIN_PICKUP_DISPLACEMENT = 3
MIN_PUTBACK_DISPLACEMENT = 3
DEADBAND = 3
MOTION_BUFFER_SIZE = 10
MOTION_MIN_DEPTH_RATIO = 0.10
ROI_STABLE_EDGE_MARGIN_PX = 140.0

# -----------------------------------------------------------------------------
# Ghost track (lost object revival)
# -----------------------------------------------------------------------------
MAX_LOST_SECONDS = 300
RESERVED_SECONDS = 60.0
REVIVAL_EMBEDDING_THRESHOLD = 0.6
REVIVAL_SPATIAL_THRESHOLD = 100
REVIVAL_ANGLE_THRESHOLD = 45

# -----------------------------------------------------------------------------
# Cross‑camera matching (homography and buffering)
# -----------------------------------------------------------------------------
HOMOGRAPHY_AVAILABLE = True
HOMOGRAPHY_THRESHOLD = 40
HOMOGRAPHY_MAX_ERROR = 50
BASE_BUFFER_MS = 2000
VELOCITY_BUFFER_FACTOR = 0.5
DESYNC_TOLERANCE_FRAMES = 2

EMBEDDING_SIM_THRESHOLDS_BY_CLASS = {
    "default": 0.50,
    "Aquafina": 0.50,
    "Skullcandy DIME 3": 0.50,
    "Quest Chips Chili Lime": 0.20,
    "Quest Chips Hot and Spicy": 0.50,
    "Quest Chips Nacho Cheese": 0.50,
    "One Bar - Reess PB Lovers": 0.20,
    "One Bar - Hersheys Cookies N Cream": 0.50,
    "Barebells Caramel Cashew": 0.10,
    "Barebells Cookies and Cream": 0.10,
    "Legendary Tasty Pastry - Strawberry": 0.50,
}

# -----------------------------------------------------------------------------
# Identity protection
# -----------------------------------------------------------------------------
ENFORCE_CLASS_CONSISTENCY = True
TEMPORAL_IOU_THRESHOLD = 0.1
LOCAL_TRACK_MAX_DISTANCE_NORM = 2.5
LOCAL_TRACK_MIN_SIMILARITY = 0.45

# -----------------------------------------------------------------------------
# Rate limiting
# -----------------------------------------------------------------------------
MAX_PICKUPS_PER_CLASS_PER_SECOND = 6

# -----------------------------------------------------------------------------
# Outputs (saving, preview, debugging)
# -----------------------------------------------------------------------------
SAVE_ANNOTATED_VIDEO = True
SHOW_PREVIEW = False
PREVIEW_WINDOW_NAME = "Vending Pipeline Preview"
PREVIEW_SCALE = 0.6
DETECTION_MODEL_CONFIDENCE = 0.6
DETECTION_BATCH_SIZE = 20
DETECTION_DEBUG_LOG_INTERVAL = 1
TRACK_CONFIDENCE_HISTORY = 10
MISSING_PENDING_CONFIRM_MS = 1500


CLASS_RETURN_MAX_GAP_MS = 120000
PRINT_DEBUG_EVENTS = True
PRINT_DETECTION_SUMMARY = False
PRINT_TRACK_STATUS = True
TRACK_STATUS_INTERVAL = 1
OVERLAY_EVENT_TTL_MS = 3000
OVERLAY_MAX_EVENT_LINES = 4

# -----------------------------------------------------------------------------
# Processing defaults
# -----------------------------------------------------------------------------
WARP_SIZE = (1920, 1080)
SHAKE_FLOW_THRESHOLD = 2.0
LOCAL_TRACK_MAX_MISSES = 45
TRACK_EMBEDDING_BANK = 3
GLOBAL_EMBEDDING_BANK = 5
ANNOTATION_FPS_FALLBACK = 30.0

# -----------------------------------------------------------------------------
# Class identity locking / class drift protection
# -----------------------------------------------------------------------------
CLASS_VOTE_HISTORY = 30
CLASS_LOCK_RATIO = 0.75
CLASS_LOCK_MIN_TOTAL_SCORE = 0.80
CLASS_VOTE_MIN_CONFIDENCE = 0.20
CLASS_VOTE_WEIGHT_STABLE = 1.00
CLASS_VOTE_WEIGHT_SAFE = 0.70
CLASS_VOTE_WEIGHT_OUTER = 0.40
CLASS_VOTE_WEIGHT_OUTSIDE = 0.20
CLASS_VOTE_RECENCY_DECAY = 0.92
CLASS_CONSISTENCY_RATIO = 0.50
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
# Adaptive thresholds container (unused)
# -----------------------------------------------------------------------------
@dataclass
class AdaptiveThresholds:
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
    root: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs")
    logs: Path = field(init=False)
    sessions: Path = field(init=False)
    annotated_videos: Path = field(init=False)

    def __post_init__(self) -> None:
        self.logs = self.root / "logs"
        self.sessions = self.root / "sessions"
        self.annotated_videos = self.root / "annotated_videos"

    def ensure(self) -> "OutputPaths":
        self.logs.mkdir(parents=True, exist_ok=True)
        self.sessions.mkdir(parents=True, exist_ok=True)
        self.annotated_videos.mkdir(parents=True, exist_ok=True)
        return self

OUTPUT_PATHS = OutputPaths().ensure()
