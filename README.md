# Viatouch CV Object Identification Pipeline

A comprehensive computer vision pipeline for detecting and tracking object movements in multi-camera vending machine scenarios. The system identifies pickup and putback events by analyzing object trajectories relative to a Region of Interest (ROI).

## Project Overview

This pipeline processes synchronized video streams from multiple cameras to:
- Detect objects using RT-DETR (a real-time object detection model)
- Track objects across frames and cameras
- Identify pickup and putback events based on motion analysis
- Generate annotated videos and session summaries

The system is designed for top-down camera views of vending machines and uses homography transformations to establish correspondences between multiple camera perspectives.

## Directory Structure

```
viatouch-cv-object-identification-pipeline/
├── config.py                          # Central configuration file with all tunable parameters
├── main.py                            # Main pipeline entry point
├── draw_roi.py                        # Interactive ROI definition tool
│
├── config/
│   └── roi/
│       ├── cam0.json                  # ROI configuration for camera 0
│       └── cam1.json                  # ROI configuration for camera 1
│
├── detection/
│   ├── __init__.py
│   └── rtdetr_wrapper.py              # RT-DETR detector wrapper and embedding extraction
│
├── tracking/
│   ├── __init__.py
│   └── single_camera_tracker.py       # Single-camera object tracker with temporal smoothing
│
├── motion/
│   ├── __init__.py
│   ├── motion_analyzer.py             # Motion analysis relative to ROI geometry
│   └── optical_flow.py                # Lucas-Kanade optical flow for motion compensation
│
├── event/
│   ├── __init__.py
│   └── event_manager.py               # Event state machine (idle -> entering -> stable -> exiting)
│
├── multicam/
│   ├── __init__.py
│   ├── homography.py                  # Homography computation and point projection
│   └── global_registry.py             # Cross-camera track association and global ID management
│
├── utils/
│   ├── __init__.py
│   ├── types.py                       # Type definitions (Detection, TrackObservation, etc.)
│   ├── logger.py                      # Logging configuration
│   ├── roi_utils.py                   # ROI polygon utilities and geometric operations
│   ├── visualization.py               # Drawing and annotation functions
│   └── video_processor.py             # Video I/O and frame synchronization
│
├── models/
│   └── best.pt                        # Fine-tuned RT-DETR model weights
│
├── outputs/                           # Generated during execution (auto-created)
│   ├── logs/                          # Execution logs with detailed debug info
│   ├── sessions/                      # JSON summaries of detected events per session
│   ├── snapshots/                     # Event-triggered image snapshots
│   └── annotated_videos/              # Output videos with visualized detections and tracks
│
└── Videos/                            # Input video storage directory
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.8+ (for GPU acceleration with PyTorch)
- FFmpeg (for video processing)

### Setup Steps

1. Clone the repository:
```bash
git clone <repository-url>
cd viatouch-cv-object-identification-pipeline
```

2. Create a Python virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install opencv-python
pip install numpy scipy scikit-learn
pip install ultralytics
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

4. Download the pre-trained model:
```bash
# Place the fine-tuned RT-DETR model at:
# models/best.pt
```

## Configuration Guide

All configurable parameters are centralized in `config.py`. Key configurations:

### Model and Detection

```python
MODEL_PATH = "models/best.pt"                 # Path to RT-DETR weights
DETECTION_MODEL_CONFIDENCE = 0.80             # Detection confidence threshold
ENABLE_ORIGINAL_VIEW_FUSION = True            # Fuse detections from original and warped views
ORIGINAL_VIEW_ROI_PAD_PX = 400                # Padding around ROI when fusing
DETECTION_FUSION_IOU = 0.5                    # IoU threshold for duplicate removal
```

### ROI and Motion

```python
ROI_MARGIN_PX = 10                            # Shrink outer polygon to create safe inner polygon
MIN_PICKUP_DISPLACEMENT = 12                  # Minimum pixels to classify as pickup
MIN_PUTBACK_DISPLACEMENT = 12                 # Minimum pixels to classify as putback
DEADBAND = 5                                  # Ignore movements smaller than this
```

### Event Detection

```python
STABILITY_FRAMES = 2                          # Frames inside safe ROI to become STABLE
LONG_BOUNDARY_STABILITY_FRAMES = 5            # Extra frames if object lingered near boundary
CONFIRM_FRAMES = 2                            # Confirmation window length for pickups/putbacks
CONF_THRESHOLD = 0.4                          # Average confidence required over window
CONF_MIN_FRAME = 0.4                          # Minimum confidence in any single frame
```

### Cross-Camera Matching

```python
HOMOGRAPHY_AVAILABLE = True                   # Use homography between cameras
HOMOGRAPHY_MAX_ERROR = 50                     # Max reprojection error (pixels)
EMBEDDING_SIM_THRESHOLD = 0.6                 # Min cosine similarity to match
BASE_BUFFER_MS = 200                          # Time to keep pending observations (ms)
DESYNC_TOLERANCE_FRAMES = 2                   # Allowed frame count difference
```

### Output Options

```python
SAVE_ANNOTATED_VIDEO = True                   # Save output videos with overlays
SAVE_SNAPSHOTS = True                         # Save snapshot images on events
SHOW_PREVIEW = True                           # Display live preview window
PREVIEW_SCALE = 0.6                           # Scale factor for preview
```

## Usage

### Step 1: Define ROI (Region of Interest)

Before running the pipeline, define the ROI boundaries for each camera:

```bash
python draw_roi.py \
  --video0 Videos/camera0.mp4 \
  --video1 Videos/camera1.mp4 \
  --output_dir config/roi
```

Interactive Instructions:
- Click 4 corners of the ROI in clockwise order on each camera view
- Press 'r' to reset points if needed
- Press Enter when all 4 points are placed
- Press 'q' to cancel

The tool will generate:
- `config/roi/cam0.json` - ROI configuration for camera 0
- `config/roi/cam1.json` - ROI configuration for camera 1

These files include:
- Polygon vertices for the outer ROI boundary
- Safe inner polygon (after applying ROI_MARGIN_PX)
- Edge normals for motion direction analysis
- Homography matrix for camera alignment
- Outward vector for distinguishing pickup vs putback

### Step 2: Define Good ROI Reference (Optional)

To establish a reference state for "good ROI" (desired object state):

```bash
python draw_roi.py \
  --video0 Videos/camera0_reference.mp4 \
  --video1 Videos/camera1_reference.mp4 \
  --output_dir config/roi_good
```

This creates a reference configuration showing the ideal state of objects in the ROI. Use this for:
- Debugging motion detection
- Validating that pickup/putback classifications are correct
- Training data generation

### Step 3: Run the Detection Pipeline

Execute the main pipeline on video files:

```bash
python main.py \
  --video0 Videos/camera0.mp4 \
  --video1 Videos/camera1.mp4 \
  --session_id test_session_001 \
  --device cuda \
  --roi_dir config/roi \
  --model_path models/best.pt \
  --det_conf 0.80 \
  --show_preview
```

Command-line Arguments:
- `--video0, --video1` (required): Paths to input video files
- `--session_id` (required): Unique identifier for this processing session
- `--device` (default: cpu): Torch device (cpu, cuda, cuda:0, etc.)
- `--roi_dir` (default: config/roi): Directory containing cam0.json and cam1.json
- `--model_path` (default: models/best.pt): Path to RT-DETR model weights
- `--det_conf` (default: 0.80): Detection confidence threshold (0.0-1.0)
- `--show_preview` (default: True): Display live preview window during processing

### Step 4: Examine Results

After processing, outputs are saved to `outputs/`:

```
outputs/
├── sessions/session_test_session_001_YYYYMMDD_HHMMSS.json    # Event summary
├── annotated_videos/
│   ├── test_session_001_cam0.mp4                              # Annotated camera 0 video
│   └── test_session_001_cam1.mp4                              # Annotated camera 1 video
├── snapshots/
│   ├── pickup_class_name_global5_frame340_cam0.jpg           # Event snapshots
│   └── putback_class_name_global5_frame385_cam0.jpg
└── logs/
    └── test_session_001_YYYYMMDD_HHMMSS.log                  # Execution log
```

#### Session JSON Structure

Example output from `session_*.json`:

```json
{
  "session_id": "test_session_001",
  "start_timestamp_ms": 1234567890,
  "end_timestamp_ms": 1234577890,
  "events": [
    {
      "type": "pickup",
      "class": "product_x",
      "class_id": 1,
      "global_id": 5,
      "camera_id": 0,
      "frame_index": 340,
      "timestamp_ms": 1234568000,
      "confidence": 0.92,
      "displacement_magnitude": 45.5,
      "snapshot_path": "outputs/snapshots/pickup_product_x_global5_frame340_cam0.jpg"
    }
  ],
  "net_change": {
    "product_x": -2,
    "product_y": 1
  },
  "total_pickups": 15,
  "total_putbacks": 13
}
```

## Tweaking Configurations for Different Scenarios

### Low-Light Conditions

If experiencing detection issues in low-light:

```python
# config.py
DETECTION_MODEL_CONFIDENCE = 0.65          # Lower threshold (more detections)
LOW_CONF_AVG_THRESHOLD = 0.45              # Trigger low-light mode earlier
LOW_CONF_FRAMES = 50                       # Increase smoothing window
CRITICAL_CONF_THRESHOLD = 0.25             # Pause when very dark
```

### Fast-Moving Objects

If objects move quickly and pickups are missed:

```python
MIN_PICKUP_DISPLACEMENT = 8                # Lower displacement threshold
CONFIRM_FRAMES = 1                         # Faster confirmation
STABILITY_FRAMES = 1                       # Faster state transitions
BASE_BUFFER_MS = 300                       # More time for cross-camera matching
```

### Cluttered Scenes

If there are many overlapping objects:

```python
LOCAL_TRACK_MAX_DISTANCE_NORM = 2.0        # Stricter distance matching
LOCAL_TRACK_MIN_SIMILARITY = 0.55          # Higher embedding similarity required
MERGE_IOU_THRESHOLD = 0.7                  # Lower merge threshold
```

### Poor Camera Synchronization

If cameras are not perfectly synchronized:

```python
DESYNC_TOLERANCE_FRAMES = 3                # Allow more frame difference
BASE_BUFFER_MS = 250                       # Increase time buffer
VELOCITY_BUFFER_FACTOR = 0.7               # More adaptive buffering
```

## Advanced Features

### Motion Compensation

The pipeline uses Lucas-Kanade optical flow to improve tracking stability:

```python
# motion/optical_flow.py
# Automatically compensates for global camera motion
# Produces motion_centroid for more stable tracking
```

### Homography-Based Cross-Camera Association

Objects are matched across cameras using:
- Homography transformation between camera planes
- Embedding similarity (histogram-based)
- Temporal buffering with adaptive delays
- Spatial distance constraints

### Adaptive Low-Light Mode

Automatically adjusts thresholds when average detection confidence drops:
- Lowers confidence thresholds
- Increases confirmation windows
- Optionally pauses event detection if conditions are critical

### Event State Machine

Each track follows a deterministic state progression:
- IDLE: Initial state, waiting to enter ROI
- ENTERING: Detected inside ROI after crossing boundary
- STABLE_INSIDE: Confirmed inside safe ROI
- EXITING: Detected moving out of ROI
- STATE_EXIT: Movement analysis for pickup/putback determination

## Performance Optimization

### GPU Acceleration

To use CUDA for faster inference:

```bash
python main.py \
  --video0 Videos/camera0.mp4 \
  --video1 Videos/camera1.mp4 \
  --session_id test_session_001 \
  --device cuda:0
```

### Batch Processing

Create a script to process multiple video pairs:

```bash
#!/bin/bash
for video_pair in data/*/; do
  base=$(basename "$video_pair")
  python main.py \
    --video0 "$video_pair/cam0.mp4" \
    --video1 "$video_pair/cam1.mp4" \
    --session_id "$base" \
    --device cuda
done
```

### Memory Management

For processing long videos:

```python
# config.py
TRACK_EMBEDDING_BANK = 3                   # Keep fewer embeddings
GLOBAL_EMBEDDING_BANK = 5
LOW_CONF_FRAMES = 20                       # Smaller history window
```

## Troubleshooting

### ROI Drawing Issues
- Ensure videos are accessible and not corrupted
- Check that video resolution is reasonable
- Try pressing 'r' to reset if clicking doesn't work

### Missing Detections
- Lower DETECTION_MODEL_CONFIDENCE threshold
- Check if objects are in the ROI bounds
- Review annotated video to see detection points

### False Pickup/Putback Events
- Increase MIN_PICKUP_DISPLACEMENT or MIN_PUTBACK_DISPLACEMENT
- Raise CONF_THRESHOLD for more confident events
- Review motion analysis in annotated videos

### Cross-Camera Matching Failures
- Check homography error (should be < 50 pixels)
- Verify cameras have sufficient overlap in ROI
- Increase EMBEDDING_SIM_THRESHOLD if too many false matches

### Video Sync Issues
- Ensure videos have same frame rate
- Check DESYNC_TOLERANCE_FRAMES in config
- Review logs for frame desync warnings

## Code Structure Overview

### Key Components

**Detection Pipeline** (`detection/rtdetr_wrapper.py`)
- Loads pre-trained RT-DETR model
- Extracts histogram embeddings for object appearance
- Runs inference on warped and original views
- Handles model device switching (CPU/GPU)

**Single-Camera Tracker** (`tracking/single_camera_tracker.py`)
- Maintains per-camera object tracks
- Matches detections to existing tracks
- Computes velocity and motion compensation
- Handles track state lifecycle (active/lost/reserved)

**Global Registry** (`multicam/global_registry.py`)
- Associates observations across cameras
- Maintains global object identities
- Handles cross-camera buffering and matching
- Manages track resurrection for lost objects

**Event Manager** (`event/event_manager.py`)
- Implements pickup/putback state machine
- Detects motion direction (inward vs outward)
- Generates events with timestamps and metadata
- Handles rate limiting and low-light adaptation

**Motion Analyzer** (`motion/motion_analyzer.py`)
- Analyzes object displacement relative to ROI
- Computes motion direction using edge normals
- Distinguishes between pickup and putback
- Filters jitter using deadband threshold

**Video Processing** (`utils/video_processor.py`)
- Synchronizes multi-camera video streams
- Applies homography warping
- Generates grayscale frames for optical flow
- Handles frame synchronization tolerances

### Data Flow

```
Input Videos
    ↓
SynchronizedVideoReader (frame alignment)
    ↓
RTDETRDetector (object detection + embedding)
    ↓
SingleCameraTracker x2 (per-camera tracking)
    ↓
GlobalRegistry (cross-camera association)
    ↓
MotionAnalyzer (motion classification)
    ↓
EventManager (event detection state machine)
    ↓
Visualizer (annotate frames)
    ↓
Output: Videos, Snapshots, Session JSON
```

## Dependencies

- **opencv-python**: Video processing and image operations
- **numpy**: Numerical computations
- **scipy**: Optimization (Hungarian algorithm for assignment)
- **scikit-learn**: Embedding similarity computations
- **ultralytics**: RT-DETR model interface
- **torch, torchvision, torchaudio**: Deep learning framework

## License

Internal Use Only

## Support

For issues or questions, refer to:
- Implementation logs in `outputs/logs/`
- Annotated videos in `outputs/annotated_videos/`
- Session summaries in `outputs/sessions/`

## Additional Notes

### Model Training

The fine-tuned RT-DETR model (`models/best.pt`) was trained on vending machine product detection. To retrain:
1. Prepare annotated dataset in YOLO format
2. Use Ultralytics YOLO training pipeline
3. Replace `models/best.pt` with new weights

### ROI Calibration Tips

- Ensure cameras capture the entire shelf area
- Include buffer space beyond shelf boundaries
- Use consistent lighting during ROI definition
- Test with reference videos before full processing
