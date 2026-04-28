# Viatouch Object Identification Pipeline Flow Document

## 1. One-line summary

This pipeline watches synchronized vending-machine videos from two cameras, detects products, tracks the same item over time, understands whether it is moving out or back in, and then converts that motion into confirmed `pickup` or `putback` events.

## 2. Simple explanation 

In simple terms, the system first detects products in each frame using a fine-tuned `RT-DETR` model. Then it tracks each detected product inside each camera view, and after that it tries to maintain one global identity for the same item across both cameras. Once an item is tracked reliably, the pipeline checks whether it is moving away from the shelf or back toward it. Based on that motion and a confirmation window, it decides whether the action is a `pickup` or a `putback`.

So the logic is not based on just one frame. It combines:

- object detection
- per-camera tracking
- cross-camera identity matching
- appearance-based ReID
- ROI-based motion understanding
- event confirmation rules

That combination is what makes the final event decision more stable.

## 3. Core modules and what each one does

| Module | Purpose |
| --- | --- |
| `detection/rtdetr_wrapper.py` | Detects products and creates appearance embeddings |
| `tracking/single_camera_tracker.py` | Maintains local tracking separately for each camera |
| `multicam/global_registry.py` | Assigns one global ID across both cameras |
| `motion/motion_analyzer.py` | Decides whether movement is outward or inward relative to ROI |
| `event/event_manager.py` | Converts stable motion into pickup / putback events |
| `utils/video_processor.py` | Reads synchronized videos, enhances frames, warps ROI, checks shake |

## 4. End-to-end flow

1. Two camera videos are read in synchronized pairs.
2. Each frame is enhanced and warped into a top-down shelf-aligned view.
3. The detector runs on the warped view.
4. If enabled, the detector also runs on the original view and those detections are projected back into the warped plane.
5. Overlapping detections from both views are fused.
6. Each camera runs its own local tracker.
7. Local observations from both cameras are sent to the global registry.
8. The global registry tries to decide whether observations belong to an existing global object or should become a new one.
9. Motion analysis checks whether the object is moving outward or inward relative to the ROI.
10. The event state machine confirms whether that motion is a real pickup or putback.
11. The system saves overlays, snapshots, logs, and a session JSON summary.

## 5. Detection model

### Model used

- Detection model: `RT-DETR`
- Wrapper used: Ultralytics `RTDETR`
- Model weights path: `models/best.pt`
- Confidence threshold: controlled by `DETECTION_MODEL_CONFIDENCE`

### Why this model is used

`RT-DETR` gives real-time object detection while still being accurate enough for shelf-product scenarios. In this pipeline it is the first stage that tells us:

- what object is present
- where it is located
- how confident the model is

### Batched inference

The pipeline reads multiple synchronized frame pairs and runs detection in batches.

- `DETECTION_BATCH_SIZE = 30`

This is mainly for throughput, so detection is faster and more efficient than running one frame at a time.

## 6. ROI and view normalization

Before tracking, each camera view is converted into a warped shelf plane using ROI corner points.

Why this matters:

- both cameras become easier to analyze in a consistent top-down-like space
- motion direction becomes easier to interpret
- pickup vs putback decisions become more stable

The ROI logic uses:

- `outer ROI`: the full working shelf area
- `safe ROI`: a slightly shrunken inner area

The safe ROI is important because the system waits until an item is stably inside this safer region before treating it as a settled shelf item.

## 7. Local tracking: what is used inside each camera

Each camera has its own `SingleCameraTracker`.

### How local tracking works

For every new frame, detections are matched to existing local tracks using the Hungarian assignment algorithm. The matching cost combines:

- class consistency
- centroid distance
- bounding-box IoU
- appearance similarity

This means a detection is matched to an existing local track only if it still looks like the same product, is spatially close enough, and overlaps reasonably over time.

### What happens if a detection is missing

If the detector misses an object for a few frames, the tracker does not immediately delete it. It tries to propagate the bounding box using Lucas-Kanade optical flow. This helps keep tracking stable during short detector drops.

### Output of local tracking

Each local tracker produces a `TrackObservation`, which includes:

- local track ID
- class name
- confidence
- centroid
- velocity
- ROI membership
- appearance embedding

## 8. ReID: how identity matching works

### Important implementation note

This pipeline does **not** currently use a separate deep ReID model.

Instead, ReID is handled using an appearance embedding created from the detected crop itself.

### What embedding is used

For each detection crop, the pipeline computes a `96-dimensional color histogram embedding`.

It is built from the product crop region using histogram bins across the B, G, and R channels. This embedding is then normalized and compared using cosine similarity.

### Where ReID is used

The embedding is used in two places:

1. local tracking, to help match a new detection to an existing track in the same camera
2. global tracking, to help decide whether observations from different cameras or different times belong to the same object

### Practical meaning

> The system gives each detected item a lightweight visual signature based on its appearance, and then uses that signature to keep the same identity over time.

### Limitation to explain honestly

Because this is histogram-based ReID, it is lighter and faster than a dedicated deep ReID network, but it can be less discriminative when products of the same class look extremely similar.

That is why the pipeline does not rely on ReID alone. It also uses:

- class consistency
- position continuity
- overlap over time
- motion direction
- ROI logic
- event confirmation windows

## 9. Global tracking: how one identity is maintained across both cameras

The `GlobalRegistry` is responsible for assigning one `global_id` to the same object, even when it is seen across both camera streams.

### How global matching works today

The global logic uses these stages:

1. If a camera/local-track pair has already been mapped to a global ID, it keeps that mapping.
2. Unmatched observations are compared to existing global tracks.
3. Matching uses:
   - class consistency
   - appearance similarity
   - spatial continuity
   - temporal IoU for same-camera continuity
4. If no existing global track matches, the observation is temporarily buffered as `pending`.
5. If an observation from the other camera appears within the time window and looks similar enough, both are merged into one new global track.
6. If no partner arrives before timeout, the pending observation is promoted into its own new global track.

The pending time window is not fixed only by time. It is extended slightly for faster-moving objects, so the registry can wait a bit longer when motion is rapid.

### Why the pending buffer is useful

The same physical product may appear in one camera slightly before the other. Instead of creating a new global identity immediately, the system briefly waits to see if the second camera confirms it. This reduces duplicate global IDs.

### Lost and revival logic

Global tracks move through lifecycle states:

- `ACTIVE`
- `LOST`
- `LOST_RESERVED`

If an object disappears, the track is not deleted immediately. It is kept for a while so a returning observation can revive the same identity if:

- appearance similarity is high enough
- spatial distance is reasonable
- motion direction is still compatible

This helps reduce identity fragmentation.

## 10. Role of homography in global tracking

Homography is used in this pipeline mainly to:

- normalize each camera into a common shelf-aligned warped view
- measure whether camera-to-camera alignment is reliable enough

If homography reprojection error becomes too high, cross-camera association is treated more conservatively.

Also, if the two camera streams become desynchronized beyond the configured tolerance, cross-camera matching is temporarily frozen and the system keeps only same-camera continuity until synchronization is acceptable again.

### Important clarification

In the current implementation, homography is **not** the main direct matching signal for global identity assignment. The actual cross-camera identity decision is driven mainly by:

- appearance similarity
- timing
- continuity rules

Homography currently acts more like an alignment and reliability layer than a full geometric matcher.

## 11. Motion analysis: how pickup vs putback is understood

Once an object has a stable track, `MotionAnalyzer` checks whether it is moving:

- outward from the ROI -> likely pickup
- inward toward the ROI -> likely putback

### How direction is decided

The pipeline looks at displacement over recent frames and compares it with:

- the ROI center
- the nearest ROI edge normal
- the configured outward vector of the shelf

This helps the system understand direction relative to the shelf, not just raw pixel movement.

### Jitter filtering

Very small motion is ignored using a deadband so camera noise or tiny frame-to-frame shifts do not become false events.

## 12. Event logic: how a pickup or putback gets confirmed

The event decision is handled by `EventManager`.

### Main event states

- `INSIDE`
- `STABLE_INSIDE`
- `PICKUP_PENDING`
- `PICKED_UP`
- `PUTBACK_PENDING`

### Pickup logic

An object is treated as a pickup candidate when:

- it was stable inside the shelf ROI
- it starts moving outward
- it leaves the safe ROI

Then the system waits for confirmation across a short confidence window before recording the pickup.

### Putback logic

An object is treated as a putback candidate when:

- it was previously picked up
- it shows sustained inward motion
- it re-enters the outer ROI / safe ROI

Again, the event is only recorded after confirmation.

### Why this matters

This prevents the system from firing events on:

- one-frame detector noise
- brief occlusions
- shaky frames
- temporary boundary crossings

## 13. Additional robustness logic

The pipeline also contains several protection layers:

- `low-light adaptation`: makes confirmation stricter when confidence drops
- `event pause`: if confidence becomes critically low, event generation can pause
- `shake detection`: shaky frames avoid aggressive state transitions
- `merge locking`: suppresses duplicate events when two same-class tracks overlap too long
- `rate limiting`: prevents unrealistic pickup bursts for the same class

## 14. Outputs delivered by the pipeline

For each run, the pipeline can generate:

- annotated output video
- event snapshots
- logs
- session JSON

The session JSON contains:

- all pickup events
- all putback events
- counts per product
- net inventory change
- warnings


> We use a fine-tuned RT-DETR model to detect products from two synchronized camera views. Each camera first tracks products locally, then we assign a global identity across cameras using appearance similarity, timing, and continuity rules. After identity is stable, we analyze whether the item is moving outward from the shelf or inward back to the shelf. A confirmation state machine then decides whether that action is a true pickup or putback, which helps reduce false events from noise, missed detections, or brief motion near the shelf boundary.


> ReID in the current implementation is appearance-based and uses histogram embeddings from the product crop. It is lightweight and efficient, and it works together with tracking, ROI logic, and event confirmation. It is not a separate deep ReID model at this stage.

## 17. Final architecture summary

`Video Sync -> ROI Warp -> RT-DETR Detection -> Local Tracking -> Global ID Association -> Motion Analysis -> Pickup/Putback State Machine -> Output JSON / Video / Snapshots`
