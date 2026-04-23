from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

# Allow script to be run directly from its directory
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    DETECTION_MODEL_CONFIDENCE,
    DETECTION_DEBUG_LOG_INTERVAL,
    DETECTION_FUSION_IOU,
    ENABLE_ORIGINAL_VIEW_FUSION,
    MODEL_PATH,
    ORIGINAL_VIEW_ROI_PAD_PX,
    OUTPUT_PATHS,
    PREVIEW_SCALE,
    PREVIEW_WINDOW_NAME,
    PRINT_DETECTION_SUMMARY,
    PRINT_TRACK_STATUS,
    ROI_CONFIG_DIR,
    SAVE_ANNOTATED_VIDEO,
    SAVE_SNAPSHOTS,
    SHOW_PREVIEW,
    TRACK_STATUS_INTERVAL,
)
from detection.rtdetr_wrapper import RTDETRDetector
from event.event_manager import EventManager
from motion.motion_analyzer import MotionAnalyzer
from multicam.global_registry import GlobalRegistry
from multicam.homography import HomographyContext
from tracking.single_camera_tracker import SingleCameraTracker
from utils.logger import setup_logger
from utils.roi_utils import (
    bbox_iou,
    clip_bbox,
    compute_edge_normals,
    expand_polygon,
    load_roi_payload,
    point_in_polygon,
    project_bbox,
    transform_point,
)
from utils.types import Detection
from utils.video_processor import AnnotatedVideoWriter, SynchronizedVideoReader
from utils.visualization import (
    compose_preview_grid,
    draw_camera_header,
    draw_event_history,
    draw_mode_banner,
    draw_original_track_motion,
    draw_polygon,
    draw_roi_edge_normals,
    draw_track_overlay,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the pipeline."""
    parser = argparse.ArgumentParser(description="Top-down multi-camera vending pickup / putback detector.")
    parser.add_argument("--video0", required=True, help="Path to camera 0 video")
    parser.add_argument("--video1", required=True, help="Path to camera 1 video")
    parser.add_argument("--session_id", required=True, help="Session identifier for outputs")
    parser.add_argument("--device", default="cpu", help="Torch device for RT-DETR")
    parser.add_argument("--roi_dir", default=ROI_CONFIG_DIR, help="Directory containing cam0.json and cam1.json")
    parser.add_argument("--model_path", default=MODEL_PATH, help="Fine-tuned RT-DETR model path")
    parser.add_argument("--det_conf", type=float, default=DETECTION_MODEL_CONFIDENCE, help="Detection confidence threshold")
    parser.add_argument("--show_preview", action=argparse.BooleanOptionalAction, default=SHOW_PREVIEW, help="Show a live preview window with both cameras")
    return parser.parse_args()


def save_snapshot(frame, event: dict, session_id: str) -> str:
    """Save a snapshot image of an event (pickup/putback)."""
    filename = (
        f"{event['type']}_{event['class']}_global{event['global_id']}_"
        f"frame{event['frame_index']}_cam{event['camera_id']}.jpg"
    )
    path = OUTPUT_PATHS.snapshots / filename
    cv2.imwrite(str(path), frame)
    return str(path)


def show_preview_window(preview_grid: np.ndarray) -> bool:
    """Display the preview grid and return False if user pressed 'q' or ESC."""
    if preview_grid.size == 0:
        return True
    if PREVIEW_SCALE != 1.0:
        preview_grid = cv2.resize(
            preview_grid,
            dsize=None,
            fx=PREVIEW_SCALE,
            fy=PREVIEW_SCALE,
            interpolation=cv2.INTER_AREA,
        )
    cv2.imshow(PREVIEW_WINDOW_NAME, preview_grid)
    key = cv2.waitKey(1) & 0xFF
    return key not in {ord("q"), 27}


def fuse_detections(detections: List[Detection], iou_threshold: float) -> List[Detection]:
    """
    Fuse overlapping detections of the same class by keeping the highest confidence.
    Used to merge warped‑view and original‑projected detections.
    """
    ordered = sorted(detections, key=lambda det: det.confidence, reverse=True)
    fused: List[Detection] = []
    for detection in ordered:
        duplicate = False
        for kept in fused:
            if kept.class_id != detection.class_id:
                continue
            if bbox_iou(kept.bbox, detection.bbox) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            fused.append(detection)
    return fused


def project_original_detection_to_warped(
    detection: Detection,
    *,
    packet,
    roi_polygon: np.ndarray,
    safe_polygon: np.ndarray,
    roi_pad_px: float,
) -> Detection | None:
    """
    Project a detection from the original (unwarped) frame into the warped shelf plane.
    Returns a new Detection with updated bbox and ROI overrides, or None if outside the expanded ROI.
    """
    # Expand ROI by a margin to catch objects near the boundary
    expanded_roi = expand_polygon(roi_polygon, roi_pad_px)
    if not point_in_polygon(detection.centroid, expanded_roi):
        return None
    projected_bbox = project_bbox(packet.warp_matrix, detection.bbox)
    if projected_bbox is None:
        return None
    clipped_bbox = clip_bbox(projected_bbox, packet.warped_frame.shape[1], packet.warped_frame.shape[0])
    if clipped_bbox is None:
        return None
    return Detection(
        bbox=clipped_bbox,
        class_id=detection.class_id,
        class_name=detection.class_name,
        confidence=detection.confidence,
        embedding=detection.embedding.copy(),
        camera_id=detection.camera_id,
        frame_index=detection.frame_index,
        timestamp_ms=detection.timestamp_ms,
        source_view="original_projected",
        original_centroid=detection.original_centroid.copy() if detection.original_centroid is not None else detection.centroid.copy(),
        in_safe_roi_override=point_in_polygon(detection.centroid, safe_polygon),
        in_outer_roi_override=point_in_polygon(detection.centroid, roi_polygon),
    )


def main() -> None:
    args = parse_args()
    timestamp_tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(OUTPUT_PATHS.logs / f"{args.session_id}_{timestamp_tag}.log")

    # Load ROI configurations for both cameras
    roi_dir = Path(args.roi_dir)
    roi0 = load_roi_payload(roi_dir / "cam0.json")
    roi1 = load_roi_payload(roi_dir / "cam1.json")

    # Compute homography between cameras (used for cross‑camera matching)
    homography = HomographyContext.from_roi_points(roi0["points"], roi1["points"])
    if not homography.active:
        logger.warning("Homography disabled because projection error is %.2f px", homography.error_px)

    # Instantiate components
    detector = RTDETRDetector(args.model_path, device=args.device, conf_threshold=args.det_conf)
    trackers = {
        0: SingleCameraTracker(camera_id=0),
        1: SingleCameraTracker(camera_id=1),
    }
    registry = GlobalRegistry()
    motion_analyzer = MotionAnalyzer()
    event_manager = EventManager(session_id=args.session_id)
    reader = SynchronizedVideoReader(args.video0, args.video1, roi0, roi1)

    # Optional video writers for annotated output
    video_writers: Dict[int, AnnotatedVideoWriter] = {}
    if SAVE_ANNOTATED_VIDEO:
        for camera_id in (0, 1):
            fps = reader.fps(camera_id)
            writer_path = OUTPUT_PATHS.annotated_videos / f"{args.session_id}_cam{camera_id}.mp4"
            video_writers[camera_id] = AnnotatedVideoWriter(writer_path, fps=fps, frame_size=reader.frame_size(camera_id))

    try:
        if args.show_preview:
            cv2.namedWindow(PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)

        # Main loop over video frames
        while True:
            bundle = reader.next()
            if bundle is None or len(bundle["packets"]) < 2:
                break
            packets = bundle["packets"]
            sync_ok = bundle["sync_ok"]
            if not sync_ok:
                logger.warning("Frame desync beyond tolerance (%s frames); freezing cross-camera matching.", bundle["desync_frames"])

            all_confidences: List[float] = []
            observations = []                     # TrackObservations from both cameras
            per_camera_detection_stats: Dict[int, Dict[str, int]] = {}
            per_camera_original_detections: Dict[int, List[Detection]] = {}

            # Process each camera independently
            for camera_id, packet in packets.items():
                # Run detector on warped (top‑down) frame
                warped_detections = detector.detect(
                    packet.warped_frame,
                    camera_id=camera_id,
                    frame_index=packet.frame_index,
                    timestamp_ms=packet.timestamp_ms,
                )
                # Store original centroid (in original image coordinates) for each detection
                inverse_warp = np.linalg.inv(packet.warp_matrix)
                for detection in warped_detections:
                    detection.original_centroid = transform_point(inverse_warp, detection.centroid)

                # Optionally, also run detector on original frame and project into warped space
                original_projected: List[Detection] = []
                if ENABLE_ORIGINAL_VIEW_FUSION:
                    original_detections = detector.detect(
                        packet.frame,
                        camera_id=camera_id,
                        frame_index=packet.frame_index,
                        timestamp_ms=packet.timestamp_ms,
                    )
                    roi_polygon = np.asarray((roi0 if camera_id == 0 else roi1)["points"], dtype=np.float32)
                    safe_polygon = np.asarray((roi0 if camera_id == 0 else roi1)["safe_polygon"], dtype=np.float32)
                    for detection in original_detections:
                        projected = project_original_detection_to_warped(
                            detection,
                            packet=packet,
                            roi_polygon=roi_polygon,
                            safe_polygon=safe_polygon,
                            roi_pad_px=ORIGINAL_VIEW_ROI_PAD_PX,
                        )
                        if projected is not None:
                            original_projected.append(projected)
                else:
                    original_detections = []

                per_camera_original_detections[camera_id] = original_projected

                # Fuse detections from warped view and projected original view
                detections = fuse_detections(warped_detections + original_projected, DETECTION_FUSION_IOU)

                per_camera_detection_stats[camera_id] = {
                    "warped": len(warped_detections),
                    "original_projected": len(original_projected),
                    "fused": len(detections),
                }
                all_confidences.extend([det.confidence for det in detections])

                # Update single‑camera tracker and get TrackObservations
                observations.extend(
                    trackers[camera_id].update(
                        detections,
                        gray=packet.warped_gray,
                        safe_roi_polygon=packet.safe_roi_polygon,
                        full_roi_polygon=packet.full_roi_polygon,
                        shaky=packet.shaky,
                    )
                )

                # Log detection summary periodically
                if packet.frame_index % DETECTION_DEBUG_LOG_INTERVAL == 0:
                    logger.debug(
                        "cam=%s frame=%s warped=%s original_projected=%s fused=%s",
                        camera_id,
                        packet.frame_index,
                        per_camera_detection_stats[camera_id]["warped"],
                        per_camera_detection_stats[camera_id]["original_projected"],
                        per_camera_detection_stats[camera_id]["fused"],
                    )
                    if PRINT_DETECTION_SUMMARY:
                        print(
                            f"[DETECTION] cam={camera_id} frame={packet.frame_index} "
                            f"warped={per_camera_detection_stats[camera_id]['warped']} "
                            f"original_projected={per_camera_detection_stats[camera_id]['original_projected']} "
                            f"fused={per_camera_detection_stats[camera_id]['fused']}"
                        )

            # Compute average confidence across all detections (used for low‑light adaptation)
            frame_avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
            timestamp_ms = max(packet.timestamp_ms for packet in packets.values())

            # Global registry: match observations to global tracks (across cameras)
            tracks = registry.update(
                observations,
                timestamp_ms=timestamp_ms,
                homography_error_px=homography.error_px,
                sync_ok=sync_ok,
                motion_lookup={},   # Motion will be computed separately
            )

            # For each global track, compute motion analysis (outward/inward)
            for track in tracks:
                if track.current_update is None:
                    continue
                packet = packets[track.current_update.camera_id]
                roi_payload = roi0 if track.current_update.camera_id == 0 else roi1
                motion = motion_analyzer.analyze(
                    track,
                    np.asarray(roi_payload["points"], dtype=np.float32),
                    np.asarray(roi_payload["edge_normals"], dtype=np.float32),
                    np.asarray(roi_payload["outward_vector"], dtype=np.float32),
                )
                # Update TrackUpdate with motion results
                track.current_update.outward_motion = motion["outward_motion"]
                track.current_update.inward_motion = motion["inward_motion"]
                track.current_update.displacement_vector = motion["displacement_vector"]
                track.current_update.displacement_magnitude = motion["displacement_magnitude"]
                track.current_update.motion_dot = motion["motion_dot"]
                track.current_update.nearest_edge_index = motion["nearest_edge_index"]
                track.current_update.edge_normal = motion["edge_normal"]

                # Optional verbose track status logging
                if PRINT_TRACK_STATUS and packet.frame_index % TRACK_STATUS_INTERVAL == 0:
                    displacement = tuple(float(v) for v in track.current_update.displacement_vector)
                    edge_normal = tuple(float(v) for v in track.current_update.edge_normal)
                    message = (
                        f"[TRACK_STATUS] cam={track.current_update.camera_id} "
                        f"gid={track.global_id} class={track.class_name} state={track.event_state} "
                        f"safe={track.current_update.in_safe_roi} outer={track.current_update.in_outer_roi} "
                        f"outward={track.current_update.outward_motion} inward={track.current_update.inward_motion} "
                        f"mag={track.current_update.displacement_magnitude:.2f} dot={track.current_update.motion_dot:.2f} "
                        f"edge={track.current_update.nearest_edge_index} "
                        f"normal={edge_normal} disp={displacement} conf={track.current_update.confidence:.2f}"
                    )
                    logger.debug(message)
                    print(message)

            # Update event manager (state machine) and get new pickup/putback events
            events = event_manager.process_frame(tracks, frame_avg_confidence, timestamp_ms)

            # ----- Visualization and output -----
            original_preview_frames: Dict[int, np.ndarray] = {}
            analyzed_preview_frames: Dict[int, np.ndarray] = {}
            overlay_lines = event_manager.overlay_event_lines(timestamp_ms)

            for camera_id, packet in packets.items():
                # --- Original view annotation ---
                original_view = packet.frame.copy()
                roi_polygon = np.asarray((roi0 if camera_id == 0 else roi1)["points"], dtype=np.float32)
                safe_polygon = np.asarray((roi0 if camera_id == 0 else roi1)["safe_polygon"], dtype=np.float32)
                original_view = draw_polygon(original_view, roi_polygon, (0, 255, 255), "ROI")
                original_view = draw_polygon(original_view, safe_polygon, (0, 255, 0), "SAFE")

                # Draw original detections (projected ones) on original view
                for detection in per_camera_original_detections[camera_id]:
                    x1, y1, x2, y2 = [int(v) for v in detection.bbox]
                    cv2.rectangle(original_view, (x1, y1), (x2, y2), (255, 200, 0), 2)
                    cv2.putText(
                        original_view,
                        f"{detection.class_name} {detection.confidence:.2f}",
                        (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 200, 0),
                        2,
                    )

                # Draw motion trails for global tracks on original view
                for track in tracks:
                    if track.current_update is not None and track.current_update.camera_id == camera_id:
                        original_view = draw_original_track_motion(original_view, track)

                original_view = draw_camera_header(
                    original_view,
                    title=f"Camera {camera_id} Original",
                    frame_index=packet.frame_index,
                    timestamp_ms=packet.timestamp_ms,
                    shaky=packet.shaky,
                    sync_ok=sync_ok,
                    detections=per_camera_detection_stats[camera_id]["fused"],
                )
                original_view = draw_event_history(original_view, overlay_lines, origin=(10, 90))
                original_preview_frames[camera_id] = original_view

                # --- Warped (analysis) view annotation ---
                annotated = packet.warped_display_frame.copy()
                annotated = draw_polygon(annotated, packet.full_roi_polygon, (255, 255, 0), "ROI")
                annotated = draw_polygon(annotated, packet.safe_roi_polygon, (0, 255, 0), "SAFE ROI")
                annotated = draw_roi_edge_normals(
                    annotated,
                    packet.full_roi_polygon,
                    compute_edge_normals(packet.full_roi_polygon),
                )
                for track in tracks:
                    if track.current_update is not None and track.current_update.camera_id == camera_id:
                        annotated = draw_track_overlay(annotated, track, event_manager.last_event_text)
                annotated = draw_mode_banner(
                    annotated,
                    low_light_mode=event_manager.low_light_mode,
                    paused=event_manager.events_paused,
                )
                annotated = draw_camera_header(
                    annotated,
                    title=f"Camera {camera_id} Analysis",
                    frame_index=packet.frame_index,
                    timestamp_ms=packet.timestamp_ms,
                    shaky=packet.shaky,
                    sync_ok=sync_ok,
                    detections=per_camera_detection_stats[camera_id]["fused"],
                )
                annotated = draw_event_history(annotated, overlay_lines, origin=(10, 90))
                analyzed_preview_frames[camera_id] = annotated

                # Save annotated video if enabled
                if SAVE_ANNOTATED_VIDEO:
                    video_writers[camera_id].write(original_view)

            # Save snapshots for any events that occurred in this frame
            for event in events:
                if SAVE_SNAPSHOTS:
                    snapshot_path = save_snapshot(original_preview_frames[event["camera_id"]], event, args.session_id)
                    event["snapshot_path"] = snapshot_path

            # Show live preview grid
            if args.show_preview:
                preview_grid = compose_preview_grid(
                    original_preview_frames,
                    analyzed_preview_frames,
                    panel_size=(960, 540),
                )
                if not show_preview_window(preview_grid):
                    logger.info("Preview window requested exit; stopping session early.")
                    break

        # End of video – finalise session and save summary
        session = event_manager.finalize()
        session_path = OUTPUT_PATHS.sessions / f"session_{args.session_id}_{timestamp_tag}.json"
        session_path.write_text(json.dumps(session.__dict__, indent=2), encoding="utf-8")
        logger.info("Session completed. Net inventory change: %s", session.net_change)
        logger.info("Session JSON written to %s", session_path)

    finally:
        # Cleanup resources
        reader.close()
        for writer in video_writers.values():
            writer.close()
        if args.show_preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()