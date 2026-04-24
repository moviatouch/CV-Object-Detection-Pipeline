from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List

import cv2
import numpy as np

# Allow script to be run directly from the project root.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

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
from utils.metrics import MetricsTracker
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


@dataclass
class PipelineStatus:
    """Live status payload emitted to UI consumers during processing."""

    session_id: str
    frames_processed: int
    total_frames: int
    current_frame_index: int
    progress: float
    preview_bgr: np.ndarray | None
    pickup_total: int
    putback_total: int
    net_inventory: int
    pickup_by_product: Dict[str, int]
    putback_by_product: Dict[str, int]
    net_by_product: Dict[str, int]
    events: List[dict]
    low_light_mode: bool
    events_paused: bool
    sync_ok: bool


@dataclass
class PipelineResult:
    """Final summary returned by a pipeline run."""

    session_id: str
    session_path: str
    log_path: str
    total_frames: int
    frames_processed: int
    pickup_total: int
    putback_total: int
    net_inventory: int
    pickup_by_product: Dict[str, int]
    putback_by_product: Dict[str, int]
    net_by_product: Dict[str, int]
    metrics_summary: Dict[str, object]
    session_summary: dict
    completed: bool


StatusCallback = Callable[[PipelineStatus], bool | None]


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
    Used to merge warped-view and original-projected detections.
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


def _camera_total_frames(reader: SynchronizedVideoReader, camera_id: int) -> int:
    """Return total frames for a given camera."""
    return int(reader.contexts[camera_id].capture.get(cv2.CAP_PROP_FRAME_COUNT))


def _overall_total_frames(reader: SynchronizedVideoReader) -> int:
    """Use the shorter video as the overall progress baseline."""
    totals = [_camera_total_frames(reader, camera_id) for camera_id in (0, 1)]
    valid_totals = [total for total in totals if total > 0]
    return min(valid_totals) if valid_totals else 0


def _net_inventory_by_product(event_manager: EventManager) -> Dict[str, int]:
    """Compute net inventory deltas per product."""
    products = set(event_manager.pickup_count) | set(event_manager.putback_count)
    return {
        product: event_manager.pickup_count.get(product, 0) - event_manager.putback_count.get(product, 0)
        for product in sorted(products)
    }


def _metrics_summary(metrics_tracker: MetricsTracker) -> Dict[str, object]:
    """Build a compact metrics summary for API/UI consumers."""
    camera_metrics = {
        camera_id: metrics.get_summary()
        for camera_id, metrics in sorted(metrics_tracker.metrics.camera_metrics.items())
    }
    return {
        "total_time_seconds": round(metrics_tracker.metrics.get_total_time_seconds(), 2),
        "cameras": camera_metrics,
        "system": metrics_tracker.metrics.system_metrics.get_summary(),
        "peak_cpu_percent": round(metrics_tracker.metrics.peak_cpu_percent, 2),
        "peak_memory_percent": round(metrics_tracker.metrics.peak_memory_percent, 2),
    }


def _build_status(
    *,
    session_id: str,
    frames_processed: int,
    total_frames: int,
    current_frame_index: int,
    preview_bgr: np.ndarray | None,
    event_manager: EventManager,
    events: List[dict],
    sync_ok: bool,
) -> PipelineStatus:
    """Build a live status payload for a UI callback."""
    pickup_by_product = dict(sorted(event_manager.pickup_count.items()))
    putback_by_product = dict(sorted(event_manager.putback_count.items()))
    net_by_product = _net_inventory_by_product(event_manager)
    progress = (frames_processed / total_frames) if total_frames > 0 else 0.0

    return PipelineStatus(
        session_id=session_id,
        frames_processed=frames_processed,
        total_frames=total_frames,
        current_frame_index=current_frame_index,
        progress=max(0.0, min(progress, 1.0)),
        preview_bgr=preview_bgr,
        pickup_total=sum(pickup_by_product.values()),
        putback_total=sum(putback_by_product.values()),
        net_inventory=sum(net_by_product.values()),
        pickup_by_product=pickup_by_product,
        putback_by_product=putback_by_product,
        net_by_product=net_by_product,
        events=events,
        low_light_mode=event_manager.low_light_mode,
        events_paused=event_manager.events_paused,
        sync_ok=sync_ok,
    )


def run_pipeline(
    *,
    video0: str,
    video1: str,
    session_id: str,
    device: str = "cpu",
    roi_dir: str = ROI_CONFIG_DIR,
    model_path: str = MODEL_PATH,
    det_conf: float = DETECTION_MODEL_CONFIDENCE,
    show_preview: bool = SHOW_PREVIEW,
    status_callback: StatusCallback | None = None,
    callback_every_n_frames: int = 1,
    preview_panel_size: tuple[int, int] = (960, 540),
) -> PipelineResult:
    """Run the pipeline and optionally emit live status updates for a UI."""
    timestamp_tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(OUTPUT_PATHS.logs / f"{session_id}_{timestamp_tag}.log")
    metrics_tracker = MetricsTracker(session_id=session_id, logger=logger)

    roi_dir_path = Path(roi_dir)
    roi0 = load_roi_payload(roi_dir_path / "cam0.json")
    roi1 = load_roi_payload(roi_dir_path / "cam1.json")

    homography = HomographyContext.from_roi_points(roi0["points"], roi1["points"])
    if not homography.active:
        logger.warning("Homography disabled because projection error is %.2f px", homography.error_px)

    detector = RTDETRDetector(model_path, device=device, conf_threshold=det_conf)
    trackers = {
        0: SingleCameraTracker(camera_id=0),
        1: SingleCameraTracker(camera_id=1),
    }
    registry = GlobalRegistry()
    motion_analyzer = MotionAnalyzer()
    event_manager = EventManager(session_id=session_id)
    reader = SynchronizedVideoReader(video0, video1, roi0, roi1)
    total_frames = _overall_total_frames(reader)
    callback_every_n_frames = max(1, int(callback_every_n_frames))
    frames_processed = 0
    completed = True

    for camera_id in (0, 1):
        fps = reader.fps(camera_id)
        resolution = reader.frame_size(camera_id)
        camera_total_frames = _camera_total_frames(reader, camera_id)
        metrics_tracker.initialize_camera(camera_id, fps, resolution, camera_total_frames)
        logger.info(
            "Camera %d initialized: FPS=%.2f, Resolution=%dx%d, Total Frames=%d",
            camera_id,
            fps,
            resolution[0],
            resolution[1],
            camera_total_frames,
        )

    video_writers: Dict[int, AnnotatedVideoWriter] = {}
    if SAVE_ANNOTATED_VIDEO:
        for camera_id in (0, 1):
            fps = reader.fps(camera_id)
            writer_path = OUTPUT_PATHS.annotated_videos / f"{session_id}_cam{camera_id}.mp4"
            video_writers[camera_id] = AnnotatedVideoWriter(
                writer_path,
                fps=fps,
                frame_size=reader.frame_size(camera_id),
            )

    try:
        if show_preview:
            cv2.namedWindow(PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)

        while True:
            metrics_tracker.start_frame()

            bundle = reader.next()
            if bundle is None or len(bundle["packets"]) < 2:
                break
            packets = bundle["packets"]
            sync_ok = bundle["sync_ok"]
            frames_processed += 1

            if not sync_ok:
                logger.warning(
                    "Frame desync beyond tolerance (%s frames); freezing cross-camera matching.",
                    bundle["desync_frames"],
                )

            all_confidences: List[float] = []
            observations = []
            per_camera_detection_stats: Dict[int, Dict[str, int]] = {}
            per_camera_original_detections: Dict[int, List[Detection]] = {}

            for camera_id, packet in packets.items():
                warped_detections = detector.detect(
                    packet.warped_frame,
                    camera_id=camera_id,
                    frame_index=packet.frame_index,
                    timestamp_ms=packet.timestamp_ms,
                )
                inverse_warp = np.linalg.inv(packet.warp_matrix)
                for detection in warped_detections:
                    detection.original_centroid = transform_point(inverse_warp, detection.centroid)

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

                per_camera_original_detections[camera_id] = original_projected
                detections = fuse_detections(warped_detections + original_projected, DETECTION_FUSION_IOU)

                per_camera_detection_stats[camera_id] = {
                    "warped": len(warped_detections),
                    "original_projected": len(original_projected),
                    "fused": len(detections),
                }
                all_confidences.extend([det.confidence for det in detections])

                observations.extend(
                    trackers[camera_id].update(
                        detections,
                        gray=packet.warped_gray,
                        safe_roi_polygon=packet.safe_roi_polygon,
                        full_roi_polygon=packet.full_roi_polygon,
                        shaky=packet.shaky,
                    )
                )

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

            frame_avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
            timestamp_ms = max(packet.timestamp_ms for packet in packets.values())

            tracks = registry.update(
                observations,
                timestamp_ms=timestamp_ms,
                homography_error_px=homography.error_px,
                sync_ok=sync_ok,
                motion_lookup={},
            )

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
                track.current_update.outward_motion = motion["outward_motion"]
                track.current_update.inward_motion = motion["inward_motion"]
                track.current_update.displacement_vector = motion["displacement_vector"]
                track.current_update.displacement_magnitude = motion["displacement_magnitude"]
                track.current_update.motion_dot = motion["motion_dot"]
                track.current_update.nearest_edge_index = motion["nearest_edge_index"]
                track.current_update.edge_normal = motion["edge_normal"]

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

            events = event_manager.process_frame(tracks, frame_avg_confidence, timestamp_ms)

            original_preview_frames: Dict[int, np.ndarray] = {}
            analyzed_preview_frames: Dict[int, np.ndarray] = {}
            overlay_lines = event_manager.overlay_event_lines(timestamp_ms)

            for camera_id, packet in packets.items():
                original_view = packet.frame.copy()
                roi_polygon = np.asarray((roi0 if camera_id == 0 else roi1)["points"], dtype=np.float32)
                safe_polygon = np.asarray((roi0 if camera_id == 0 else roi1)["safe_polygon"], dtype=np.float32)
                original_view = draw_polygon(original_view, roi_polygon, (0, 255, 255), "ROI")
                original_view = draw_polygon(original_view, safe_polygon, (0, 255, 0), "SAFE")

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

                if SAVE_ANNOTATED_VIDEO:
                    video_writers[camera_id].write(original_view)

            for event in events:
                if SAVE_SNAPSHOTS:
                    snapshot_path = save_snapshot(original_preview_frames[event["camera_id"]], event, session_id)
                    event["snapshot_path"] = snapshot_path

            preview_grid: np.ndarray | None = None
            should_render_preview = show_preview or status_callback is not None
            if should_render_preview:
                preview_grid = compose_preview_grid(
                    original_preview_frames,
                    analyzed_preview_frames,
                    panel_size=preview_panel_size,
                )

            if show_preview and preview_grid is not None:
                if not show_preview_window(preview_grid):
                    logger.info("Preview window requested exit; stopping session early.")
                    completed = False
                    break

            current_frame_index = packets[0].frame_index if 0 in packets else max(
                packet.frame_index for packet in packets.values()
            )
            if status_callback is not None and (
                frames_processed % callback_every_n_frames == 0 or bool(events)
            ):
                status = _build_status(
                    session_id=session_id,
                    frames_processed=frames_processed,
                    total_frames=total_frames,
                    current_frame_index=current_frame_index,
                    preview_bgr=preview_grid,
                    event_manager=event_manager,
                    events=events,
                    sync_ok=sync_ok,
                )
                keep_running = status_callback(status)
                if keep_running is False:
                    logger.info("Status callback requested early stop for session %s.", session_id)
                    completed = False
                    break

            for camera_id in packets.keys():
                metrics_tracker.end_frame(camera_id)

            if current_frame_index % 100 == 0:
                metrics_tracker.update_system_metrics()

        session = event_manager.finalize()
        session_path = OUTPUT_PATHS.sessions / f"session_{session_id}_{timestamp_tag}.json"
        session_payload = {
            "session_id": session.session_id,
            "events": session.events,
            "pickup_records": session.pickup_records,
            "putback_records": session.putback_records,
            "pickup_count": session.pickup_count,
            "putback_count": session.putback_count,
            "net_change": session.net_change,
            "warnings": session.warnings,
        }
        session_path.write_text(json.dumps(session_payload, indent=2), encoding="utf-8")
        logger.info("Session completed. Net inventory change: %s", session.net_change)
        logger.info("Session JSON written to %s", session_path)

        metrics_tracker.finalize()
        metrics_tracker.print_summary()

        metrics_summary = _metrics_summary(metrics_tracker)
        net_by_product = _net_inventory_by_product(event_manager)

        return PipelineResult(
            session_id=session_id,
            session_path=str(session_path),
            log_path=str(OUTPUT_PATHS.logs / f"{session_id}_{timestamp_tag}.log"),
            total_frames=total_frames,
            frames_processed=frames_processed,
            pickup_total=sum(event_manager.pickup_count.values()),
            putback_total=sum(event_manager.putback_count.values()),
            net_inventory=sum(net_by_product.values()),
            pickup_by_product=dict(sorted(event_manager.pickup_count.items())),
            putback_by_product=dict(sorted(event_manager.putback_count.items())),
            net_by_product=net_by_product,
            metrics_summary=metrics_summary,
            session_summary=session_payload,
            completed=completed,
        )
    finally:
        reader.close()
        for writer in video_writers.values():
            writer.close()
        if show_preview:
            cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    run_pipeline(
        video0=args.video0,
        video1=args.video1,
        session_id=args.session_id,
        device=args.device,
        roi_dir=args.roi_dir,
        model_path=args.model_path,
        det_conf=args.det_conf,
        show_preview=args.show_preview,
    )


if __name__ == "__main__":
    main()
