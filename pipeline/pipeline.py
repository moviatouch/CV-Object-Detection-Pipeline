from __future__ import annotations


import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Generator, List, Tuple

import cv2
import numpy as np

# Allow script to be run directly from the project root.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    DETECTION_BATCH_SIZE,
    DETECTION_MODEL_CONFIDENCE,
    DETECTION_DEBUG_LOG_INTERVAL,
    MODEL_PATH,
    OUTPUT_PATHS,
    PREVIEW_SCALE,
    PREVIEW_WINDOW_NAME,
    PRINT_DETECTION_SUMMARY,
    PRINT_TRACK_STATUS,
    ROI_CONFIG_DIR,
    SAVE_ANNOTATED_VIDEO,
    SHOW_PREVIEW,
    TRACK_STATUS_INTERVAL,
    WARP_SIZE,
    RESIZE_VIDEO
)
from detection.rtdetr_wrapper import YOLODetector
from detection.reid import ReIDModel
import torch
# from detection.reid import DINOv2ReID
from event.event_manager import EventManager
from motion.motion_analyzer import MotionAnalyzer
from multicam.global_registry import GlobalRegistry
from multicam.homography import HomographyContext
from tracking.single_camera_tracker import SingleCameraTracker
from utils.cumulative_stats import CumulativeStats
from utils.logger import setup_logger
from utils.metrics import MetricsTracker
from utils.roi_utils import (
    clip_bbox,
    compute_edge_normals,
    load_roi_payload,
    point_in_polygon,
    project_bbox,
    transform_point,
)
from utils.types import Detection, GlobalTrack, TrackObservation
from utils.video_processor import SynchronizedVideoReader, AnnotatedVideoWriter
from utils.visualization import (
    compose_preview_grid,
    draw_camera_header,
    draw_cumulative_overlay,
    draw_event_panel,
    draw_original_local_track,
    draw_polygon,
    draw_roi_edge_normals,
    draw_track_overlay,
)

# [DEBUG] Import the debug logger
from utils.debug_logger import DebugLogger


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


def show_preview_window(preview_grid: np.ndarray) -> bool:
    if preview_grid.size == 0:
        return True
    if PREVIEW_SCALE != 1.0:
        preview_grid = cv2.resize(preview_grid, dsize=None, fx=PREVIEW_SCALE, fy=PREVIEW_SCALE, interpolation=cv2.INTER_AREA)
    cv2.imshow(PREVIEW_WINDOW_NAME, preview_grid)
    key = cv2.waitKey(1) & 0xFF
    return key not in {ord("q"), 27}


def _camera_total_frames(reader: SynchronizedVideoReader, camera_id: int) -> int:
    return int(reader.contexts[camera_id].capture.get(cv2.CAP_PROP_FRAME_COUNT))


def _overall_total_frames(reader: SynchronizedVideoReader) -> int:
    totals = [_camera_total_frames(reader, cam) for cam in (0, 1)]
    valid = [t for t in totals if t > 0]
    return min(valid) if valid else 0


def _next_bundle_batch(reader: SynchronizedVideoReader, batch_size: int) -> List[dict]:
    bundles = []
    while len(bundles) < batch_size:
        bundle = reader.next()
        if bundle is None:
            break
        if not isinstance(bundle, dict) or "packets" not in bundle:
            break
        if len(bundle["packets"]) < 2:
            break
        bundles.append(bundle)
    return bundles


def _iter_detection_ready_bundles(
    reader: SynchronizedVideoReader,
    detector: YOLODetector,
    *,
    batch_size: int,
    logger:logging.Logger,
    # [DEBUG] removed debug_logger param – we log detections in the main loop
) -> Generator[dict, None, None]:
    """
    Read bundles, run detector in batches, cache frames and detections,
    project detections to warped coordinates, then yield bundles.
    """
    while True:
        bundles = _next_bundle_batch(reader, batch_size)
        if not bundles:
            return

        # ==========================================================
        # CACHES
        # ==========================================================
        frame_cache: Dict[Tuple[int, int], object] = {}
        detection_cache: Dict[Tuple[int, int], List[Detection]] = {}

        # ==========================================================
        # DETECTOR INPUT COLLECTION
        # ==========================================================
        original_frames: List[np.ndarray] = []
        camera_ids: List[int] = []
        frame_indices: List[int] = []
        timestamps: List[float] = []
        keys: List[Tuple[int, int]] = []

        warp_matrices: Dict[Tuple[int, int], np.ndarray] = {}
        frame_shapes: Dict[Tuple[int, int], Tuple[int, int]] = {}
        warped_safe_roi_polygons: Dict[Tuple[int, int], np.ndarray] = {}
        warped_full_roi_polygons: Dict[Tuple[int, int], np.ndarray] = {}

        for batch_idx, bundle in enumerate(bundles):
            for camera_id, packet in bundle["packets"].items():

                key = (camera_id, packet.frame_index)

                frame_cache[key] = packet

                keys.append(key)

                original_frames.append(packet.frame)
                camera_ids.append(camera_id)
                frame_indices.append(packet.frame_index)
                timestamps.append(packet.timestamp_ms)

                warp_matrices[key] = packet.warp_matrix

                frame_shapes[key] = (
                    packet.warped_frame.shape[1],
                    packet.warped_frame.shape[0],
                )

                warped_safe_roi_polygons[key] = packet.safe_roi_polygon
                warped_full_roi_polygons[key] = packet.full_roi_polygon

        # ==========================================================
        # BATCH DETECTION
        # ==========================================================
        logger.info(
            f"[BATCH DETECTION] "
            f"frames={len(original_frames)} "
            f"bundles={len(bundles)}"
        )
        results = detector.detect_batch(
            original_frames,
            camera_ids=camera_ids,
            frame_indices=frame_indices,
            timestamp_ms_list=timestamps,
        )

        # ==========================================================
        # STORE DETECTIONS IN CACHE (no logging here anymore)
        # ==========================================================
        for key, detections in zip(keys, results):
            detection_cache[key] = detections

        # ==========================================================
        # PROJECT DETECTIONS
        # ==========================================================
        projected_by_key: Dict[Tuple[int, int], List[Detection]] = {}

        for key in keys:

            detections = detection_cache.get(key, [])

            warped_dets = []

            for det in detections:

                projected_bbox = project_bbox(
                    warp_matrices[key],
                    det.bbox,
                )

                if projected_bbox is None:
                    continue

                clipped = clip_bbox(
                    projected_bbox,
                    *frame_shapes[key],
                )

                if clipped is None:
                    continue

                if (clipped[2] - clipped[0]) < 32 or (clipped[3] - clipped[1]) < 32:
                    continue

                projected_centroid = np.array(
                    [
                        (clipped[0] + clipped[2]) / 2.0,
                        (clipped[1] + clipped[3]) / 2.0,
                    ],
                    dtype=np.float32,
                )

                warped_det = Detection(
                    bbox=clipped,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    embedding=det.embedding.copy(),
                    camera_id=det.camera_id,
                    frame_index=det.frame_index,
                    timestamp_ms=det.timestamp_ms,
                    source_view="original_projected",
                    original_centroid=det.centroid.copy(),
                    display_bbox=det.bbox,
                    in_safe_roi_override=point_in_polygon(
                        projected_centroid,
                        warped_safe_roi_polygons[key],
                    ),
                    in_outer_roi_override=point_in_polygon(
                        projected_centroid,
                        warped_full_roi_polygons[key],
                    ),
                )

                warped_dets.append(warped_det)

            projected_by_key[key] = warped_dets

        # ==========================================================
        # ATTACH DETECTIONS BACK TO BUNDLES
        # ==========================================================
        for bundle in bundles:

            detections_per_camera = {}

            for camera_id, packet in bundle["packets"].items():

                key = (camera_id, packet.frame_index)

                detections_per_camera[camera_id] = (
                    projected_by_key.get(key, [])
                )

            bundle["detections"] = detections_per_camera

            yield bundle


def _net_inventory_by_product(event_manager: EventManager) -> Dict[str, int]:
    products = set(event_manager.pickup_count) | set(event_manager.putback_count)
    return {p: event_manager.pickup_count.get(p, 0) - event_manager.putback_count.get(p, 0) for p in sorted(products)}


def _metrics_summary(metrics_tracker: MetricsTracker) -> Dict[str, object]:
    camera_metrics = {cid: m.get_summary() for cid, m in sorted(metrics_tracker.metrics.camera_metrics.items())}
    return {
        "total_time_seconds": round(metrics_tracker.metrics.get_total_time_seconds(), 2),
        "cameras": camera_metrics,
        "system": metrics_tracker.metrics.system_metrics.get_summary(),
        "peak_cpu_percent": round(metrics_tracker.metrics.peak_cpu_percent, 2),
        "peak_memory_percent": round(metrics_tracker.metrics.peak_memory_percent, 2),
    }


def _build_status(
    session_id: str,
    frames_processed: int,
    total_frames: int,
    current_frame_index: int,
    preview_bgr: np.ndarray | None,
    event_manager: EventManager,
    events: List[dict],
    sync_ok: bool,
) -> PipelineStatus:
    pickup_by_product = dict(sorted(event_manager.pickup_count.items()))
    putback_by_product = dict(sorted(event_manager.putback_count.items()))
    net_by_product = _net_inventory_by_product(event_manager)
    progress = frames_processed / total_frames if total_frames > 0 else 0.0
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
        sync_ok=sync_ok,
    )


def run_pipeline(
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
    preview_panel_size: Tuple[int, int] = (960, 540),
) -> PipelineResult:
    timestamp_tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(OUTPUT_PATHS.logs / f"{session_id}_{timestamp_tag}.log")
    metrics_tracker = MetricsTracker(session_id=session_id, logger=logger)

    # [DEBUG] Create debug logger
    debug_logger = DebugLogger(
        session_id=session_id,
        output_dir=OUTPUT_PATHS.root,
    )

    # Load cumulative statistics
    cumulative_stats_path = OUTPUT_PATHS.root / "cumulative_stats.json"
    cumulative_stats = CumulativeStats(cumulative_stats_path)
    cumulative_lines = cumulative_stats.get_display_lines()
    if cumulative_stats.data["videos_processed"] > 0:
        logger.info(f"Previously processed videos: {cumulative_stats.data['videos_processed']}")
        logger.info(cumulative_stats.get_summary_line())

    roi_dir_path = Path(roi_dir)
    roi0 = load_roi_payload(roi_dir_path / "cam0.json")
    roi1 = load_roi_payload(roi_dir_path / "cam1.json")

    # --- FIX: extract per-camera outward vectors from ROI config ---
    outward_vec0 = np.asarray(roi0.get("outward_vector", [0.0, 1.0]), dtype=np.float32)
    outward_vec1 = np.asarray(roi1.get("outward_vector", [0.0, 1.0]), dtype=np.float32)

    outward_vec0 = -outward_vec0   # because pickup is upward
    outward_vec1 = -outward_vec1

    homography = HomographyContext.from_roi_points(roi0["points"], roi1["points"])
    if not homography.active:
        logger.warning("Homography disabled (error %.2f px)", homography.error_px)

    detector = YOLODetector(model_path, device=device, conf_threshold=det_conf)
    reid = ReIDModel(Path(model_path).parent / "mobilenet_v3_small-047dcff4.pth", device=device)
    # reid_model = DINOv2ReID(model_name="facebook/dinov2-large", device="cuda" if torch.cuda.is_available() else "cpu")
    trackers = {0: SingleCameraTracker(0), 1: SingleCameraTracker(1)}
    registry = GlobalRegistry()
    motion_analyzer = MotionAnalyzer()
    event_manager = EventManager(session_id=session_id, debug_logger=debug_logger)   # [DEBUG] pass logger
    reader = SynchronizedVideoReader(video0, video1, roi0, roi1)
    total_frames = _overall_total_frames(reader)
    det_batch_size = max(1, int(DETECTION_BATCH_SIZE))
    frames_processed = 0
    completed = True

    for cam in (0, 1):
        fps = reader.fps(cam)
        res = reader.frame_size(cam)
        cam_total = _camera_total_frames(reader, cam)
        metrics_tracker.initialize_camera(cam, fps, res, cam_total)
        logger.info("Camera %d: FPS=%.2f, Resolution=%dx%d, Total Frames=%d", cam, fps, res[0], res[1], cam_total)

    video_writers = {}
    if SAVE_ANNOTATED_VIDEO:
        for cam in (0, 1):
            fps = reader.fps(cam)
            writer_path = OUTPUT_PATHS.annotated_videos / f"{session_id}_cam{cam}.mp4"
            video_writers[cam] = AnnotatedVideoWriter(writer_path, fps=fps, frame_size=reader.frame_size(cam), resize_to=RESIZE_VIDEO)

    try:
        if show_preview:
            cv2.namedWindow(PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)

        bundle_iterator = _iter_detection_ready_bundles(reader, detector, batch_size=det_batch_size,logger=logger)   # no debug_logger param

        while True:
            bundle = next(bundle_iterator, None)
            if bundle is None:
                break
            metrics_tracker.start_frame()
            packets = bundle["packets"]
            sync_ok = bundle["sync_ok"]
            desync_frames = bundle.get("desync_frames", 0)
            frames_processed += 1

            if not sync_ok:
                logger.warning("Frame desync (%s frames) – freezing cross‑camera matching", desync_frames)

            # [DEBUG] We'll set frame context after we compute frame_avg_conf
            # But we need to log detections early – we'll set a placeholder then update later.

            all_confidences = []
            observations = []
            per_camera_detection_stats = {}

            # First, collect all detections and confidences
            for camera_id, packet in packets.items():
                detections = bundle["detections"].get(camera_id, [])
                per_camera_detection_stats[camera_id] = {"original_projected": len(detections)}

                # Optional ReID embedding refinement (using warped frame for consistency)
                for det in detections:
                    try:
                        det.embedding = reid.embed_crop(packet.warped_frame, det.bbox)
                    except Exception:
                        pass

                all_confidences.extend([d.confidence for d in detections])

            # Compute average confidence
            frame_avg_conf = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

            # [DEBUG] Set frame context now (with frame_avg_conf)
            first_packet = packets[0]
            debug_logger.set_frame_context(
                camera_id=first_packet.camera_id,
                frame_index=first_packet.frame_index,
                timestamp_ms=first_packet.timestamp_ms,
                sync_ok=sync_ok,
                desync_frames=desync_frames,
                homography_error_px=homography.error_px,
                frame_avg_conf=frame_avg_conf,
                total_frames=total_frames,
            )

            # [DEBUG] Now log detections for each camera
            for camera_id, packet in packets.items():
                detections = bundle["detections"].get(camera_id, [])
                for det in detections:
                    debug_logger.log_detection(det)

            # Now update trackers
            for camera_id, packet in packets.items():
                detections = bundle["detections"].get(camera_id, [])
                obs_list = trackers[camera_id].update(
                    detections,
                    safe_roi_polygon=packet.safe_roi_polygon,
                    full_roi_polygon=packet.full_roi_polygon,
                )
                # [DEBUG] Log tracker observations
                for obs in obs_list:
                    debug_logger.log_tracker_observation(obs)
                observations.extend(obs_list)

                if packet.frame_index % DETECTION_DEBUG_LOG_INTERVAL == 0:
                    logger.debug("cam=%s frame=%s projected=%s", camera_id, packet.frame_index, len(detections))
                    if PRINT_DETECTION_SUMMARY:
                        logger.info(f"[DETECTION] cam={camera_id} frame={packet.frame_index} projected={len(detections)}")

            timestamp_ms = max(p.timestamp_ms for p in packets.values())

            # Update global registry
            tracks = registry.update(
                observations,
                timestamp_ms=timestamp_ms,
                homography_error_px=homography.error_px,
                sync_ok=sync_ok,
                motion_lookup={},
            )

            # [DEBUG] Optional: log global registry matches – removed for now to avoid errors
            # (We can add later if needed)

            # Run motion analysis on each track
            for track in tracks:
                if track.current_update is None:
                    continue
                camera_id = track.current_update.camera_id
                packet = packets[camera_id]
                canonical_edge_normals = compute_edge_normals(packet.full_roi_polygon)

                # --- FIX: use the per-camera outward vector ---
                outward_vec = outward_vec0 if camera_id == 0 else outward_vec1

                motion = motion_analyzer.analyze(
                    track,
                    packet.full_roi_polygon,
                    np.asarray(canonical_edge_normals, dtype=np.float32),
                    outward_vec,   # <-- now geometry-aware
                )
                track.current_update.outward_motion = motion["outward_motion"]
                track.current_update.inward_motion = motion["inward_motion"]
                track.current_update.displacement_vector = motion["displacement_vector"]
                track.current_update.displacement_magnitude = motion["displacement_magnitude"]
                track.current_update.motion_dot = motion["motion_dot"]
                track.current_update.nearest_edge_index = motion["nearest_edge_index"]
                track.current_update.edge_normal = motion["edge_normal"]

                # [DEBUG] Log motion analysis
                debug_logger.log_motion_analysis(track, motion, track.current_update)

                if PRINT_TRACK_STATUS and packet.frame_index % TRACK_STATUS_INTERVAL == 0:
                    dvec = tuple(float(v) for v in track.current_update.displacement_vector)
                    enormal = tuple(float(v) for v in track.current_update.edge_normal)
                    logger.info(f"[TRACK_STATUS] cam={track.current_update.camera_id} gid={track.global_id} "
                          f"class={track.class_name} state={track.event_state} "
                          f"safe={track.current_update.in_safe_roi} outer={track.current_update.in_outer_roi} "
                          f"outward={track.current_update.outward_motion} inward={track.current_update.inward_motion} "
                          f"mag={track.current_update.displacement_magnitude:.2f} dot={track.current_update.motion_dot:.2f} "
                          f"edge={track.current_update.nearest_edge_index} normal={enormal} disp={dvec} "
                          f"conf={track.current_update.confidence:.2f}")

            # Process events (debug logging is inside EventManager)
            events = event_manager.process_frame(tracks, frame_avg_conf, timestamp_ms)

            # --- Visualisation ---
            original_preview_frames = {}
            analyzed_preview_frames = {}
            overlay_lines = event_manager.overlay_event_lines(timestamp_ms)
            tracks_by_source = {}
            for track in tracks:
                if track.current_update is None:
                    continue
                local_id = track.source_local_ids.get(track.current_update.camera_id)
                if local_id is not None:
                    tracks_by_source[(track.current_update.camera_id, local_id)] = track

            for camera_id, packet in packets.items():
                # Original view
                orig = packet.frame.copy()
                roi_poly = np.asarray((roi0 if camera_id == 0 else roi1)["points"], dtype=np.float32)
                safe_poly = np.asarray((roi0 if camera_id == 0 else roi1)["safe_polygon"], dtype=np.float32)
                orig = draw_polygon(orig, roi_poly, (0, 255, 255), "ROI")
                orig = draw_polygon(orig, safe_poly, (0, 255, 0), "SAFE")
                inv_warp = np.linalg.inv(packet.warp_matrix)
                for obs in observations:
                    if obs.camera_id == camera_id:
                        orig = draw_original_local_track(
                            orig, obs, inverse_warp=inv_warp,
                            linked_track=tracks_by_source.get((camera_id, obs.local_track_id))
                        )
                orig = draw_camera_header(orig, title=f"Camera {camera_id} Original",
                                          frame_index=packet.frame_index,
                                          timestamp_ms=packet.timestamp_ms,
                                          sync_ok=sync_ok,
                                          detections=per_camera_detection_stats[camera_id]["original_projected"])
                orig = draw_event_panel(orig, overlay_lines, top_offset=78)
                # Build current video stats (real-time)
                current_pickups = sum(event_manager.pickup_count.values())
                current_putbacks = sum(event_manager.putback_count.values())
                current_lines = [f"Current video: Pickups {current_pickups}, Putbacks {current_putbacks}"]

                orig = draw_cumulative_overlay(orig, current_lines, [])
                original_preview_frames[camera_id] = orig

                # Warped analysis view
                warped = packet.warped_display_frame.copy()
                warped = draw_polygon(warped, packet.full_roi_polygon, (255, 255, 0), "ROI")
                warped = draw_polygon(warped, packet.safe_roi_polygon, (0, 255, 0), "SAFE ROI")
                warped = draw_roi_edge_normals(warped, packet.full_roi_polygon, compute_edge_normals(packet.full_roi_polygon))
                for track in tracks:
                    if track.current_update and track.current_update.camera_id == camera_id:
                        warped = draw_track_overlay(warped, track)
                warped = draw_camera_header(warped, title=f"Camera {camera_id} Analysis",
                                            frame_index=packet.frame_index,
                                            timestamp_ms=packet.timestamp_ms,
                                            sync_ok=sync_ok,
                                            detections=per_camera_detection_stats[camera_id]["original_projected"])
                warped = draw_event_panel(warped, overlay_lines, top_offset=78)
                analyzed_preview_frames[camera_id] = warped

                if SAVE_ANNOTATED_VIDEO:
                    video_writers[camera_id].write(orig)

            preview_grid = None
            if show_preview or status_callback is not None:
                preview_grid = compose_preview_grid(original_preview_frames, analyzed_preview_frames, panel_size=preview_panel_size)

            if show_preview and preview_grid is not None:
                if not show_preview_window(preview_grid):
                    logger.info("Preview exit requested")
                    completed = False
                    break

            current_idx = packets[0].frame_index if 0 in packets else max(p.frame_index for p in packets.values())
            if status_callback is not None and (frames_processed % callback_every_n_frames == 0 or events):
                status = _build_status(session_id, frames_processed, total_frames, current_idx,
                                       preview_grid, event_manager, events, sync_ok)
                if status_callback(status) is False:
                    logger.info("Callback requested early stop")
                    completed = False
                    break

            for cam in packets:
                metrics_tracker.end_frame(cam)
            if current_idx % 100 == 0:
                metrics_tracker.update_system_metrics()

        session = event_manager.finalize()
        session_path = OUTPUT_PATHS.sessions / f"session_{session_id}_{timestamp_tag}.json"
        session_payload = {
            "session_id": session.session_id,
            "events": session.events,
            "pickup_count": session.pickup_count,
            "putback_count": session.putback_count,
            "net_change": session.net_change,
            "total_pickups": session.total_pickups,
            "total_putbacks": session.total_putbacks,
            "warnings": session.warnings,
        }
        session_path.write_text(json.dumps(session_payload, indent=2), encoding="utf-8")
        logger.info("Session completed. Net change: %s", session.net_change)
        logger.info("Session JSON: %s", session_path)

        metrics_tracker.finalize()
        metrics_tracker.print_summary()
        metrics_summary = _metrics_summary(metrics_tracker)
        net_by_product = _net_inventory_by_product(event_manager)

        # Update cumulative statistics with this session
        video_name = session_id  # or construct from video0/video1 paths
        total_pickups = sum(event_manager.pickup_count.values())
        total_putbacks = sum(event_manager.putback_count.values())
        cumulative_stats.add_session(
            video_name=video_name,
            pickups=total_pickups,
            putbacks=total_putbacks,
            session_id=session_id,
        )
        print(f"\nUpdated cumulative stats: {cumulative_stats.get_summary_line()}")

        # [DEBUG] Close the debug logger to flush remaining rows
        debug_logger.close()

        return PipelineResult(
            session_id=session_id,
            session_path=str(session_path),
            log_path=str(OUTPUT_PATHS.logs / f"{session_id}_{timestamp_tag}.log"),
            total_frames=total_frames,
            frames_processed=frames_processed,
            pickup_total=total_pickups,
            putback_total=total_putbacks,
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
        # Ensure debug logger is closed even if exception
        debug_logger.close()


