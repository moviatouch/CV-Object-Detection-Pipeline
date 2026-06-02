from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from utils.types import GlobalTrack, TrackObservation


def _as_int_box(bbox: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    """Convert a float bounding box (x1,y1,x2,y2) to integers for drawing."""
    return tuple(int(v) for v in bbox)


def _motion_color(*, outward_motion: bool, inward_motion: bool) -> tuple[int, int, int]:
    """Return a consistent overlay color for motion state."""
    if outward_motion:
        return (0, 255, 0)
    if inward_motion:
        return (0, 90, 255)
    return (0, 255, 255)


def _event_color(line: str) -> tuple[int, int, int]:
    """Highlight pickup and putback lines with distinct colors."""
    upper = line.upper()
    if "PICKUP" in upper:
        return (80, 255, 120)
    if "PUTBACK" in upper:
        return (0, 180, 255)
    return (0, 255, 255)


def _put_text_block(
    frame: np.ndarray,
    lines: list[str],
    *,
    origin: tuple[int, int] = (10, 24),
    color: tuple[int, int, int] = (255, 255, 255),
    font_scale: float = 0.58,
    thickness: int = 1,
    shadow_thickness: int | None = None,
    line_spacing: int = 22,
) -> None:
    """
    Draw a block of text lines on the frame, with a black shadow for contrast.
    """
    shadow = max(2, thickness + 2) if shadow_thickness is None else shadow_thickness
    x, y = origin
    for line in lines:
        # Draw black outline (shadow) for readability
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), shadow)
        # Draw actual text on top
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        y += line_spacing


def _project_bbox_polygon(matrix: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray | None:
    """Project bbox corners through a homography and return the resulting quadrilateral."""
    x1, y1, x2, y2 = [float(v) for v in bbox]
    corners = np.array(
        [[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]],
        dtype=np.float32,
    )
    projected = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)
    if not np.isfinite(projected).all():
        return None
    return projected.astype(np.float32)


def draw_polygon(frame: np.ndarray, polygon: np.ndarray, color: tuple[int, int, int], label: str | None = None) -> np.ndarray:
    """
    Draw a polygon (e.g., ROI) on a copy of the frame, optionally with a label.
    """
    annotated = frame.copy()
    pts = polygon.reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(annotated, [pts], True, color, 2)
    if label is not None:
        anchor = tuple(pts[0, 0])
        cv2.putText(annotated, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return annotated


def draw_track_overlay(frame: np.ndarray, track: GlobalTrack, last_event_text: str = "") -> np.ndarray:
    """
    Draw a global track on the frame:
    - Bounding box (colour indicates motion direction: green=outward, red=inward, yellow=static)
    - Track ID, class, confidence, state
    - Arrow showing motion direction (if moving)
    - Last event text (e.g., "PICKUP Coke G1")
    """
    annotated = frame.copy()
    update = track.current_update
    if update is None:
        return annotated

    x1, y1, x2, y2 = _as_int_box(update.bbox)
    color = _motion_color(
        outward_motion=update.outward_motion,
        inward_motion=update.inward_motion,
    )

    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

    label = (
        f"G{track.global_id} {track.class_name} "
        f"{update.confidence:.2f} {track.event_state}/{track.lifecycle_state}"
    )
    cv2.putText(annotated, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Draw arrow from centroid in motion direction (if moving)
    c_x, c_y = (int(update.centroid[0]), int(update.centroid[1]))
    end_x, end_y = c_x, c_y
    if update.outward_motion or update.inward_motion:
        direction = np.array([0.0, 1.0]) if update.outward_motion else np.array([0.0, -1.0])
        end = update.centroid + direction * 40.0
        end_x, end_y = int(end[0]), int(end[1])
        cv2.arrowedLine(annotated, (c_x, c_y), (end_x, end_y), color, 2, tipLength=0.25)

    if last_event_text:
        _put_text_block(annotated, [last_event_text], origin=(10, 25))
    return annotated


def draw_roi_edge_normals(
    frame: np.ndarray,
    polygon: np.ndarray,
    edge_normals: np.ndarray,
    *,
    color: tuple[int, int, int] = (255, 80, 255),
    arrow_length: int = 40,
) -> np.ndarray:
    """
    Draw outward normals from the midpoints of each polygon edge.
    Useful for visualising the ROI geometry used in motion analysis.
    """
    annotated = frame.copy()
    pts = polygon.astype(np.float32)
    normals = np.asarray(edge_normals, dtype=np.float32)
    for idx in range(len(pts)):
        start = pts[idx]
        end = pts[(idx + 1) % len(pts)]
        midpoint = ((start + end) / 2.0).astype(np.int32)
        tip = (midpoint + normals[idx] * arrow_length).astype(np.int32)
        cv2.arrowedLine(annotated, tuple(midpoint), tuple(tip), color, 2, tipLength=0.25)
        cv2.putText(
            annotated,
            f"E{idx}",
            (int(midpoint[0]) + 4, int(midpoint[1]) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    return annotated


def draw_original_track_motion(frame: np.ndarray, track: GlobalTrack) -> np.ndarray:
    """
    Draw motion analysis information on the original (unwarped) view:
    - Motion trail (centroid history)
    - Motion centroid as a filled circle
    - Arrow for displacement vector
    - Arrow for the nearest edge normal
    - Edge index and track label
    """
    annotated = frame.copy()
    update = track.current_update
    if update is None:
        return annotated

    history = track.motion_history_by_camera.get(update.camera_id)
    if history and len(history) >= 2:
        points = np.asarray(list(history), dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(annotated, [points], False, (0, 200, 255), 2)

    center = tuple(np.asarray(update.motion_centroid, dtype=np.int32))
    color = _motion_color(
        outward_motion=update.outward_motion,
        inward_motion=update.inward_motion,
    )
    cv2.circle(annotated, center, 6, color, -1)

    displacement = np.asarray(update.displacement_vector, dtype=np.float32)
    disp_norm = np.linalg.norm(displacement)
    if disp_norm > 1e-3:
        tip = np.asarray(update.motion_centroid, dtype=np.float32) + (displacement / disp_norm) * 45.0
        cv2.arrowedLine(annotated, center, tuple(tip.astype(np.int32)), color, 2, tipLength=0.25)

    normal = np.asarray(update.edge_normal, dtype=np.float32)
    normal_norm = np.linalg.norm(normal)
    if normal_norm > 1e-3:
        normal_tip = np.asarray(update.motion_centroid, dtype=np.float32) + (normal / normal_norm) * 32.0
        cv2.arrowedLine(annotated, center, tuple(normal_tip.astype(np.int32)), (255, 120, 0), 2, tipLength=0.25)

    label = f"G{track.global_id} {track.class_name} E{update.nearest_edge_index}"
    cv2.putText(annotated, label, (center[0] + 8, max(20, center[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return annotated


def draw_original_local_track(
    frame: np.ndarray,
    observation: TrackObservation,
    *,
    inverse_warp: np.ndarray,
    linked_track: GlobalTrack | None = None,
) -> np.ndarray:
    """
    Draw a local track in the original camera view using the exact original-frame
    detector bbox when one is available.
    """
    annotated = frame.copy()
    update = linked_track.current_update if linked_track is not None else None
    outward_motion = bool(update.outward_motion) if update is not None else False
    inward_motion = bool(update.inward_motion) if update is not None else False
    color = _motion_color(
        outward_motion=outward_motion,
        inward_motion=inward_motion,
    )

    render_bbox = observation.display_bbox
    if render_bbox is None:
        return annotated

    x1, y1, x2, y2 = _as_int_box(render_bbox)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

    label_lines = [
        f"{observation.class_name} {observation.confidence:.2f} | L{observation.local_track_id}",
    ]
    if linked_track is not None:
        label_lines.append(f"G{linked_track.global_id} | {linked_track.event_state}")
    label_y = y1 - 12
    if label_y < 150:
        label_y = min(annotated.shape[0] - 24, y2 + 24)
    if label_y >= annotated.shape[0] - (24 * len(label_lines)):
        label_y = max(28, y1 - 12)
    _put_text_block(
        annotated,
        label_lines,
        origin=(max(10, x1), label_y),
        color=color,
        font_scale=0.58,
        thickness=2,
        line_spacing=24,
    )
    return annotated


def draw_event_history(frame: np.ndarray, lines: list[str], *, origin: tuple[int, int] = (10, 110)) -> np.ndarray:
    """Overlay a list of recent event lines (e.g., pickups/putbacks) on the frame."""
    if not lines:
        return frame
    annotated = frame.copy()
    _put_text_block(annotated, lines, origin=origin, color=(0, 255, 255))
    return annotated


def draw_event_panel(
    frame: np.ndarray,
    lines: list[str],
    *,
    top_offset: int = 78,
) -> np.ndarray:
    """Draw a more legible event panel beneath the camera header."""
    if not lines or top_offset >= frame.shape[0]:
        return frame

    annotated = frame.copy()
    recent_lines = lines[-3:]
    latest_line = recent_lines[-1]
    history_lines = list(reversed(recent_lines[:-1]))

    panel_height = 66 + (24 * len(history_lines))
    bottom = min(frame.shape[0] - 1, top_offset + panel_height)

    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, top_offset), (annotated.shape[1], bottom), (22, 22, 22), -1)
    cv2.addWeighted(overlay, 0.74, annotated, 0.26, 0, annotated)

    _put_text_block(
        annotated,
        ["Recent Events"],
        origin=(10, top_offset + 20),
        color=(230, 230, 230),
        font_scale=0.52,
        thickness=1,
        line_spacing=18,
    )
    _put_text_block(
        annotated,
        [latest_line],
        origin=(10, top_offset + 48),
        color=_event_color(latest_line),
        font_scale=0.82,
        thickness=2,
        line_spacing=30,
    )
    if history_lines:
        for idx, line in enumerate(history_lines, start=1):
            _put_text_block(
                annotated,
                [line],
                origin=(10, top_offset + 48 + (idx * 24)),
                color=_event_color(line),
                font_scale=0.6,
                thickness=1,
                line_spacing=22,
            )
    return annotated


def draw_mode_banner(
    frame: np.ndarray,
    *,
    low_light_mode: bool = False,
    paused: bool = False,
) -> np.ndarray:
    """
    No‑op function: mode banner is removed (shake detection and low‑light adaptation gone).
    Kept for compatibility; always returns the frame unchanged.
    """
    return frame


def draw_camera_header(
    frame: np.ndarray,
    *,
    title: str,
    frame_index: int,
    timestamp_ms: float,
    sync_ok: bool,
    detections: int,
) -> np.ndarray:
    """
    Draw a header with camera metadata: title, frame number, timestamp,
    detection count, and cross‑camera sync status.
    """
    annotated = frame.copy()
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 78), (30, 30, 30), -1)
    lines = [
        f"{title} | frame={frame_index} | t={timestamp_ms/1000.0:.2f}s",
        f"detections={detections} | sync={'OK' if sync_ok else 'FROZEN'}",
    ]
    _put_text_block(annotated, lines, origin=(10, 24), color=(230, 230, 230))
    return annotated


def compose_preview_grid(
    original_frames: Dict[int, np.ndarray],
    analyzed_frames: Dict[int, np.ndarray],
    *,
    panel_size: tuple[int, int],
) -> np.ndarray:
    """
    Compose a 2×2 grid:
    - Top row: original frames from camera 0 (left) and camera 1 (right)
    - Bottom row: analysed (annotated) frames from camera 0 and camera 1
    Missing frames are replaced by placeholder text panels.
    """
    def get_panel(source: Dict[int, np.ndarray], camera_id: int, fallback_text: str) -> np.ndarray:
        if camera_id in source:
            return cv2.resize(source[camera_id], panel_size, interpolation=cv2.INTER_AREA)
        panel = np.zeros((panel_size[1], panel_size[0], 3), dtype=np.uint8)
        _put_text_block(panel, [fallback_text], origin=(20, panel_size[1] // 2), color=(220, 220, 220))
        return panel

    top = np.hstack(
        [
            get_panel(original_frames, 0, "Camera 0 unavailable"),
            get_panel(original_frames, 1, "Camera 1 unavailable"),
        ]
    )
    bottom = np.hstack(
        [
            get_panel(analyzed_frames, 0, "Analysis 0 unavailable"),
            get_panel(analyzed_frames, 1, "Analysis 1 unavailable"),
        ]
    )
    return np.vstack([top, bottom])


# ═══════════════════════════════════════════════════════════════════════════
# ROI Visualization Integration
# ═══════════════════════════════════════════════════════════════════════════

def draw_frame_with_roi_overlay(
    frame: np.ndarray,
    outer_roi: Sequence[Sequence[float]],
    safe_roi: Sequence[Sequence[float]],
    *,
    edge_normals: Optional[List[np.ndarray]] = None,
    outward_vector: Optional[np.ndarray] = None,
    camera_label: str = "",
    show_normals: bool = True,
    show_outward: bool = True,
    show_customer: bool = True,
) -> np.ndarray:
    """
    Draw ROI polygons with edge normals and outward vector on a frame.
    
    Integrated from roi_visualization module for visualization workflows.
    
    Args:
        frame: Input frame.
        outer_roi: Outer ROI polygon points.
        safe_roi: Safe (inner) ROI polygon points.
        edge_normals: Edge normal vectors (computed if None).
        outward_vector: Outward direction vector (computed if None).
        camera_label: Camera identifier.
        show_normals: Draw edge normal arrows.
        show_outward: Draw outward vector.
        show_customer: Draw customer dot.
    
    Returns:
        Frame with ROI overlays.
    """
    from utils.roi_visualization import draw_comprehensive_roi
    
    return draw_comprehensive_roi(
        frame,
        outer_roi,
        safe_roi,
        camera_label=camera_label,
        edge_normals=edge_normals,
        outward_vector=outward_vector,
        show_normals=show_normals,
        show_outward=show_outward,
        show_customer=show_customer,
        show_motion_hints=True,
    )


def draw_roi_diagnostic_frame(
    frame: np.ndarray,
    outer_roi: Sequence[Sequence[float]],
    safe_roi: Sequence[Sequence[float]],
    detections: List[Tuple[Sequence[float], str, float]],
    *,
    edge_normals: Optional[List[np.ndarray]] = None,
    outward_vector: Optional[np.ndarray] = None,
    camera_id: int = 0,
) -> np.ndarray:
    """
    Create a diagnostic frame showing ROI, detections, and motion analysis.
    
    Useful for debugging motion classification and ROI configuration.
    
    Args:
        frame: Input frame.
        outer_roi: Outer ROI polygon.
        safe_roi: Safe ROI polygon.
        detections: List of (bbox, class_name, confidence) tuples.
        edge_normals: Edge normal vectors.
        outward_vector: Outward vector.
        camera_id: Camera identifier.
    
    Returns:
        Diagnostic frame.
    """
    from utils.roi_visualization import (
        draw_comprehensive_roi,
        COLOR_NEUTRAL,
        COLOR_OUTWARD,
        COLOR_INWARD,
    )
    
    annotated = draw_comprehensive_roi(
        frame,
        outer_roi,
        safe_roi,
        camera_label=f"Camera {camera_id} - Diagnostic",
        edge_normals=edge_normals,
        outward_vector=outward_vector,
    )
    
    # Draw detections
    for bbox, class_name, confidence in detections:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_NEUTRAL, 2)
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLOR_NEUTRAL,
            2,
        )
    
    return annotated