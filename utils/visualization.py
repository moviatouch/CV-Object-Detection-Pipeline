from __future__ import annotations

from typing import Dict

import cv2
import numpy as np

from vending_pickup_putback.utils.types import GlobalTrack


def _as_int_box(bbox: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    """Convert a float bounding box (x1,y1,x2,y2) to integers for drawing."""
    return tuple(int(v) for v in bbox)


def _put_text_block(
    frame: np.ndarray,
    lines: list[str],
    *,
    origin: tuple[int, int] = (10, 24),
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    """
    Draw a block of text lines on the frame, with a black shadow for contrast.
    """
    x, y = origin
    for line in lines:
        # Draw black outline (shadow) for readability
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3)
        # Draw actual text on top
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1)
        y += 22  # Line spacing


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
    # Default colour: yellow (no significant motion)
    color = (0, 255, 255)
    if update.outward_motion:
        color = (0, 255, 0)      # Green for pickup motion
    elif update.inward_motion:
        color = (0, 0, 255)      # Red for putback motion

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
        # Outward = downward (assuming ROI bottom is outward), inward = upward
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

    # Draw motion trail (centroid history) for this camera
    history = track.motion_history_by_camera.get(update.camera_id)
    if history and len(history) >= 2:
        points = np.asarray(list(history), dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(annotated, [points], False, (0, 200, 255), 2)

    center = tuple(np.asarray(update.motion_centroid, dtype=np.int32))
    color = (0, 255, 255)
    if update.outward_motion:
        color = (0, 255, 0)
    elif update.inward_motion:
        color = (0, 0, 255)
    cv2.circle(annotated, center, 6, color, -1)

    # Draw displacement vector arrow
    displacement = np.asarray(update.displacement_vector, dtype=np.float32)
    disp_norm = np.linalg.norm(displacement)
    if disp_norm > 1e-3:
        tip = np.asarray(update.motion_centroid, dtype=np.float32) + (displacement / disp_norm) * 45.0
        cv2.arrowedLine(annotated, center, tuple(tip.astype(np.int32)), color, 2, tipLength=0.25)

    # Draw nearest edge normal arrow (orange)
    normal = np.asarray(update.edge_normal, dtype=np.float32)
    normal_norm = np.linalg.norm(normal)
    if normal_norm > 1e-3:
        normal_tip = np.asarray(update.motion_centroid, dtype=np.float32) + (normal / normal_norm) * 32.0
        cv2.arrowedLine(annotated, center, tuple(normal_tip.astype(np.int32)), (255, 120, 0), 2, tipLength=0.25)

    label = f"G{track.global_id} {track.class_name} E{update.nearest_edge_index}"
    cv2.putText(annotated, label, (center[0] + 8, max(20, center[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return annotated


def draw_event_history(frame: np.ndarray, lines: list[str], *, origin: tuple[int, int] = (10, 110)) -> np.ndarray:
    """Overlay a list of recent event lines (e.g., pickups/putbacks) on the frame."""
    if not lines:
        return frame
    annotated = frame.copy()
    _put_text_block(annotated, lines, origin=origin, color=(0, 255, 255))
    return annotated


def draw_mode_banner(
    frame: np.ndarray,
    *,
    low_light_mode: bool,
    paused: bool,
) -> np.ndarray:
    """Draw a top banner indicating low‑light adaptation and/or event pause status."""
    annotated = frame.copy()
    messages = []
    if low_light_mode:
        messages.append("LOW LIGHT ADAPT")
    if paused:
        messages.append("EVENTS PAUSED")
    if not messages:
        return annotated
    text = " | ".join(messages)
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 40), (40, 40, 40), -1)
    cv2.putText(annotated, text, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
    return annotated


def draw_camera_header(
    frame: np.ndarray,
    *,
    title: str,
    frame_index: int,
    timestamp_ms: float,
    shaky: bool,
    sync_ok: bool,
    detections: int,
) -> np.ndarray:
    """
    Draw a header with camera metadata: title, frame number, timestamp,
    detection count, shaky flag, and cross‑camera sync status.
    """
    annotated = frame.copy()
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 78), (30, 30, 30), -1)
    lines = [
        f"{title} | frame={frame_index} | t={timestamp_ms/1000.0:.2f}s",
        f"detections={detections} | shaky={'YES' if shaky else 'NO'} | sync={'OK' if sync_ok else 'FROZEN'}",
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