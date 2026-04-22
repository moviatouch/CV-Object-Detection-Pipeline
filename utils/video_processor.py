from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from vending_pickup_putback.config import (
    ANNOTATION_FPS_FALLBACK,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID,
    DESYNC_TOLERANCE_FRAMES,
)
from vending_pickup_putback.motion.optical_flow import GlobalMotionEstimator
from vending_pickup_putback.utils.roi_utils import as_np, warp_frame
from vending_pickup_putback.utils.types import FramePacket


@dataclass
class CameraContext:
    """
    Holds per‑camera state needed during video reading.
    """
    video_path: str
    roi_payload: dict                # ROI definition (polygon, safe polygon, etc.)
    capture: cv2.VideoCapture
    prev_gray: Optional[np.ndarray] = None
    motion_estimator: GlobalMotionEstimator = field(default_factory=GlobalMotionEstimator)


class SynchronizedVideoReader:
    """
    Reads two video files in a synchronised manner.

    For each frame, it reads the next frame from both cameras (if available),
    preprocesses each frame (CLAHE, warping, shake detection), and returns
    a pair of FramePacket objects together with a sync status.
    """

    def __init__(self, video0: str, video1: str, roi0: dict, roi1: dict) -> None:
        """
        Args:
            video0: Path to first camera's video.
            video1: Path to second camera's video.
            roi0: ROI payload for camera 0 (from load_roi_payload).
            roi1: ROI payload for camera 1.
        """
        self.contexts: Dict[int, CameraContext] = {
            0: CameraContext(video_path=video0, roi_payload=roi0, capture=cv2.VideoCapture(video0)),
            1: CameraContext(video_path=video1, roi_payload=roi1, capture=cv2.VideoCapture(video1)),
        }
        # Verify both videos opened successfully
        for camera_id, context in self.contexts.items():
            if not context.capture.isOpened():
                raise RuntimeError(f"Could not open video for camera {camera_id}: {context.video_path}")

    def close(self) -> None:
        """Release both video captures."""
        for context in self.contexts.values():
            context.capture.release()

    def fps(self, camera_id: int) -> float:
        """Return the frames per second of the given camera's video."""
        fps = float(self.contexts[camera_id].capture.get(cv2.CAP_PROP_FPS))
        return fps if fps > 0 else ANNOTATION_FPS_FALLBACK

    def frame_size(self, camera_id: int) -> tuple[int, int]:
        """Return the (width, height) of the video for the given camera."""
        capture = self.contexts[camera_id].capture
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            return (1280, 720)          # fallback if property read fails
        return (width, height)

    def _timestamp_ms(self, capture: cv2.VideoCapture, frame_index: int, fps: float) -> float:
        """
        Compute timestamp in milliseconds for a frame.
        Prefers the video's native POS_MSEC, otherwise estimates from frame index and fps.
        """
        position_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
        if position_ms > 0:
            return position_ms
        return (frame_index / fps) * 1000.0

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply CLAHE to the grayscale and colour versions of the frame.

        Returns:
            - enhanced grayscale (for optical flow)
            - enhanced BGR (for detection)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
        enhanced = clahe.apply(gray)                     # grayscale enhanced

        # Enhance colour frame: convert to LAB, apply CLAHE only on L channel, then back to BGR
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        enhanced_l = clahe.apply(l_channel)
        enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        return enhanced, enhanced_bgr

    def _warp_polygon(self, polygon: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """
        Apply perspective transform to a polygon (e.g., safe ROI polygon)
        using the same homography that warps the full frame.
        """
        pts = polygon.reshape(-1, 1, 2).astype(np.float32)
        warped = cv2.perspectiveTransform(pts, matrix)
        return warped.reshape(-1, 2)

    def next(self) -> Optional[dict]:
        """
        Read the next frame pair from both cameras.

        Returns:
            None if camera 0 has no more frames (end of video).
            Otherwise a dictionary with:
                - "packets": Dict[int, FramePacket] for camera 0 and 1
                - "sync_ok": bool (true if desync ≤ DESYNC_TOLERANCE_FRAMES)
                - "desync_frames": int (absolute difference in frame indices)
        """
        packets: Dict[int, FramePacket] = {}
        frame_indices: Dict[int, int] = {}

        for camera_id, context in self.contexts.items():
            ok, frame = context.capture.read()
            if not ok:
                # If camera 0 fails, stop completely; if camera 1 fails but camera 0 succeeded,
                # return whatever we have (may be incomplete, but caller can handle).
                return None if camera_id == 0 else packets or None

            # Frame index (0‑based)
            frame_index = int(context.capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            fps = self.fps(camera_id)
            timestamp_ms = self._timestamp_ms(context.capture, frame_index, fps)

            # Preprocess: enhanced grayscale and enhanced BGR
            gray, enhanced_bgr = self._preprocess(frame)

            # Detect global shake using optical flow between consecutive frames
            shaky = context.motion_estimator.is_shaky(context.prev_gray, gray)
            context.prev_gray = gray

            # Warp the enhanced frame so that the ROI becomes a rectangle of size WARP_SIZE
            warped_detection_frame, matrix = warp_frame(enhanced_bgr, context.roi_payload["points"])
            warped_display_frame, _ = warp_frame(frame, context.roi_payload["points"])
            warped_gray = cv2.cvtColor(warped_detection_frame, cv2.COLOR_BGR2GRAY)

            # In the warped coordinate system, the ROI becomes the whole image
            full_roi_polygon = np.array(
                [
                    [0, 0],
                    [warped_detection_frame.shape[1] - 1, 0],
                    [warped_detection_frame.shape[1] - 1, warped_detection_frame.shape[0] - 1],
                    [0, warped_detection_frame.shape[0] - 1],
                ],
                dtype=np.float32,
            )
            # Warp the safe (inner) ROI polygon using the same homography
            safe_roi_polygon = self._warp_polygon(as_np(context.roi_payload["safe_polygon"]), matrix)

            packets[camera_id] = FramePacket(
                camera_id=camera_id,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                frame=frame,
                gray=gray,
                enhanced_frame=enhanced_bgr,
                warped_frame=warped_detection_frame,
                warped_display_frame=warped_display_frame,
                warped_gray=warped_gray,
                warp_matrix=matrix,
                shaky=shaky,
                full_roi_polygon=full_roi_polygon,
                safe_roi_polygon=safe_roi_polygon,
            )
            frame_indices[camera_id] = frame_index

        # Check synchronisation between the two cameras
        desync_frames = abs(frame_indices[0] - frame_indices[1])
        return {
            "packets": packets,
            "sync_ok": desync_frames <= DESYNC_TOLERANCE_FRAMES,
            "desync_frames": desync_frames,
        }


class AnnotatedVideoWriter:
    """
    Simple wrapper for writing annotated output videos (e.g., with drawn tracks and events).
    """

    def __init__(self, path: str | Path, fps: float, frame_size: tuple[int, int]) -> None:
        """
        Args:
            path: Output file path.
            fps: Frames per second.
            frame_size: (width, height) of the video.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(path), fourcc, fps, frame_size)
        self.path = path

    def write(self, frame: np.ndarray) -> None:
        """Write a single frame."""
        self.writer.write(frame)

    def close(self) -> None:
        """Release the video writer."""
        self.writer.release()