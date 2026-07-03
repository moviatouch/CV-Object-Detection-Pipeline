from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np


import subprocess
from pathlib import Path


from config import DESYNC_TOLERANCE_FRAMES  # Only sync tolerance remains
from utils.roi_utils import as_np, warp_frame
from utils.types import FramePacket


@dataclass
class CameraContext:
    """
    Holds per‑camera state needed during video reading.
    """
    video_path: str
    roi_payload: dict                # ROI definition (polygon, safe polygon, etc.)
    capture: cv2.VideoCapture
    # No prev_gray, no motion_estimator – shake detection removed


class SynchronizedVideoReader:
    """
    Reads two video files in a synchronised manner.

    For each frame, it reads the next frame from both cameras,
    warps each original frame to a top‑down view, and returns
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
        return fps if fps > 0 else 30.0  # fallback

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

            # ----- Image processing: only perspective warp (no CLAHE, no grayscale for detection) -----
            # Warp the original BGR frame to a top‑down view (used for detection)
            warped_detection_frame, matrix = warp_frame(frame, context.roi_payload["points"])
            # Warp the original frame also for display (same transform)
            warped_display_frame, _ = warp_frame(frame, context.roi_payload["points"])

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
                frame=frame,                               # original BGR (for display/overlay)
                warped_frame=warped_detection_frame,       # warped BGR → used for detection
                warped_display_frame=warped_display_frame, # warped BGR for visualisation
                warp_matrix=matrix,
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

    Pure cv2.VideoWriter - no ffmpeg subprocess. NOTE: as established earlier, whether
    this actually produces real H.264 depends entirely on how your OpenCV build's
    FFmpeg backend was compiled (most pip opencv-python wheels do NOT include libx264,
    see prior discussion). Verify with verify_avc1.py on your target machine before
    relying on this for dashboard uploads - if it doesn't work, use the
    ffmpeg-subprocess version (annotated_video_writer.py) instead.
    """

    def __init__(
        self,
        path: str | Path,
        fps: float,
        frame_size: tuple[int, int],
        resize_to: tuple[int, int] | None = None,
        fourcc: str = "mp4v",
    ) -> None:
        """
        Args:
            path: Output file path.
            fps: Frames per second.
            frame_size: (width, height) of the frames you will pass to write().
                This is the ORIGINAL frame size, e.g. straight from your source video.
            resize_to: Optional (width, height) from config, e.g. RESIZE_WIDTH/RESIZE_HEIGHT.
                If set, each frame passed to write() is resized to fit within this box
                (aspect ratio preserved, letterboxed with black padding to hit the exact
                size) before encoding. The output video's actual dimensions become
                resize_to. If None, frames are encoded at frame_size unchanged.
            fourcc: Codec fourcc string. Default "mp4v" (matches original behavior,
                guaranteed to work everywhere but is NOT H.264). Try "avc1" for H.264 -
                verify it actually opens and check the resulting file with ffprobe,
                since it can silently fail or fall back on some builds.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

        self.input_size = frame_size          # (width, height) caller will write() at
        self.resize_to = resize_to            # (width, height) target, or None
        self.output_size = resize_to or frame_size

        fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
        self.writer = cv2.VideoWriter(str(path), fourcc_code, fps, self.output_size)

        if not self.writer.isOpened():
            raise RuntimeError(
                f"Failed to open cv2.VideoWriter with fourcc='{fourcc}' for {path}. "
                "Your OpenCV build likely doesn't support this codec - "
                "check with verify_avc1.py, or use the ffmpeg-subprocess writer instead."
            )

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize a frame to fit within self.resize_to, preserving aspect ratio,
        letterboxed with black padding to hit the exact target size."""
        target_w, target_h = self.resize_to
        h, w = frame.shape[:2]

        scale = min(target_w / w, target_h / h)
        new_w, new_h = max(int(w * scale), 1), max(int(h * scale), 1)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x_off = (target_w - new_w) // 2
        y_off = (target_h - new_h) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return canvas

    def write(self, frame: np.ndarray) -> None:
        """Write a single BGR frame (resized first if resize_to was set)."""
        if self.resize_to is not None:
            frame = self._resize_frame(frame)
        self.writer.write(frame)

    def close(self) -> None:
        """Release the video writer."""
        self.writer.release()


# class AnnotatedVideoWriter:
#     """
#     Simple wrapper for writing annotated output videos (e.g., with drawn tracks and events).

#     Streams frames directly to an ffmpeg subprocess encoding real H.264 (libx264,
#     yuv420p) - no cv2 codec fallback issues, no intermediate file, constant memory
#     regardless of video length.

#     Requires the system `ffmpeg` binary to be installed and on PATH
#     (e.g. `apt-get install -y ffmpeg`), with libx264 support
#     (check via `ffmpeg -codecs | grep libx264`).
#     """

#     def __init__(
#         self,
#         path: str | Path,
#         fps: float,
#         frame_size: tuple[int, int],
#         resize_to: tuple[int, int] | None = None,
#     ) -> None:
#         """
#         Args:
#             path: Output file path.
#             fps: Frames per second.
#             frame_size: (width, height) of the frames you will pass to write().
#                 This is the ORIGINAL frame size, e.g. straight from your source video.
#             resize_to: Optional (width, height) from config, e.g. RESIZE_WIDTH/RESIZE_HEIGHT.
#                 If set, each frame passed to write() is resized to fit within this box
#                 (aspect ratio preserved, letterboxed with black padding to hit the exact
#                 size) before encoding. The output video's actual dimensions become
#                 resize_to. If None, frames are encoded at frame_size unchanged.
#         """
#         path = Path(path)
#         path.parent.mkdir(parents=True, exist_ok=True)
#         self.path = path
#         self._closed = False

#         self.input_size = frame_size          # (width, height) caller will write() at
#         self.resize_to = resize_to            # (width, height) target, or None
#         self.output_size = resize_to or frame_size

#         out_width, out_height = self.output_size

#         cmd = [
#             "ffmpeg", "-y",
#             "-f", "rawvideo",
#             "-vcodec", "rawvideo",
#             "-pix_fmt", "bgr24",          # matches cv2's native frame format
#             "-s", f"{out_width}x{out_height}",
#             "-r", str(fps),
#             "-i", "-",
#             "-an",
#             "-c:v", "libx264",
#             "-pix_fmt", "yuv420p",        # web/browser/dashboard compatible
#             "-movflags", "+faststart",
#             str(path),
#         ]

#         try:
#             self.proc = subprocess.Popen(
#                 cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE
#             )
#         except FileNotFoundError as e:
#             raise RuntimeError(
#                 "ffmpeg binary not found. Install it with `apt-get install -y ffmpeg` "
#                 "(or your distro's equivalent) and ensure it's on PATH."
#             ) from e

#     def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
#         """Resize a frame to fit within self.resize_to, preserving aspect ratio,
#         letterboxed with black padding to hit the exact target size (ffmpeg
#         requires every frame to match the -s dimensions exactly)."""
#         target_w, target_h = self.resize_to
#         h, w = frame.shape[:2]

#         scale = min(target_w / w, target_h / h)
#         new_w, new_h = max(int(w * scale), 1), max(int(h * scale), 1)

#         resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

#         canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
#         x_off = (target_w - new_w) // 2
#         y_off = (target_h - new_h) // 2
#         canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
#         return canvas

#     def write(self, frame: np.ndarray) -> None:
#         """Write a single BGR frame (resized first if resize_to was set)."""
#         if self._closed:
#             raise RuntimeError("Cannot write: writer is already closed.")
#         if self.resize_to is not None:
#             frame = self._resize_frame(frame)
#         self.proc.stdin.write(frame.tobytes())

#     def close(self) -> None:
#         """Flush and release the video writer, raising if ffmpeg failed."""
#         if self._closed:
#             return
#         self._closed = True

#         self.proc.stdin.close()
#         stderr = self.proc.stderr.read()
#         ret = self.proc.wait()

#         if ret != 0:
#             raise RuntimeError(
#                 f"ffmpeg failed (exit code {ret}) writing {self.path}:\n"
#                 f"{stderr.decode(errors='ignore')}"
#             )