from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow script to be run directly from its directory
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ROI_CONFIG_DIR
from multicam.homography import HomographyContext
from utils.roi_utils import roi_payload, save_roi_payload


class RoiCollector:
    """Interactive mouse‑driven ROI point collector for a single camera view."""

    def __init__(self, window_name: str, frame: np.ndarray) -> None:
        self.window_name = window_name
        self.original_frame = frame.copy()   # Keep original for reset
        self.frame = frame.copy()            # Working copy with drawings
        self.points: list[tuple[int, int]] = []   # Stores clicked (x, y)

    def on_mouse(self, event, x, y, _flags, _param) -> None:
        """Callback for mouse clicks. Adds a point (up to 4) and draws it."""
        if event != cv2.EVENT_LBUTTONDOWN or len(self.points) >= 4:
            return
        self.points.append((x, y))
        cv2.circle(self.frame, (x, y), 5, (0, 255, 0), -1)          # Green dot
        if len(self.points) > 1:
            cv2.line(self.frame, self.points[-2], self.points[-1], (255, 255, 0), 2)  # Yellow edge
        if len(self.points) == 4:
            cv2.line(self.frame, self.points[-1], self.points[0], (255, 255, 0), 2)   # Close polygon

    def collect(self) -> list[tuple[int, int]]:
        """Show the window and collect 4 clicks. Returns points in order clicked."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        while True:
            display = self.frame.copy()
            # Instruction overlay
            cv2.putText(
                display,
                f"Click 4 ROI corners clockwise ({len(self.points)}/4). Press r to reset, q to quit.",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"):
                raise KeyboardInterrupt("ROI drawing cancelled by user.")
            if key == ord("r"):
                self.frame = self.original_frame.copy()
                self.points.clear()
            if key == 13 and len(self.points) == 4:   # Enter key
                break
        cv2.destroyWindow(self.window_name)
        return self.points


def first_frame(path: str) -> np.ndarray:
    """Read and return the first frame of a video file."""
    capture = cv2.VideoCapture(path)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read first frame from {path}")
    return frame


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Draw ROIs for both cameras and save calibration JSON.")
    parser.add_argument("--video0", required=True, help="Path to camera 0 video")
    parser.add_argument("--video1", required=True, help="Path to camera 1 video")
    parser.add_argument("--output_dir", default=ROI_CONFIG_DIR, help="Directory to write ROI JSON files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load first frames
    frame0 = first_frame(args.video0)
    frame1 = first_frame(args.video1)

    # Collect ROI points for each camera
    collector0 = RoiCollector("Camera 0 ROI", frame0)
    points0 = collector0.collect()
    collector1 = RoiCollector("Camera 1 ROI", frame1)
    points1 = collector1.collect()

    # Compute homography between the two sets of points
    homography = HomographyContext.from_roi_points(points0, points1)

    # Build ROI payloads (polygon, centroid, outward vector, safe polygon, normals, etc.)
    payload0 = roi_payload(camera_id=0, points=points0, homography_to_other=homography.matrix_01)
    payload1 = roi_payload(camera_id=1, points=points1, homography_to_other=homography.matrix_10)

    # Add homography quality metadata
    payload0["homography_error_px"] = homography.error_px
    payload1["homography_error_px"] = homography.error_px
    payload0["homography_active"] = homography.active
    payload1["homography_active"] = homography.active

    # Save JSON files
    output_dir = Path(args.output_dir)
    save_roi_payload(output_dir / "cam0.json", payload0)
    save_roi_payload(output_dir / "cam1.json", payload1)

    print(f"Saved ROI files to {output_dir}")
    print(f"Homography error: {homography.error_px:.2f} px | active={homography.active}")


if __name__ == "__main__":
    main()