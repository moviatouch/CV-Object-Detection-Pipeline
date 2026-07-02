from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

# Allow script to be run directly from its directory
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PROJECT_ROOT, ROI_CONFIG_DIR
from multicam.homography import HomographyContext
from utils.roi_utils import ensure_polygon_clockwise, point_in_polygon, roi_payload, save_roi_payload


POINT_ORDER = ["top_left", "top_right", "bottom_right", "bottom_left"]
POINT_COLORS = [
    (255, 0, 0),
    (255, 255, 0),
    (0, 255, 0),
    (0, 128, 255),
]


def path_for_json(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


class PolygonCollector:
    """Interactive mouse-driven polygon collector for a single camera view."""

    def __init__(
        self,
        window_name: str,
        frame: np.ndarray,
        *,
        polygon_name: str,
        guide_polygons: list[tuple[list[tuple[int, int]], tuple[int, int, int], str]] | None = None,
        validator: Callable[[list[tuple[int, int]]], str | None] | None = None,
    ) -> None:
        self.window_name = window_name
        self.polygon_name = polygon_name
        self.original_frame = frame.copy()
        self.guide_polygons = guide_polygons or []
        self.validator = validator
        self.points: list[tuple[int, int]] = []
        self.error_message: str | None = None
        self.frame = self.original_frame.copy()
        self._redraw()

    def _draw_polygon(
        self,
        canvas: np.ndarray,
        points: list[tuple[int, int]],
        *,
        color: tuple[int, int, int],
        prefix: str,
        thickness: int,
    ) -> None:
        if points:
            pts = np.asarray(points, dtype=np.int32)
            if len(points) > 1:
                cv2.polylines(canvas, [pts], len(points) == 4, color, thickness)
            for idx, (x, y) in enumerate(points):
                point_color = POINT_COLORS[idx % len(POINT_COLORS)]
                cv2.circle(canvas, (x, y), 6, point_color, -1)
                cv2.putText(
                    canvas,
                    f"{prefix}{idx}",
                    (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    point_color,
                    2,
                )

    def _redraw(self) -> None:
        self.frame = self.original_frame.copy()
        for points, color, prefix in self.guide_polygons:
            self._draw_polygon(self.frame, points, color=color, prefix=prefix, thickness=2)
        self._draw_polygon(self.frame, self.points, color=(255, 255, 0), prefix="", thickness=2)

    def on_mouse(self, event, x, y, _flags, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or len(self.points) >= 4:
            return
        self.points.append((x, y))
        self.error_message = None
        self._redraw()

    def collect(self) -> list[tuple[int, int]]:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        while True:
            display = self.frame.copy()
            next_label = POINT_ORDER[min(len(self.points), len(POINT_ORDER) - 1)]
            cv2.putText(
                display,
                (
                    f"{self.polygon_name}: click 4 corners in order "
                    "top_left -> top_right -> bottom_right -> bottom_left"
                ),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                display,
                (
                    f"Points: {len(self.points)}/4"
                    + (f" | next: {next_label}" if len(self.points) < 4 else " | press Enter to confirm")
                    + " | r=reset q=quit"
                ),
                (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            if self.error_message:
                cv2.putText(
                    display,
                    self.error_message,
                    (10, 86),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"):
                raise KeyboardInterrupt("ROI drawing cancelled by user.")
            if key == ord("r"):
                self.points.clear()
                self.error_message = None
                self._redraw()
            if key == 13 and len(self.points) == 4:
                ordered_points = [
                    (int(round(x)), int(round(y)))
                    for x, y in ensure_polygon_clockwise(self.points).tolist()
                ]
                if self.validator is not None:
                    self.error_message = self.validator(ordered_points)
                    if self.error_message:
                        continue
                self.points = ordered_points
                self._redraw()
                break
        cv2.destroyWindow(self.window_name)
        return self.points


def get_frame_by_position(video_path: str, position: str) -> tuple[np.ndarray, int]:
    """
    Extract a frame from a video file based on position string.

    Args:
        video_path: Path to the video file.
        position: One of 'first', 'mid', 'last'.

    Returns:
        Requested frame as a BGR numpy array and its 0-based frame index.

    Raises:
        RuntimeError: If the frame cannot be read.
        ValueError: If position is invalid.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise RuntimeError(f"Could not determine frame count for {video_path}")

    if position == "first":
        frame_index = 0
    elif position == "mid":
        frame_index = total_frames // 2
    elif position == "last":
        frame_index = max(0, total_frames - 1)
    else:
        cap.release()
        raise ValueError(f"Invalid position: {position}. Choose 'first', 'mid', or 'last'.")

    # Seek to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")

    return frame, frame_index


def build_safe_roi_validator(outer_points: list[tuple[int, int]]) -> Callable[[list[tuple[int, int]]], str | None]:
    outer_polygon = np.asarray(ensure_polygon_clockwise(outer_points), dtype=np.float32)

    def validator(candidate_points: list[tuple[int, int]]) -> str | None:
        if any(not point_in_polygon(point, outer_polygon) for point in candidate_points):
            return "Safe ROI corners must stay inside the outer ROI. Press r to redraw."
        return None

    return validator


def save_reference_overlay(
    path: Path,
    frame: np.ndarray,
    *,
    camera_label: str,
    roi_points: list[tuple[int, int]],
    safe_points: list[tuple[int, int]],
) -> None:
    canvas = frame.copy()
    roi_pts = np.asarray(roi_points, dtype=np.int32)
    safe_pts = np.asarray(safe_points, dtype=np.int32)
    cv2.polylines(canvas, [roi_pts], True, (0, 180, 255), 3)
    cv2.polylines(canvas, [safe_pts], True, (0, 255, 0), 2)

    for idx, point in enumerate(roi_points):
        color = POINT_COLORS[idx % len(POINT_COLORS)]
        cv2.circle(canvas, point, 8, color, -1)
        cv2.putText(
            canvas,
            f"R{idx}:{POINT_ORDER[idx]}",
            (point[0] + 10, point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    for idx, point in enumerate(safe_points):
        color = POINT_COLORS[idx % len(POINT_COLORS)]
        cv2.circle(canvas, point, 6, color, 2)
        cv2.putText(
            canvas,
            f"S{idx}",
            (point[0] + 10, point[1] + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    cv2.putText(canvas, camera_label, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    cv2.putText(
        canvas,
        "Orange=ROI  Green=SAFE ROI  Matching colors show corresponding point order",
        (20, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"Could not save reference overlay to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw outer ROI and manual safe ROI for both cameras, then save JSON and reference overlays. "
                    "Choose which frame to use for drawing (first, middle, or last)."
    )
    parser.add_argument("--video0", required=True, help="Path to camera 0 video")
    parser.add_argument("--video1", required=True, help="Path to camera 1 video")
    parser.add_argument(
        "--output_dir",
        default=ROI_CONFIG_DIR,
        help="Directory to write ROI JSON files and drawn reference frames",
    )
    parser.add_argument(
        "--frame",
        choices=["first", "mid", "last"],
        default="first",
        help="Which frame to use for ROI drawing: first, middle (mid), or last",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load the requested frames
    frame0, frame_index0 = get_frame_by_position(args.video0, args.frame)
    frame1, frame_index1 = get_frame_by_position(args.video1, args.frame)

    print(f"Using {args.frame.upper()} frame for ROI drawing on both cameras.")

    # Collect outer and safe ROI points for each camera
    collector0 = PolygonCollector("Camera 0 ROI", frame0, polygon_name="Camera 0 ROI")
    points0 = collector0.collect()
    safe_collector0 = PolygonCollector(
        "Camera 0 Safe ROI",
        frame0,
        polygon_name="Camera 0 Safe ROI",
        guide_polygons=[(points0, (0, 180, 255), "R")],
        validator=build_safe_roi_validator(points0),
    )
    safe_points0 = safe_collector0.collect()

    collector1 = PolygonCollector("Camera 1 ROI", frame1, polygon_name="Camera 1 ROI")
    points1 = collector1.collect()
    safe_collector1 = PolygonCollector(
        "Camera 1 Safe ROI",
        frame1,
        polygon_name="Camera 1 Safe ROI",
        guide_polygons=[(points1, (0, 180, 255), "R")],
        validator=build_safe_roi_validator(points1),
    )
    safe_points1 = safe_collector1.collect()

    # Compute homography between the two point sets
    homography = HomographyContext.from_roi_points(points0, points1)

    # Build ROI payloads
    payload0 = roi_payload(
        camera_id=0,
        points=points0,
        safe_points=safe_points0,
        safe_polygon_mode="manual",
        roi_margin_px=None,
        homography_to_other=homography.matrix_01,
    )
    payload1 = roi_payload(
        camera_id=1,
        points=points1,
        safe_points=safe_points1,
        safe_polygon_mode="manual",
        roi_margin_px=None,
        homography_to_other=homography.matrix_10,
    )

    # Add homography quality metadata
    payload0["homography_error_px"] = homography.error_px
    payload1["homography_error_px"] = homography.error_px
    payload0["homography_active"] = homography.active
    payload1["homography_active"] = homography.active
    payload0["point_order"] = POINT_ORDER
    payload1["point_order"] = POINT_ORDER

    # Save JSON files and drawn reference frames
    output_dir = Path(args.output_dir)
    reference_path0 = output_dir / "cam0_reference.jpg"
    reference_path1 = output_dir / "cam1_reference.jpg"
    payload0["source_video"] = args.video0
    payload1["source_video"] = args.video1
    payload0["reference_frame_index"] = frame_index0
    payload1["reference_frame_index"] = frame_index1
    payload0["reference_frame_selector"] = args.frame
    payload1["reference_frame_selector"] = args.frame
    payload0["reference_image"] = path_for_json(reference_path0)
    payload1["reference_image"] = path_for_json(reference_path1)

    save_roi_payload(output_dir / "cam0.json", payload0)
    save_roi_payload(output_dir / "cam1.json", payload1)
    save_reference_overlay(
        reference_path0,
        frame0,
        camera_label=f"Camera 0 | frame {frame_index0}",
        roi_points=points0,
        safe_points=safe_points0,
    )
    save_reference_overlay(
        reference_path1,
        frame1,
        camera_label=f"Camera 1 | frame {frame_index1}",
        roi_points=points1,
        safe_points=safe_points1,
    )

    print(f"Saved ROI files to {output_dir}")
    print(f"Saved reference overlays: {reference_path0.name}, {reference_path1.name}")
    print(f"Homography error: {homography.error_px:.2f} px | active={homography.active}")


if __name__ == "__main__":
    main()
