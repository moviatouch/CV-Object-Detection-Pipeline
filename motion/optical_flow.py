from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

# SHAKE_FLOW_THRESHOLD and GlobalMotionEstimator removed


def propagate_bbox_lk(
    prev_gray: np.ndarray | None,
    gray: np.ndarray | None,
    bbox: Sequence[float],
) -> Optional[tuple[float, float, float, float]]:
    """
    Propagate a bounding box from the previous frame to the current frame using
    Lucas‑Kanade optical flow (sparse feature tracking).

    Args:
        prev_gray: Previous grayscale frame.
        gray: Current grayscale frame.
        bbox: Bounding box in (x1, y1, x2, y2) format.

    Returns:
        New bbox (x1, y1, x2, y2) if propagation succeeds, else None.
    """
    # Need both frames
    if prev_gray is None or gray is None:
        return None

    # Convert bbox to integer indices
    x1, y1, x2, y2 = [int(v) for v in bbox]

    # Extract region of interest from previous frame, clamp to image boundaries
    roi = prev_gray[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]

    # Empty ROI → cannot propagate
    if roi.size == 0:
        return None

    # Detect strong corners in the ROI (Shi‑Tomasi)
    points = cv2.goodFeaturesToTrack(roi, maxCorners=20, qualityLevel=0.01, minDistance=3)

    if points is None or len(points) == 0:
        return None

    # Convert points from ROI‑relative to full‑image coordinates
    points[:, 0, 0] += x1
    points[:, 0, 1] += y1

    # Track points to current frame using Lucas‑Kanade optical flow
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, points, None)

    if next_points is None or status is None:
        return None

    # Keep only successfully tracked points (status == 1)
    valid_prev = points[status.flatten() == 1]
    valid_next = next_points[status.flatten() == 1]

    if len(valid_prev) == 0 or len(valid_next) == 0:
        return None

    # Compute displacement vectors for each tracked point
    shifts = (valid_next - valid_prev).reshape(-1, 2)

    if shifts.size == 0:
        return None

    # Median shift (robust to outliers)
    median_shift = np.median(shifts, axis=0)

    if median_shift.shape[0] != 2:
        return None

    dx, dy = float(median_shift[0]), float(median_shift[1])

    # Apply the median shift to all corners of the bbox
    return float(x1 + dx), float(y1 + dy), float(x2 + dx), float(y2 + dy)