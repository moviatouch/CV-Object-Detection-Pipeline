from __future__ import annotations  # Allow forward references in type hints

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from vending_pickup_putback.config import HOMOGRAPHY_MAX_ERROR


def compute_homography(src_points: Sequence[Sequence[float]], dst_points: Sequence[Sequence[float]]) -> np.ndarray:
    """
    Compute a homography matrix mapping points from source image to destination image.

    Args:
        src_points: List of source points (e.g., ROI corners in camera 0), each (x, y).
        dst_points: Corresponding destination points (e.g., same ROI corners in camera 1).

    Returns:
        3x3 homography matrix (np.float32).

    Raises:
        RuntimeError: If homography computation fails (e.g., insufficient points).
    """
    # Convert to numpy arrays of shape (N,2) and compute homography
    matrix, _ = cv2.findHomography(
        np.asarray(src_points, dtype=np.float32),
        np.asarray(dst_points, dtype=np.float32)
    )
    if matrix is None:
        raise RuntimeError("Unable to compute homography from the provided ROI points.")
    return matrix


def project_point(matrix: np.ndarray, point: Sequence[float]) -> np.ndarray:
    """
    Project a single point from one image plane to the other using a homography.

    Args:
        matrix: 3x3 homography matrix.
        point: (x, y) point in the source image.

    Returns:
        Projected point (x', y') as a numpy array of shape (2,).
    """
    # cv2.perspectiveTransform expects an array of shape (1,1,2)
    pts = np.asarray([[point]], dtype=np.float32)
    projected = cv2.perspectiveTransform(pts, matrix)
    return projected[0, 0]  # Extract the point coordinates


def projection_error(matrix: np.ndarray,
                     src_points: Sequence[Sequence[float]],
                     dst_points: Sequence[Sequence[float]]) -> float:
    """
    Compute the mean reprojection error (in pixels) of a homography.

    Args:
        matrix: 3x3 homography matrix.
        src_points: Source points.
        dst_points: Ground truth destination points.

    Returns:
        Average Euclidean distance between projected source points and destination points.
    """
    # Reshape to shape (N,1,2) required by cv2.perspectiveTransform
    src = np.asarray(src_points, dtype=np.float32).reshape(-1, 1, 2)
    dst = np.asarray(dst_points, dtype=np.float32).reshape(-1, 1, 2)

    # Project source points and compute per‑point Euclidean errors
    projected = cv2.perspectiveTransform(src, matrix)
    errors = np.linalg.norm(projected - dst, axis=2)  # shape (N,)

    # Return the mean error
    return float(np.mean(errors))


@dataclass
class HomographyContext:
    """
    Container for homography transformations between two cameras (camera 0 and camera 1).

    Attributes:
        matrix_01: Homography from camera 0 to camera 1 (3x3), or None if not computed.
        matrix_10: Homography from camera 1 to camera 0 (3x3), or None.
        active: Boolean indicating whether the homography is considered reliable (error <= threshold).
        error_px: Mean reprojection error in pixels.
    """
    matrix_01: np.ndarray | None
    matrix_10: np.ndarray | None
    active: bool = True
    error_px: float = 0.0

    @classmethod
    def from_roi_points(cls,
                        cam0_points: Sequence[Sequence[float]],
                        cam1_points: Sequence[Sequence[float]]) -> "HomographyContext":
        """
        Create a HomographyContext from corresponding ROI points in two camera views.

        Args:
            cam0_points: Points in camera 0 (e.g., shelf corners).
            cam1_points: Corresponding points in camera 1.

        Returns:
            HomographyContext with both matrices, error, and active flag.
        """
        # Compute forward and backward homographies
        matrix_01 = compute_homography(cam0_points, cam1_points)
        matrix_10 = compute_homography(cam1_points, cam0_points)

        # Compute reprojection error (using forward homography on the ROI points)
        error = projection_error(matrix_01, cam0_points, cam1_points)

        # Mark as active only if error is within the allowed maximum
        active = error <= HOMOGRAPHY_MAX_ERROR

        return cls(matrix_01=matrix_01, matrix_10=matrix_10, active=active, error_px=error)