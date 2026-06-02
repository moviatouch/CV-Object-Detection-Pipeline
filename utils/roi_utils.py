# from __future__ import annotations  # Allow forward references in type hints

# import json
# from pathlib import Path
# from typing import Dict, Iterable, List, Sequence, Tuple

# import cv2
# import numpy as np

# from config import WARP_SIZE


# def as_np(points: Sequence[Sequence[float]]) -> np.ndarray:
#     """Convert a sequence of points to a float32 numpy array."""
#     return np.asarray(points, dtype=np.float32)


# def order_quadrilateral(points: Sequence[Sequence[float]]) -> np.ndarray:
#     """
#     Canonically order a 4-point ROI as:
#     top-left, top-right, bottom-right, bottom-left.

#     The pipeline assumes a stable corner order for perspective warping,
#     outward-vector inference, and cross-camera homography. This keeps ROI
#     behavior robust even if the user clicks the corners starting from a
#     different point.
#     """
#     pts = as_np(points)
#     if len(pts) != 4:
#         raise ValueError("order_quadrilateral expects exactly 4 points.")

#     sorted_by_y = pts[np.lexsort((pts[:, 0], pts[:, 1]))]
#     top = sorted_by_y[:2][np.argsort(sorted_by_y[:2, 0])]
#     bottom = sorted_by_y[2:][np.argsort(sorted_by_y[2:, 0])]
#     return np.asarray([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)


# def polygon_centroid(points: Sequence[Sequence[float]]) -> np.ndarray:
#     """
#     Compute the centroid of a polygon using image moments.

#     Args:
#         points: Polygon vertices as list of (x, y).

#     Returns:
#         Centroid as (cx, cy).
#     """
#     pts = as_np(points)
#     moments = cv2.moments(pts)
#     if abs(moments["m00"]) < 1e-6:          # Degenerate polygon → fallback to mean
#         return np.mean(pts, axis=0)
#     return np.array(
#         [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
#         dtype=np.float32,
#     )


# def normalize(vector: Sequence[float]) -> np.ndarray:
#     """Normalise a vector to unit length. Returns zero vector if input is zero."""
#     arr = np.asarray(vector, dtype=np.float32)
#     norm = np.linalg.norm(arr)
#     if norm < 1e-6:
#         return np.zeros_like(arr)
#     return arr / norm


# def compute_outward_vector(points: Sequence[Sequence[float]]) -> np.ndarray:
#     """
#     Compute an outward‑pointing vector from the polygon's centroid to the
#     middle of the bottom edge (assumed to be the side facing the customer).
#     """
#     pts = ensure_polygon_clockwise(points)
#     centroid = polygon_centroid(pts)
#     bottom_mid = (pts[2] + pts[3]) / 2.0      # Indices assume a 4‑point polygon
#     return normalize(bottom_mid - centroid)


# def shrink_polygon(points: Sequence[Sequence[float]], margin_px: float) -> np.ndarray:
#     """
#     Shrink a polygon inward by moving each vertex toward the centroid by a
#     given margin (capped at 25% of the original distance).
    
#     This function is kept for compatibility but is no longer used in the pipeline
#     because we now require manual safe polygons.
#     """
#     pts = as_np(points)
#     centroid = polygon_centroid(pts)
#     shrunken = []
#     for point in pts:
#         direction = centroid - point
#         dist = np.linalg.norm(direction)
#         if dist < 1e-6:
#             shrunken.append(point)
#             continue
#         shrunken.append(point + normalize(direction) * min(margin_px, dist * 0.25))
#     return np.asarray(shrunken, dtype=np.float32)


# def expand_polygon(points: Sequence[Sequence[float]], margin_px: float) -> np.ndarray:
#     """Expand a polygon outward by moving each vertex away from the centroid."""
#     pts = as_np(points)
#     centroid = polygon_centroid(pts)
#     expanded = []
#     for point in pts:
#         direction = point - centroid
#         dist = np.linalg.norm(direction)
#         if dist < 1e-6:
#             expanded.append(point)
#             continue
#         expanded.append(point + normalize(direction) * margin_px)
#     return np.asarray(expanded, dtype=np.float32)


# def compute_edge_normals(points: Sequence[Sequence[float]]) -> List[np.ndarray]:
#     """
#     Compute outward‑pointing normals for each edge of a convex polygon.

#     For each edge, two candidate normals (perpendicular to the edge) are
#     considered; the one that points away from the polygon's centroid is chosen.
#     """
#     pts = as_np(points)
#     centroid = polygon_centroid(pts)
#     normals: List[np.ndarray] = []
#     for idx in range(len(pts)):
#         start = pts[idx]
#         end = pts[(idx + 1) % len(pts)]
#         edge = end - start
#         candidate_a = normalize(np.array([-edge[1], edge[0]], dtype=np.float32))
#         candidate_b = -candidate_a
#         midpoint = (start + end) / 2.0
#         reference = midpoint - centroid
#         normal = candidate_a if np.dot(reference, candidate_a) >= 0 else candidate_b
#         normals.append(normal.astype(np.float32))
#     return normals


# def point_in_polygon(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> bool:
#     """Test if a point lies inside a polygon (including boundary)."""
#     return cv2.pointPolygonTest(as_np(polygon), tuple(map(float, point)), False) >= 0


# def nearest_edge_normal(
#     point: Sequence[float],
#     polygon: Sequence[Sequence[float]],
#     normals: Sequence[Sequence[float]],
# ) -> Tuple[int, np.ndarray]:
#     """
#     Find the nearest edge of a polygon to a given point and return the edge index
#     and its outward normal.
#     """
#     pts = as_np(polygon)
#     point_arr = np.asarray(point, dtype=np.float32)
#     min_distance = float("inf")
#     min_index = 0
#     for idx in range(len(pts)):
#         start = pts[idx]
#         end = pts[(idx + 1) % len(pts)]
#         edge = end - start
#         denom = max(np.dot(edge, edge), 1e-6)
#         t = float(np.clip(np.dot(point_arr - start, edge) / denom, 0.0, 1.0))
#         projection = start + t * edge
#         distance = float(np.linalg.norm(point_arr - projection))
#         if distance < min_distance:
#             min_distance = distance
#             min_index = idx
#     return min_index, normalize(normals[min_index])


# def bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
#     """Compute Intersection over Union of two axis‑aligned bounding boxes."""
#     ax1, ay1, ax2, ay2 = box_a
#     bx1, by1, bx2, by2 = box_b
#     ix1 = max(ax1, bx1)
#     iy1 = max(ay1, by1)
#     ix2 = min(ax2, bx2)
#     iy2 = min(ay2, by2)
#     iw = max(0.0, ix2 - ix1)
#     ih = max(0.0, iy2 - iy1)
#     inter = iw * ih
#     area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
#     area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
#     union = area_a + area_b - inter
#     if union <= 0:
#         return 0.0
#     return float(inter / union)


# def cosine_similarity(vec_a: np.ndarray | None, vec_b: np.ndarray | None) -> float:
#     """Compute cosine similarity between two vectors. Returns 0 if either is None."""
#     if vec_a is None or vec_b is None:
#         return 0.0
#     a = np.asarray(vec_a, dtype=np.float32)
#     b = np.asarray(vec_b, dtype=np.float32)
#     denom = np.linalg.norm(a) * np.linalg.norm(b)
#     if denom < 1e-6:
#         return 0.0
#     return float(np.dot(a, b) / denom)


# def angle_between(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
#     """Compute the angle (in degrees) between two vectors."""
#     a = normalize(vec_a)
#     b = normalize(vec_b)
#     dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
#     return float(np.degrees(np.arccos(dot)))


# def warp_frame(frame: np.ndarray, source_polygon: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Warp an entire frame so that the source polygon becomes a rectangle of size
#     WARP_SIZE. Returns the warped image and the perspective transform matrix.
#     """
#     src = ensure_polygon_clockwise(source_polygon)
#     dst = np.array(
#         [[0, 0], [WARP_SIZE[0] - 1, 0], [WARP_SIZE[0] - 1, WARP_SIZE[1] - 1], [0, WARP_SIZE[1] - 1]],
#         dtype=np.float32,
#     )
#     matrix = cv2.getPerspectiveTransform(src, dst)
#     warped = cv2.warpPerspective(frame, matrix, WARP_SIZE)
#     return warped, matrix


# def project_bbox(matrix: np.ndarray, bbox: Sequence[float]) -> tuple[float, float, float, float] | None:
#     """
#     Project a bounding box from one image plane to another using a perspective
#     transform. Returns the projected bbox or None if invalid.
#     """
#     x1, y1, x2, y2 = [float(v) for v in bbox]
#     corners = np.array(
#         [[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]],
#         dtype=np.float32,
#     )
#     projected = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)
#     if not np.isfinite(projected).all():
#         return None
#     px1 = float(np.min(projected[:, 0]))
#     py1 = float(np.min(projected[:, 1]))
#     px2 = float(np.max(projected[:, 0]))
#     py2 = float(np.max(projected[:, 1]))
#     if px2 <= px1 or py2 <= py1:
#         return None
#     return px1, py1, px2, py2


# def transform_point(matrix: np.ndarray, point: Sequence[float]) -> np.ndarray:
#     """Apply a perspective transform to a single point."""
#     pts = np.asarray([[list(point)]], dtype=np.float32)
#     transformed = cv2.perspectiveTransform(pts, matrix)
#     return transformed[0, 0]


# def clip_bbox(bbox: Sequence[float], width: int, height: int) -> tuple[float, float, float, float] | None:
#     """Clip a bounding box to the image boundaries. Returns None if it becomes empty."""
#     x1, y1, x2, y2 = [float(v) for v in bbox]
#     x1 = max(0.0, min(x1, width - 1.0))
#     y1 = max(0.0, min(y1, height - 1.0))
#     x2 = max(0.0, min(x2, width - 1.0))
#     y2 = max(0.0, min(y2, height - 1.0))
#     if x2 <= x1 or y2 <= y1:
#         return None
#     return x1, y1, x2, y2


# def ensure_polygon_clockwise(points: Sequence[Sequence[float]]) -> np.ndarray:
#     """
#     Ensure polygon vertices are ordered clockwise. If they are counter‑clockwise,
#     reverse the order.
#     """
#     pts = as_np(points)
#     if len(pts) == 4:
#         return order_quadrilateral(pts)
#     area = 0.0
#     for idx in range(len(pts)):
#         x1, y1 = pts[idx]
#         x2, y2 = pts[(idx + 1) % len(pts)]
#         area += (x2 - x1) * (y2 + y1)
#     if area > 0:
#         return pts          # already clockwise
#     return pts[::-1]        # reverse to make clockwise


# def roi_payload(
#     camera_id: int,
#     points: Sequence[Sequence[float]],
#     homography_to_other: np.ndarray | None,
#     *,
#     safe_points: Sequence[Sequence[float]],   # required – manual safe polygon
#     safe_polygon_mode: str | None = None,
# ) -> Dict[str, object]:
#     """
#     Create a serialisable dictionary containing all ROI information for one camera.

#     safe_points must be provided (manually drawn safe polygon). 
#     """
#     pts = ensure_polygon_clockwise(points)
#     centroid = polygon_centroid(pts)
#     safe_polygon = ensure_polygon_clockwise(safe_points)   # use manual polygon
#     normals = compute_edge_normals(pts)
#     resolved_safe_mode = safe_polygon_mode or "manual"

#     payload = {
#         "camera_id": camera_id,
#         "points": pts.tolist(),
#         "centroid": centroid.tolist(),
#         "outward_vector": compute_outward_vector(pts).tolist(),
#         "safe_polygon": safe_polygon.tolist(),
#         "safe_polygon_mode": resolved_safe_mode,
#         "edge_normals": [normal.tolist() for normal in normals],
#         "warp_size": list(WARP_SIZE),
#         "homography_to_other": None if homography_to_other is None else np.asarray(homography_to_other).tolist(),
#     }
#     return payload


# def save_roi_payload(path: Path, payload: Dict[str, object]) -> None:
#     """Save ROI payload to a JSON file."""
#     path.parent.mkdir(parents=True, exist_ok=True)
#     path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# def load_roi_payload(path: str | Path) -> Dict[str, object]:
#     """
#     Load ROI payload from JSON. Requires a manually defined 'safe_polygon' in the file.
#     Raises ValueError if missing.
#     """
#     raw_payload = json.loads(Path(path).read_text(encoding="utf-8"))
#     raw_points = raw_payload["points"]
#     raw_safe_polygon = raw_payload.get("safe_polygon")
#     if raw_safe_polygon is None:
#         raise ValueError(f"ROI JSON {path} does not contain a manual 'safe_polygon'. "
#                          "Please run the ROI calibrator to draw the safe inner polygon.")

#     # Rebuild payload using the manual safe polygon (no shrinking)
#     payload = roi_payload(
#         camera_id=int(raw_payload.get("camera_id", 0)),
#         points=raw_points,
#         homography_to_other=raw_payload.get("homography_to_other"),
#         safe_points=raw_safe_polygon,
#         safe_polygon_mode="manual",
#     )
#     # Preserve any extra metadata (homography error, etc.) that might be in the file
#     for key, value in raw_payload.items():
#         if key not in payload:
#             payload[key] = value
#     return payload




from __future__ import annotations  # Allow forward references in type hints

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from config import WARP_SIZE


def as_np(points: Sequence[Sequence[float]]) -> np.ndarray:
    """Convert a sequence of points to a float32 numpy array."""
    return np.asarray(points, dtype=np.float32)


def order_quadrilateral(points: Sequence[Sequence[float]]) -> np.ndarray:
    """
    Canonically order a 4-point ROI as:
    top-left, top-right, bottom-right, bottom-left.

    The pipeline assumes a stable corner order for perspective warping,
    outward-vector inference, and cross-camera homography. This keeps ROI
    behavior robust even if the user clicks the corners starting from a
    different point.
    """
    pts = as_np(points)
    if len(pts) != 4:
        raise ValueError("order_quadrilateral expects exactly 4 points.")

    sorted_by_y = pts[np.lexsort((pts[:, 0], pts[:, 1]))]
    top = sorted_by_y[:2][np.argsort(sorted_by_y[:2, 0])]
    bottom = sorted_by_y[2:][np.argsort(sorted_by_y[2:, 0])]
    return np.asarray([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)


def polygon_centroid(points: Sequence[Sequence[float]]) -> np.ndarray:
    """
    Compute the centroid of a polygon using image moments.

    Args:
        points: Polygon vertices as list of (x, y).

    Returns:
        Centroid as (cx, cy).
    """
    pts = as_np(points)
    moments = cv2.moments(pts)
    if abs(moments["m00"]) < 1e-6:          # Degenerate polygon → fallback to mean
        return np.mean(pts, axis=0)
    return np.array(
        [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
        dtype=np.float32,
    )


def normalize(vector: Sequence[float]) -> np.ndarray:
    """Normalise a vector to unit length. Returns zero vector if input is zero."""
    arr = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm < 1e-6:
        return np.zeros_like(arr)
    return arr / norm


def compute_outward_vector(points: Sequence[Sequence[float]]) -> np.ndarray:
    """
    Compute an outward‑pointing vector from the polygon's centroid to the
    middle of the bottom edge (assumed to be the side facing the customer).
    """
    pts = ensure_polygon_clockwise(points)
    centroid = polygon_centroid(pts)
    bottom_mid = (pts[2] + pts[3]) / 2.0      # Indices assume a 4‑point polygon
    return normalize(bottom_mid - centroid)


def shrink_polygon(points: Sequence[Sequence[float]], margin_px: float) -> np.ndarray:
    """
    Shrink a polygon inward by moving each vertex toward the centroid by a
    given margin (capped at 25% of the original distance).
    
    This function is kept for compatibility but is no longer used in the pipeline
    because we now require manual safe polygons.
    """
    pts = as_np(points)
    centroid = polygon_centroid(pts)
    shrunken = []
    for point in pts:
        direction = centroid - point
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            shrunken.append(point)
            continue
        shrunken.append(point + normalize(direction) * min(margin_px, dist * 0.25))
    return np.asarray(shrunken, dtype=np.float32)


def expand_polygon(points: Sequence[Sequence[float]], margin_px: float) -> np.ndarray:
    """Expand a polygon outward by moving each vertex away from the centroid."""
    pts = as_np(points)
    centroid = polygon_centroid(pts)
    expanded = []
    for point in pts:
        direction = point - centroid
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            expanded.append(point)
            continue
        expanded.append(point + normalize(direction) * margin_px)
    return np.asarray(expanded, dtype=np.float32)


def compute_edge_normals(points: Sequence[Sequence[float]]) -> List[np.ndarray]:
    """
    Compute outward‑pointing normals for each edge of a convex polygon.

    For each edge, two candidate normals (perpendicular to the edge) are
    considered; the one that points away from the polygon's centroid is chosen.
    """
    pts = as_np(points)
    centroid = polygon_centroid(pts)
    normals: List[np.ndarray] = []
    for idx in range(len(pts)):
        start = pts[idx]
        end = pts[(idx + 1) % len(pts)]
        edge = end - start
        candidate_a = normalize(np.array([-edge[1], edge[0]], dtype=np.float32))
        candidate_b = -candidate_a
        midpoint = (start + end) / 2.0
        reference = midpoint - centroid
        normal = candidate_a if np.dot(reference, candidate_a) >= 0 else candidate_b
        normals.append(normal.astype(np.float32))
    return normals


def point_in_polygon(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> bool:
    """Test if a point lies inside a polygon (including boundary)."""
    return cv2.pointPolygonTest(as_np(polygon), tuple(map(float, point)), False) >= 0


def signed_distance_to_polygon(
    point: Sequence[float],
    polygon: Sequence[Sequence[float]],
) -> float:
    """
    Return signed distance from a point to a polygon boundary.

    Positive values are inside the polygon, zero is on the boundary, and
    negative values are outside. OpenCV computes the nearest-edge distance.
    """
    return float(cv2.pointPolygonTest(as_np(polygon), tuple(map(float, point)), True))


def nearest_edge_normal(
    point: Sequence[float],
    polygon: Sequence[Sequence[float]],
    normals: Sequence[Sequence[float]],
) -> Tuple[int, np.ndarray]:
    """
    Find the nearest edge of a polygon to a given point and return the edge index
    and its outward normal.
    """
    pts = as_np(polygon)
    point_arr = np.asarray(point, dtype=np.float32)
    min_distance = float("inf")
    min_index = 0
    for idx in range(len(pts)):
        start = pts[idx]
        end = pts[(idx + 1) % len(pts)]
        edge = end - start
        denom = max(np.dot(edge, edge), 1e-6)
        t = float(np.clip(np.dot(point_arr - start, edge) / denom, 0.0, 1.0))
        projection = start + t * edge
        distance = float(np.linalg.norm(point_arr - projection))
        if distance < min_distance:
            min_distance = distance
            min_index = idx
    return min_index, normalize(normals[min_index])


def bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute Intersection over Union of two axis‑aligned bounding boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def cosine_similarity(vec_a: np.ndarray | None, vec_b: np.ndarray | None) -> float:
    """Compute cosine similarity between two vectors. Returns 0 if either is None."""
    if vec_a is None or vec_b is None:
        return 0.0
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-6:
        return 0.0
    return float(np.dot(a, b) / denom)


def angle_between(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute the angle (in degrees) between two vectors."""
    a = normalize(vec_a)
    b = normalize(vec_b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def warp_frame(frame: np.ndarray, source_polygon: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
    """
    Warp an entire frame so that the source polygon becomes a rectangle of size
    WARP_SIZE. Returns the warped image and the perspective transform matrix.
    """
    src = ensure_polygon_clockwise(source_polygon)
    dst = np.array(
        [[0, 0], [WARP_SIZE[0] - 1, 0], [WARP_SIZE[0] - 1, WARP_SIZE[1] - 1], [0, WARP_SIZE[1] - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame, matrix, WARP_SIZE)
    return warped, matrix


def project_bbox(matrix: np.ndarray, bbox: Sequence[float]) -> tuple[float, float, float, float] | None:
    """
    Project a bounding box from one image plane to another using a perspective
    transform. Returns the projected bbox or None if invalid.
    """
    x1, y1, x2, y2 = [float(v) for v in bbox]
    corners = np.array(
        [[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]],
        dtype=np.float32,
    )
    projected = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)
    if not np.isfinite(projected).all():
        return None
    px1 = float(np.min(projected[:, 0]))
    py1 = float(np.min(projected[:, 1]))
    px2 = float(np.max(projected[:, 0]))
    py2 = float(np.max(projected[:, 1]))
    if px2 <= px1 or py2 <= py1:
        return None
    return px1, py1, px2, py2


def transform_point(matrix: np.ndarray, point: Sequence[float]) -> np.ndarray:
    """Apply a perspective transform to a single point."""
    pts = np.asarray([[list(point)]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pts, matrix)
    return transformed[0, 0]


def clip_bbox(bbox: Sequence[float], width: int, height: int) -> tuple[float, float, float, float] | None:
    """Clip a bounding box to the image boundaries. Returns None if it becomes empty."""
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = max(0.0, min(x1, width - 1.0))
    y1 = max(0.0, min(y1, height - 1.0))
    x2 = max(0.0, min(x2, width - 1.0))
    y2 = max(0.0, min(y2, height - 1.0))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def ensure_polygon_clockwise(points: Sequence[Sequence[float]]) -> np.ndarray:
    """
    Ensure polygon vertices are ordered clockwise. If they are counter‑clockwise,
    reverse the order.
    """
    pts = as_np(points)
    if len(pts) == 4:
        return order_quadrilateral(pts)
    area = 0.0
    for idx in range(len(pts)):
        x1, y1 = pts[idx]
        x2, y2 = pts[(idx + 1) % len(pts)]
        area += (x2 - x1) * (y2 + y1)
    if area > 0:
        return pts          # already clockwise
    return pts[::-1]        # reverse to make clockwise


def roi_payload(
    camera_id: int,
    points: Sequence[Sequence[float]],
    homography_to_other: np.ndarray | None,
    *,
    safe_points: Sequence[Sequence[float]],   # required – manual safe polygon
    safe_polygon_mode: str | None = None,
) -> Dict[str, object]:
    """
    Create a serialisable dictionary containing all ROI information for one camera.

    safe_points must be provided (manually drawn safe polygon). 
    """
    pts = ensure_polygon_clockwise(points)
    centroid = polygon_centroid(pts)
    safe_polygon = ensure_polygon_clockwise(safe_points)   # use manual polygon
    normals = compute_edge_normals(pts)
    resolved_safe_mode = safe_polygon_mode or "manual"

    payload = {
        "camera_id": camera_id,
        "points": pts.tolist(),
        "centroid": centroid.tolist(),
        "outward_vector": compute_outward_vector(pts).tolist(),
        "safe_polygon": safe_polygon.tolist(),
        "safe_polygon_mode": resolved_safe_mode,
        "edge_normals": [normal.tolist() for normal in normals],
        "warp_size": list(WARP_SIZE),
        "homography_to_other": None if homography_to_other is None else np.asarray(homography_to_other).tolist(),
    }
    return payload


def save_roi_payload(path: Path, payload: Dict[str, object]) -> None:
    """Save ROI payload to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_roi_payload(path: str | Path) -> Dict[str, object]:
    """
    Load ROI payload from JSON. Requires a manually defined 'safe_polygon' in the file.
    Raises ValueError if missing.
    """
    raw_payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_points = raw_payload["points"]
    raw_safe_polygon = raw_payload.get("safe_polygon")
    if raw_safe_polygon is None:
        raise ValueError(f"ROI JSON {path} does not contain a manual 'safe_polygon'. "
                         "Please run the ROI calibrator to draw the safe inner polygon.")

    # Rebuild payload using the manual safe polygon (no shrinking)
    payload = roi_payload(
        camera_id=int(raw_payload.get("camera_id", 0)),
        points=raw_points,
        homography_to_other=raw_payload.get("homography_to_other"),
        safe_points=raw_safe_polygon,
        safe_polygon_mode="manual",
    )
    # Preserve any extra metadata (homography error, etc.) that might be in the file
    for key, value in raw_payload.items():
        if key not in payload:
            payload[key] = value
    return payload
