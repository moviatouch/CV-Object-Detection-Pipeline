from __future__ import annotations  # Allow forward references in type hints

import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np

from config import DETECTION_MODEL_CONFIDENCE
from utils.types import Detection

# Set up module-level logger
LOGGER = logging.getLogger("vending_pipeline.detector")

# Try to import YOLO from Ultralytics; if missing, set YOLO to None
try:
    from ultralytics import YOLO
except Exception:  
    YOLO = None


def crop_histogram_embedding(frame: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    """
    Extract a 96‑dimensional histogram embedding from the image region defined by bbox.

    Args:
        frame: Input image (BGR format).
        bbox: Bounding box as (x1, y1, x2, y2).

    Returns:
        Flattened normalized histogram (float32) of shape (96,).
    """
    # Convert coordinates to integers and clamp to image boundaries
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], max(x1 + 1, x2))
    y2 = min(frame.shape[0], max(y1 + 1, y2))

    # Crop the region of interest
    crop = frame[y1:y2, x1:x2]

    # If crop is empty, return a zero embedding
    if crop.size == 0:
        return np.zeros(96, dtype=np.float32)

    # Compute 3D histogram: 4 bins for blue, 4 for green, 6 for red → 4*4*6 = 96 bins
    hist = cv2.calcHist([crop], [0, 1, 2], None, [4, 4, 6], [0, 256, 0, 256, 0, 256])

    # Normalize, flatten, and convert to float32
    hist = cv2.normalize(hist, None).flatten().astype(np.float32)
    return hist


class RTDETRDetector:
    """Object detector using an RT‑DETR model via Ultralytics YOLO interface."""

    def __init__(self, model_path: str, device: str = "cpu", conf_threshold: float = DETECTION_MODEL_CONFIDENCE) -> None:
        """
        Initialize the detector.

        Args:
            model_path: Path to the model weights file.
            device: Computation device ("cpu" or "cuda").
            conf_threshold: Minimum confidence score for detections.
        """
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.model = None  # Will be set if model loads successfully

        path = Path(model_path)
        # Load model only if Ultralytics is available and the file exists
        if YOLO is not None and path.exists():
            self.model = YOLO(str(path))
            LOGGER.info("Loaded RT-DETR model from %s with confidence threshold %.2f", path, self.conf_threshold)
        else:
            LOGGER.warning(
                "RT-DETR model unavailable. The pipeline will run with empty detections until a model is provided."
            )

    def detect(
        self,
        frame: np.ndarray,
        *,
        camera_id: int,
        frame_index: int,
        timestamp_ms: float,
    ) -> List[Detection]:
        """
        Run detection on a single frame.

        Args:
            frame: Input image (BGR numpy array).
            camera_id: Identifier of the camera that captured the frame.
            frame_index: Sequential frame number.
            timestamp_ms: Timestamp of the frame in milliseconds.

        Returns:
            List of Detection objects (may be empty if no model or no detections).
        """
        if self.model is None:
            return []  # No model loaded → no detections

        # Run inference; verbose=False suppresses extra output
        results = self.model.predict(frame, verbose=False, device=self.device, conf=self.conf_threshold)

        detections: List[Detection] = []

        for result in results:
            # Get class name mapping (e.g., {0: "person", 1: "bottle"})
            names = getattr(result, "names", {}) or {}
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue  # No bounding boxes in this result

            # Extract detection data as numpy arrays (move from GPU if needed)
            xyxy = boxes.xyxy.cpu().numpy()          # shape (N,4)
            cls = boxes.cls.cpu().numpy().astype(int)  # shape (N,)
            conf = boxes.conf.cpu().numpy()            # shape (N,)

            # Iterate over each detection
            for bbox, class_id, confidence in zip(xyxy, cls, conf):
                bbox_tuple = tuple(float(v) for v in bbox)  # (x1,y1,x2,y2)
                class_name = names.get(int(class_id), str(class_id))

                # Compute centroid from the bounding box
                centroid = np.array(
                    [(bbox_tuple[0] + bbox_tuple[2]) / 2.0, (bbox_tuple[1] + bbox_tuple[3]) / 2.0],
                    dtype=np.float32,
                )

                # Create a Detection object with all required fields
                detections.append(
                    Detection(
                        bbox=bbox_tuple,
                        class_id=int(class_id),
                        class_name=class_name,
                        confidence=float(confidence),
                        embedding=crop_histogram_embedding(frame, bbox_tuple),
                        camera_id=camera_id,
                        frame_index=frame_index,
                        timestamp_ms=timestamp_ms,
                        original_centroid=centroid,
                    )
                )

        return detections