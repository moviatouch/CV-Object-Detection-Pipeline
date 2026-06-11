from __future__ import annotations

from pathlib import Path
import logging
from typing import Tuple

import numpy as np

try:
    import torch
    import torchvision
    from torchvision import transforms
except Exception:  # pragma: no cover - optional dependency
    torch = None
    torchvision = None

from detection.rtdetr_wrapper import crop_histogram_embedding

import cv2
from config import MOBILENET_MODEL_PATH

LOGGER = logging.getLogger("vending_pipeline.reid")


class ReIDModel:
    """Simple ReID wrapper around a MobileNetV3 small backbone.

    If PyTorch and torchvision are available and a model file exists at model_path,
    the wrapper will attempt to load the weights (with strict=False). If loading
    fails or torch is unavailable, the wrapper falls back to the histogram
    embedding implemented in the detector (96‑d).
    """

    def __init__(self, model_path: str | Path, device: str = "cpu") -> None:
        self.device = device
        self.model_path = Path(model_path) if model_path is not None else None
        self.model = None
        self.pool = None
        self.transform = None
        self.available = False

        if torch is None or torchvision is None:
            LOGGER.warning("PyTorch/torchvision not available; ReID disabled, using histogram fallback.")
            return

        # Build a MobileNetV3 small backbone and global pooling
        try:
            base = torchvision.models.mobilenet_v3_small(pretrained=False)
            # We'll use the feature extractor portion and an adaptive pool
            self.model = base.features
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            # Preprocessing similar to ImageNet models
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            if self.model_path is not None and self.model_path.exists():
                try:
                    state = torch.load(str(self.model_path), map_location=torch.device(self.device))
                    # Try to load state into the whole base model if possible; allow missing keys
                    try:
                        # If the checkpoint contains a 'model' or 'state_dict' key, pick it
                        if isinstance(state, dict) and ("model" in state or "state_dict" in state):
                            sd = state.get("model", state.get("state_dict", state))
                        else:
                            sd = state
                        # Attempt to load into the full mobilenet to maximize key matches
                        full_model = torchvision.models.mobilenet_v3_small(pretrained=False)
                        full_model.load_state_dict(sd, strict=False)
                        self.model = full_model.features
                        LOGGER.info("Loaded ReID weights into MobileNetV3 backbone (strict=False).")
                    except Exception:
                        LOGGER.warning("Failed to strictly load ReID weights; continuing with backbone and strict=False.")
                        # best-effort: try to load partial keys into features
                        try:
                            self.model.load_state_dict(sd, strict=False)
                        except Exception:
                            LOGGER.debug("Partial load into features also failed; using uninitialized backbone.")

                except Exception:
                    LOGGER.exception("Failed to load ReID checkpoint '%s'; using uninitialized backbone.", self.model_path)

            # Move model to device and set eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            self.pool = self.pool.to(self.device)
            self.available = True
        except Exception:
            LOGGER.exception("Failed to construct MobileNetV3 backbone for ReID; falling back to histogram.")
            self.model = None
            self.pool = None
            self.transform = None
            self.available = False

    def embed_crop(self, frame_bgr: np.ndarray, bbox: Tuple[float, float, float, float]) -> np.ndarray:
        """Compute an embedding for the image crop defined by bbox.

        Returns an L2-normalized float32 numpy vector. If the deep model is not
        available, falls back to the histogram embedding used by the detector
        (96‑dim float32 vector).
        """
        if not self.available:
            return crop_histogram_embedding(frame_bgr, bbox)

        # Crop and convert to RGB
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_bgr.shape[1], max(x1 + 1, x2))
        y2 = min(frame_bgr.shape[0], max(y1 + 1, y2))
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            # Empty crop: return zeros of reasonable size (match histogram fallback shape)
            return crop_histogram_embedding(frame_bgr, bbox)

        # BGR -> RGB
        crop_rgb = crop[:, :, ::-1]

        try:
            tensor = self.transform(crop_rgb)
        except Exception:
            # Fallback if PIL conversion fails for some reason
            crop_resized = cv2.resize(crop_rgb, (224, 224))
            tensor = transforms.ToTensor()(crop_resized)
            tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor)

        tensor = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.model(tensor)
            pooled = self.pool(feats).squeeze()  # (C,)
            emb = pooled.cpu().numpy().astype(np.float32)

        # L2 normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb


# class ReIDModel:
#     """
#     Feature extraction backbone for Appearance Re-Identification.
    
#     Dynamically loads either OSNet (for fine-grained metric learning) or 
#     MobileNetV3 (legacy baseline) based on the REID_MODEL_TYPE config flag.
#     Provides L2-normalized embeddings for cross-camera cosine similarity matching.
#     """

#     def __init__(self, model_path: str | Path = None, device: str = "cpu") -> None:
#         self.device = device
#         self.fallback_model_path = Path(model_path) if model_path is not None else None
#         self.model_type = str(REID_MODEL_TYPE).lower()
#         self.model = None
#         self.pool = None
#         self.transform = None
#         self.available = False

#         if torch is None or torchvision is None:
#             LOGGER.warning("PyTorch/torchvision not available; ReID disabled, using histogram fallback.")
#             return

#         try:
#             if self.model_type == "osnet":
#                 LOGGER.info("Initializing OSNet ReID Backbone from config path.")
#                 self.model = osnet_x1_0(pretrained_path=OSNET_MODEL_PATH)
#                 self.input_size = (256, 256)
#                 self.pool = torch.nn.Identity() # OSNet handles its own pooling
                
#             elif self.model_type == "mobilenet":
#                 LOGGER.info("Initializing MobileNetV3 ReID Backbone.")
#                 base = torchvision.models.mobilenet_v3_small(pretrained=False)
#                 self.model = base.features
#                 self.pool = torch.nn.AdaptiveAvgPool2d(1)
#                 self.input_size = (224, 224)
                
#                 # Load weights for MobileNet if path exists
#                 target_path = MOBILENET_MODEL_PATH if Path(MOBILENET_MODEL_PATH).exists() else self.fallback_model_path
#                 if target_path and Path(target_path).exists():
#                     try:
#                         state = torch.load(str(target_path), map_location=torch.device(self.device))
#                         sd = state.get("model", state.get("state_dict", state)) if isinstance(state, dict) else state
#                         full_model = torchvision.models.mobilenet_v3_small(pretrained=False)
#                         full_model.load_state_dict(sd, strict=False)
#                         self.model = full_model.features
#                     except Exception as e:
#                         LOGGER.warning(f"Failed to load MobileNet weights strictly: {e}")
#             else:
#                 raise ValueError(f"Unknown REID_MODEL_TYPE: {self.model_type}")

#             # Standard ImageNet Transforms
#             self.transform = torchvision.transforms.Compose([
#                 torchvision.transforms.ToPILImage(),
#                 torchvision.transforms.Resize(self.input_size),
#                 torchvision.transforms.ToTensor(),
#                 torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             ])

#             self.model = self.model.to(self.device)
#             self.model.eval()
#             if self.pool is not None:
#                 self.pool = self.pool.to(self.device)
#             self.available = True
#             LOGGER.info(f"Successfully loaded {self.model_type.upper()} ReID model.")

#         except Exception as e:
#             LOGGER.exception(f"Failed to construct {self.model_type.upper()} backbone; falling back to histogram. Error: {e}")
#             self.available = False

#     def embed_crop(self, frame_bgr: np.ndarray, bbox: Tuple[float, float, float, float]) -> np.ndarray:
#         """
#         Compute an embedding for the image crop defined by bbox.
#         Returns an L2-normalized float32 numpy vector.
#         """
#         if not self.available:
#             return crop_histogram_embedding(frame_bgr, bbox)

#         x1, y1, x2, y2 = [int(v) for v in bbox]
#         x1, y1 = max(0, x1), max(0, y1)
#         x2 = min(frame_bgr.shape[1], max(x1 + 1, x2))
#         y2 = min(frame_bgr.shape[0], max(y1 + 1, y2))
        
#         crop = frame_bgr[y1:y2, x1:x2]
#         if crop.size == 0:
#             return crop_histogram_embedding(frame_bgr, bbox)

#         # OpenCV BGR to RGB
#         crop_rgb = crop[:, :, ::-1]

#         try:
#             tensor = self.transform(crop_rgb)
#         except Exception:
#             # Fallback resize if transform fails
#             crop_resized = cv2.resize(crop_rgb, self.input_size)
#             tensor = torchvision.transforms.ToTensor()(crop_resized)
#             tensor = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor)

#         tensor = tensor.unsqueeze(0).to(self.device)
        
#         with torch.no_grad():
#             feats = self.model(tensor)
#             if self.model_type == "mobilenet":
#                 feats = self.pool(feats).squeeze()
            
#             emb = feats.cpu().numpy().flatten().astype(np.float32)

#         # Strict L2 Normalization
#         norm = np.linalg.norm(emb)
#         if norm > 0:
#             emb = emb / norm
            
#         return emb