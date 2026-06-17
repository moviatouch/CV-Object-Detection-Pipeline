from __future__ import annotations

import csv
import hashlib
import logging
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from utils.types import Detection, TrackObservation, TrackUpdate, GlobalTrack


LOGGER = logging.getLogger("debug_logger")


class DebugLogger:
    """
    Ultra‑detailed debug logger that writes one row per stage per frame.
    All columns are included to enable deep forensic analysis.
    """

    def __init__(
        self,
        session_id: str,
        output_dir: Path,
        max_rows_in_memory: int = 10000,
    ):
        self.session_id = session_id
        self.output_path = output_dir / f"debug_{session_id}.csv"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.max_rows_in_memory = max_rows_in_memory
        self._rows: List[Dict[str, Any]] = []
        self._row_count = 0

        # Frame context (shared across all rows of a frame)
        self._frame_context: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _hash_embedding(emb: Optional[np.ndarray]) -> Optional[str]:
        if emb is None or not isinstance(emb, np.ndarray):
            return None
        return hashlib.sha256(emb.astype(np.float32).tobytes()).hexdigest()[:8]

    @staticmethod
    def _list_to_str(lst: Optional[list]) -> Optional[str]:
        if lst is None:
            return None
        return ", ".join(str(v) for v in lst)

    @staticmethod
    def _dict_to_str(d: Optional[dict]) -> Optional[str]:
        if d is None:
            return None
        return "; ".join(f"{k}:{v:.3f}" for k, v in sorted(d.items()) if isinstance(v, (int, float)))

    def _get_frame_context(self) -> Dict[str, Any]:
        return self._frame_context.copy()

    def set_frame_context(
        self,
        camera_id: int,
        frame_index: int,
        timestamp_ms: float,
        sync_ok: bool,
        desync_frames: int,
        homography_error_px: float,
        frame_avg_conf: float,
        total_frames: int,
    ) -> None:
        self._frame_context.update({
            "camera_id": camera_id,
            "frame_number": frame_index,
            "timestamp": datetime.fromtimestamp(timestamp_ms / 1000.0).isoformat(),
            "timestamp_ms": timestamp_ms,
            "sync_ok": sync_ok,
            "desync_frames": desync_frames,
            "homography_error_px": homography_error_px,
            "frame_avg_conf": frame_avg_conf,
            "total_frames": total_frames,
        })

    def _add_row(self, row: Dict[str, Any]) -> None:
        base = self._get_frame_context()
        base.update(row)
        # Keep only columns that are defined in fieldnames
        fieldnames = self._get_all_columns()
        filtered = {k: base.get(k) for k in fieldnames}
        self._rows.append(filtered)
        self._row_count += 1
        if len(self._rows) >= self.max_rows_in_memory:
            self.flush()

    def flush(self) -> None:
        if not self._rows:
            return
        write_header = not self.output_path.exists() or self.output_path.stat().st_size == 0
        with open(self.output_path, "a" if not write_header else "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._get_all_columns())
            if write_header:
                writer.writeheader()
            writer.writerows(self._rows)
        self._rows.clear()

    def close(self) -> None:
        self.flush()

    # -------------------------------------------------------------------------
    # All column names (must include every key used in any row)
    # -------------------------------------------------------------------------
    def _get_all_columns(self) -> List[str]:
        return [
            # ----- Frame context -----
            "timestamp", "camera_id", "frame_number", "timestamp_ms",
            "sync_ok", "desync_frames", "homography_error_px", "frame_avg_conf", "total_frames",

            # ----- Track identifiers -----
            "global_track_id", "local_track_id",

            # ----- Detection raw -----
            "detected_class", "detection_confidence",
            "reid_embedding_hash", "reid_confidence",
            "source_view",
            "original_centroid_x", "original_centroid_y",
            "display_bbox_x1", "display_bbox_y1", "display_bbox_x2", "display_bbox_y2",
            "in_safe_roi_override", "in_outer_roi_override",

            # ----- BBox & centroid (common) -----
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "centroid_x", "centroid_y",

            # ----- Tracker observation -----
            "velocity_x", "velocity_y",
            "motion_centroid_x", "motion_centroid_y",
            "temporal_iou",

            # ----- ROI status -----
            "is_safe", "is_outer", "is_stable",
            "safe_distance",

            # ----- FSM state -----
            "fsm_state",
            "transition_old_state", "transition_new_state", "transition_reason",

            # ----- Motion analysis -----
            "outward_flag", "inward_flag",
            "displacement_magnitude", "motion_dot",
            "displacement_vector_x", "displacement_vector_y",
            "nearest_edge_index",
            "edge_normal_x", "edge_normal_y",

            # ----- Motion history (as string) -----
            "motion_history",

            # ----- Event manager -----
            "event_type", "event_reason",
            "suppressed", "suppression_reason",
            "cooldown_remaining",
            "pending_since_ms",
            "inside_counter",
            "boundary_linger_start_ms",
            "boundary_linger_flagged",
            "was_stable",
            "stability_boosted",
            "hesitation_resets",
            "long_hold_pending",

            # ----- Class identity -----
            "class_locked", "resolved_class",
            "class_drift_flagged",
            "class_scores",          # dict as string
            "class_votes",           # list of recent votes as string
            "confirmation_confidences",  # list as string
            "confirmation_class_votes",  # list as string

            # ----- Thresholds used (for this decision) -----
            "thresholds_used",

            # ----- Ledger & cross‑camera -----
            "ledger_id",
            "cross_camera_match",
            "cross_camera_score",

            # ----- Module identifier (detector/tracker/registry/motion/event) -----
            "module",

            # ----- Additional fields used in some rows -----
            "motion_score",          # used in log_motion_analysis
        ]

    # -------------------------------------------------------------------------
    # Logging methods for each module
    # -------------------------------------------------------------------------

    def log_detection(
        self,
        detection: Detection,
        local_track_id: Optional[int] = None,
        global_track_id: Optional[int] = None,
    ) -> None:
        x1, y1, x2, y2 = detection.bbox
        cx, cy = detection.centroid
        row = {
            "module": "detector",
            "local_track_id": local_track_id,
            "global_track_id": global_track_id,
            "detected_class": detection.class_name,
            "detection_confidence": detection.confidence,
            "reid_embedding_hash": self._hash_embedding(detection.embedding),
            "reid_confidence": None,
            "source_view": detection.source_view,
            "original_centroid_x": detection.original_centroid[0] if detection.original_centroid is not None else None,
            "original_centroid_y": detection.original_centroid[1] if detection.original_centroid is not None else None,
            "display_bbox_x1": detection.display_bbox[0] if detection.display_bbox else None,
            "display_bbox_y1": detection.display_bbox[1] if detection.display_bbox else None,
            "display_bbox_x2": detection.display_bbox[2] if detection.display_bbox else None,
            "display_bbox_y2": detection.display_bbox[3] if detection.display_bbox else None,
            "in_safe_roi_override": detection.in_safe_roi_override,
            "in_outer_roi_override": detection.in_outer_roi_override,
            "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
            "centroid_x": cx, "centroid_y": cy,
        }
        self._add_row(row)

    def log_tracker_observation(
        self,
        obs: TrackObservation,
        global_track: Optional[GlobalTrack] = None,
    ) -> None:
        x1, y1, x2, y2 = obs.bbox
        cx, cy = obs.centroid
        row = {
            "module": "tracker",
            "local_track_id": obs.local_track_id,
            "global_track_id": global_track.global_id if global_track else None,
            "detected_class": obs.class_name,
            "detection_confidence": obs.confidence,
            "reid_embedding_hash": self._hash_embedding(obs.embedding),
            "reid_confidence": None,
            "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
            "centroid_x": cx, "centroid_y": cy,
            "velocity_x": obs.velocity[0] if obs.velocity is not None else None,
            "velocity_y": obs.velocity[1] if obs.velocity is not None else None,
            "motion_centroid_x": obs.motion_centroid[0] if obs.motion_centroid is not None else None,
            "motion_centroid_y": obs.motion_centroid[1] if obs.motion_centroid is not None else None,
            "temporal_iou": obs.temporal_iou,
            "is_safe": obs.in_safe_roi,
            "is_outer": obs.in_outer_roi,
            "is_stable": obs.in_stable_roi,
            "safe_distance": obs.safe_roi_distance if hasattr(obs, "safe_roi_distance") else None,
            "display_bbox_x1": obs.display_bbox[0] if obs.display_bbox else None,
            "display_bbox_y1": obs.display_bbox[1] if obs.display_bbox else None,
            "display_bbox_x2": obs.display_bbox[2] if obs.display_bbox else None,
            "display_bbox_y2": obs.display_bbox[3] if obs.display_bbox else None,
        }
        self._add_row(row)

    def log_global_registry(
        self,
        track: GlobalTrack,
        obs: TrackObservation,  # matched observation
        cross_camera_match: bool = False,
        cross_camera_score: Optional[float] = None,
    ) -> None:
        row = {
            "module": "global_registry",
            "local_track_id": obs.local_track_id,
            "global_track_id": track.global_id,
            "detected_class": obs.class_name,
            "detection_confidence": obs.confidence,
            "bbox_x1": obs.bbox[0], "bbox_y1": obs.bbox[1],
            "bbox_x2": obs.bbox[2], "bbox_y2": obs.bbox[3],
            "centroid_x": obs.centroid[0], "centroid_y": obs.centroid[1],
            "cross_camera_match": cross_camera_match,
            "cross_camera_score": cross_camera_score,
            "fsm_state": track.event_state,
            "class_locked": track.locked_class_name,
            "is_safe": obs.in_safe_roi,
            "is_outer": obs.in_outer_roi,
            "is_stable": obs.in_stable_roi,
            "safe_distance": obs.safe_roi_distance if hasattr(obs, "safe_roi_distance") else None,
        }
        self._add_row(row)

    def log_motion_analysis(
        self,
        track: GlobalTrack,
        motion: dict,
        update: TrackUpdate,
    ) -> None:
        row = {
            "module": "motion_analyzer",
            "global_track_id": track.global_id,
            "detected_class": update.class_name,
            "detection_confidence": update.confidence,
            "centroid_x": update.centroid[0], "centroid_y": update.centroid[1],
            "is_safe": update.in_safe_roi,
            "is_outer": update.in_outer_roi,
            "is_stable": update.in_stable_roi,
            "safe_distance": update.safe_roi_distance,
            "motion_score": motion.get("displacement_magnitude", 0.0),
            "motion_dot": motion.get("motion_dot", 0.0),
            "displacement_vector_x": motion.get("displacement_vector", [0,0])[0],
            "displacement_vector_y": motion.get("displacement_vector", [0,0])[1],
            "nearest_edge_index": motion.get("nearest_edge_index", -1),
            "edge_normal_x": motion.get("edge_normal", [0,0])[0],
            "edge_normal_y": motion.get("edge_normal", [0,0])[1],
            "outward_flag": motion.get("outward_motion", False),
            "inward_flag": motion.get("inward_motion", False),
            "motion_history": self._list_to_str([d for d in track.motion_direction_history]),
            "fsm_state": track.event_state,
            "inside_counter": track.inside_counter,
            "boundary_linger_start_ms": track.boundary_linger_start_ms,
            "boundary_linger_flagged": track.boundary_linger_flagged,
            "was_stable": track.was_stable,
            "stability_boosted": track.stability_boosted,
            "hesitation_resets": track.hesitation_resets,
            "class_locked": track.locked_class_name,
        }
        self._add_row(row)

    def log_event_manager_state(
        self,
        track: GlobalTrack,
        old_state: str,
        new_state: str,
        reason: str,
        timestamp_ms: float,
    ) -> None:
        update = track.current_update
        row = {
            "module": "event_manager",
            "global_track_id": track.global_id,
            "detected_class": update.class_name if update else track.class_name,
            "detection_confidence": update.confidence if update else track.last_confidence,
            "centroid_x": update.centroid[0] if update else None,
            "centroid_y": update.centroid[1] if update else None,
            "fsm_state": new_state,
            "transition_old_state": old_state,
            "transition_new_state": new_state,
            "transition_reason": reason,
            "is_safe": update.in_safe_roi if update else None,
            "is_outer": update.in_outer_roi if update else None,
            "is_stable": update.in_stable_roi if update else None,
            "safe_distance": update.safe_roi_distance if update else None,
            "motion_score": update.displacement_magnitude if update else None,
            "motion_history": self._list_to_str([d for d in track.motion_direction_history]),
            "outward_flag": update.outward_motion if update else None,
            "inward_flag": update.inward_motion if update else None,
            "cooldown_remaining": None,  # filled when applicable
            "pending_since_ms": track.pending_since_ms,
            "inside_counter": track.inside_counter,
            "boundary_linger_start_ms": track.boundary_linger_start_ms,
            "boundary_linger_flagged": track.boundary_linger_flagged,
            "was_stable": track.was_stable,
            "stability_boosted": track.stability_boosted,
            "hesitation_resets": track.hesitation_resets,
            "long_hold_pending": track.long_hold_pending,
            "class_locked": track.locked_class_name,
            "resolved_class": track.resolved_class_name,
            "class_drift_flagged": track.class_drift_flagged,
            "class_scores": None,  # not computed here
            "class_votes": self._list_to_str([v["class"] for v in track.class_votes]),
            "confirmation_confidences": self._list_to_str(track.confirmation_confidences),
            "confirmation_class_votes": self._list_to_str([v["class"] for v in track.confirmation_class_votes]),
            "thresholds_used": None,  # can be added if needed
        }
        self._add_row(row)

    def log_event_fired(
        self,
        track: GlobalTrack,
        event_type: str,  # "pickup" or "putback"
        event: dict,
        timestamp_ms: float,
        suppressed: bool = False,
        suppression_reason: Optional[str] = None,
        cooldown_remaining: Optional[float] = None,
    ) -> None:
        update = track.current_update
        row = {
            "module": "event_manager",
            "global_track_id": track.global_id,
            "event_type": event_type,
            "event_reason": event.get("reason"),
            "suppressed": suppressed,
            "suppression_reason": suppression_reason,
            "cooldown_remaining": cooldown_remaining,
            "detected_class": event.get("detected_class"),
            "detection_confidence": event.get("confidence"),
            "centroid_x": update.centroid[0] if update else None,
            "centroid_y": update.centroid[1] if update else None,
            "fsm_state": track.event_state,
            "is_safe": update.in_safe_roi if update else None,
            "is_outer": update.in_outer_roi if update else None,
            "is_stable": update.in_stable_roi if update else None,
            "safe_distance": update.safe_roi_distance if update else None,
            "motion_score": update.displacement_magnitude if update else None,
            "motion_history": self._list_to_str([d for d in track.motion_direction_history]),
            "outward_flag": update.outward_motion if update else None,
            "inward_flag": update.inward_motion if update else None,
            "pending_since_ms": track.pending_since_ms,
            "inside_counter": track.inside_counter,
            "boundary_linger_start_ms": track.boundary_linger_start_ms,
            "boundary_linger_flagged": track.boundary_linger_flagged,
            "was_stable": track.was_stable,
            "stability_boosted": track.stability_boosted,
            "hesitation_resets": track.hesitation_resets,
            "long_hold_pending": track.long_hold_pending,
            "class_locked": track.locked_class_name,
            "resolved_class": track.resolved_class_name,
            "class_drift_flagged": track.class_drift_flagged,
            "class_scores": self._dict_to_str(event.get("class_scores")),
            "class_votes": self._list_to_str([v["class"] for v in track.class_votes]),
            "confirmation_confidences": self._list_to_str(track.confirmation_confidences),
            "confirmation_class_votes": self._list_to_str([v["class"] for v in track.confirmation_class_votes]),
            "ledger_id": event.get("ledger_id"),
            "thresholds_used": None,
        }
        self._add_row(row)

    def log_blocked_inward(
        self,
        track: GlobalTrack,
        reason: str,
        timestamp_ms: float,
        prev_safe: bool,
        curr_safe: bool,
        prev_outer: bool,
        curr_outer: bool,
        prev_stable: bool,
        curr_stable: bool,
        motion_history: list,
    ) -> None:
        update = track.current_update
        row = {
            "module": "event_manager",
            "global_track_id": track.global_id,
            "event_type": "PICKUP_BLOCKED_INWARD",
            "event_reason": reason,
            "suppressed": True,
            "suppression_reason": "inward motion blocked",
            "detected_class": update.class_name if update else track.class_name,
            "detection_confidence": update.confidence if update else track.last_confidence,
            "centroid_x": update.centroid[0] if update else None,
            "centroid_y": update.centroid[1] if update else None,
            "fsm_state": track.event_state,
            "is_safe": curr_safe,
            "is_outer": curr_outer,
            "is_stable": curr_stable,
            "safe_distance": update.safe_roi_distance if update else None,
            "motion_score": update.displacement_magnitude if update else None,
            "motion_history": self._list_to_str(motion_history),
            "outward_flag": update.outward_motion if update else None,
            "inward_flag": update.inward_motion if update else None,
            "inside_counter": track.inside_counter,
            "boundary_linger_start_ms": track.boundary_linger_start_ms,
            "boundary_linger_flagged": track.boundary_linger_flagged,
            "was_stable": track.was_stable,
            "class_locked": track.locked_class_name,
            "thresholds_used": {
                "prev_safe": prev_safe, "curr_safe": curr_safe,
                "prev_outer": prev_outer, "curr_outer": curr_outer,
                "prev_stable": prev_stable, "curr_stable": curr_stable,
            },
        }
        self._add_row(row)

    def log_cooldown_suppression(
        self,
        track: GlobalTrack,
        cooldown_ms: float,
        elapsed_ms: float,
        timestamp_ms: float,
        reason: str = "cooldown",
    ) -> None:
        update = track.current_update
        row = {
            "module": "event_manager",
            "global_track_id": track.global_id,
            "event_type": "PICKUP_SUPPRESSED",
            "event_reason": reason,
            "suppressed": True,
            "suppression_reason": f"{reason} elapsed={elapsed_ms:.0f}ms cooldown={cooldown_ms:.0f}ms",
            "cooldown_remaining": max(0.0, cooldown_ms - elapsed_ms),
            "detected_class": update.class_name if update else track.class_name,
            "detection_confidence": update.confidence if update else track.last_confidence,
            "centroid_x": update.centroid[0] if update else None,
            "centroid_y": update.centroid[1] if update else None,
            "fsm_state": track.event_state,
            "is_safe": update.in_safe_roi if update else None,
            "is_outer": update.in_outer_roi if update else None,
            "is_stable": update.in_stable_roi if update else None,
            "safe_distance": update.safe_roi_distance if update else None,
            "motion_score": update.displacement_magnitude if update else None,
            "motion_history": self._list_to_str([d for d in track.motion_direction_history]),
            "outward_flag": update.outward_motion if update else None,
            "inward_flag": update.inward_motion if update else None,
            "inside_counter": track.inside_counter,
            "boundary_linger_start_ms": track.boundary_linger_start_ms,
            "boundary_linger_flagged": track.boundary_linger_flagged,
            "was_stable": track.was_stable,
            "class_locked": track.locked_class_name,
            "thresholds_used": {"cooldown_ms": cooldown_ms, "elapsed_ms": elapsed_ms},
        }
        self._add_row(row)