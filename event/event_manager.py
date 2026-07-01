from __future__ import annotations

import logging
import uuid
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from config import (
    CLASS_RETURN_MAX_GAP_MS,
    CONFIRM_FRAMES,
    CONF_MIN_FRAME,
    CONF_THRESHOLD,
    LONG_BOUNDARY_STABILITY_FRAMES,
    MISSING_PENDING_CONFIRM_MS,
    OVERLAY_EVENT_TTL_MS,
    OVERLAY_MAX_EVENT_LINES,
    POST_PUTBACK_COOLDOWN_MS,
    PRINT_DEBUG_EVENTS,
    QUICK_RETURN_THRESHOLD_MS,
    QUICK_RETURN_MIN_OUTSIDE_FRAMES,
    STABILITY_FRAMES,

    CLASS_VOTE_HISTORY,
    CLASS_LOCK_RATIO,
    CLASS_LOCK_MIN_TOTAL_SCORE,
    CLASS_VOTE_MIN_CONFIDENCE,
    CLASS_VOTE_WEIGHT_STABLE,
    CLASS_VOTE_WEIGHT_SAFE,
    CLASS_VOTE_WEIGHT_OUTER,
    CLASS_VOTE_WEIGHT_OUTSIDE,
    CLASS_VOTE_RECENCY_DECAY,
    CLASS_CONSISTENCY_RATIO,
    UNKNOWN_UNSTABLE_CLASS,
    ENABLE_CLASS_DRIFT_WARNINGS,

    PENDING_LEDGER_MAX_AGE_MS,
    PUTBACK_LEDGER_MATCH_THRESHOLD,
    LEDGER_MATCH_WEIGHT_CLASS,
    LEDGER_MATCH_WEIGHT_TIME,
    LEDGER_MATCH_WEIGHT_TRACK,
    LEDGER_MATCH_WEIGHT_CLASS_CANDIDATE,
    LEDGER_MATCH_WEIGHT_MOTION,
    LEDGER_MATCH_WEIGHT_EMBEDDING,
    LEDGER_MATCH_WEIGHT_SPATIAL,
    LEDGER_SAME_CLASS_SCORE,
    LEDGER_DIFFERENT_CLASS_SCORE,
    LEDGER_SPATIAL_MAX_DISTANCE_PX,
    EMBEDDING_SIM_THRESHOLDS_BY_CLASS,

    PUTBACK_STABLE_FRAMES,
)

from utils.types import SessionSummary


try:
    from debug_logger import DebugLogger
except ImportError:
    DebugLogger = None


LOGGER = logging.getLogger("vending_pipeline.events")


class EventManager:
    """
    Production-hardened pickup/putback EventManager.

    Main goals:
    - prevent false putback at pickup time
    - prevent same-class stable product from closing pickup
    - reduce missed simple pickups/putbacks
    - prevent duplicate events
    - make ledger the source of truth
    - keep existing architecture mostly unchanged
    """

    def __init__(self, session_id: str, debug_logger: Optional[Any] = None):
        self.session = SessionSummary(session_id=session_id)

        self.pickup_count: Dict[str, int] = defaultdict(int)
        self.putback_count: Dict[str, int] = defaultdict(int)
        self.class_pickup_timestamps: Dict[str, Deque[float]] = defaultdict(deque)

        self.pending_pickups: List[dict] = []

        self._frame_net_snapshot: Dict[str, int] = {}
        self._frame_putback_used: Dict[str, int] = defaultdict(int)

        self.last_putback_by_class_ms: Dict[str, float] = {}

        self.min_valid_putback_hold_ms = 500.0

        self.last_event_text = ""
        self.recent_overlay_events: Deque[Tuple[float, str]] = deque(maxlen=20)

        self.debug_logger = debug_logger

    # ---------------------------------------------------------------------
    # Safety / initialization helpers
    # ---------------------------------------------------------------------

    def _ensure_identity_fields(self, track):
        if not hasattr(track, "class_votes") or track.class_votes is None:
            track.class_votes = deque(maxlen=CLASS_VOTE_HISTORY)

        if not hasattr(track, "confirmation_confidences") or track.confirmation_confidences is None:
            track.confirmation_confidences = deque(maxlen=CONFIRM_FRAMES)

        if not hasattr(track, "confirmation_class_votes") or track.confirmation_class_votes is None:
            track.confirmation_class_votes = deque(maxlen=CONFIRM_FRAMES)

        if not hasattr(track, "locked_class_name"):
            track.locked_class_name = None

        if not hasattr(track, "resolved_class_name"):
            track.resolved_class_name = None

        if not hasattr(track, "class_drift_flagged"):
            track.class_drift_flagged = False

        if not hasattr(track, "visual_embedding"):
            track.visual_embedding = None

        if not hasattr(track, "frames_outside_stable"):
            track.frames_outside_stable = 0

        if not hasattr(track, "pickup_cancel_counter"):
            track.pickup_cancel_counter = 0

        if not hasattr(track, "current_stable_counter"):
            track.current_stable_counter = 0

        if not hasattr(track, "active_pickup_ledger_id"):
            track.active_pickup_ledger_id = None

        if not hasattr(track, "candidate_putback_ledger_id"):
            track.candidate_putback_ledger_id = None

        if not hasattr(track, "interaction_cycle_id"):
            track.interaction_cycle_id = 0

        if not hasattr(track, "was_stable"):
            track.was_stable = False

        if not hasattr(track, "pickup_confirmed_ms"):
            track.pickup_confirmed_ms = None

        if not hasattr(track, "last_putback_confirmed_ms"):
            track.last_putback_confirmed_ms = None

        # For the strict INSIDE edge-outward fallback
        if not hasattr(track, "inside_outward_pickup_counter"):
            track.inside_outward_pickup_counter = 0

        # -------------------------------------------------------------
        # FIXED: Force set created_ms to current update timestamp if not set
        # -------------------------------------------------------------
        if not hasattr(track, "created_ms") or track.created_ms is None:
            # Use the first detection timestamp
            if track.current_update is not None:
                track.created_ms = track.current_update.timestamp_ms
            elif hasattr(track, "last_seen_ms") and track.last_seen_ms is not None:
                track.created_ms = track.last_seen_ms
            else:
                track.created_ms = 0.0

        if not hasattr(track, "is_picked"):
            track.is_picked = False

    def _is_valid_class(self, class_name):
        return bool(class_name) and class_name != UNKNOWN_UNSTABLE_CLASS

    def _family_base_class(self, class_name):
        if not class_name:
            return UNKNOWN_UNSTABLE_CLASS

        lower = class_name.lower()

        if lower.startswith("barebells"):
            return "Barebells"
        if lower.startswith("quest chips"):
            return "Quest Chips"
        if lower.startswith("one bar"):
            return "One Bar"
        if lower.startswith("legendary tasty pastry"):
            return "Legendary Tasty Pastry"
        if lower.startswith("aquafina"):
            return "Aquafina"
        if lower.startswith("skullcandy"):
            return "Skullcandy DIME 3"

        return class_name

    def _detected_class(self, track):
        update = track.current_update
        if update is not None and getattr(update, "class_name", None):
            return update.class_name
        return getattr(track, "class_name", None) or UNKNOWN_UNSTABLE_CLASS

    # ---------------------------------------------------------------------
    # Logging
    # ---------------------------------------------------------------------

    def _emit_debug(self, message, level="debug"):
        try:
            getattr(LOGGER, level)(message)
        except Exception:
            LOGGER.debug(message)

        if PRINT_DEBUG_EVENTS:
            print(message)

    def _push_overlay_event(self, timestamp_ms, message):
        self.recent_overlay_events.append((timestamp_ms, message))
        self.last_event_text = message

    def _replace_last_overlay(self, message):
        if self.recent_overlay_events:
            ts, _ = self.recent_overlay_events[-1]
            self.recent_overlay_events[-1] = (ts, message)

    def _log_transition(self, track, old_state, new_state, reason):
        message = (
            f"[TRACK {track.global_id}] {old_state} -> {new_state} | "
            f"event_class={self._event_class(track)} | "
            f"detected_class={self._detected_class(track)} | "
            f"locked_class={getattr(track, 'locked_class_name', None)} | "
            f"resolved_class={getattr(track, 'resolved_class_name', None)} | "
            f"reason={reason}"
        )

        track.last_debug_reason = reason
        self._emit_debug(message, level="debug")

    def _transition(self, track, new_state, reason):
        self._ensure_identity_fields(track)

        if track.event_state == new_state:
            return

        old_state = track.event_state
        track.event_state = new_state

        if new_state == "STABLE_INSIDE":
            track.was_stable = True
            track.is_picked = False
            self._lock_class_if_possible(track, reason="entered STABLE_INSIDE")

        if new_state in {"PICKUP_PENDING", "PUTBACK_PENDING"}:
            track.confirmation_confidences.clear()
            track.confirmation_class_votes.clear()
            # Reset stable counter on entering PUTBACK_PENDING
            if new_state == "PUTBACK_PENDING":
                track.current_stable_counter = 0
        else:
            track.pending_since_ms = None

        self._log_transition(track, old_state, new_state, reason)

        if self.debug_logger is not None and hasattr(self.debug_logger, "log_event_manager_state"):
            timestamp_ms = track.current_update.timestamp_ms if track.current_update else track.last_seen_ms
            self.debug_logger.log_event_manager_state(
                track=track,
                old_state=old_state,
                new_state=new_state,
                reason=reason,
                timestamp_ms=timestamp_ms,
            )

    # ---------------------------------------------------------------------
    # Class voting
    # ---------------------------------------------------------------------

    def _roi_weight_for_vote(self, update):
        if update is None:
            return CLASS_VOTE_WEIGHT_OUTSIDE
        if update.in_stable_roi:
            return CLASS_VOTE_WEIGHT_STABLE
        if update.in_safe_roi:
            return CLASS_VOTE_WEIGHT_SAFE
        if update.in_outer_roi:
            return CLASS_VOTE_WEIGHT_OUTER
        return CLASS_VOTE_WEIGHT_OUTSIDE

    def _append_class_vote(self, track, update):
        self._ensure_identity_fields(track)

        if update is None:
            return

        confidence = float(update.confidence or 0.0)

        if confidence < CLASS_VOTE_MIN_CONFIDENCE:
            return

        detected_class = getattr(update, "class_name", None) or self._detected_class(track)

        track.class_votes.append({
            "class": detected_class,
            "confidence": confidence,
            "timestamp_ms": float(update.timestamp_ms),
            "roi_weight": self._roi_weight_for_vote(update),
            "in_stable_roi": bool(update.in_stable_roi),
            "in_safe_roi": bool(update.in_safe_roi),
            "in_outer_roi": bool(update.in_outer_roi),
        })

    def _compute_weighted_class_vote(self, track, stable_only=False, include_safe=True):
        self._ensure_identity_fields(track)

        votes = list(track.class_votes)

        if stable_only:
            votes = [v for v in votes if v.get("in_stable_roi")]
        elif include_safe:
            votes = [
                v for v in votes
                if v.get("in_stable_roi") or v.get("in_safe_roi") or v.get("in_outer_roi")
            ]

        if not votes:
            return {
                "class": UNKNOWN_UNSTABLE_CLASS,
                "score": 0.0,
                "total_score": 0.0,
                "ratio": 0.0,
                "scores": {},
            }

        scores = defaultdict(float)
        n = len(votes)

        for idx, vote in enumerate(votes):
            age_from_latest = n - 1 - idx
            recency_weight = CLASS_VOTE_RECENCY_DECAY ** age_from_latest
            cls = vote["class"]
            score = (
                float(vote.get("confidence", 0.0))
                * float(vote.get("roi_weight", 1.0))
                * recency_weight
            )
            scores[cls] += score

        total_score = sum(scores.values())

        if total_score <= 0:
            return {
                "class": UNKNOWN_UNSTABLE_CLASS,
                "score": 0.0,
                "total_score": 0.0,
                "ratio": 0.0,
                "scores": dict(scores),
            }

        sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_class, best_score = sorted_scores[0]
        ratio = best_score / total_score

        # Family merge only for consistency support, not aggressive billing rewrite.
        if len(sorted_scores) >= 2:
            runner_up_class, runner_up_score = sorted_scores[1]
            runner_up_ratio = runner_up_score / total_score

            if ratio >= 0.35 and runner_up_ratio >= 0.35:
                base_a = self._family_base_class(best_class)
                base_b = self._family_base_class(runner_up_class)

                if base_a == base_b and base_a != UNKNOWN_UNSTABLE_CLASS:
                    best_score += runner_up_score
                    ratio = best_score / total_score
                    # Keep original SKU candidate as best_class.
                    # Do not replace final SKU with family name here.

        selected_class = (
            best_class
            if total_score >= CLASS_LOCK_MIN_TOTAL_SCORE and ratio >= CLASS_LOCK_RATIO
            else UNKNOWN_UNSTABLE_CLASS
        )

        return {
            "class": selected_class,
            "score": best_score,
            "total_score": total_score,
            "ratio": ratio,
            "scores": dict(scores),
        }

    def _lock_class_if_possible(self, track, reason):
        self._ensure_identity_fields(track)

        if self._is_valid_class(track.locked_class_name):
            return

        result = self._compute_weighted_class_vote(track, stable_only=True)

        if result["class"] == UNKNOWN_UNSTABLE_CLASS:
            result = self._compute_weighted_class_vote(track, stable_only=False, include_safe=True)

        locked_class = result["class"]

        if locked_class == UNKNOWN_UNSTABLE_CLASS:
            track.locked_class_name = UNKNOWN_UNSTABLE_CLASS
            track.resolved_class_name = UNKNOWN_UNSTABLE_CLASS
            self._emit_debug(
                f"[CLASS-LOCK-DELAYED] track={track.global_id} "
                f"reason={reason} ratio={result['ratio']:.2f} scores={result['scores']}",
                level="info",
            )
            return

        track.locked_class_name = locked_class
        track.resolved_class_name = locked_class

        detected_class = self._detected_class(track)
        if detected_class != locked_class:
            track.class_drift_flagged = True

        self._emit_debug(
            f"[CLASS-LOCKED] track={track.global_id} locked={locked_class} "
            f"detected={detected_class} reason={reason} ratio={result['ratio']:.2f} "
            f"scores={result['scores']}",
            level="info",
        )

    def _voted_event_class(self, track):
        vote_result = self._compute_weighted_class_vote(track, stable_only=False, include_safe=True)
        voted_class = vote_result.get("class")

        if self._is_valid_class(voted_class):
            return voted_class

        return self._detected_class(track)

    def _pickup_event_class(self, track):
        self._ensure_identity_fields(track)

        if self._is_valid_class(track.locked_class_name):
            return track.locked_class_name

        return self._voted_event_class(track)

    def _putback_event_class(self, track):
        self._ensure_identity_fields(track)

        if self._is_valid_class(track.resolved_class_name):
            return track.resolved_class_name

        if self._is_valid_class(track.locked_class_name):
            return track.locked_class_name

        return self._voted_event_class(track)

    def _event_class(self, track, purpose="generic"):
        self._ensure_identity_fields(track)

        if purpose == "pickup":
            return self._pickup_event_class(track)

        if purpose == "putback":
            return self._putback_event_class(track)

        if self._is_valid_class(track.locked_class_name):
            return track.locked_class_name

        if self._is_valid_class(track.resolved_class_name):
            return track.resolved_class_name

        return self._voted_event_class(track)

    # ---------------------------------------------------------------------
    # Confirmation
    # ---------------------------------------------------------------------

    def _append_confirmation(self, track, confidence):
        self._ensure_identity_fields(track)

        if track.confirmation_confidences.maxlen != CONFIRM_FRAMES:
            track.confirmation_confidences = deque(
                track.confirmation_confidences,
                maxlen=CONFIRM_FRAMES,
            )

        if track.confirmation_class_votes.maxlen != CONFIRM_FRAMES:
            track.confirmation_class_votes = deque(
                track.confirmation_class_votes,
                maxlen=CONFIRM_FRAMES,
            )

        track.confirmation_confidences.append(float(confidence))

        update = track.current_update
        if update is not None:
            detected_class = getattr(update, "class_name", None) or self._detected_class(track)
            track.confirmation_class_votes.append({
                "class": detected_class,
                "confidence": float(confidence),
                "timestamp_ms": float(update.timestamp_ms),
            })

    def _clear_confirmation(self, track):
        self._ensure_identity_fields(track)
        track.confirmation_confidences.clear()
        track.confirmation_class_votes.clear()

    def _seed_class_return_confirmation(self, track):
        self._ensure_identity_fields(track)

        history = list(getattr(track, "confidence_history", []))[-CONFIRM_FRAMES:]

        if not history:
            history = [getattr(track, "last_confidence", 0.0)]

        while len(history) < CONFIRM_FRAMES:
            history.append(history[-1])

        track.confirmation_confidences = deque(history[-CONFIRM_FRAMES:], maxlen=CONFIRM_FRAMES)

        class_history = list(track.class_votes)[-CONFIRM_FRAMES:]

        if not class_history:
            detected_class = self._detected_class(track)
            class_history = [{
                "class": detected_class,
                "confidence": getattr(track, "last_confidence", 0.0),
                "timestamp_ms": getattr(track, "last_seen_ms", 0.0),
            }]

        while len(class_history) < CONFIRM_FRAMES:
            class_history.append(class_history[-1])

        track.confirmation_class_votes = deque(
            [
                {
                    "class": item["class"],
                    "confidence": item["confidence"],
                    "timestamp_ms": item["timestamp_ms"],
                }
                for item in class_history[-CONFIRM_FRAMES:]
            ],
            maxlen=CONFIRM_FRAMES,
        )

    def _pending_timeout_elapsed(self, track, timestamp_ms):
        if track.pending_since_ms is None:
            return False
        return (timestamp_ms - track.pending_since_ms) >= MISSING_PENDING_CONFIRM_MS

    def _confidence_ok(self, track):
        if len(track.confirmation_confidences) < CONFIRM_FRAMES:
            return False

        confidences = list(track.confirmation_confidences)
        avg_conf = sum(confidences) / len(confidences)
        min_conf = min(confidences)

        if avg_conf >= CONF_THRESHOLD and min_conf >= CONF_MIN_FRAME:
            return True

        # Allow one weak outlier only if overall average remains healthy.
        if len(confidences) > 2:
            sorted_confs = sorted(confidences)
            without_lowest = sorted_confs[1:]

            avg_without_lowest = sum(without_lowest) / len(without_lowest)
            min_without_lowest = min(without_lowest)

            if (
                avg_without_lowest >= CONF_THRESHOLD
                and min_without_lowest >= CONF_MIN_FRAME
                and avg_conf >= CONF_THRESHOLD * 0.90
            ):
                self._emit_debug(
                    f"[CONF-LENIENT] track={track.global_id} "
                    f"orig_avg={avg_conf:.2f} orig_min={min_conf:.2f} "
                    f"adj_avg={avg_without_lowest:.2f} adj_min={min_without_lowest:.2f}",
                    level="debug",
                )
                return True

        return False

    def _class_consistency_ok(self, track):
        self._ensure_identity_fields(track)

        event_class = self._event_class(track, purpose="pickup")

        if not event_class or event_class == UNKNOWN_UNSTABLE_CLASS:
            return False

        votes = list(track.confirmation_class_votes)

        if not votes:
            return True

        event_family = self._family_base_class(event_class)

        total_weight = 0.0
        exact_weight = 0.0
        family_weight = 0.0

        for vote in votes:
            cls = vote.get("class")
            conf = float(vote.get("confidence", 0.0) or 0.0)

            total_weight += conf

            if cls == event_class:
                exact_weight += conf
                family_weight += conf
            else:
                vote_family = self._family_base_class(cls) if cls else None
                if vote_family == event_family and event_family != UNKNOWN_UNSTABLE_CLASS:
                    family_weight += conf * 0.7

        if total_weight <= 0:
            return False

        exact_ratio = exact_weight / total_weight
        family_ratio = family_weight / total_weight

        passes = (
            exact_ratio >= CLASS_CONSISTENCY_RATIO
            or family_ratio >= CLASS_CONSISTENCY_RATIO
        )

        if not passes:
            track.class_drift_flagged = True
            self._emit_debug(
                f"[CLASS-DRIFT] track={track.global_id} event_class={event_class} "
                f"exact_ratio={exact_ratio:.2f} family_ratio={family_ratio:.2f} "
                f"votes={votes}",
                level="info",
            )

        return passes

    def _confirmation_ok(self, track, require_class_consistency=True):
        if not self._confidence_ok(track):
            return False

        if not require_class_consistency:
            return True

        return self._class_consistency_ok(track)

    # ---------------------------------------------------------------------
    # Motion / ROI helpers
    # ---------------------------------------------------------------------

    def _required_stability_frames(self, track, timestamp_ms):
        return LONG_BOUNDARY_STABILITY_FRAMES if track.boundary_linger_flagged else STABILITY_FRAMES

    def _has_any_recent_outward_motion(self, track, look_back=3):
        if not track.motion_direction_history:
            return False
        return any(d is True for d in list(track.motion_direction_history)[-look_back:])

    def _has_any_recent_inward_motion(self, track, look_back=3):
        if not track.motion_direction_history:
            return False
        return any(d is False for d in list(track.motion_direction_history)[-look_back:])

    def _has_sustained_outward_motion(self, track, min_consecutive=2):
        if len(track.motion_direction_history) < min_consecutive:
            return False
        return all(d is True for d in list(track.motion_direction_history)[-min_consecutive:])

    def _has_sustained_inward_motion(self, track, min_consecutive=2):
        if len(track.motion_direction_history) < min_consecutive:
            return False
        return all(d is False for d in list(track.motion_direction_history)[-min_consecutive:])

    @staticmethod
    def _roi_exit_occurred(track, update, prev_stable):
        return bool(track.was_stable) and prev_stable is True and not update.in_stable_roi

    @staticmethod
    def _roi_reentry_occurred(update, prev_safe, prev_outer):
        was_fully_absent = prev_safe is False and prev_outer is False
        now_in_roi = update.in_safe_roi or update.in_outer_roi
        return was_fully_absent and now_in_roi

    def _is_inward_or_return_motion(self, track, update, prev_safe, prev_outer, prev_stable):
        if update is None:
            return False

        current_outward = bool(update.outward_motion)
        current_inward = bool(update.inward_motion)

        sustained_outward = self._has_sustained_outward_motion(track, min_consecutive=2)
        sustained_inward = self._has_sustained_inward_motion(track, min_consecutive=2)

        if current_outward or sustained_outward:
            return False

        reentered_roi = self._roi_reentry_occurred(update, prev_safe, prev_outer)

        return bool(current_inward or sustained_inward or (reentered_roi and (current_inward or sustained_inward)))

    # ---------------------------------------------------------------------
    # Ledger helpers
    # ---------------------------------------------------------------------

    def _has_outstanding_pickup(self, class_name):
        return (self.pickup_count[class_name] - self.putback_count[class_name]) > 0

    def _latest_pickup_timestamp(self, class_name):
        history = self.class_pickup_timestamps[class_name]
        return history[-1] if history else None

    def _active_pending_pickups(self, timestamp_ms):
        active = []

        for item in self.pending_pickups:
            if item.get("closed"):
                continue

            age = timestamp_ms - float(item["pickup_time_ms"])

            if age <= PENDING_LEDGER_MAX_AGE_MS:
                active.append(item)

        return active

    def _cleanup_pending_pickups(self, timestamp_ms):
        cleaned = []

        for item in self.pending_pickups:
            if item.get("closed"):
                continue

            age = timestamp_ms - float(item["pickup_time_ms"])

            if age <= PENDING_LEDGER_MAX_AGE_MS:
                cleaned.append(item)

        self.pending_pickups = cleaned

    def _class_similarity_score(self, a, b):
        if not a or not b:
            return LEDGER_DIFFERENT_CLASS_SCORE

        if a == b:
            return LEDGER_SAME_CLASS_SCORE

        if self._family_base_class(a) == self._family_base_class(b):
            return max(LEDGER_DIFFERENT_CLASS_SCORE, 0.65)

        return LEDGER_DIFFERENT_CLASS_SCORE

    def _ledger_class_candidate_score(self, track, pending_item):
        current_vote = self._compute_weighted_class_vote(track, stable_only=False, include_safe=True)
        current_scores = current_vote.get("scores", {}) or {}
        pickup_scores = pending_item.get("class_candidates", {}) or {}

        if not current_scores or not pickup_scores:
            return 0.0

        current_total = sum(current_scores.values()) or 1.0
        pickup_total = sum(pickup_scores.values()) or 1.0

        overlap = set(current_scores.keys()) & set(pickup_scores.keys())

        if not overlap:
            return 0.0

        score = 0.0

        for cls in overlap:
            score += min(
                current_scores.get(cls, 0.0) / current_total,
                pickup_scores.get(cls, 0.0) / pickup_total,
            )

        return min(1.0, score)

    def _time_match_score(self, pickup_time_ms, return_time_ms):
        gap = max(0.0, return_time_ms - pickup_time_ms)

        if gap >= PENDING_LEDGER_MAX_AGE_MS:
            return 0.0

        return 1.0 - (gap / PENDING_LEDGER_MAX_AGE_MS)

    def _track_match_score(self, track, pending_item):
        return 1.0 if pending_item.get("track_id") == track.global_id else 0.0

    def _motion_match_score_for_putback(self, track):
        update = getattr(track, "current_update", None)

        if update is None:
            return 0.0

        current_outward = bool(getattr(update, "outward_motion", False))
        current_inward = bool(getattr(update, "inward_motion", False))
        sustained_inward = self._has_sustained_inward_motion(track, min_consecutive=2)
        recent_outward = self._has_any_recent_outward_motion(track, look_back=3)

        if (current_inward or sustained_inward) and not current_outward and not recent_outward:
            return 1.0

        return 0.2

    def _track_embedding(self, track):
        if getattr(track, "visual_embedding", None) is not None:
            return np.asarray(track.visual_embedding, dtype=np.float32)

        if hasattr(track, "mean_embedding"):
            emb = track.mean_embedding()
            if emb is not None:
                return np.asarray(emb, dtype=np.float32)

        return None

    def _embedding_similarity_score(self, track, pending_item, expected_class=None):
        emb_a = self._track_embedding(track)
        emb_b = pending_item.get("visual_embedding")

        if emb_a is None or emb_b is None:
            return 0.0

        emb_b = np.asarray(emb_b, dtype=np.float32)

        denom = float(np.linalg.norm(emb_a) * np.linalg.norm(emb_b))

        if denom <= 1e-6:
            return 0.0

        cosine = float(np.dot(emb_a, emb_b) / denom)
        cosine = max(-1.0, min(1.0, cosine))

        normalized = (cosine + 1.0) / 2.0

        if expected_class and expected_class in EMBEDDING_SIM_THRESHOLDS_BY_CLASS:
            min_sim = EMBEDDING_SIM_THRESHOLDS_BY_CLASS[expected_class]
        else:
            min_sim = EMBEDDING_SIM_THRESHOLDS_BY_CLASS.get("default", 0.65)

        return normalized if normalized >= min_sim else 0.0

    def _spatial_match_score(self, track, pending_item):
        """
        Spatial score is intentionally weak because products may be returned to
        different shelf positions.
        """
        update = track.current_update
        pickup_centroid = pending_item.get("pickup_centroid")

        if update is None or pickup_centroid is None:
            return 0.0

        try:
            return_centroid = np.asarray(update.centroid, dtype=np.float32)
            pickup_centroid_np = np.asarray(pickup_centroid, dtype=np.float32)
        except Exception:
            return 0.0

        dist = float(np.linalg.norm(return_centroid - pickup_centroid_np))

        if dist >= LEDGER_SPATIAL_MAX_DISTANCE_PX:
            return 0.10

        return max(0.10, 1.0 - (dist / LEDGER_SPATIAL_MAX_DISTANCE_PX))

    def _find_ledger_by_id(self, ledger_id):
        if not ledger_id:
            return None

        for item in self.pending_pickups:
            if item.get("ledger_id") == ledger_id and not item.get("closed"):
                return item

        return None

    def _match_putback_to_pending_pickup(self, track, timestamp_ms):
        self._ensure_identity_fields(track)

        # Prefer a previously selected candidate ledger.
        if getattr(track, "candidate_putback_ledger_id", None):
            item = self._find_ledger_by_id(track.candidate_putback_ledger_id)
            if item is not None:
                return item

        candidates = self._active_pending_pickups(timestamp_ms)

        if not candidates:
            return None

        detected_return_class = self._detected_class(track)
        resolved_return_class = self._event_class(track, purpose="putback")

        scored = []

        for item in candidates:
            picked_class = item["event_class"]

            class_score = max(
                self._class_similarity_score(detected_return_class, picked_class),
                self._class_similarity_score(resolved_return_class, picked_class),
            )

            time_score = self._time_match_score(item["pickup_time_ms"], timestamp_ms)
            track_score = self._track_match_score(track, item)
            candidate_score = self._ledger_class_candidate_score(track, item)
            motion_score = self._motion_match_score_for_putback(track)
            embedding_score = self._embedding_similarity_score(track, item, expected_class=picked_class)
            spatial_score = self._spatial_match_score(track, item)

            total_score = (
                LEDGER_MATCH_WEIGHT_CLASS * class_score
                + LEDGER_MATCH_WEIGHT_TIME * time_score
                + LEDGER_MATCH_WEIGHT_TRACK * track_score
                + LEDGER_MATCH_WEIGHT_CLASS_CANDIDATE * candidate_score
                + LEDGER_MATCH_WEIGHT_MOTION * motion_score
                + LEDGER_MATCH_WEIGHT_EMBEDDING * embedding_score
                + LEDGER_MATCH_WEIGHT_SPATIAL * spatial_score
            )

            breakdown = {
                "total": total_score,
                "picked_class": picked_class,
                "detected_return_class": detected_return_class,
                "resolved_return_class": resolved_return_class,
                "class_score": class_score,
                "time_score": time_score,
                "track_score": track_score,
                "class_candidate_score": candidate_score,
                "motion_score": motion_score,
                "embedding_score": embedding_score,
                "spatial_score": spatial_score,
            }

            scored.append((total_score, item, breakdown))

        scored.sort(key=lambda x: x[0], reverse=True)

        best_score, best_item, best_breakdown = scored[0]

        # Ambiguity protection.
        if len(scored) > 1:
            second_score = scored[1][0]
            if best_score - second_score < 0.08:
                self._emit_debug(
                    f"[LEDGER-AMBIGUOUS] track={track.global_id} "
                    f"best={best_score:.2f} second={second_score:.2f} "
                    f"best_breakdown={best_breakdown}",
                    level="warning",
                )
                return None

        if best_score >= PUTBACK_LEDGER_MATCH_THRESHOLD:
            best_item["_last_match_score"] = best_score
            best_item["_last_match_breakdown"] = best_breakdown
            return best_item

        self._emit_debug(
            f"[LEDGER-NO-MATCH] track={track.global_id} "
            f"detected={detected_return_class} best_score={best_score:.2f} "
            f"threshold={PUTBACK_LEDGER_MATCH_THRESHOLD} breakdown={best_breakdown}",
            level="info",
        )

        return None

    # ---------------------------------------------------------------------
    # Class-level return candidate (FIXED – brand-new check first)
    # ---------------------------------------------------------------------

    def _is_class_return_candidate(self, track, prev_safe=None, prev_outer=None):
        """
        Class-level return is only for fresh/reappearing tracks.
        It must never fire for already stable shelf products.
        """
        self._ensure_identity_fields(track)

        if getattr(track, "was_stable", False):
            return False

        update = track.current_update
        if update is None:
            return False

        candidate_class = (
            track.locked_class_name
            if self._is_valid_class(track.locked_class_name)
            else self._voted_event_class(track)
        )

        if not self._is_valid_class(candidate_class):
            return False

        if not self._has_outstanding_pickup(candidate_class):
            return False

        latest_pickup_ts = self._latest_pickup_timestamp(candidate_class)
        if latest_pickup_ts is None:
            return False

        created_ms = getattr(track, "created_ms", None)

        # -----------------------------------------------------------------
        # DEBUG: log creation timestamp and latest pickup
        # -----------------------------------------------------------------
        self._emit_debug(
            f"[RETURN-CANDIDATE] track={track.global_id} class={candidate_class} "
            f"created_ms={created_ms} latest_pickup_ts={latest_pickup_ts}",
            level="debug"
        )

        # ------------------------------------------------------------
        # IMPORTANT: Brand-new tracks (created after the latest pickup)
        # should NEVER be treated as returns, regardless of ledger match.
        # ------------------------------------------------------------
        if created_ms is not None and created_ms > latest_pickup_ts:
            self._emit_debug(
                f"[RETURN-CANDIDATE] track={track.global_id} created_ms > latest_pickup, not a return",
                level="debug"
            )
            return False

        # ------------------------------------------------------------
        # Now check for an actual ledger match (track ID, embedding, etc.)
        # This is safe because the track is NOT brand new.
        # ------------------------------------------------------------
        ledger_match = self._match_putback_to_pending_pickup(track, update.timestamp_ms)
        if ledger_match is not None:
            track.resolved_class_name = ledger_match["event_class"]
            track.candidate_putback_ledger_id = ledger_match["ledger_id"]
            return True

        # ------------------------------------------------------------
        # No ledger match – treat as a possible class‑level return only if:
        #   - there is inward or re‑entry evidence.
        # ------------------------------------------------------------
        current_outward = bool(getattr(update, "outward_motion", False))
        current_inward = bool(getattr(update, "inward_motion", False))
        sustained_inward = self._has_sustained_inward_motion(track, min_consecutive=2)
        recent_outward = self._has_any_recent_outward_motion(track, look_back=3)

        has_inward = (
            (current_inward or sustained_inward)
            and not current_outward
            and not recent_outward
        )
        was_absent = prev_safe is False and prev_outer is False
        now_inside = update.in_outer_roi or update.in_safe_roi or update.in_stable_roi
        has_reentry = was_absent and now_inside

        if not has_inward and not has_reentry:
            return False

        # Fallback to class-level return
        track.resolved_class_name = candidate_class
        track.candidate_putback_ledger_id = None
        return True

    # ---------------------------------------------------------------------
    # Neighbor stability boost
    # ---------------------------------------------------------------------

    def _check_neighbor_stability_boost(self, track, all_tracks):
        update = track.current_update

        if update is None:
            return

        for neighbor in all_tracks:
            self._ensure_identity_fields(neighbor)

            if neighbor.global_id == track.global_id:
                continue

            if neighbor.event_state not in {"INSIDE", "STABLE_INSIDE"}:
                continue

            if getattr(neighbor, "stability_boosted", False):
                continue

            neighbor_update = neighbor.current_update

            if neighbor_update is None:
                continue

            same_zone = (
                update.in_stable_roi and neighbor_update.in_stable_roi
            ) or (
                update.in_outer_roi and neighbor_update.in_outer_roi
            )

            half_stability = neighbor.inside_counter >= max(1, STABILITY_FRAMES // 2)

            if same_zone and half_stability:
                required = self._required_stability_frames(neighbor, update.timestamp_ms)
                neighbor.inside_counter = required
                neighbor.event_state = "STABLE_INSIDE"
                neighbor.stability_boosted = True
                neighbor.was_stable = True
                self._lock_class_if_possible(neighbor, reason="neighbor stability boost")
                self._emit_debug(
                    f"[BOOST] neighbor track={neighbor.global_id} -> STABLE_INSIDE by track={track.global_id}",
                    level="debug",
                )

    # ---------------------------------------------------------------------
    # Strict INSIDE edge-outward pickup fallback
    # ---------------------------------------------------------------------

    def _is_inside_edge_pickup_candidate(self, track, update):
        """
        Strict fallback for products that never reached STABLE_INSIDE but are
        clearly leaving from shelf edge.

        This is intentionally conservative to avoid:
        - false pickup during putback
        - duplicate pickup
        - random outer ROI flicker
        """
        if update is None:
            return False

        if track.event_state != "INSIDE":
            return False

        # Do not interfere with normal stable-product pickup path.
        if getattr(track, "was_stable", False):
            return False

        # Duplicate safety.
        if getattr(track, "active_pickup_ledger_id", None):
            return False

        # Prevent pickup echo right after a putback.
        if (
            getattr(track, "last_putback_confirmed_ms", None) is not None
            and (update.timestamp_ms - track.last_putback_confirmed_ms) < POST_PUTBACK_COOLDOWN_MS
        ):
            return False

        # This fallback is only for shelf-edge/outer ROI pickup.
        if update.in_stable_roi:
            return False

        if update.in_safe_roi:
            return False

        if not update.in_outer_roi:
            return False

        confidence = float(getattr(update, "confidence", 0.0) or 0.0)
        if confidence < CONF_THRESHOLD:
            return False

        detected_class = self._detected_class(track)
        if not self._is_valid_class(detected_class):
            return False

        has_outward = bool(getattr(update, "outward_motion", False)) or self._has_any_recent_outward_motion(
            track,
            look_back=3,
        )

        has_inward = bool(getattr(update, "inward_motion", False)) or self._has_any_recent_inward_motion(
            track,
            look_back=2,
        )

        if not has_outward:
            return False

        # Critical: do not trigger pickup fallback while product is returning.
        if has_inward:
            return False

        return True

    # ---------------------------------------------------------------------
    # Event recording
    # ---------------------------------------------------------------------

    def _pickup_ledger_item_from_event(self, track, event):
        update = track.current_update
        vote_result = self._compute_weighted_class_vote(track, stable_only=False, include_safe=True)

        pickup_centroid = None
        pickup_bbox = None

        if update is not None:
            try:
                pickup_centroid = np.asarray(update.centroid, dtype=np.float32).tolist()
            except Exception:
                pickup_centroid = None

            pickup_bbox = tuple(update.bbox) if getattr(update, "bbox", None) is not None else None

        return {
            "ledger_id": str(uuid.uuid4()),
            "pickup_event_id": event["event_id"],
            "event_class": event["class"],
            "locked_class": event.get("locked_class"),
            "detected_class_at_pickup": event.get("detected_class"),
            "track_id": track.global_id,
            "pickup_time_ms": event["timestamp_ms"],
            "camera_id": event["camera_id"],
            "frame_index": event["frame_index"],
            "class_candidates": vote_result.get("scores", {}),
            "visual_embedding": self._track_embedding(track),
            "pickup_centroid": pickup_centroid,
            "pickup_bbox": pickup_bbox,
            "closed": False,
            "closed_by_event_id": None,
        }

    def _record_pickup(self, track):
        self._ensure_identity_fields(track)

        # Idempotency guard.
        if getattr(track, "active_pickup_ledger_id", None):
            self._emit_debug(
                f"[DUPLICATE-PICKUP-SUPPRESSED] track={track.global_id} "
                f"active_ledger={track.active_pickup_ledger_id}",
                level="warning",
            )
            return None

        update = track.current_update

        event_class = self._event_class(track, purpose="pickup")

        if event_class == UNKNOWN_UNSTABLE_CLASS:
            self._lock_class_if_possible(track, reason="pickup final lock attempt")
            event_class = self._event_class(track, purpose="pickup")

        if not self._is_valid_class(event_class):
            event_class = self._detected_class(track)

        detected_class = self._detected_class(track)

        if self._is_valid_class(track.locked_class_name) and event_class != track.locked_class_name:
            self._emit_debug(
                f"[PICKUP-CLASS-CORRECTED] track={track.global_id} "
                f"event_class={event_class} -> locked_class={track.locked_class_name} "
                f"detected={detected_class}",
                level="warning",
            )
            event_class = track.locked_class_name
            track.resolved_class_name = event_class

        event_timestamp = track.last_seen_ms if update is None else update.timestamp_ms
        event_camera_id = track.last_camera_id if update is None else update.camera_id
        event_frame_index = track.last_frame_index if update is None else update.frame_index

        self.pickup_count[event_class] += 1

        # Only place where frame snapshot increments.
        self._frame_net_snapshot[event_class] = self._frame_net_snapshot.get(event_class, 0) + 1

        self.class_pickup_timestamps[event_class].append(event_timestamp)

        track.pickup_confirmed_ms = event_timestamp
        track.resolved_class_name = event_class
        track.interaction_cycle_id += 1

        class_drift = detected_class != event_class

        conf_values = list(track.confirmation_confidences)
        event_conf = float(sum(conf_values) / max(len(conf_values), 1))

        event = {
            "event_id": str(uuid.uuid4()),
            "type": "pickup",
            "class": event_class,
            "event_class": event_class,
            "detected_class": detected_class,
            "locked_class": track.locked_class_name,
            "resolved_class": event_class,
            "class_drift": bool(class_drift or track.class_drift_flagged),
            "global_id": track.global_id,
            "camera_id": event_camera_id,
            "frame_index": event_frame_index,
            "timestamp_ms": event_timestamp,
            "confidence": event_conf,
            "confidence_window": conf_values,
            "confirmation_class_votes": list(track.confirmation_class_votes),
            "interaction_cycle_id": track.interaction_cycle_id,
        }

        if ENABLE_CLASS_DRIFT_WARNINGS and event["class_drift"]:
            event["warnings"] = [
                f"class drift observed on pickup: detected={detected_class}, resolved={event_class}"
            ]

        self.session.events.append(event)

        if hasattr(self.session, "pickup_records"):
            self.session.pickup_records.append(event)

        ledger_item = self._pickup_ledger_item_from_event(track, event)
        self.pending_pickups.append(ledger_item)

        track.active_pickup_ledger_id = ledger_item["ledger_id"]

        overlay = f"PICKUP {event_class} G{track.global_id}"
        if event["class_drift"]:
            overlay += f" [detected:{detected_class}]"

        self._push_overlay_event(event_timestamp, overlay)

        self._emit_debug(
            f"[EVENT] {overlay} | cam={event_camera_id} frame={event_frame_index} "
            f"conf={conf_values} ledger_id={ledger_item['ledger_id']}",
            level="info",
        )
        track.is_picked = True

        if self.debug_logger is not None and hasattr(self.debug_logger, "log_event_fired"):
            self.debug_logger.log_event_fired(
                track=track,
                event_type="pickup",
                event=event,
                timestamp_ms=event_timestamp,
                suppressed=False,
                suppression_reason=None,
            )

        return event

    def _record_putback(self, track):
        self._ensure_identity_fields(track)

        update = track.current_update

        # =============================================================
        # FINAL SAFETY GUARD (Triple-check before firing)
        # =============================================================
        if update is not None:
            current_outward = bool(getattr(update, "outward_motion", False))
            current_inward = bool(getattr(update, "inward_motion", False))
            sustained_inward = self._has_sustained_inward_motion(track, min_consecutive=2)
            recent_outward = self._has_any_recent_outward_motion(track, look_back=8)

            has_inward_evidence = current_inward or sustained_inward

            # 1. Block if actively moving outward
            if current_outward and not has_inward_evidence:
                self._suppress_putback(
                    track,
                    self._event_class(track, purpose="putback"),
                    self._detected_class(track),
                    "outward motion at putback record time"
                )
                return None

            # 2. Block if recent outward trail exists without inward evidence
            if recent_outward and not has_inward_evidence:
                self._suppress_putback(
                    track,
                    self._event_class(track, purpose="putback"),
                    self._detected_class(track),
                    "recent outward motion without inward evidence"
                )
                return None

            # 3. Block if outside safe ROI without inward evidence
            if not getattr(update, "in_safe_roi", False) and not has_inward_evidence:
                self._suppress_putback(
                    track,
                    self._event_class(track, purpose="putback"),
                    self._detected_class(track),
                    "outside safe ROI without inward evidence"
                )
                return None
        # =============================================================

        event_timestamp = track.last_seen_ms if update is None else update.timestamp_ms
        event_camera_id = track.last_camera_id if update is None else update.camera_id
        event_frame_index = track.last_frame_index if update is None else update.frame_index
        detected_class = self._detected_class(track)

        matched_item = self._match_putback_to_pending_pickup(track, event_timestamp)

        if matched_item is not None:
            event_class = matched_item["event_class"]
            track.resolved_class_name = event_class
            ledger_match_score = matched_item.get("_last_match_score")
            ledger_match_breakdown = matched_item.get("_last_match_breakdown")
        else:
            event_class = self._event_class(track, purpose="putback")
            ledger_match_score = None
            ledger_match_breakdown = None

            same_track_was_picked = track.pickup_confirmed_ms is not None and track.active_pickup_ledger_id is not None

            if not self._is_valid_class(event_class):
                self._suppress_putback(track, event_class, detected_class, "invalid class")
                return None

            if getattr(track, "was_stable", False) and not same_track_was_picked:
                self._suppress_putback(track, event_class, detected_class, "track was stable and no ledger/same-track pickup")
                return None

            if not self._has_outstanding_pickup(event_class) and not same_track_was_picked:
                self._suppress_putback(track, event_class, detected_class, "no outstanding pickup")
                return None

        current_net = self._frame_net_snapshot.get(event_class, 0) - self._frame_putback_used.get(event_class, 0)

        if current_net <= 0:
            self._suppress_putback(
                track,
                event_class,
                detected_class,
                f"no outstanding pickup to offset current_net={current_net}",
            )
            return None

        self.putback_count[event_class] += 1
        self._frame_putback_used[event_class] += 1

        track.last_putback_confirmed_ms = event_timestamp
        track.resolved_class_name = event_class

        self.last_putback_by_class_ms[event_class] = event_timestamp

        if matched_item is not None:
            matched_item["closed"] = True

        track.active_pickup_ledger_id = None
        track.candidate_putback_ledger_id = None

        is_long_hold = (
            track.pickup_confirmed_ms is not None
            and (event_timestamp - track.pickup_confirmed_ms) > CLASS_RETURN_MAX_GAP_MS
        )

        class_drift = detected_class != event_class

        ledger_score_text = f"{ledger_match_score:.2f}" if ledger_match_score is not None else "N/A"
        global_id_match = matched_item is not None and matched_item.get("track_id") == track.global_id

        self._emit_debug(
            f"[PUTBACK-EVIDENCE] track={track.global_id} "
            f"event_class={event_class} detected={detected_class} "
            f"ledger_matched={matched_item is not None} "
            f"global_id_match={global_id_match} "
            f"ledger_score={ledger_score_text} "
            f"class_drift={class_drift} "
            f"fallback_used={matched_item is None}",
            level="info",
        )

        conf_values = list(track.confirmation_confidences)
        event_conf = float(sum(conf_values) / max(len(conf_values), 1))

        event = {
            "event_id": str(uuid.uuid4()),
            "type": "putback",
            "class": event_class,
            "event_class": event_class,
            "detected_class": detected_class,
            "locked_class": track.locked_class_name,
            "resolved_class": event_class,
            "class_drift": bool(class_drift or track.class_drift_flagged),
            "global_id": track.global_id,
            "camera_id": event_camera_id,
            "frame_index": event_frame_index,
            "timestamp_ms": event_timestamp,
            "confidence": event_conf,
            "confidence_window": conf_values,
            "confirmation_class_votes": list(track.confirmation_class_votes),
            "interaction_cycle_id": track.interaction_cycle_id,
        }

        if matched_item is not None:
            event["matched_pickup_ledger_id"] = matched_item["ledger_id"]
            event["matched_pickup_event_id"] = matched_item["pickup_event_id"]
            event["ledger_match_score"] = ledger_match_score
            event["ledger_match_breakdown"] = ledger_match_breakdown
            matched_item["closed_by_event_id"] = event["event_id"]

        if is_long_hold:
            event["long_hold_return"] = True

        if ENABLE_CLASS_DRIFT_WARNINGS and event["class_drift"]:
            event["warnings"] = [
                f"class drift observed on putback: detected={detected_class}, resolved={event_class}"
            ]

        self.session.events.append(event)

        if hasattr(self.session, "putback_records"):
            self.session.putback_records.append(event)

        overlay = f"PUTBACK {event_class} G{track.global_id}"
        if event["class_drift"]:
            overlay += f" [detected:{detected_class}]"
        if is_long_hold:
            overlay += " [long-hold]"

        self._push_overlay_event(event_timestamp, overlay)

        self._emit_debug(
            f"[EVENT] {overlay} | cam={event_camera_id} frame={event_frame_index} "
            f"conf={conf_values} matched_ledger={matched_item['ledger_id'] if matched_item else None}",
            level="info",
        )
        track.is_picked = False

        if self.debug_logger is not None and hasattr(self.debug_logger, "log_event_fired"):
            self.debug_logger.log_event_fired(
                track=track,
                event_type="putback",
                event=event,
                timestamp_ms=event_timestamp,
                suppressed=False,
                suppression_reason=None,
            )

        return event

    def _suppress_putback(self, track, event_class, detected_class, reason):
        warning = (
            f"Putback suppressed class={event_class} detected={detected_class} "
            f"track={track.global_id} — {reason}"
        )
        self.session.warnings.append(warning)
        LOGGER.warning(warning)
        self._emit_debug(f"[PUTBACK-SUPPRESSED] {warning}", level="warning")

    # ---------------------------------------------------------------------
    # Overlay
    # ---------------------------------------------------------------------

    def overlay_event_lines(self, timestamp_ms):
        while (
            self.recent_overlay_events
            and (timestamp_ms - self.recent_overlay_events[0][0]) > OVERLAY_EVENT_TTL_MS
        ):
            self.recent_overlay_events.popleft()

        return [item[1] for item in list(self.recent_overlay_events)[-OVERLAY_MAX_EVENT_LINES:]]

    # ---------------------------------------------------------------------
    # Core FSM
    # ---------------------------------------------------------------------

    def _process_track(self, track, timestamp_ms, all_tracks=None):
        self._ensure_identity_fields(track)

        update = track.current_update
        prev_outside_frames = track.frames_outside_stable

        if update is not None:
            if update.in_stable_roi:
                track.frames_outside_stable = 0
                track.current_stable_counter += 1
            else:
                track.frames_outside_stable += 1
                track.current_stable_counter = 0
        else:
            track.frames_outside_stable += 1
            track.current_stable_counter = 0

        # -------------------------------------------------------------
        # No detection
        # -------------------------------------------------------------
        if update is None:
            if track.event_state == "PICKUP_PENDING":
                if self._pending_timeout_elapsed(track, timestamp_ms):
                    if self._has_any_recent_outward_motion(track, look_back=5):
                        self._transition(track, "PICKED_UP", "pickup confirmed: lost after outward motion")
                        event = self._record_pickup(track)
                        return [] if event is None else [event]

                    if track.was_stable:
                        self._transition(track, "PICKED_UP", "pickup confirmed: stable track lost / occlusion")
                        event = self._record_pickup(track)
                        if event is not None:
                            event["occlusion_pickup"] = True
                            self._replace_last_overlay(f"PICKUP {event['class']} G{track.global_id} [occluded]")
                            return [event]
                        return []

                    self._emit_debug(
                        f"[SILENT-CANCEL] track={track.global_id} never reached stability — suppressed",
                        level="debug",
                    )
                    self._transition(track, "INSIDE", "pickup cancelled: never reached stability")
                    return []

            if track.event_state == "PUTBACK_PENDING":
                if self._pending_timeout_elapsed(track, timestamp_ms):
                    if track.last_seen_in_stable_roi is True and self._has_any_recent_inward_motion(track, look_back=5):
                        event = self._record_putback(track)
                        if event is not None:
                            self._transition(track, "STABLE_INSIDE", "putback confirmed: lost after stable ROI")
                            return [event]
                        self._transition(track, "PICKED_UP", "putback validation failed after lost stable")
                        return []

                    self._emit_debug(
                        f"[TIMEOUT] putback absent track={track.global_id} "
                        f"last_stable={track.last_seen_in_stable_roi}",
                        level="info",
                    )
                    self._transition(track, "PICKED_UP", "putback cancelled: lost without entering stable ROI")

            return []

        # -------------------------------------------------------------
        # Detection exists
        # -------------------------------------------------------------
        self._append_class_vote(track, update)

        prev_safe = track.last_seen_in_safe_roi
        prev_outer = track.last_seen_in_outer_roi
        prev_stable = track.last_seen_in_stable_roi

        track.last_seen_in_safe_roi = update.in_safe_roi
        track.last_seen_in_outer_roi = update.in_outer_roi
        track.last_seen_in_stable_roi = update.in_stable_roi

        if update.displacement_magnitude > 0:
            if update.outward_motion:
                motion_direction = True
            elif update.inward_motion:
                motion_direction = False
            else:
                motion_direction = None
        else:
            motion_direction = None

        track.motion_direction_history.append(motion_direction)

        if update.in_outer_roi and not update.in_stable_roi:
            if track.boundary_linger_start_ms is None:
                track.boundary_linger_start_ms = update.timestamp_ms
            elif (update.timestamp_ms - track.boundary_linger_start_ms) > 1000.0:
                track.boundary_linger_flagged = True
        else:
            track.boundary_linger_start_ms = None

        required_stability = self._required_stability_frames(track, update.timestamp_ms)

        # -------------------------------------------------------------
        # INSIDE
        # -------------------------------------------------------------
        if track.event_state == "INSIDE":
            if self._is_class_return_candidate(track, prev_safe=prev_safe, prev_outer=prev_outer):
                self._transition(track, "PUTBACK_PENDING", "fresh/reappearing class-level return candidate")
                track.pending_since_ms = update.timestamp_ms
                self._seed_class_return_confirmation(track)
                self._emit_debug(
                    f"[PENDING] putback track={track.global_id} "
                    f"event_class={self._event_class(track, purpose='putback')} "
                    f"detected={self._detected_class(track)}",
                    level="info",
                )
                return []

            # ---------------------------------------------------------------
            # Shelf-edge outward pickup fallback
            # ---------------------------------------------------------------
            if self._is_inside_edge_pickup_candidate(track, update):
                track.inside_outward_pickup_counter += 1
            else:
                track.inside_outward_pickup_counter = 0

            required_edge_pickup_frames = max(2, min(CONFIRM_FRAMES, 3))

            if track.inside_outward_pickup_counter >= required_edge_pickup_frames:
                # Treat this as a shelf-origin pickup candidate.
                # Set was_stable=True only after multi-frame outward evidence,
                # so existing PICKUP_PENDING confirmation logic can reuse normal
                # pickup validation without changing ledger/putback logic.
                track.was_stable = True
                track.inside_counter = 0

                self._lock_class_if_possible(track, reason="inside edge outward pickup trigger")

                self._transition(
                    track,
                    "PICKUP_PENDING",
                    "inside edge outward pickup trigger"
                )

                track.pending_since_ms = update.timestamp_ms
                self._clear_confirmation(track)

                self._emit_debug(
                    f"[PENDING] pickup track={track.global_id} "
                    f"event_class={self._event_class(track, purpose='pickup')} "
                    f"detected={self._detected_class(track)} "
                    f"reason=inside_edge_outward "
                    f"counter={track.inside_outward_pickup_counter} "
                    f"safe={update.in_safe_roi} outer={update.in_outer_roi} "
                    f"outward={update.outward_motion} inward={update.inward_motion} "
                    f"conf={update.confidence:.2f}",
                    level="info",
                )

                return []

            if update.in_stable_roi and update.confidence >= CONF_THRESHOLD:
                track.inside_counter += 1

                if track.inside_counter >= required_stability:
                    self._transition(track, "STABLE_INSIDE", "stable in shelf ROI")
                    track.boundary_linger_flagged = False

                    if all_tracks is not None:
                        self._check_neighbor_stability_boost(track, all_tracks)
            else:
                track.inside_counter = 0

            return []

        # -------------------------------------------------------------
        # STABLE_INSIDE
        # -------------------------------------------------------------
        if track.event_state == "STABLE_INSIDE":
            if track.locked_class_name == UNKNOWN_UNSTABLE_CLASS:
                self._lock_class_if_possible(track, reason="additional stable evidence")

            # Define motion/exit variables consistently
            has_outward = self._has_any_recent_outward_motion(track, look_back=3)
            has_sustained_outward = self._has_sustained_outward_motion(track, min_consecutive=2)
            has_roi_exit = self._roi_exit_occurred(track, update, prev_stable)
            is_inward_or_return = self._is_inward_or_return_motion(track, update, prev_safe, prev_outer, prev_stable)

            if not update.in_safe_roi:
                pickup_zone_or_motion_exit = (
                    not update.in_safe_roi
                    or has_roi_exit
                    or has_sustained_outward
                )
            else:
                pickup_zone_or_motion_exit = False

            if pickup_zone_or_motion_exit:
                just_exited_safe = prev_safe is True and not update.in_safe_roi

                if (
                    is_inward_or_return
                    and not has_outward
                    and not has_roi_exit
                    and not just_exited_safe
                ):
                    self._emit_debug(
                        f"[PICKUP-BLOCKED-INWARD] track={track.global_id} "
                        f"prev_safe={prev_safe} curr_safe={update.in_safe_roi} "
                        f"motion_history={list(track.motion_direction_history)}",
                        level="info",
                    )
                    return []

                pickup_like_exit = has_sustained_outward or has_outward or has_roi_exit

                if pickup_like_exit:
                    if (
                        track.last_putback_confirmed_ms is not None
                        and (update.timestamp_ms - track.last_putback_confirmed_ms) < POST_PUTBACK_COOLDOWN_MS
                    ):
                        self._emit_debug(
                            f"[COOLDOWN] pickup suppressed track={track.global_id} "
                            f"elapsed={update.timestamp_ms - track.last_putback_confirmed_ms:.0f}ms",
                            level="debug",
                        )
                        return []

                    self._lock_class_if_possible(track, reason="pickup trigger")

                    if (
                        self._is_valid_class(track.locked_class_name)
                        and self._is_valid_class(track.resolved_class_name)
                        and track.resolved_class_name != track.locked_class_name
                    ):
                        track.resolved_class_name = track.locked_class_name

                    trigger_reason = (
                        "outward exit + motion"
                        if (has_outward or has_sustained_outward)
                        else "ROI exit without strong motion / vertical lift"
                    )

                    self._transition(track, "PICKUP_PENDING", trigger_reason)
                    track.pending_since_ms = update.timestamp_ms

                    # Important: for pickup, do NOT seed from stable history.
                    self._clear_confirmation(track)

                    if all_tracks is not None:
                        self._check_neighbor_stability_boost(track, all_tracks)

                    self._emit_debug(
                        f"[PENDING] pickup track={track.global_id} "
                        f"event_class={self._event_class(track, purpose='pickup')} "
                        f"detected={self._detected_class(track)} "
                        f"outward={has_outward} roi_exit={has_roi_exit} "
                        f"prev_safe={prev_safe} curr_safe={update.in_safe_roi}",
                        level="info",
                    )
                    return []

            if update.in_stable_roi and update.confidence >= CONF_THRESHOLD:
                track.inside_counter = required_stability
            else:
                track.inside_counter = 0

            return []

        # -------------------------------------------------------------
        # PICKUP_PENDING
        # -------------------------------------------------------------
        if track.event_state == "PICKUP_PENDING":
            if update.in_stable_roi and not update.outward_motion:
                time_since_pending_ms = update.timestamp_ms - (track.pending_since_ms or update.timestamp_ms)
                was_outside_stable = prev_outside_frames >= QUICK_RETURN_MIN_OUTSIDE_FRAMES

                if was_outside_stable and time_since_pending_ms < QUICK_RETURN_THRESHOLD_MS:
                    self._transition(track, "PICKED_UP", "quick pickup part confirmed")
                    pickup_event = self._record_pickup(track)

                    if pickup_event is None:
                        return []

                    track.resolved_class_name = pickup_event["class"]
                    self._transition(track, "PUTBACK_PENDING", "quick return: fast re-entry")
                    track.pending_since_ms = update.timestamp_ms
                    self._seed_class_return_confirmation(track)

                    self._emit_debug(
                        f"[QUICK-RETURN] track={track.global_id} "
                        f"time_ms={time_since_pending_ms:.0f} "
                        f"outside_frames={prev_outside_frames}",
                        level="info",
                    )

                    track.pickup_cancel_counter = 0
                    return [pickup_event]

                track.pickup_cancel_counter += 1

                if track.pickup_cancel_counter >= 3:
                    self._transition(
                        track,
                        "STABLE_INSIDE",
                        "pickup cancelled: sustained stable ROI re-entry",
                    )
                    track.pickup_cancel_counter = 0
                    return []

                self._emit_debug(
                    f"[PICKUP-FLICKER] track={track.global_id} "
                    f"stable re-entry not sustained counter={track.pickup_cancel_counter}",
                    level="debug",
                )
                return []

            track.pickup_cancel_counter = 0

            if self._has_sustained_inward_motion(track, min_consecutive=2) and (
                update.in_safe_roi or update.in_stable_roi
            ):
                self._transition(track, "STABLE_INSIDE", "pickup cancelled: sustained inward return")
                return []

            outside_safe = not update.in_safe_roi
            has_outward = self._has_any_recent_outward_motion(track, look_back=5)
            has_roi_exit = self._roi_exit_occurred(track, update, prev_stable)
            sustained_outside = track.frames_outside_stable >= 2

            if outside_safe and track.was_stable and (has_outward or (has_roi_exit and sustained_outside)):
                self._append_confirmation(track, update.confidence)

                if self._confidence_ok(track):
                    self._transition(track, "PICKED_UP", "pickup confirmed: outside safe ROI")
                    event = self._record_pickup(track)
                    track.pickup_cancel_counter = 0
                    return [] if event is None else [event]

            if not update.in_outer_roi:
                self._append_confirmation(track, update.confidence)

                confirmation_ok = self._confirmation_ok(track, require_class_consistency=False)

                timeout_absent_ok = (
                    track.was_stable
                    and (update.timestamp_ms - (track.pending_since_ms or update.timestamp_ms)) > 100
                    and not self._has_any_recent_inward_motion(track, look_back=5)
                )

                if confirmation_ok or timeout_absent_ok:
                    self._transition(track, "PICKED_UP", "pickup confirmed: outer ROI absent")
                    event = self._record_pickup(track)
                    return [] if event is None else [event]

                return []

            if update.in_outer_roi:
                self._append_confirmation(track, update.confidence)

                if len(track.confirmation_confidences) >= CONFIRM_FRAMES:
                    conf_ok = self._confidence_ok(track)
                    class_ok = self._class_consistency_ok(track)

                    if not conf_ok:
                        self._transition(track, "STABLE_INSIDE", "pickup cancelled: confidence too low")
                        return []

                    if not class_ok:
                        if track.was_stable and self._has_any_recent_outward_motion(track, look_back=5):
                            self._emit_debug(
                                f"[CLASS-DRIFT-TOLERATED] pickup track={track.global_id}",
                                level="info",
                            )
                            return []

                        self._transition(track, "STABLE_INSIDE", "pickup cancelled: class consistency too low")
                        return []

                if self._has_any_recent_outward_motion(track, look_back=3):
                    return []

                if self._pending_timeout_elapsed(track, update.timestamp_ms):
                    self._transition(track, "STABLE_INSIDE", "pickup cancelled: timeout in outer ROI")
                    return []

                return []

            return []

        # -------------------------------------------------------------
        # PICKED_UP
        # -------------------------------------------------------------
        if track.event_state == "PICKED_UP":
            track_pickup_ts = track.pickup_confirmed_ms
            time_since_pickup_ms = (
                update.timestamp_ms - track_pickup_ts
                if track_pickup_ts is not None
                else 0.0
            )

            if time_since_pickup_ms < self.min_valid_putback_hold_ms:
                return []

            current_outward = bool(getattr(update, "outward_motion", False))
            current_inward = bool(getattr(update, "inward_motion", False))

            sustained_inward = self._has_sustained_inward_motion(track, min_consecutive=2)
            recent_outward = self._has_any_recent_outward_motion(track, look_back=8)

            # Putback must be based on CURRENT/SUSTAINED inward evidence.
            has_inward = (
                (current_inward or sustained_inward)
                and not current_outward
                and not recent_outward
            )

            has_reentry = (
                self._roi_reentry_occurred(update, prev_safe, prev_outer)
                and not current_outward
            )

            matched = self._match_putback_to_pending_pickup(track, update.timestamp_ms)

            if matched is not None:
                track.resolved_class_name = matched["event_class"]
                track.candidate_putback_ledger_id = matched["ledger_id"]

            # =============================================================
            # STRICT STABLE RETURN FALLBACK
            # Now requires: in_safe_roi, no recent outward, and stable_counter >= 5
            # =============================================================
            stable_return_fallback = (
                update.in_stable_roi
                and update.in_safe_roi                     # <-- Must be safely inside
                and not current_outward
                and not recent_outward                     # <-- No recent outward trail
                and time_since_pickup_ms >= self.min_valid_putback_hold_ms
                and track.current_stable_counter >= max(5, STABILITY_FRAMES)
                and (
                    matched is not None
                    or getattr(track, "active_pickup_ledger_id", None) is not None
                )
            )
            # =============================================================

            long_hold = (
                track_pickup_ts is not None
                and time_since_pickup_ms > CLASS_RETURN_MAX_GAP_MS
                and not current_outward
                and (has_inward or has_reentry or stable_return_fallback)
            )

            in_return_roi = update.in_stable_roi or update.in_safe_roi

            putback_like = (
                (has_inward and in_return_roi)
                or has_reentry
                or stable_return_fallback
                or long_hold
            )

            if putback_like:
                reason = (
                    "inward return motion"
                    if has_inward
                    else "ROI re-entry"
                    if has_reentry
                    else "stable return fallback with ledger"
                )

                self._transition(track, "PUTBACK_PENDING", reason)
                track.pending_since_ms = update.timestamp_ms
                self._seed_class_return_confirmation(track)

                self._emit_debug(
                    f"[PENDING] putback track={track.global_id} "
                    f"event_class={self._event_class(track, purpose='putback')} "
                    f"ledger_matched={matched is not None} "
                    f"active_ledger={track.active_pickup_ledger_id} "
                    f"current_inward={current_inward} "
                    f"sustained_inward={sustained_inward} "
                    f"current_outward={current_outward} "
                    f"recent_outward={recent_outward} "
                    f"reentry={has_reentry} "
                    f"stable_counter={track.current_stable_counter}",
                    level="info",
                )
                return []

            return []

        # -------------------------------------------------------------
        # PUTBACK_PENDING
        # -------------------------------------------------------------
        if track.event_state == "PUTBACK_PENDING":
            current_outward = bool(getattr(update, "outward_motion", False))
            current_inward = bool(getattr(update, "inward_motion", False))
            sustained_outward = self._has_sustained_outward_motion(track, min_consecutive=2)
            sustained_inward = self._has_sustained_inward_motion(track, min_consecutive=2)

            recent_outward = self._has_any_recent_outward_motion(track, look_back=8)

            # If product is still moving outward and not stable, this is not putback.
            if (current_outward or sustained_outward) and not update.in_stable_roi:
                self._transition(
                    track,
                    "PICKED_UP",
                    "putback cancelled: outward motion while not stable"
                )
                return []

            if update.in_stable_roi:
                # =============================================================
                # STRICT STABLE CONFIRMATION
                # Prevent confirmation if outside safe ROI or recent outward without inward
                # =============================================================
                if not update.in_safe_roi and not current_inward and not sustained_inward:
                    self._transition(
                        track,
                        "PICKED_UP",
                        "putback cancelled: stable ROI but outside safe ROI without inward motion"
                    )
                    return []

                if recent_outward and not current_inward and not sustained_inward:
                    self._transition(
                        track,
                        "PICKED_UP",
                        "putback cancelled: recent outward motion without inward evidence"
                    )
                    return []

                if current_outward and not current_inward:
                    self._emit_debug(
                        f"[PUTBACK-STABLE-BLOCKED] track={track.global_id} "
                        f"reason=outward_motion_in_stable "
                        f"outward={current_outward} inward={current_inward} "
                        f"stable_counter={track.current_stable_counter}",
                        level="info",
                    )
                    return []

                # Append confidence for this frame
                self._append_confirmation(track, update.confidence)

                # =============================================================
                # ADDED: Require sustained inward or at least 2 frames of inward
                # before we even consider stable count.
                # =============================================================
                if current_inward and not sustained_inward and track.current_stable_counter < 2:
                    # First inward frame but not sustained yet; keep waiting.
                    return []

                # =============================================================
                # FIX: Use reset stable counter (now counts only frames in this state)
                # =============================================================
                if track.current_stable_counter >= PUTBACK_STABLE_FRAMES:
                    # Confidence window must also pass
                    if self._confidence_ok(track):
                        matched = self._match_putback_to_pending_pickup(track, update.timestamp_ms)
                        if matched is not None:
                            track.resolved_class_name = matched["event_class"]
                            track.candidate_putback_ledger_id = matched["ledger_id"]

                        event = self._record_putback(track)
                        if event is not None:
                            self._transition(track, "STABLE_INSIDE", "putback confirmed: stable in shelf ROI")
                            return [event]
                        else:
                            self._transition(track, "PICKED_UP", "putback validation failed")
                            return []
                    else:
                        # Confidence failed – cancel putback
                        self._transition(track, "PICKED_UP", "putback cancelled: confidence window failed")
                        return []

                # Not enough stable frames yet; keep pending.
                return []

            if update.in_outer_roi:
                if current_outward and not current_inward:
                    self._transition(
                        track,
                        "PICKED_UP",
                        "putback cancelled: outward motion in outer ROI"
                    )
                    return []

                if not current_inward and not sustained_inward:
                    if self._pending_timeout_elapsed(track, update.timestamp_ms):
                        self._transition(
                            track,
                            "PICKED_UP",
                            "putback cancelled: outer ROI timeout without inward motion"
                        )
                    return []

                self._append_confirmation(track, update.confidence)

                if len(track.confirmation_confidences) >= CONFIRM_FRAMES:
                    if not self._confidence_ok(track):
                        self._transition(track, "PICKED_UP", "putback cancelled: confidence too low in outer")
                        return []

                if self._pending_timeout_elapsed(track, update.timestamp_ms):
                    if current_inward or sustained_inward:
                        track.pending_since_ms = update.timestamp_ms
                    else:
                        self._transition(
                            track,
                            "PICKED_UP",
                            "putback cancelled: timeout with no inward motion"
                        )

                return []

            if self._pending_timeout_elapsed(track, update.timestamp_ms):
                self._transition(track, "PICKED_UP", "putback cancelled: timeout while absent")
                return []

            return []

        # Final fallback (should never be reached)
        return []

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def process_frame(self, tracks, frame_avg_confidence, timestamp_ms):
        for track in tracks:
            self._ensure_identity_fields(track)

        self._cleanup_pending_pickups(timestamp_ms)

        self._frame_net_snapshot = {
            class_name: self.pickup_count[class_name] - self.putback_count[class_name]
            for class_name in set(self.pickup_count) | set(self.putback_count)
        }

        self._frame_putback_used = defaultdict(int)

        events = []

        for track in tracks:
            produced = self._process_track(track, timestamp_ms, all_tracks=tracks)
            if produced:
                events.extend(produced)

        self._frame_net_snapshot = {}
        self._frame_putback_used = defaultdict(int)

        return events

    def finalize(self):
        classes = set(self.pickup_count) | set(self.putback_count)

        self.session.pickup_count = dict(self.pickup_count)
        self.session.putback_count = dict(self.putback_count)
        self.session.net_change = {
            cls: self.pickup_count[cls] - self.putback_count[cls]
            for cls in classes
        }
        self.session.total_pickups = sum(self.pickup_count.values())
        self.session.total_putbacks = sum(self.putback_count.values())

        if hasattr(self.session, "pending_pickups"):
            self.session.pending_pickups = [
                item for item in self.pending_pickups
                if not item.get("closed")
            ]

        return self.session