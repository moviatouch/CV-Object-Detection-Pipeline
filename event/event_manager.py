from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple

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
    STABILITY_FRAMES,
)
from utils.types import GlobalTrack, SessionSummary

LOGGER = logging.getLogger("vending_pipeline.events")


class EventManager:
    """
    Manages pickup/putback event detection for tracked objects in a vending machine scene.
    """

    def __init__(self, session_id: str) -> None:
        self.session = SessionSummary(session_id=session_id)

        self.pickup_count: Dict[str, int] = defaultdict(int)
        self.putback_count: Dict[str, int] = defaultdict(int)
        self.class_pickup_timestamps: Dict[str, Deque[float]] = defaultdict(deque)

        # Pre-frame net snapshot — computed once before the per-track loop so
        # simultaneous putbacks in the same frame all see the same baseline.
        self._frame_net_snapshot: Dict[str, int] = {}

        self.last_event_text = ""
        self.recent_overlay_events: Deque[tuple[float, str]] = deque(maxlen=20)

    # ─────────────────────────────────────────────────────────────────────────
    # Logging / overlay
    # ─────────────────────────────────────────────────────────────────────────

    def _emit_debug(self, message: str, *, level: str = "debug") -> None:
        getattr(LOGGER, level)(message)
        if PRINT_DEBUG_EVENTS:
            print(message)

    def _push_overlay_event(self, timestamp_ms: float, message: str) -> None:
        self.recent_overlay_events.append((timestamp_ms, message))
        self.last_event_text = message

    def _replace_last_overlay(self, message: str) -> None:
        if self.recent_overlay_events:
            ts, _ = self.recent_overlay_events[-1]
            self.recent_overlay_events[-1] = (ts, message)
            self.last_event_text = message

    def _log_transition(
        self, track: GlobalTrack, old_state: str, new_state: str, reason: str
    ) -> None:
        message = (
            f"[TRACK {track.global_id}] {old_state} -> {new_state} | "
            f"class={track.class_name} | reason={reason}"
        )
        track.last_debug_reason = reason
        self._emit_debug(message, level="debug")

    def _transition(self, track: GlobalTrack, new_state: str, reason: str) -> None:
        """Change track FSM state with consistent side-effects."""
        if track.event_state == new_state:
            return
        old_state = track.event_state
        track.event_state = new_state

        if new_state == "STABLE_INSIDE":
            track.was_stable = True

        # Hesitation counter no longer used – removed reset logic
        # (previously track.hesitation_resets = 0 for PICKUP/PUTBACK_PENDING)

        if new_state in {"PICKUP_PENDING", "PUTBACK_PENDING"}:
            track.confirmation_confidences.clear()
        else:
            track.pending_since_ms = None

        self._log_transition(track, old_state, new_state, reason)

    # ─────────────────────────────────────────────────────────────────────────
    # Motion helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _required_stability_frames(self, track: GlobalTrack, timestamp_ms: float) -> int:
        return (
            LONG_BOUNDARY_STABILITY_FRAMES
            if track.boundary_linger_flagged
            else STABILITY_FRAMES
        )

    def _has_any_recent_outward_motion(self, track: GlobalTrack, look_back: int = 3) -> bool:
        if not track.motion_direction_history:
            return False
        recent = list(track.motion_direction_history)[-look_back:]
        return any(d is True for d in recent)

    def _has_any_recent_inward_motion(self, track: GlobalTrack, look_back: int = 3) -> bool:
        if not track.motion_direction_history:
            return False
        recent = list(track.motion_direction_history)[-look_back:]
        return any(d is False for d in recent)

    def _has_sustained_outward_motion(self, track: GlobalTrack, min_consecutive: int = 2) -> bool:
        if len(track.motion_direction_history) < min_consecutive:
            return False
        return all(d is True for d in list(track.motion_direction_history)[-min_consecutive:])

    def _has_sustained_inward_motion(self, track: GlobalTrack, min_consecutive: int = 2) -> bool:
        if len(track.motion_direction_history) < min_consecutive:
            return False
        return all(d is False for d in list(track.motion_direction_history)[-min_consecutive:])

    # ─────────────────────────────────────────────────────────────────────────
    # ROI-transition evidence helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _roi_exit_occurred(
        track: GlobalTrack,
        update,
        prev_stable: Optional[bool],
    ) -> bool:
        """
        True when a confirmed-stable product has just left the stable shelf zone.
        """
        return (
            track.was_stable
            and prev_stable is True
            and not update.in_stable_roi
        )

    @staticmethod
    def _roi_reentry_occurred(
        update,
        prev_safe: Optional[bool],
        prev_outer: Optional[bool],
    ) -> bool:
        """
        True when a product re-enters any ROI zone after being fully absent.
        """
        was_fully_absent = (prev_safe is False) and (prev_outer is False)
        now_in_roi = update.in_safe_roi or update.in_outer_roi
        return was_fully_absent and now_in_roi

    # ─────────────────────────────────────────────────────────────────────────
    # Confirmation window helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _append_confirmation(self, track: GlobalTrack, confidence: float) -> None:
        if track.confirmation_confidences.maxlen != CONFIRM_FRAMES:
            track.confirmation_confidences = deque(
                track.confirmation_confidences, maxlen=CONFIRM_FRAMES
            )
        track.confirmation_confidences.append(confidence)

    def _seed_confirmation_from_history(self, track: GlobalTrack) -> None:
        history = list(track.confidence_history)[-CONFIRM_FRAMES:]
        track.confirmation_confidences = deque(history, maxlen=CONFIRM_FRAMES)

    def _seed_class_return_confirmation(self, track: GlobalTrack) -> None:
        history = list(track.confidence_history)[-CONFIRM_FRAMES:]
        if not history:
            history = [track.last_confidence]
        while len(history) < CONFIRM_FRAMES:
            history.append(history[-1])
        track.confirmation_confidences = deque(history, maxlen=CONFIRM_FRAMES)

    def _pending_timeout_elapsed(self, track: GlobalTrack, timestamp_ms: float) -> bool:
        if track.pending_since_ms is None:
            return False
        return (timestamp_ms - track.pending_since_ms) >= MISSING_PENDING_CONFIRM_MS

    def _confirmation_ok(self, track: GlobalTrack) -> bool:
        if len(track.confirmation_confidences) < CONFIRM_FRAMES:
            return False
        avg_conf = sum(track.confirmation_confidences) / len(track.confirmation_confidences)
        min_conf = min(track.confirmation_confidences)
        return avg_conf >= CONF_THRESHOLD and min_conf >= CONF_MIN_FRAME

    # ─────────────────────────────────────────────────────────────────────────
    # Class-level putback helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _has_outstanding_pickup(self, class_name: str) -> bool:
        return (self.pickup_count[class_name] - self.putback_count[class_name]) > 0

    def _latest_pickup_timestamp(self, class_name: str) -> Optional[float]:
        history = self.class_pickup_timestamps[class_name]
        return history[-1] if history else None

    def _is_class_return_candidate(self, track: GlobalTrack) -> bool:
        """
        True when a newly-appearing track looks like the return of a recently
        picked item from the same class.
        """
        update = track.current_update
        if update is None or not self._has_outstanding_pickup(track.class_name):
            return False
        latest_pickup_ts = self._latest_pickup_timestamp(track.class_name)
        if latest_pickup_ts is None:
            return False
        within_gap = (update.timestamp_ms - latest_pickup_ts) <= CLASS_RETURN_MAX_GAP_MS
        born_after_pickup = track.created_ms >= latest_pickup_ts
        not_previously_stable = not track.was_stable
        has_inward = (
            update.inward_motion
            or self._has_any_recent_inward_motion(track, look_back=5)
        )
        return within_gap and born_after_pickup and not_previously_stable and has_inward

    # ─────────────────────────────────────────────────────────────────────────
    # Neighbor stability boost
    # ─────────────────────────────────────────────────────────────────────────

    def _check_neighbor_stability_boost(
        self, track: GlobalTrack, all_tracks: List[GlobalTrack]
    ) -> None:
        """
        Promote adjacent half-stable INSIDE tracks to STABLE_INSIDE when one
        track is grabbed, preventing a grab disturbance from resetting neighbours.
        """
        update = track.current_update
        if update is None:
            return
        for neighbor in all_tracks:
            if neighbor.global_id == track.global_id:
                continue
            if neighbor.event_state not in {"INSIDE", "STABLE_INSIDE"}:
                continue
            if neighbor.stability_boosted:
                continue
            neighbor_update = neighbor.current_update
            if neighbor_update is None:
                continue
            same_zone = (
                (update.in_stable_roi and neighbor_update.in_stable_roi)
                or (update.in_outer_roi and neighbor_update.in_outer_roi)
            )
            half_stability = neighbor.inside_counter >= max(1, STABILITY_FRAMES // 2)
            if same_zone and half_stability:
                required = self._required_stability_frames(neighbor, update.timestamp_ms)
                neighbor.inside_counter = required
                neighbor.event_state = "STABLE_INSIDE"
                neighbor.stability_boosted = True
                neighbor.was_stable = True
                self._emit_debug(
                    f"[BOOST] neighbor track={neighbor.global_id} → STABLE_INSIDE "
                    f"by track={track.global_id}",
                    level="debug",
                )

    # ─────────────────────────────────────────────────────────────────────────
    # Recording events
    # ─────────────────────────────────────────────────────────────────────────

    def _record_pickup(self, track: GlobalTrack) -> dict:
        """
        Record a pickup event.
        """
        update = track.current_update
        self.pickup_count[track.class_name] += 1

        event_timestamp   = track.last_seen_ms      if update is None else update.timestamp_ms
        event_camera_id   = track.last_camera_id    if update is None else update.camera_id
        event_frame_index = track.last_frame_index  if update is None else update.frame_index

        self.class_pickup_timestamps[track.class_name].append(event_timestamp)
        track.pickup_confirmed_ms = event_timestamp

        event: dict = {
            "type": "pickup",
            "class": track.class_name,
            "global_id": track.global_id,
            "camera_id": event_camera_id,
            "frame_index": event_frame_index,
            "timestamp_ms": event_timestamp,
            "confidence": float(
                sum(track.confirmation_confidences)
                / max(len(track.confirmation_confidences), 1)
            ),
            "confidence_window": list(track.confirmation_confidences),
        }
        self.session.events.append(event)

        overlay = f"PICKUP {track.class_name} G{track.global_id}"
        self._push_overlay_event(event_timestamp, overlay)
        self._emit_debug(
            f"[EVENT] {overlay} | cam={event_camera_id} frame={event_frame_index} "
            f"conf={list(track.confirmation_confidences)}",
            level="info",
        )
        return event

    def _record_putback(self, track: GlobalTrack) -> Optional[dict]:
        """
        Record a putback event.
        """
        update = track.current_update

        current_net = self._frame_net_snapshot.get(track.class_name, 0)
        if current_net <= 0:
            warning = (
                f"Putback ignored class={track.class_name} track={track.global_id} "
                "— no outstanding pickup to offset."
            )
            self.session.warnings.append(warning)
            LOGGER.warning(warning)
            return None

        event_timestamp   = track.last_seen_ms      if update is None else update.timestamp_ms
        event_camera_id   = track.last_camera_id    if update is None else update.camera_id
        event_frame_index = track.last_frame_index  if update is None else update.frame_index

        self.putback_count[track.class_name] += 1
        track.last_putback_confirmed_ms = event_timestamp

        is_long_hold = (
            track.pickup_confirmed_ms is not None
            and (event_timestamp - track.pickup_confirmed_ms) > CLASS_RETURN_MAX_GAP_MS
        )

        event: dict = {
            "type": "putback",
            "class": track.class_name,
            "global_id": track.global_id,
            "camera_id": event_camera_id,
            "frame_index": event_frame_index,
            "timestamp_ms": event_timestamp,
            "confidence": float(
                sum(track.confirmation_confidences)
                / max(len(track.confirmation_confidences), 1)
            ),
            "confidence_window": list(track.confirmation_confidences),
        }
        if is_long_hold:
            event["long_hold_return"] = True

        self.session.events.append(event)

        overlay = f"PUTBACK {track.class_name} G{track.global_id}"
        if is_long_hold:
            overlay += " [long-hold]"
        self._push_overlay_event(event_timestamp, overlay)
        self._emit_debug(
            f"[EVENT] {overlay} | cam={event_camera_id} frame={event_frame_index} "
            f"conf={list(track.confirmation_confidences)}",
            level="info",
        )
        return event

    # ─────────────────────────────────────────────────────────────────────────
    # Overlay
    # ─────────────────────────────────────────────────────────────────────────

    def overlay_event_lines(self, timestamp_ms: float) -> List[str]:
        while (
            self.recent_overlay_events
            and (timestamp_ms - self.recent_overlay_events[0][0]) > OVERLAY_EVENT_TTL_MS
        ):
            self.recent_overlay_events.popleft()
        return [
            item[1]
            for item in list(self.recent_overlay_events)[-OVERLAY_MAX_EVENT_LINES:]
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # Core state machine — one track per call
    # ─────────────────────────────────────────────────────────────────────────

    def _process_track(
        self,
        track: GlobalTrack,
        timestamp_ms: float,
        all_tracks: Optional[List[GlobalTrack]] = None,
    ) -> List[dict]:
        """
        Run one FSM step for a single track. Returns new event dicts (may be empty).
        """
        update = track.current_update

        # =====================================================================
        # BRANCH A: no detection this frame
        # =====================================================================
        if update is None:
            if track.event_state == "PICKUP_PENDING":
                if self._pending_timeout_elapsed(track, timestamp_ms):

                    # A1 — outward motion history present
                    if self._has_any_recent_outward_motion(track, look_back=5):
                        self._transition(
                            track, "PICKED_UP",
                            "pickup confirmed: lost (outward motion)",
                        )
                        return [self._record_pickup(track)]

                    # A2 — no motion but was_stable: occlusion / vertical lift
                    elif track.was_stable:
                        self._transition(
                            track, "PICKED_UP",
                            "pickup confirmed: lost (was stable — occlusion/vertical)",
                        )
                        event = self._record_pickup(track)
                        event["occlusion_pickup"] = True
                        self._replace_last_overlay(
                            f"PICKUP {track.class_name} G{track.global_id} [occluded]"
                        )
                        return [event]

                    # A3 — never stable: silent cancel
                    else:
                        self._emit_debug(
                            f"[SILENT-CANCEL] track={track.global_id} "
                            "never reached stability — suppressed",
                            level="debug",
                        )
                        self._transition(
                            track, "INSIDE",
                            "pickup cancelled: never reached stability",
                        )
                        return []

            elif track.event_state == "PUTBACK_PENDING":
                if self._pending_timeout_elapsed(track, timestamp_ms):
                    if (
                        track.last_seen_in_stable_roi is True
                        and self._has_any_recent_inward_motion(track, look_back=5)
                    ):
                        self._transition(
                            track, "STABLE_INSIDE",
                            "putback confirmed: lost in stable ROI",
                        )
                        event = self._record_putback(track)
                        return [] if event is None else [event]
                    self._emit_debug(
                        f"[TIMEOUT] putback absent track={track.global_id} "
                        f"last_stable={track.last_seen_in_stable_roi}",
                        level="info",
                    )
                    self._transition(
                        track, "PICKED_UP",
                        "putback cancelled: lost without entering stable ROI",
                    )
            return []

        # =====================================================================
        # BRANCH B: detection exists this frame
        # =====================================================================

        # Capture PREVIOUS frame ROI state before overwriting
        prev_safe:  Optional[bool] = track.last_seen_in_safe_roi
        prev_outer: Optional[bool] = track.last_seen_in_outer_roi
        prev_stable: Optional[bool] = track.last_seen_in_stable_roi

        # Update for future frames and for BRANCH A lost-track logic
        track.last_seen_in_safe_roi  = update.in_safe_roi
        track.last_seen_in_outer_roi = update.in_outer_roi
        track.last_seen_in_stable_roi = update.in_stable_roi

        # Update motion-direction history
        if update.displacement_magnitude > 0:
            if update.outward_motion:
                motion_direction: Optional[bool] = True
            elif update.inward_motion:
                motion_direction = False
            else:
                motion_direction = None
        else:
            motion_direction = None
        track.motion_direction_history.append(motion_direction)

        # Boundary-linger detection
        if update.in_outer_roi and not update.in_stable_roi:
            if track.boundary_linger_start_ms is None:
                track.boundary_linger_start_ms = update.timestamp_ms
            elif (update.timestamp_ms - track.boundary_linger_start_ms) > 1000.0:
                track.boundary_linger_flagged = True
        else:
            track.boundary_linger_start_ms = None
        required_stability = self._required_stability_frames(track, update.timestamp_ms)

        # =====================================================================
        # STATE: INSIDE
        # =====================================================================
        if track.event_state == "INSIDE":
            if self._is_class_return_candidate(track):
                self._transition(
                    track, "PUTBACK_PENDING", "class-level inward re-entry candidate"
                )
                track.pending_since_ms = update.timestamp_ms
                self._seed_class_return_confirmation(track)
                self._emit_debug(
                    f"[PENDING] class-putback track={track.global_id} "
                    f"class={track.class_name}",
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

        # =====================================================================
        # STATE: STABLE_INSIDE
        # =====================================================================
        if track.event_state == "STABLE_INSIDE":
            if self._is_class_return_candidate(track):
                self._transition(
                    track, "PUTBACK_PENDING", "class-level inward re-entry candidate"
                )
                track.pending_since_ms = update.timestamp_ms
                self._seed_class_return_confirmation(track)
                self._emit_debug(
                    f"[PENDING] class-putback track={track.global_id} "
                    f"class={track.class_name}",
                    level="info",
                )
                return []

            # ── PICKUP TRIGGER ────────────────────────────────────────────────
            if not update.in_safe_roi:
                has_motion   = self._has_any_recent_outward_motion(track, look_back=3)
                has_roi_exit = self._roi_exit_occurred(track, update, prev_stable)

                if has_motion or has_roi_exit:
                    # Post-putback cooldown (per-track, not cross-class)
                    if (
                        track.last_putback_confirmed_ms is not None
                        and (update.timestamp_ms - track.last_putback_confirmed_ms)
                        < POST_PUTBACK_COOLDOWN_MS
                    ):
                        self._emit_debug(
                            f"[COOLDOWN] pickup suppressed track={track.global_id} "
                            f"elapsed={update.timestamp_ms - track.last_putback_confirmed_ms:.0f}ms "
                            f"cooldown={POST_PUTBACK_COOLDOWN_MS}ms",
                            level="debug",
                        )
                        return []

                    trigger_reason = (
                        "outward exit + motion"
                        if has_motion
                        else "ROI exit without motion (vertical/edge/fast/low-tray pick)"
                    )
                    self._transition(track, "PICKUP_PENDING", trigger_reason)
                    track.pending_since_ms = update.timestamp_ms
                    self._seed_confirmation_from_history(track)
                    if all_tracks is not None:
                        self._check_neighbor_stability_boost(track, all_tracks)
                    self._emit_debug(
                        f"[PENDING] pickup track={track.global_id} "
                        f"class={track.class_name} "
                        f"motion={has_motion} roi_exit={has_roi_exit} "
                        f"prev_safe={prev_safe} curr_safe={update.in_safe_roi} "
                        f"prev_stable={prev_stable} curr_stable={update.in_stable_roi} "
                        f"safe_dist={update.safe_roi_distance:.1f} "
                        f"motion_history={list(track.motion_direction_history)}",
                        level="info",
                    )
                    return []

            if update.in_stable_roi and update.confidence >= CONF_THRESHOLD:
                track.inside_counter = required_stability
            else:
                track.inside_counter = 0
            return []

        # =====================================================================
        # STATE: PICKUP_PENDING
        # =====================================================================
        if track.event_state == "PICKUP_PENDING":

            # ── Quick pick-and-return ─────────────────────────────────────────
            if update.in_stable_roi and not update.outward_motion:
                time_since_pending_ms = update.timestamp_ms - (
                    track.pending_since_ms or update.timestamp_ms
                )
                if time_since_pending_ms < QUICK_RETURN_THRESHOLD_MS:
                    self._transition(
                        track, "PICKED_UP",
                        "quick pickup: fast re-entry within threshold",
                    )
                    pickup_event = self._record_pickup(track)
                    self._frame_net_snapshot[track.class_name] = (
                        self._frame_net_snapshot.get(track.class_name, 0) + 1
                    )
                    self._transition(
                        track, "PUTBACK_PENDING",
                        "quick return: fast re-entry",
                    )
                    track.pending_since_ms = update.timestamp_ms
                    self._seed_confirmation_from_history(track)
                    self._emit_debug(
                        f"[QUICK-RETURN] track={track.global_id} "
                        f"time_ms={time_since_pending_ms:.0f}",
                        level="info",
                    )
                    return [pickup_event]

                # Re-entry after threshold → false alarm
                self._emit_debug(
                    f"[CANCEL] pickup track={track.global_id} "
                    "re-entered stable ROI after threshold",
                    level="info",
                )
                self._transition(
                    track, "STABLE_INSIDE",
                    "pickup cancelled: returned to stable ROI",
                )
                return []

            # ── Fully absent from both ROIs ───────────────────────────────────
            if not update.in_outer_roi:
                self._append_confirmation(track, update.confidence)
                if (
                    self._confirmation_ok(track)
                    or (update.timestamp_ms - (track.pending_since_ms or update.timestamp_ms)) > 100
                ):
                    self._transition(
                        track, "PICKED_UP",
                        "pickup confirmed: SAFE → OUTER → ABSENT",
                    )
                    return [self._record_pickup(track)]
                return []

            # ── In OUTER ROI ──────────────────────────────────────────────────
            if update.in_outer_roi:
                self._append_confirmation(track, update.confidence)

                if (
                    len(track.confirmation_confidences) >= CONFIRM_FRAMES
                    and not self._confirmation_ok(track)
                ):
                    self._emit_debug(
                        f"[CANCEL] pickup track={track.global_id} "
                        "confidence too low in OUTER",
                        level="info",
                    )
                    self._transition(
                        track, "STABLE_INSIDE",
                        "pickup cancelled: confidence too low",
                    )
                    return []

                if self._has_any_recent_outward_motion(track, look_back=3):
                    return []

                # HESITATION RESETS REMOVED: timeout now immediately cancels
                if self._pending_timeout_elapsed(track, update.timestamp_ms):
                    self._emit_debug(
                        f"[TIMEOUT] pickup OUTER track={track.global_id} "
                        "timeout elapsed",
                        level="info",
                    )
                    self._transition(
                        track, "STABLE_INSIDE",
                        "pickup cancelled: timeout in OUTER ROI",
                    )
                    return []

                return []

            return []

        # =====================================================================
        # STATE: PICKED_UP
        # =====================================================================
        if track.event_state == "PICKED_UP":

            # Long-hold fallback
            track_pickup_ts: Optional[float] = track.pickup_confirmed_ms
            is_long_hold = (
                track_pickup_ts is not None
                and (timestamp_ms - track_pickup_ts) > CLASS_RETURN_MAX_GAP_MS
            )
            if is_long_hold and (update.in_stable_roi or update.in_outer_roi):
                track.long_hold_pending = True
                self._transition(
                    track, "PUTBACK_PENDING",
                    "long-hold fallback: elapsed > CLASS_RETURN_MAX_GAP_MS",
                )
                track.pending_since_ms = update.timestamp_ms
                self._seed_confirmation_from_history(track)
                self._emit_debug(
                    f"[LONG-HOLD] track={track.global_id} class={track.class_name} "
                    f"elapsed_ms={timestamp_ms - track_pickup_ts:.0f}",
                    level="info",
                )
                return []

            # ── PUTBACK TRIGGER ───────────────────────────────────────────────
            if update.in_stable_roi or update.in_outer_roi:
                has_inward  = self._has_any_recent_inward_motion(track, look_back=3)
                has_reentry = self._roi_reentry_occurred(update, prev_safe, prev_outer)

                trigger_reason = (
                    "inward re-entry + motion" if has_inward
                    else "ROI re-entry from absent (vertical/edge/fast return)"
                    if has_reentry
                    else "ROI presence after pickup (product returned to zone)"
                )
                self._transition(track, "PUTBACK_PENDING", trigger_reason)
                track.pending_since_ms = update.timestamp_ms
                self._seed_confirmation_from_history(track)
                self._emit_debug(
                    f"[PENDING] putback track={track.global_id} "
                    f"class={track.class_name} "
                    f"safe={update.in_safe_roi} stable={update.in_stable_roi} "
                    f"outer={update.in_outer_roi} "
                    f"motion={has_inward} reentry={has_reentry} "
                    f"prev_safe={prev_safe} prev_outer={prev_outer}",
                    level="info",
                )
                return []

            return []

        # =====================================================================
        # STATE: PUTBACK_PENDING
        # =====================================================================
        if track.event_state == "PUTBACK_PENDING":

            # ── Stable ROI: accumulate and confirm ───────────────────────────
            if update.in_stable_roi:
                self._append_confirmation(track, update.confidence)
                if len(track.confirmation_confidences) >= CONFIRM_FRAMES:
                    if not self._confirmation_ok(track):
                        self._emit_debug(
                            f"[CANCEL] putback confidence track={track.global_id} "
                            f"conf={list(track.confirmation_confidences)}",
                            level="info",
                        )
                        self._transition(
                            track, "PICKED_UP",
                            "putback cancelled by confidence window",
                        )
                        return []
                    self._transition(
                        track, "STABLE_INSIDE",
                        "putback confirmed: stable in shelf ROI",
                    )
                    event = self._record_putback(track)
                    return [] if event is None else [event]
                return []

            # ── Outer ROI: accumulate unconditionally, single extension on timeout with inward motion
            if update.in_outer_roi:
                self._append_confirmation(track, update.confidence)
                if (
                    len(track.confirmation_confidences) >= CONFIRM_FRAMES
                    and not self._confirmation_ok(track)
                ):
                    self._emit_debug(
                        f"[CANCEL] putback OUTER confidence track={track.global_id} "
                        f"conf={list(track.confirmation_confidences)}",
                        level="info",
                    )
                    self._transition(
                        track, "PICKED_UP",
                        "putback cancelled: confidence too low in OUTER",
                    )
                    return []

                if self._pending_timeout_elapsed(track, update.timestamp_ms):
                    if not self._has_any_recent_inward_motion(track, look_back=5):
                        self._emit_debug(
                            f"[TIMEOUT] putback OUTER track={track.global_id} "
                            "no inward motion",
                            level="info",
                        )
                        self._transition(
                            track, "PICKED_UP",
                            "putback cancelled: timeout with no inward motion",
                        )
                        return []
                    # Extend window once (single extension, not multiple resets)
                    track.pending_since_ms = update.timestamp_ms
                return []

            # ── Absent from both zones ────────────────────────────────────────
            if self._pending_timeout_elapsed(track, update.timestamp_ms):
                self._emit_debug(
                    f"[TIMEOUT] putback absent track={track.global_id}",
                    level="info",
                )
                self._transition(
                    track, "PICKED_UP",
                    "putback cancelled: timeout while absent",
                )
                return []
            return []

        return []

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def process_frame(
        self,
        tracks: List[GlobalTrack],
        frame_avg_confidence: float,
        timestamp_ms: float,
    ) -> List[dict]:
        """
        Process all active tracks for one frame. Returns all new events generated.
        """
        self._frame_net_snapshot = {
            class_name: self.pickup_count[class_name] - self.putback_count[class_name]
            for class_name in set(self.pickup_count) | set(self.putback_count)
        }
        events: List[dict] = []
        for track in tracks:
            events.extend(self._process_track(track, timestamp_ms, all_tracks=tracks))
        self._frame_net_snapshot = {}
        return events

    def finalize(self) -> SessionSummary:
        """Compute net changes and totals, return the completed session summary."""
        classes = set(self.pickup_count) | set(self.putback_count)
        self.session.pickup_count = dict(self.pickup_count)
        self.session.putback_count = dict(self.putback_count)
        self.session.net_change = {
            cls: self.pickup_count[cls] - self.putback_count[cls]
            for cls in classes
        }
        self.session.total_pickups  = sum(self.pickup_count.values())
        self.session.total_putbacks = sum(self.putback_count.values())
        return self.session


