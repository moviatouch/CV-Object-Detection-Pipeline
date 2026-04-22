from __future__ import annotations  # Allow forward references in type hints

import logging
from collections import defaultdict, deque
from typing import Deque, Dict, List

from vending_pickup_putback.config import (
    CLASS_RETURN_MAX_GAP_MS,
    CONFIRM_FRAMES,
    CONF_MIN_FRAME,
    CONF_THRESHOLD,
    CRITICAL_CONF_THRESHOLD,
    LONG_BOUNDARY_STABILITY_FRAMES,
    LOW_CONF_AVG_THRESHOLD,
    LOW_CONF_FRAMES,
    MAX_PICKUPS_PER_CLASS_PER_SECOND,
    MERGE_FRAMES,
    MERGE_IOU_THRESHOLD,
    MISSING_PENDING_CONFIRM_MS,
    OVERLAY_EVENT_TTL_MS,
    OVERLAY_MAX_EVENT_LINES,
    PRINT_DEBUG_EVENTS,
    STABILITY_FRAMES,
)
from vending_pickup_putback.utils.roi_utils import bbox_iou
from vending_pickup_putback.utils.types import GlobalTrack, SessionSummary

LOGGER = logging.getLogger("vending_pipeline.events")


class EventManager:
    """Manages pickup/putback event detection for tracked objects in a vending machine scene."""

    def __init__(self, session_id: str) -> None:
        # Core session data
        self.session = SessionSummary(session_id=session_id)

        # Counters per product class
        self.pickup_count: Dict[str, int] = defaultdict(int)
        self.putback_count: Dict[str, int] = defaultdict(int)

        # Rate limiting: store timestamps of recent pickups per class
        self.class_pickup_timestamps: Dict[str, Deque[float]] = defaultdict(deque)

        # Low‑light detection: sliding window of frame‑average confidences
        self.low_light_confidences: Deque[float] = deque(maxlen=LOW_CONF_FRAMES)

        # Dynamic thresholds (may be changed in low‑light mode)
        self.frame_confirm_frames = CONFIRM_FRAMES
        self.frame_conf_threshold = CONF_THRESHOLD
        self.frame_conf_min = CONF_MIN_FRAME
        self.frame_stability_frames = STABILITY_FRAMES

        # Mode flags
        self.low_light_mode = False      # Currently in low‑light adaptation
        self.events_paused = False       # Extremely low confidence → no events

        # Overlay / debug
        self.last_event_text = ""
        self.recent_overlay_events: Deque[tuple[float, str]] = deque(maxlen=20)

    # -------------------------------------------------------------------------
    # Helper: logging and overlay
    # -------------------------------------------------------------------------
    def _emit_debug(self, message: str, *, level: str = "debug") -> None:
        """Log a message and optionally print to console."""
        getattr(LOGGER, level)(message)
        if PRINT_DEBUG_EVENTS:
            print(message)

    def _push_overlay_event(self, timestamp_ms: float, message: str) -> None:
        """Store an event string for on‑screen overlay."""
        self.recent_overlay_events.append((timestamp_ms, message))
        self.last_event_text = message

    def _log_transition(self, track: GlobalTrack, old_state: str, new_state: str, reason: str) -> None:
        """Log a state transition for a track."""
        message = (
            f"[TRACK {track.global_id}] {old_state} -> {new_state} | "
            f"class={track.class_name} | reason={reason}"
        )
        track.last_debug_reason = reason
        self._emit_debug(message, level="debug")

    def _transition(self, track: GlobalTrack, new_state: str, reason: str) -> None:
        """Change a track's state, resetting confirmation window or pending timer as needed."""
        if track.event_state == new_state:
            return
        old_state = track.event_state
        track.event_state = new_state
        # Clear confirmation window when entering a pending state; clear pending timer otherwise
        if new_state in {"PICKUP_PENDING", "PUTBACK_PENDING"}:
            track.confirmation_confidences.clear()
        else:
            track.pending_since_ms = None
        self._log_transition(track, old_state, new_state, reason)

    # -------------------------------------------------------------------------
    # Low‑light mode adaptation
    # -------------------------------------------------------------------------
    def _update_low_light_mode(self, frame_avg_confidence: float) -> None:
        """Adjust detection thresholds based on recent frame‑average confidence."""
        self.low_light_confidences.append(frame_avg_confidence)

        # Not enough history → use defaults, no pause
        if len(self.low_light_confidences) < LOW_CONF_FRAMES:
            self.events_paused = False
            self.low_light_mode = False
            self.frame_confirm_frames = CONFIRM_FRAMES
            self.frame_conf_threshold = CONF_THRESHOLD
            self.frame_conf_min = CONF_MIN_FRAME
            self.frame_stability_frames = STABILITY_FRAMES
            return

        running_avg = sum(self.low_light_confidences) / len(self.low_light_confidences)

        # Critical: pause all events
        if running_avg < CRITICAL_CONF_THRESHOLD:
            self.events_paused = True
            self.low_light_mode = True
            self.frame_confirm_frames = 7
            self.frame_conf_threshold = 0.7
            self.frame_conf_min = CONF_MIN_FRAME
            self.frame_stability_frames = 5
            return

        # Moderate low‑light: stricter confirmation, no pause
        if running_avg < LOW_CONF_AVG_THRESHOLD:
            self.events_paused = False
            self.low_light_mode = True
            self.frame_confirm_frames = 7
            self.frame_conf_threshold = 0.7
            self.frame_conf_min = CONF_MIN_FRAME
            self.frame_stability_frames = 5
            return

        # Normal confidence: restore defaults
        self.events_paused = False
        self.low_light_mode = False
        self.frame_confirm_frames = CONFIRM_FRAMES
        self.frame_conf_threshold = CONF_THRESHOLD
        self.frame_conf_min = CONF_MIN_FRAME
        self.frame_stability_frames = STABILITY_FRAMES

    # -------------------------------------------------------------------------
    # Merge locking – prevent duplicate events for overlapping tracks
    # -------------------------------------------------------------------------
    def _update_merge_locks(self, tracks: List[GlobalTrack]) -> None:
        """Detect persistent overlap between same‑class tracks and lock them together."""
        # Reset merge flags
        for track in tracks:
            track.merge_locked = False
            track.merge_owner_id = None

        # Compare every pair of tracks
        for idx, track in enumerate(tracks):
            if track.current_update is None:
                continue
            for other in tracks[idx + 1 :]:
                if other.current_update is None:
                    continue
                if track.class_name != other.class_name:
                    continue

                overlap = bbox_iou(track.current_update.bbox, other.current_update.bbox)

                # High IoU → increase overlap counter
                if overlap > MERGE_IOU_THRESHOLD:
                    track.overlap_history[other.global_id] = track.overlap_history.get(other.global_id, 0) + 1
                    other.overlap_history[track.global_id] = other.overlap_history.get(track.global_id, 0) + 1

                    # If high IoU persists for MERGE_FRAMES, lock both tracks
                    if track.overlap_history[other.global_id] > MERGE_FRAMES:
                        track.merge_locked = True
                        other.merge_locked = True
                        owner_id = min(track.global_id, other.global_id)  # older track owns the lock
                        track.merge_owner_id = owner_id
                        other.merge_owner_id = owner_id

                # Low IoU → reset counter and unlock
                elif overlap < 0.5:
                    track.overlap_history[other.global_id] = 0
                    other.overlap_history[track.global_id] = 0
                    track.merge_owner_id = None
                    other.merge_owner_id = None

    # -------------------------------------------------------------------------
    # Motion and confirmation helpers
    # -------------------------------------------------------------------------
    def _required_stability_frames(self, track: GlobalTrack, timestamp_ms: float) -> int:
        """How many stable frames are needed to become STABLE_INSIDE."""
        if track.boundary_linger_flagged:
            return LONG_BOUNDARY_STABILITY_FRAMES
        return self.frame_stability_frames

    def _has_sustained_outward_motion(self, track: GlobalTrack, min_consecutive: int = 2) -> bool:
        """Check if the track has been moving outward for at least min_consecutive frames."""
        if len(track.motion_direction_history) < min_consecutive:
            return False
        recent = list(track.motion_direction_history)[-min_consecutive:]
        return all(direction is True for direction in recent)

    def _has_sustained_inward_motion(self, track: GlobalTrack, min_consecutive: int = 2) -> bool:
        """Check if the track has been moving inward for at least min_consecutive frames."""
        if len(track.motion_direction_history) < min_consecutive:
            return False
        recent = list(track.motion_direction_history)[-min_consecutive:]
        return all(direction is False for direction in recent)

    def _append_confirmation(self, track: GlobalTrack, confidence: float) -> None:
        """Add a confidence value to the track's confirmation window, adjusting maxlen if needed."""
        if track.confirmation_confidences.maxlen != self.frame_confirm_frames:
            track.confirmation_confidences = deque(track.confirmation_confidences, maxlen=self.frame_confirm_frames)
        track.confirmation_confidences.append(confidence)

    def _seed_confirmation_from_history(self, track: GlobalTrack) -> None:
        """Fill the confirmation window with recent confidence history (for pickup pending)."""
        history = list(track.confidence_history)[-self.frame_confirm_frames :]
        track.confirmation_confidences = deque(history, maxlen=self.frame_confirm_frames)

    def _seed_class_return_confirmation(self, track: GlobalTrack) -> None:
        """Fill the confirmation window for a class‑level putback candidate (may need padding)."""
        history = list(track.confidence_history)[-self.frame_confirm_frames :]
        if not history:
            history = [track.last_confidence]
        while len(history) < self.frame_confirm_frames:
            history.append(history[-1])
        track.confirmation_confidences = deque(history, maxlen=self.frame_confirm_frames)

    def _pending_timeout_elapsed(self, track: GlobalTrack, timestamp_ms: float) -> bool:
        """Check if a pending event (pickup/putback) has timed out."""
        if track.pending_since_ms is None:
            return False
        return (timestamp_ms - track.pending_since_ms) >= MISSING_PENDING_CONFIRM_MS

    def _confirmation_ok(self, track: GlobalTrack) -> bool:
        """Return True if the track's confirmation window meets the current thresholds."""
        if len(track.confirmation_confidences) < self.frame_confirm_frames:
            return False
        avg_conf = sum(track.confirmation_confidences) / len(track.confirmation_confidences)
        min_conf = min(track.confirmation_confidences)
        return avg_conf >= self.frame_conf_threshold and min_conf >= self.frame_conf_min

    # -------------------------------------------------------------------------
    # Rate limiting and class‑level putback helpers
    # -------------------------------------------------------------------------
    def _rate_limited(self, class_name: str, timestamp_ms: float) -> bool:
        """True if too many pickups of this class occurred in the last second."""
        history = self.class_pickup_timestamps[class_name]
        while history and (timestamp_ms - history[0]) > 1000.0:
            history.popleft()
        return len(history) >= MAX_PICKUPS_PER_CLASS_PER_SECOND

    def _has_outstanding_pickup(self, class_name: str) -> bool:
        """Return True if there are more pickups than putbacks for this class."""
        return (self.pickup_count[class_name] - self.putback_count[class_name]) > 0

    def _latest_pickup_timestamp(self, class_name: str) -> float | None:
        """Return the timestamp of the most recent pickup for this class, or None."""
        history = self.class_pickup_timestamps[class_name]
        if not history:
            return None
        return history[-1]

    def _is_class_return_candidate(self, track: GlobalTrack) -> bool:
        """
        Determine if a track that is currently inside the ROI could be a putback
        of a previously picked item (class‑level logic).
        """
        update = track.current_update
        if update is None or not self._has_outstanding_pickup(track.class_name):
            return False
        latest_pickup_ts = self._latest_pickup_timestamp(track.class_name)
        if latest_pickup_ts is None:
            return False
        within_gap = (update.timestamp_ms - latest_pickup_ts) <= CLASS_RETURN_MAX_GAP_MS
        return within_gap and track.created_ms >= latest_pickup_ts and update.inward_motion

    # -------------------------------------------------------------------------
    # Recording events
    # -------------------------------------------------------------------------
    def _record_pickup(self, track: GlobalTrack) -> dict:
        """Record a pickup event, update counters, and return the event dict."""
        update = track.current_update
        self.pickup_count[track.class_name] += 1
        event_timestamp = track.last_seen_ms if update is None else update.timestamp_ms
        event_camera_id = track.last_camera_id if update is None else update.camera_id
        event_frame_index = track.last_frame_index if update is None else update.frame_index

        self.class_pickup_timestamps[track.class_name].append(event_timestamp)

        event = {
            "type": "pickup",
            "class": track.class_name,
            "global_id": track.global_id,
            "camera_id": event_camera_id,
            "frame_index": event_frame_index,
            "timestamp_ms": event_timestamp,
            "confidence": float(sum(track.confirmation_confidences) / max(len(track.confirmation_confidences), 1)),
            "confidence_window": list(track.confirmation_confidences),
        }

        self.session.events.append(event)
        self.session.pickup_records.append(event)

        overlay = f"PICKUP {track.class_name} G{track.global_id}"
        self._push_overlay_event(event_timestamp, overlay)
        self._emit_debug(
            f"[EVENT] {overlay} | cam={event_camera_id} frame={event_frame_index} conf={list(track.confirmation_confidences)}",
            level="info",
        )
        return event

    def _record_putback(self, track: GlobalTrack) -> dict | None:
        """Record a putback event if there is an outstanding pickup to offset, else log warning."""
        update = track.current_update
        current_net = self.pickup_count[track.class_name] - self.putback_count[track.class_name]
        if current_net <= 0:
            warning = (
                f"Putback ignored for class {track.class_name} on track {track.global_id} because "
                "there is no prior pickup to offset."
            )
            self.session.warnings.append(warning)
            LOGGER.warning(warning)
            return None

        event_timestamp = track.last_seen_ms if update is None else update.timestamp_ms
        event_camera_id = track.last_camera_id if update is None else update.camera_id
        event_frame_index = track.last_frame_index if update is None else update.frame_index

        self.putback_count[track.class_name] += 1

        event = {
            "type": "putback",
            "class": track.class_name,
            "global_id": track.global_id,
            "camera_id": event_camera_id,
            "frame_index": event_frame_index,
            "timestamp_ms": event_timestamp,
            "confidence": float(sum(track.confirmation_confidences) / max(len(track.confirmation_confidences), 1)),
            "confidence_window": list(track.confirmation_confidences),
        }

        self.session.events.append(event)
        self.session.putback_records.append(event)

        overlay = f"PUTBACK {track.class_name} G{track.global_id}"
        self._push_overlay_event(event_timestamp, overlay)
        self._emit_debug(
            f"[EVENT] {overlay} | cam={event_camera_id} frame={event_frame_index} conf={list(track.confirmation_confidences)}",
            level="info",
        )
        return event

    # -------------------------------------------------------------------------
    # Overlay lines (for GUI)
    # -------------------------------------------------------------------------
    def overlay_event_lines(self, timestamp_ms: float) -> list[str]:
        """Return the most recent event strings that are still within the TTL."""
        while self.recent_overlay_events and (timestamp_ms - self.recent_overlay_events[0][0]) > OVERLAY_EVENT_TTL_MS:
            self.recent_overlay_events.popleft()
        return [item[1] for item in list(self.recent_overlay_events)[-OVERLAY_MAX_EVENT_LINES:]]

    # -------------------------------------------------------------------------
    # Core state machine for one track
    # -------------------------------------------------------------------------
    def _process_track(self, track: GlobalTrack, timestamp_ms: float) -> List[dict]:
        """Update the track's state and return any events generated in this frame."""
        # Global pause overrides everything
        if self.events_paused:
            return []

        update = track.current_update

        # --- No detection in this frame (object lost) ---
        if update is None:
            # Pending pickup: confirm if timeout and confirmation OK, otherwise cancel
            if track.event_state == "PICKUP_PENDING" and self._pending_timeout_elapsed(track, timestamp_ms):
                if self._confirmation_ok(track) and track.last_seen_in_safe_roi is False:
                    self._transition(track, "PICKED_UP", "pickup confirmed after non-return")
                    return [self._record_pickup(track)]
                self._emit_debug(
                    f"[SUPPRESS] pickup timeout track={track.global_id} class={track.class_name} "
                    f"last_safe={track.last_seen_in_safe_roi} conf={list(track.confirmation_confidences)}",
                    level="info",
                )
                self._transition(track, "STABLE_INSIDE", "pickup cancelled after pending timeout")

            # Pending putback: confirm if timeout and confirmation OK, otherwise cancel
            elif track.event_state == "PUTBACK_PENDING" and self._pending_timeout_elapsed(track, timestamp_ms):
                if self._confirmation_ok(track) and track.last_seen_in_safe_roi is True:
                    self._transition(track, "STABLE_INSIDE", "putback confirmed after non-return")
                    event = self._record_putback(track)
                    return [] if event is None else [event]
                self._emit_debug(
                    f"[SUPPRESS] putback timeout track={track.global_id} class={track.class_name} "
                    f"last_safe={track.last_seen_in_safe_roi} conf={list(track.confirmation_confidences)}",
                    level="info",
                )
                self._transition(track, "PICKED_UP", "putback cancelled after pending timeout")

            return []

        # --- Detection exists: update motion direction history ---
        if update.displacement_magnitude > 0:
            # outward_motion and inward_motion are mutually exclusive in the update object
            if update.outward_motion:
                motion_direction = True
            elif update.inward_motion:
                motion_direction = False
            else:
                motion_direction = None
        else:
            motion_direction = None
        track.motion_direction_history.append(motion_direction)

        # --- Boundary linger detection ---
        if update.in_outer_roi and not update.in_safe_roi:
            if track.boundary_linger_start_ms is None:
                track.boundary_linger_start_ms = update.timestamp_ms
            elif (update.timestamp_ms - track.boundary_linger_start_ms) > 1000.0:
                track.boundary_linger_flagged = True
        else:
            track.boundary_linger_start_ms = None

        # --- Shaky detections: only update confidence, no state changes ---
        if update.shaky:
            if track.event_state in {"PICKUP_PENDING", "PUTBACK_PENDING"}:
                self._append_confirmation(track, update.confidence)
            return []

        required_stability = self._required_stability_frames(track, update.timestamp_ms)

        # ========== State machine ==========
        # State: INSIDE (not yet stable)
        if track.event_state == "INSIDE":
            # Class‑level putback candidate overrides normal inside logic
            if self._is_class_return_candidate(track):
                self._transition(track, "PUTBACK_PENDING", "class-level inward re-entry candidate")
                track.pending_since_ms = update.timestamp_ms
                self._seed_class_return_confirmation(track)
                self._emit_debug(
                    f"[PENDING] class-putback track={track.global_id} class={track.class_name} "
                    f"safe={update.in_safe_roi} outer={update.in_outer_roi} conf_seed={list(track.confirmation_confidences)}",
                    level="info",
                )
                return []
            # Otherwise, count stable frames inside safe ROI
            if update.in_safe_roi and update.confidence >= self.frame_conf_threshold:
                track.inside_counter += 1
                if track.inside_counter >= required_stability:
                    self._transition(track, "STABLE_INSIDE", "stable inside ROI")
                    track.boundary_linger_flagged = False
            else:
                track.inside_counter = 0
            return []

        # State: STABLE_INSIDE
        if track.event_state == "STABLE_INSIDE":
            # Class‑level putback candidate (rare, but possible)
            if self._is_class_return_candidate(track):
                self._transition(track, "PUTBACK_PENDING", "class-level inward re-entry candidate")
                track.pending_since_ms = update.timestamp_ms
                self._seed_class_return_confirmation(track)
                self._emit_debug(
                    f"[PENDING] class-putback track={track.global_id} class={track.class_name} "
                    f"safe={update.in_safe_roi} outer={update.in_outer_roi} conf_seed={list(track.confirmation_confidences)}",
                    level="info",
                )
                return []
            # Outward motion and leaving safe ROI → potential pickup
            if update.outward_motion and not update.in_safe_roi:
                self._transition(track, "PICKUP_PENDING", "outward exit candidate")
                track.pending_since_ms = update.timestamp_ms
                self._seed_confirmation_from_history(track)
                self._emit_debug(
                    f"[PENDING] pickup track={track.global_id} class={track.class_name} "
                    f"safe={update.in_safe_roi} outer={update.in_outer_roi} conf_seed={list(track.confirmation_confidences)}",
                    level="info",
                )
            return []

        # State: PICKUP_PENDING
        if track.event_state == "PICKUP_PENDING":
            # Cancel if object re‑enters safe ROI or stops outward motion
            if update.in_safe_roi or not update.outward_motion:
                self._emit_debug(
                    f"[CANCEL] pickup track={track.global_id} class={track.class_name} "
                    f"reentered={update.in_safe_roi} outward={update.outward_motion}",
                    level="info",
                )
                self._transition(track, "STABLE_INSIDE", "pickup cancelled by re-entry or weak motion")
                return []
            # Add confidence to window
            self._append_confirmation(track, update.confidence)
            # If window full and fails thresholds → cancel
            if len(track.confirmation_confidences) >= self.frame_confirm_frames and not self._confirmation_ok(track):
                self._emit_debug(
                    f"[CANCEL] pickup confidence track={track.global_id} conf={list(track.confirmation_confidences)}",
                    level="info",
                )
                self._transition(track, "STABLE_INSIDE", "pickup cancelled by confidence window")
                return []
            if not self._confirmation_ok(track):
                return []
            # Merge lock suppression: if this track is a child of a merge, do not confirm
            if track.merge_locked and track.merge_owner_id not in {None, track.global_id}:
                self._emit_debug(f"[SUPPRESS] pickup merge-lock track={track.global_id}", level="info")
                self._transition(track, "STABLE_INSIDE", "pickup suppressed by merge lock")
                return []
            # Rate limit per class
            if self._rate_limited(track.class_name, update.timestamp_ms):
                self._emit_debug(f"[SUPPRESS] pickup rate-limit class={track.class_name}", level="info")
                self._transition(track, "STABLE_INSIDE", "pickup suppressed by rate limit")
                return []
            # Confirmation successful
            self._transition(track, "PICKED_UP", "pickup confirmed")
            return [self._record_pickup(track)]

        # State: PICKED_UP (item has been taken, now looking for return)
        if track.event_state == "PICKED_UP":
            # Sustained inward motion while in outer or safe ROI → potential putback
            if self._has_sustained_inward_motion(track, min_consecutive=2) and (update.in_outer_roi or update.in_safe_roi):
                self._transition(track, "PUTBACK_PENDING", "sustained inward motion detected")
                track.pending_since_ms = update.timestamp_ms
                self._seed_confirmation_from_history(track)
                self._emit_debug(
                    f"[PENDING] putback track={track.global_id} class={track.class_name} "
                    f"safe={update.in_safe_roi} outer={update.in_outer_roi} "
                    f"motion_history={list(track.motion_direction_history)} "
                    f"conf_seed={list(track.confirmation_confidences)}",
                    level="info",
                )
            return []

        # State: PUTBACK_PENDING
        if track.event_state == "PUTBACK_PENDING":
            # If inside safe ROI: build confirmation window
            if update.in_safe_roi:
                self._append_confirmation(track, update.confidence)
                if len(track.confirmation_confidences) >= self.frame_confirm_frames and not self._confirmation_ok(track):
                    self._emit_debug(
                        f"[CANCEL] putback confidence track={track.global_id} conf={list(track.confirmation_confidences)}",
                        level="info",
                    )
                    self._transition(track, "PICKED_UP", "putback cancelled by confidence window")
                    return []
                if self._confirmation_ok(track):
                    self._transition(track, "STABLE_INSIDE", "putback confirmed")
                    event = self._record_putback(track)
                    return [] if event is None else [event]
                return []

            # Outside safe ROI: require inward motion or two consecutive inward directions
            if update.inward_motion or (len(track.motion_direction_history) >= 2 and
                                        list(track.motion_direction_history)[-2:] == [False, False]):
                self._append_confirmation(track, update.confidence)
                if len(track.confirmation_confidences) >= self.frame_confirm_frames and not self._confirmation_ok(track):
                    self._emit_debug(
                        f"[CANCEL] putback confidence track={track.global_id} conf={list(track.confirmation_confidences)}",
                        level="info",
                    )
                    self._transition(track, "PICKED_UP", "putback cancelled by confidence window")
                    return []
            # Timeout if no re‑entry and no inward motion for too long
            elif self._pending_timeout_elapsed(track, update.timestamp_ms):
                self._emit_debug(
                    f"[CANCEL] putback timeout track={track.global_id} class={track.class_name} "
                    f"safe={update.in_safe_roi} inward={update.inward_motion} "
                    f"motion_history={list(track.motion_direction_history)} "
                    f"disp={tuple(float(v) for v in update.displacement_vector)}",
                    level="info",
                )
                self._transition(track, "PICKED_UP", "putback cancelled after no re-entry")
                return []

            return []

        # Should never reach here
        return []

    # -------------------------------------------------------------------------
    # Public frame processing and finalization
    # -------------------------------------------------------------------------
    def process_frame(self, tracks: List[GlobalTrack], frame_avg_confidence: float, timestamp_ms: float) -> List[dict]:
        """Process all tracks for one frame, update low‑light mode and merge locks, return new events."""
        self._update_low_light_mode(frame_avg_confidence)
        self._update_merge_locks(tracks)
        events: List[dict] = []
        for track in tracks:
            events.extend(self._process_track(track, timestamp_ms))
        return events

    def finalize(self) -> SessionSummary:
        """Finish the session, compute net changes, and return the session summary."""
        classes = set(self.pickup_count) | set(self.putback_count)
        self.session.pickup_count = dict(self.pickup_count)
        self.session.putback_count = dict(self.putback_count)
        self.session.net_change = {
            class_name: self.pickup_count[class_name] - self.putback_count[class_name]
            for class_name in classes
        }
        return self.session