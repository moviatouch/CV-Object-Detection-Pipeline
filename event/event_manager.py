# from __future__ import annotations

# import logging
# from collections import defaultdict, deque
# from typing import Deque, Dict, List, Optional, Tuple

# from config import (
#     CLASS_RETURN_MAX_GAP_MS,
#     CONFIRM_FRAMES,
#     CONF_MIN_FRAME,
#     CONF_THRESHOLD,
#     LONG_BOUNDARY_STABILITY_FRAMES,
#     MISSING_PENDING_CONFIRM_MS,
#     OVERLAY_EVENT_TTL_MS,
#     OVERLAY_MAX_EVENT_LINES,
#     POST_PUTBACK_COOLDOWN_MS,
#     PRINT_DEBUG_EVENTS,
#     QUICK_RETURN_THRESHOLD_MS,
#     STABILITY_FRAMES,
# )
# from utils.types import GlobalTrack, SessionSummary

# LOGGER = logging.getLogger("vending_pipeline.events")


# class EventManager:
#     """
#     Manages pickup/putback event detection for tracked objects in a vending machine scene.
#     """

#     def __init__(self, session_id: str) -> None:
#         self.session = SessionSummary(session_id=session_id)

#         self.pickup_count: Dict[str, int] = defaultdict(int)
#         self.putback_count: Dict[str, int] = defaultdict(int)
#         self.class_pickup_timestamps: Dict[str, Deque[float]] = defaultdict(deque)

#         # Pre-frame net snapshot — computed once before the per-track loop so
#         # simultaneous putbacks in the same frame all see the same baseline.
#         self._frame_net_snapshot: Dict[str, int] = {}

#         self.last_event_text = ""
#         self.recent_overlay_events: Deque[tuple[float, str]] = deque(maxlen=20)

#     # ─────────────────────────────────────────────────────────────────────────
#     # Logging / overlay
#     # ─────────────────────────────────────────────────────────────────────────

#     def _emit_debug(self, message: str, *, level: str = "debug") -> None:
#         getattr(LOGGER, level)(message)
#         if PRINT_DEBUG_EVENTS:
#             print(message)

#     def _push_overlay_event(self, timestamp_ms: float, message: str) -> None:
#         self.recent_overlay_events.append((timestamp_ms, message))
#         self.last_event_text = message

#     def _replace_last_overlay(self, message: str) -> None:
#         if self.recent_overlay_events:
#             ts, _ = self.recent_overlay_events[-1]
#             self.recent_overlay_events[-1] = (ts, message)
#             self.last_event_text = message

#     def _log_transition(
#         self, track: GlobalTrack, old_state: str, new_state: str, reason: str
#     ) -> None:
#         message = (
#             f"[TRACK {track.global_id}] {old_state} -> {new_state} | "
#             f"class={track.class_name} | reason={reason}"
#         )
#         track.last_debug_reason = reason
#         self._emit_debug(message, level="debug")

#     def _transition(self, track: GlobalTrack, new_state: str, reason: str) -> None:
#         """Change track FSM state with consistent side-effects."""
#         if track.event_state == new_state:
#             return
#         old_state = track.event_state
#         track.event_state = new_state

#         if new_state == "STABLE_INSIDE":
#             track.was_stable = True

#         # Hesitation counter no longer used – removed reset logic
#         # (previously track.hesitation_resets = 0 for PICKUP/PUTBACK_PENDING)

#         if new_state in {"PICKUP_PENDING", "PUTBACK_PENDING"}:
#             track.confirmation_confidences.clear()
#         else:
#             track.pending_since_ms = None

#         self._log_transition(track, old_state, new_state, reason)

#     # ─────────────────────────────────────────────────────────────────────────
#     # Motion helpers
#     # ─────────────────────────────────────────────────────────────────────────

#     def _required_stability_frames(self, track: GlobalTrack, timestamp_ms: float) -> int:
#         return (
#             LONG_BOUNDARY_STABILITY_FRAMES
#             if track.boundary_linger_flagged
#             else STABILITY_FRAMES
#         )

#     def _has_any_recent_outward_motion(self, track: GlobalTrack, look_back: int = 3) -> bool:
#         if not track.motion_direction_history:
#             return False
#         recent = list(track.motion_direction_history)[-look_back:]
#         return any(d is True for d in recent)

#     def _has_any_recent_inward_motion(self, track: GlobalTrack, look_back: int = 3) -> bool:
#         if not track.motion_direction_history:
#             return False
#         recent = list(track.motion_direction_history)[-look_back:]
#         return any(d is False for d in recent)

#     def _has_sustained_outward_motion(self, track: GlobalTrack, min_consecutive: int = 2) -> bool:
#         if len(track.motion_direction_history) < min_consecutive:
#             return False
#         return all(d is True for d in list(track.motion_direction_history)[-min_consecutive:])

#     def _has_sustained_inward_motion(self, track: GlobalTrack, min_consecutive: int = 2) -> bool:
#         if len(track.motion_direction_history) < min_consecutive:
#             return False
#         return all(d is False for d in list(track.motion_direction_history)[-min_consecutive:])

#     # ─────────────────────────────────────────────────────────────────────────
#     # ROI-transition evidence helpers
#     # ─────────────────────────────────────────────────────────────────────────

#     @staticmethod
#     def _roi_exit_occurred(
#         track: GlobalTrack,
#         update,
#         prev_stable: Optional[bool],
#     ) -> bool:
#         """
#         True when a confirmed-stable product has just left the stable shelf zone.
#         """
#         return (
#             track.was_stable
#             and prev_stable is True
#             and not update.in_stable_roi
#         )

#     @staticmethod
#     def _roi_reentry_occurred(
#         update,
#         prev_safe: Optional[bool],
#         prev_outer: Optional[bool],
#     ) -> bool:
#         """
#         True when a product re-enters any ROI zone after being fully absent.
#         """
#         was_fully_absent = (prev_safe is False) and (prev_outer is False)
#         now_in_roi = update.in_safe_roi or update.in_outer_roi
#         return was_fully_absent and now_in_roi

#     # ─────────────────────────────────────────────────────────────────────────
#     # Confirmation window helpers
#     # ─────────────────────────────────────────────────────────────────────────

#     def _append_confirmation(self, track: GlobalTrack, confidence: float) -> None:
#         if track.confirmation_confidences.maxlen != CONFIRM_FRAMES:
#             track.confirmation_confidences = deque(
#                 track.confirmation_confidences, maxlen=CONFIRM_FRAMES
#             )
#         track.confirmation_confidences.append(confidence)

#     def _seed_confirmation_from_history(self, track: GlobalTrack) -> None:
#         history = list(track.confidence_history)[-CONFIRM_FRAMES:]
#         track.confirmation_confidences = deque(history, maxlen=CONFIRM_FRAMES)

#     def _seed_class_return_confirmation(self, track: GlobalTrack) -> None:
#         history = list(track.confidence_history)[-CONFIRM_FRAMES:]
#         if not history:
#             history = [track.last_confidence]
#         while len(history) < CONFIRM_FRAMES:
#             history.append(history[-1])
#         track.confirmation_confidences = deque(history, maxlen=CONFIRM_FRAMES)

#     def _pending_timeout_elapsed(self, track: GlobalTrack, timestamp_ms: float) -> bool:
#         if track.pending_since_ms is None:
#             return False
#         return (timestamp_ms - track.pending_since_ms) >= MISSING_PENDING_CONFIRM_MS

#     def _confirmation_ok(self, track: GlobalTrack) -> bool:
#         if len(track.confirmation_confidences) < CONFIRM_FRAMES:
#             return False
#         avg_conf = sum(track.confirmation_confidences) / len(track.confirmation_confidences)
#         min_conf = min(track.confirmation_confidences)
#         return avg_conf >= CONF_THRESHOLD and min_conf >= CONF_MIN_FRAME

#     # ─────────────────────────────────────────────────────────────────────────
#     # Class-level putback helpers
#     # ─────────────────────────────────────────────────────────────────────────

#     def _has_outstanding_pickup(self, class_name: str) -> bool:
#         return (self.pickup_count[class_name] - self.putback_count[class_name]) > 0

#     def _latest_pickup_timestamp(self, class_name: str) -> Optional[float]:
#         history = self.class_pickup_timestamps[class_name]
#         return history[-1] if history else None

#     def _is_class_return_candidate(self, track: GlobalTrack) -> bool:
#         """
#         True when a newly-appearing track looks like the return of a recently
#         picked item from the same class.
#         """
#         update = track.current_update
#         if update is None or not self._has_outstanding_pickup(track.class_name):
#             return False
#         latest_pickup_ts = self._latest_pickup_timestamp(track.class_name)
#         if latest_pickup_ts is None:
#             return False
#         within_gap = (update.timestamp_ms - latest_pickup_ts) <= CLASS_RETURN_MAX_GAP_MS
#         born_after_pickup = track.created_ms >= latest_pickup_ts
#         not_previously_stable = not track.was_stable
#         has_inward = (
#             update.inward_motion
#             or self._has_any_recent_inward_motion(track, look_back=5)
#         )
#         return within_gap and born_after_pickup and not_previously_stable and has_inward

#     # ─────────────────────────────────────────────────────────────────────────
#     # Neighbor stability boost
#     # ─────────────────────────────────────────────────────────────────────────

#     def _check_neighbor_stability_boost(
#         self, track: GlobalTrack, all_tracks: List[GlobalTrack]
#     ) -> None:
#         """
#         Promote adjacent half-stable INSIDE tracks to STABLE_INSIDE when one
#         track is grabbed, preventing a grab disturbance from resetting neighbours.
#         """
#         update = track.current_update
#         if update is None:
#             return
#         for neighbor in all_tracks:
#             if neighbor.global_id == track.global_id:
#                 continue
#             if neighbor.event_state not in {"INSIDE", "STABLE_INSIDE"}:
#                 continue
#             if neighbor.stability_boosted:
#                 continue
#             neighbor_update = neighbor.current_update
#             if neighbor_update is None:
#                 continue
#             same_zone = (
#                 (update.in_stable_roi and neighbor_update.in_stable_roi)
#                 or (update.in_outer_roi and neighbor_update.in_outer_roi)
#             )
#             half_stability = neighbor.inside_counter >= max(1, STABILITY_FRAMES // 2)
#             if same_zone and half_stability:
#                 required = self._required_stability_frames(neighbor, update.timestamp_ms)
#                 neighbor.inside_counter = required
#                 neighbor.event_state = "STABLE_INSIDE"
#                 neighbor.stability_boosted = True
#                 neighbor.was_stable = True
#                 self._emit_debug(
#                     f"[BOOST] neighbor track={neighbor.global_id} → STABLE_INSIDE "
#                     f"by track={track.global_id}",
#                     level="debug",
#                 )

#     # ─────────────────────────────────────────────────────────────────────────
#     # Recording events
#     # ─────────────────────────────────────────────────────────────────────────

#     def _record_pickup(self, track: GlobalTrack) -> dict:
#         """
#         Record a pickup event.
#         """
#         update = track.current_update
#         self.pickup_count[track.class_name] += 1

#         event_timestamp   = track.last_seen_ms      if update is None else update.timestamp_ms
#         event_camera_id   = track.last_camera_id    if update is None else update.camera_id
#         event_frame_index = track.last_frame_index  if update is None else update.frame_index

#         self.class_pickup_timestamps[track.class_name].append(event_timestamp)
#         track.pickup_confirmed_ms = event_timestamp

#         event: dict = {
#             "type": "pickup",
#             "class": track.class_name,
#             "global_id": track.global_id,
#             "camera_id": event_camera_id,
#             "frame_index": event_frame_index,
#             "timestamp_ms": event_timestamp,
#             "confidence": float(
#                 sum(track.confirmation_confidences)
#                 / max(len(track.confirmation_confidences), 1)
#             ),
#             "confidence_window": list(track.confirmation_confidences),
#         }
#         self.session.events.append(event)

#         overlay = f"PICKUP {track.class_name} G{track.global_id}"
#         self._push_overlay_event(event_timestamp, overlay)
#         self._emit_debug(
#             f"[EVENT] {overlay} | cam={event_camera_id} frame={event_frame_index} "
#             f"conf={list(track.confirmation_confidences)}",
#             level="info",
#         )
#         return event

#     def _record_putback(self, track: GlobalTrack) -> Optional[dict]:
#         """
#         Record a putback event.
#         """
#         update = track.current_update

#         current_net = self._frame_net_snapshot.get(track.class_name, 0)
#         if current_net <= 0:
#             warning = (
#                 f"Putback ignored class={track.class_name} track={track.global_id} "
#                 "— no outstanding pickup to offset."
#             )
#             self.session.warnings.append(warning)
#             LOGGER.warning(warning)
#             return None

#         event_timestamp   = track.last_seen_ms      if update is None else update.timestamp_ms
#         event_camera_id   = track.last_camera_id    if update is None else update.camera_id
#         event_frame_index = track.last_frame_index  if update is None else update.frame_index

#         self.putback_count[track.class_name] += 1
#         track.last_putback_confirmed_ms = event_timestamp

#         is_long_hold = (
#             track.pickup_confirmed_ms is not None
#             and (event_timestamp - track.pickup_confirmed_ms) > CLASS_RETURN_MAX_GAP_MS
#         )

#         event: dict = {
#             "type": "putback",
#             "class": track.class_name,
#             "global_id": track.global_id,
#             "camera_id": event_camera_id,
#             "frame_index": event_frame_index,
#             "timestamp_ms": event_timestamp,
#             "confidence": float(
#                 sum(track.confirmation_confidences)
#                 / max(len(track.confirmation_confidences), 1)
#             ),
#             "confidence_window": list(track.confirmation_confidences),
#         }
#         if is_long_hold:
#             event["long_hold_return"] = True

#         self.session.events.append(event)

#         overlay = f"PUTBACK {track.class_name} G{track.global_id}"
#         if is_long_hold:
#             overlay += " [long-hold]"
#         self._push_overlay_event(event_timestamp, overlay)
#         self._emit_debug(
#             f"[EVENT] {overlay} | cam={event_camera_id} frame={event_frame_index} "
#             f"conf={list(track.confirmation_confidences)}",
#             level="info",
#         )
#         return event

#     # ─────────────────────────────────────────────────────────────────────────
#     # Overlay
#     # ─────────────────────────────────────────────────────────────────────────

#     def overlay_event_lines(self, timestamp_ms: float) -> List[str]:
#         while (
#             self.recent_overlay_events
#             and (timestamp_ms - self.recent_overlay_events[0][0]) > OVERLAY_EVENT_TTL_MS
#         ):
#             self.recent_overlay_events.popleft()
#         return [
#             item[1]
#             for item in list(self.recent_overlay_events)[-OVERLAY_MAX_EVENT_LINES:]
#         ]

#     # ─────────────────────────────────────────────────────────────────────────
#     # Core state machine — one track per call
#     # ─────────────────────────────────────────────────────────────────────────

#     def _process_track(
#         self,
#         track: GlobalTrack,
#         timestamp_ms: float,
#         all_tracks: Optional[List[GlobalTrack]] = None,
#     ) -> List[dict]:
#         """
#         Run one FSM step for a single track. Returns new event dicts (may be empty).
#         """
#         update = track.current_update

#         # =====================================================================
#         # BRANCH A: no detection this frame
#         # =====================================================================
#         if update is None:
#             if track.event_state == "PICKUP_PENDING":
#                 if self._pending_timeout_elapsed(track, timestamp_ms):

#                     # A1 — outward motion history present
#                     if self._has_any_recent_outward_motion(track, look_back=5):
#                         self._transition(
#                             track, "PICKED_UP",
#                             "pickup confirmed: lost (outward motion)",
#                         )
#                         return [self._record_pickup(track)]

#                     # A2 — no motion but was_stable: occlusion / vertical lift
#                     elif track.was_stable:
#                         self._transition(
#                             track, "PICKED_UP",
#                             "pickup confirmed: lost (was stable — occlusion/vertical)",
#                         )
#                         event = self._record_pickup(track)
#                         event["occlusion_pickup"] = True
#                         self._replace_last_overlay(
#                             f"PICKUP {track.class_name} G{track.global_id} [occluded]"
#                         )
#                         return [event]

#                     # A3 — never stable: silent cancel
#                     else:
#                         self._emit_debug(
#                             f"[SILENT-CANCEL] track={track.global_id} "
#                             "never reached stability — suppressed",
#                             level="debug",
#                         )
#                         self._transition(
#                             track, "INSIDE",
#                             "pickup cancelled: never reached stability",
#                         )
#                         return []

#             elif track.event_state == "PUTBACK_PENDING":
#                 if self._pending_timeout_elapsed(track, timestamp_ms):
#                     if (
#                         track.last_seen_in_stable_roi is True
#                         and self._has_any_recent_inward_motion(track, look_back=5)
#                     ):
#                         self._transition(
#                             track, "STABLE_INSIDE",
#                             "putback confirmed: lost in stable ROI",
#                         )
#                         event = self._record_putback(track)
#                         return [] if event is None else [event]
#                     self._emit_debug(
#                         f"[TIMEOUT] putback absent track={track.global_id} "
#                         f"last_stable={track.last_seen_in_stable_roi}",
#                         level="info",
#                     )
#                     self._transition(
#                         track, "PICKED_UP",
#                         "putback cancelled: lost without entering stable ROI",
#                     )
#             return []

#         # =====================================================================
#         # BRANCH B: detection exists this frame
#         # =====================================================================

#         # Capture PREVIOUS frame ROI state before overwriting
#         prev_safe:  Optional[bool] = track.last_seen_in_safe_roi
#         prev_outer: Optional[bool] = track.last_seen_in_outer_roi
#         prev_stable: Optional[bool] = track.last_seen_in_stable_roi

#         # Update for future frames and for BRANCH A lost-track logic
#         track.last_seen_in_safe_roi  = update.in_safe_roi
#         track.last_seen_in_outer_roi = update.in_outer_roi
#         track.last_seen_in_stable_roi = update.in_stable_roi

#         # Update motion-direction history
#         if update.displacement_magnitude > 0:
#             if update.outward_motion:
#                 motion_direction: Optional[bool] = True
#             elif update.inward_motion:
#                 motion_direction = False
#             else:
#                 motion_direction = None
#         else:
#             motion_direction = None
#         track.motion_direction_history.append(motion_direction)

#         # Boundary-linger detection
#         if update.in_outer_roi and not update.in_stable_roi:
#             if track.boundary_linger_start_ms is None:
#                 track.boundary_linger_start_ms = update.timestamp_ms
#             elif (update.timestamp_ms - track.boundary_linger_start_ms) > 1000.0:
#                 track.boundary_linger_flagged = True
#         else:
#             track.boundary_linger_start_ms = None
#         required_stability = self._required_stability_frames(track, update.timestamp_ms)

#         # =====================================================================
#         # STATE: INSIDE
#         # =====================================================================
#         if track.event_state == "INSIDE":
#             if self._is_class_return_candidate(track):
#                 self._transition(
#                     track, "PUTBACK_PENDING", "class-level inward re-entry candidate"
#                 )
#                 track.pending_since_ms = update.timestamp_ms
#                 self._seed_class_return_confirmation(track)
#                 self._emit_debug(
#                     f"[PENDING] class-putback track={track.global_id} "
#                     f"class={track.class_name}",
#                     level="info",
#                 )
#                 return []

#             if update.in_stable_roi and update.confidence >= CONF_THRESHOLD:
#                 track.inside_counter += 1
#                 if track.inside_counter >= required_stability:
#                     self._transition(track, "STABLE_INSIDE", "stable in shelf ROI")
#                     track.boundary_linger_flagged = False
#                     if all_tracks is not None:
#                         self._check_neighbor_stability_boost(track, all_tracks)
#             else:
#                 track.inside_counter = 0
#             return []

#         # =====================================================================
#         # STATE: STABLE_INSIDE
#         # =====================================================================
#         if track.event_state == "STABLE_INSIDE":
#             if self._is_class_return_candidate(track):
#                 self._transition(
#                     track, "PUTBACK_PENDING", "class-level inward re-entry candidate"
#                 )
#                 track.pending_since_ms = update.timestamp_ms
#                 self._seed_class_return_confirmation(track)
#                 self._emit_debug(
#                     f"[PENDING] class-putback track={track.global_id} "
#                     f"class={track.class_name}",
#                     level="info",
#                 )
#                 return []

#             # ── PICKUP TRIGGER ────────────────────────────────────────────────
#             if not update.in_safe_roi:
#                 has_motion   = self._has_any_recent_outward_motion(track, look_back=3)
#                 has_roi_exit = self._roi_exit_occurred(track, update, prev_stable)

#                 if has_motion or has_roi_exit:
#                     # Post-putback cooldown (per-track, not cross-class)
#                     if (
#                         track.last_putback_confirmed_ms is not None
#                         and (update.timestamp_ms - track.last_putback_confirmed_ms)
#                         < POST_PUTBACK_COOLDOWN_MS
#                     ):
#                         self._emit_debug(
#                             f"[COOLDOWN] pickup suppressed track={track.global_id} "
#                             f"elapsed={update.timestamp_ms - track.last_putback_confirmed_ms:.0f}ms "
#                             f"cooldown={POST_PUTBACK_COOLDOWN_MS}ms",
#                             level="debug",
#                         )
#                         return []

#                     trigger_reason = (
#                         "outward exit + motion"
#                         if has_motion
#                         else "ROI exit without motion (vertical/edge/fast/low-tray pick)"
#                     )
#                     self._transition(track, "PICKUP_PENDING", trigger_reason)
#                     track.pending_since_ms = update.timestamp_ms
#                     self._seed_confirmation_from_history(track)
#                     if all_tracks is not None:
#                         self._check_neighbor_stability_boost(track, all_tracks)
#                     self._emit_debug(
#                         f"[PENDING] pickup track={track.global_id} "
#                         f"class={track.class_name} "
#                         f"motion={has_motion} roi_exit={has_roi_exit} "
#                         f"prev_safe={prev_safe} curr_safe={update.in_safe_roi} "
#                         f"prev_stable={prev_stable} curr_stable={update.in_stable_roi} "
#                         f"safe_dist={update.safe_roi_distance:.1f} "
#                         f"motion_history={list(track.motion_direction_history)}",
#                         level="info",
#                     )
#                     return []

#             if update.in_stable_roi and update.confidence >= CONF_THRESHOLD:
#                 track.inside_counter = required_stability
#             else:
#                 track.inside_counter = 0
#             return []

#         # =====================================================================
#         # STATE: PICKUP_PENDING
#         # =====================================================================
#         if track.event_state == "PICKUP_PENDING":

#             # ── Quick pick-and-return ─────────────────────────────────────────
#             if update.in_stable_roi and not update.outward_motion:
#                 time_since_pending_ms = update.timestamp_ms - (
#                     track.pending_since_ms or update.timestamp_ms
#                 )
#                 if time_since_pending_ms < QUICK_RETURN_THRESHOLD_MS:
#                     self._transition(
#                         track, "PICKED_UP",
#                         "quick pickup: fast re-entry within threshold",
#                     )
#                     pickup_event = self._record_pickup(track)
#                     self._frame_net_snapshot[track.class_name] = (
#                         self._frame_net_snapshot.get(track.class_name, 0) + 1
#                     )
#                     self._transition(
#                         track, "PUTBACK_PENDING",
#                         "quick return: fast re-entry",
#                     )
#                     track.pending_since_ms = update.timestamp_ms
#                     self._seed_confirmation_from_history(track)
#                     self._emit_debug(
#                         f"[QUICK-RETURN] track={track.global_id} "
#                         f"time_ms={time_since_pending_ms:.0f}",
#                         level="info",
#                     )
#                     return [pickup_event]

#                 # Re-entry after threshold → false alarm
#                 self._emit_debug(
#                     f"[CANCEL] pickup track={track.global_id} "
#                     "re-entered stable ROI after threshold",
#                     level="info",
#                 )
#                 self._transition(
#                     track, "STABLE_INSIDE",
#                     "pickup cancelled: returned to stable ROI",
#                 )
#                 return []

#             # ── Fully absent from both ROIs ───────────────────────────────────
#             if not update.in_outer_roi:
#                 self._append_confirmation(track, update.confidence)
#                 if (
#                     self._confirmation_ok(track)
#                     or (update.timestamp_ms - (track.pending_since_ms or update.timestamp_ms)) > 100
#                 ):
#                     self._transition(
#                         track, "PICKED_UP",
#                         "pickup confirmed: SAFE → OUTER → ABSENT",
#                     )
#                     return [self._record_pickup(track)]
#                 return []

#             # ── In OUTER ROI ──────────────────────────────────────────────────
#             if update.in_outer_roi:
#                 self._append_confirmation(track, update.confidence)

#                 if (
#                     len(track.confirmation_confidences) >= CONFIRM_FRAMES
#                     and not self._confirmation_ok(track)
#                 ):
#                     self._emit_debug(
#                         f"[CANCEL] pickup track={track.global_id} "
#                         "confidence too low in OUTER",
#                         level="info",
#                     )
#                     self._transition(
#                         track, "STABLE_INSIDE",
#                         "pickup cancelled: confidence too low",
#                     )
#                     return []

#                 if self._has_any_recent_outward_motion(track, look_back=3):
#                     return []

#                 # HESITATION RESETS REMOVED: timeout now immediately cancels
#                 if self._pending_timeout_elapsed(track, update.timestamp_ms):
#                     self._emit_debug(
#                         f"[TIMEOUT] pickup OUTER track={track.global_id} "
#                         "timeout elapsed",
#                         level="info",
#                     )
#                     self._transition(
#                         track, "STABLE_INSIDE",
#                         "pickup cancelled: timeout in OUTER ROI",
#                     )
#                     return []

#                 return []

#             return []

#         # =====================================================================
#         # STATE: PICKED_UP
#         # =====================================================================
#         if track.event_state == "PICKED_UP":

#             # Long-hold fallback
#             track_pickup_ts: Optional[float] = track.pickup_confirmed_ms
#             is_long_hold = (
#                 track_pickup_ts is not None
#                 and (timestamp_ms - track_pickup_ts) > CLASS_RETURN_MAX_GAP_MS
#             )
#             if is_long_hold and (update.in_stable_roi or update.in_outer_roi):
#                 track.long_hold_pending = True
#                 self._transition(
#                     track, "PUTBACK_PENDING",
#                     "long-hold fallback: elapsed > CLASS_RETURN_MAX_GAP_MS",
#                 )
#                 track.pending_since_ms = update.timestamp_ms
#                 self._seed_confirmation_from_history(track)
#                 self._emit_debug(
#                     f"[LONG-HOLD] track={track.global_id} class={track.class_name} "
#                     f"elapsed_ms={timestamp_ms - track_pickup_ts:.0f}",
#                     level="info",
#                 )
#                 return []

#             # ── PUTBACK TRIGGER ───────────────────────────────────────────────
#             if update.in_stable_roi or update.in_outer_roi:
#                 has_inward  = self._has_any_recent_inward_motion(track, look_back=3)
#                 has_reentry = self._roi_reentry_occurred(update, prev_safe, prev_outer)

#                 trigger_reason = (
#                     "inward re-entry + motion" if has_inward
#                     else "ROI re-entry from absent (vertical/edge/fast return)"
#                     if has_reentry
#                     else "ROI presence after pickup (product returned to zone)"
#                 )
#                 self._transition(track, "PUTBACK_PENDING", trigger_reason)
#                 track.pending_since_ms = update.timestamp_ms
#                 self._seed_confirmation_from_history(track)
#                 self._emit_debug(
#                     f"[PENDING] putback track={track.global_id} "
#                     f"class={track.class_name} "
#                     f"safe={update.in_safe_roi} stable={update.in_stable_roi} "
#                     f"outer={update.in_outer_roi} "
#                     f"motion={has_inward} reentry={has_reentry} "
#                     f"prev_safe={prev_safe} prev_outer={prev_outer}",
#                     level="info",
#                 )
#                 return []

#             return []

#         # =====================================================================
#         # STATE: PUTBACK_PENDING
#         # =====================================================================
#         if track.event_state == "PUTBACK_PENDING":

#             # ── Stable ROI: accumulate and confirm ───────────────────────────
#             if update.in_stable_roi:
#                 self._append_confirmation(track, update.confidence)
#                 if len(track.confirmation_confidences) >= CONFIRM_FRAMES:
#                     if not self._confirmation_ok(track):
#                         self._emit_debug(
#                             f"[CANCEL] putback confidence track={track.global_id} "
#                             f"conf={list(track.confirmation_confidences)}",
#                             level="info",
#                         )
#                         self._transition(
#                             track, "PICKED_UP",
#                             "putback cancelled by confidence window",
#                         )
#                         return []
#                     self._transition(
#                         track, "STABLE_INSIDE",
#                         "putback confirmed: stable in shelf ROI",
#                     )
#                     event = self._record_putback(track)
#                     return [] if event is None else [event]
#                 return []

#             # ── Outer ROI: accumulate unconditionally, single extension on timeout with inward motion
#             if update.in_outer_roi:
#                 self._append_confirmation(track, update.confidence)
#                 if (
#                     len(track.confirmation_confidences) >= CONFIRM_FRAMES
#                     and not self._confirmation_ok(track)
#                 ):
#                     self._emit_debug(
#                         f"[CANCEL] putback OUTER confidence track={track.global_id} "
#                         f"conf={list(track.confirmation_confidences)}",
#                         level="info",
#                     )
#                     self._transition(
#                         track, "PICKED_UP",
#                         "putback cancelled: confidence too low in OUTER",
#                     )
#                     return []

#                 if self._pending_timeout_elapsed(track, update.timestamp_ms):
#                     if not self._has_any_recent_inward_motion(track, look_back=5):
#                         self._emit_debug(
#                             f"[TIMEOUT] putback OUTER track={track.global_id} "
#                             "no inward motion",
#                             level="info",
#                         )
#                         self._transition(
#                             track, "PICKED_UP",
#                             "putback cancelled: timeout with no inward motion",
#                         )
#                         return []
#                     # Extend window once (single extension, not multiple resets)
#                     track.pending_since_ms = update.timestamp_ms
#                 return []

#             # ── Absent from both zones ────────────────────────────────────────
#             if self._pending_timeout_elapsed(track, update.timestamp_ms):
#                 self._emit_debug(
#                     f"[TIMEOUT] putback absent track={track.global_id}",
#                     level="info",
#                 )
#                 self._transition(
#                     track, "PICKED_UP",
#                     "putback cancelled: timeout while absent",
#                 )
#                 return []
#             return []

#         return []

#     # ─────────────────────────────────────────────────────────────────────────
#     # Public API
#     # ─────────────────────────────────────────────────────────────────────────

#     def process_frame(
#         self,
#         tracks: List[GlobalTrack],
#         frame_avg_confidence: float,
#         timestamp_ms: float,
#     ) -> List[dict]:
#         """
#         Process all active tracks for one frame. Returns all new events generated.
#         """
#         self._frame_net_snapshot = {
#             class_name: self.pickup_count[class_name] - self.putback_count[class_name]
#             for class_name in set(self.pickup_count) | set(self.putback_count)
#         }
#         events: List[dict] = []
#         for track in tracks:
#             events.extend(self._process_track(track, timestamp_ms, all_tracks=tracks))
#         self._frame_net_snapshot = {}
#         return events

#     def finalize(self) -> SessionSummary:
#         """Compute net changes and totals, return the completed session summary."""
#         classes = set(self.pickup_count) | set(self.putback_count)
#         self.session.pickup_count = dict(self.pickup_count)
#         self.session.putback_count = dict(self.putback_count)
#         self.session.net_change = {
#             cls: self.pickup_count[cls] - self.putback_count[cls]
#             for cls in classes
#         }
#         self.session.total_pickups  = sum(self.pickup_count.values())
#         self.session.total_putbacks = sum(self.putback_count.values())
#         return self.session


# from __future__ import annotations

# import logging
# import uuid
# from collections import defaultdict, deque
# from typing import Deque, Dict, List, Optional

# import numpy as np

# from config import (
#     CLASS_RETURN_MAX_GAP_MS,
#     CONFIRM_FRAMES,
#     CONF_MIN_FRAME,
#     CONF_THRESHOLD,
#     LONG_BOUNDARY_STABILITY_FRAMES,
#     MISSING_PENDING_CONFIRM_MS,
#     OVERLAY_EVENT_TTL_MS,
#     OVERLAY_MAX_EVENT_LINES,
#     POST_PUTBACK_COOLDOWN_MS,
#     PRINT_DEBUG_EVENTS,
#     QUICK_RETURN_THRESHOLD_MS,
#     STABILITY_FRAMES,

#     CLASS_VOTE_HISTORY,
#     CLASS_LOCK_RATIO,
#     CLASS_LOCK_MIN_TOTAL_SCORE,
#     CLASS_VOTE_MIN_CONFIDENCE,
#     CLASS_VOTE_WEIGHT_STABLE,
#     CLASS_VOTE_WEIGHT_SAFE,
#     CLASS_VOTE_WEIGHT_OUTER,
#     CLASS_VOTE_WEIGHT_OUTSIDE,
#     CLASS_VOTE_RECENCY_DECAY,
#     CLASS_CONSISTENCY_RATIO,
#     UNKNOWN_UNSTABLE_CLASS,
#     ENABLE_CLASS_DRIFT_WARNINGS,

#     PENDING_LEDGER_MAX_AGE_MS,
#     PUTBACK_LEDGER_MATCH_THRESHOLD,
#     LEDGER_MATCH_WEIGHT_CLASS,
#     LEDGER_MATCH_WEIGHT_TIME,
#     LEDGER_MATCH_WEIGHT_TRACK,
#     LEDGER_MATCH_WEIGHT_CLASS_CANDIDATE,
#     LEDGER_MATCH_WEIGHT_MOTION,
#     LEDGER_MATCH_WEIGHT_EMBEDDING,
#     LEDGER_MATCH_WEIGHT_SPATIAL,
#     LEDGER_SAME_CLASS_SCORE,
#     LEDGER_DIFFERENT_CLASS_SCORE,
#     LEDGER_SPATIAL_MAX_DISTANCE_PX,
#     LEDGER_EMBEDDING_MIN_SIMILARITY,
# )
# from utils.types import GlobalTrack, SessionSummary

# LOGGER = logging.getLogger("vending_pipeline.events")


# class EventManager:
#     """
#     Pickup/putback FSM with class-drift protection.

#     Fixes:
#     1. Product misclassified while moving outward.
#     2. Product class flickers inside ROI.
#     3. Product misclassified during confirmation window.
#     4. Product returns with different detected class.
#     """

#     def __init__(self, session_id: str) -> None:
#         self.session = SessionSummary(session_id=session_id)

#         self.pickup_count: Dict[str, int] = defaultdict(int)
#         self.putback_count: Dict[str, int] = defaultdict(int)
#         self.class_pickup_timestamps: Dict[str, Deque[float]] = defaultdict(deque)

#         # Item-level pickup ledger.
#         # Every confirmed pickup opens one ledger item.
#         # Putback closes the matched ledger item.
#         self.pending_pickups: List[dict] = []

#         self._frame_net_snapshot: Dict[str, int] = {}
#         self._frame_putback_used: Dict[str, int] = defaultdict(int)

#         self.last_event_text = ""
#         self.recent_overlay_events: Deque[tuple[float, str]] = deque(maxlen=20)

#     # ─────────────────────────────────────────────────────────────────────────
#     # Compatibility / identity helpers
#     # ─────────────────────────────────────────────────────────────────────────

#     def _ensure_identity_fields(self, track: GlobalTrack) -> None:
#         if not hasattr(track, "class_votes") or track.class_votes is None:
#             track.class_votes = deque(maxlen=CLASS_VOTE_HISTORY)

#         if not hasattr(track, "confirmation_class_votes") or track.confirmation_class_votes is None:
#             track.confirmation_class_votes = deque(maxlen=CONFIRM_FRAMES)

#         if not hasattr(track, "locked_class_name"):
#             track.locked_class_name = None

#         if not hasattr(track, "resolved_class_name"):
#             track.resolved_class_name = None

#         if not hasattr(track, "class_drift_flagged"):
#             track.class_drift_flagged = False

#         if not hasattr(track, "visual_embedding"):
#             track.visual_embedding = None

#     def _detected_class(self, track: GlobalTrack) -> str:
#         update = track.current_update
#         if update is not None and getattr(update, "class_name", None):
#             return update.class_name
#         return getattr(track, "class_name", None) or UNKNOWN_UNSTABLE_CLASS

#     def _event_class(self, track: GlobalTrack) -> str:
#         """
#         Final inventory/billing class.

#         Priority:
#         1. resolved_class_name
#         2. locked_class_name
#         3. weighted voting result
#         4. latest detected class
#         """
#         self._ensure_identity_fields(track)

#         if track.resolved_class_name and track.resolved_class_name != UNKNOWN_UNSTABLE_CLASS:
#             return track.resolved_class_name

#         if track.locked_class_name and track.locked_class_name != UNKNOWN_UNSTABLE_CLASS:
#             return track.locked_class_name

#         vote_result = self._compute_weighted_class_vote(
#             track,
#             stable_only=False,
#             include_safe=True,
#         )
#         voted_class = vote_result.get("class")

#         if voted_class and voted_class != UNKNOWN_UNSTABLE_CLASS:
#             return voted_class

#         return self._detected_class(track)

#     # ─────────────────────────────────────────────────────────────────────────
#     # Logging / overlay
#     # ─────────────────────────────────────────────────────────────────────────

#     def _emit_debug(self, message: str, *, level: str = "debug") -> None:
#         getattr(LOGGER, level)(message)
#         if PRINT_DEBUG_EVENTS:
#             print(message)

#     def _push_overlay_event(self, timestamp_ms: float, message: str) -> None:
#         self.recent_overlay_events.append((timestamp_ms, message))
#         self.last_event_text = message

#     def _replace_last_overlay(self, message: str) -> None:
#         if self.recent_overlay_events:
#             ts, _ = self.recent_overlay_events[-1]
#             self.recent_overlay_events[-1] = (ts, message)
#             self.last_event_text = message

#     def _log_transition(
#         self,
#         track: GlobalTrack,
#         old_state: str,
#         new_state: str,
#         reason: str,
#     ) -> None:
#         message = (
#             f"[TRACK {track.global_id}] {old_state} -> {new_state} | "
#             f"event_class={self._event_class(track)} | "
#             f"detected_class={self._detected_class(track)} | "
#             f"locked_class={getattr(track, 'locked_class_name', None)} | "
#             f"reason={reason}"
#         )
#         track.last_debug_reason = reason
#         self._emit_debug(message, level="debug")

#     def _transition(self, track: GlobalTrack, new_state: str, reason: str) -> None:
#         self._ensure_identity_fields(track)

#         if track.event_state == new_state:
#             return

#         old_state = track.event_state
#         track.event_state = new_state

#         if new_state == "STABLE_INSIDE":
#             track.was_stable = True
#             self._lock_class_if_possible(track, reason="entered STABLE_INSIDE")

#         if new_state in {"PICKUP_PENDING", "PUTBACK_PENDING"}:
#             track.confirmation_confidences.clear()
#             track.confirmation_class_votes.clear()
#         else:
#             track.pending_since_ms = None

#         self._log_transition(track, old_state, new_state, reason)

#     # ─────────────────────────────────────────────────────────────────────────
#     # Class voting / class locking
#     # ─────────────────────────────────────────────────────────────────────────

#     def _roi_weight_for_vote(self, update) -> float:
#         if update is None:
#             return CLASS_VOTE_WEIGHT_OUTSIDE

#         if update.in_stable_roi:
#             return CLASS_VOTE_WEIGHT_STABLE

#         if update.in_safe_roi:
#             return CLASS_VOTE_WEIGHT_SAFE

#         if update.in_outer_roi:
#             return CLASS_VOTE_WEIGHT_OUTER

#         return CLASS_VOTE_WEIGHT_OUTSIDE

#     def _append_class_vote(self, track: GlobalTrack, update) -> None:
#         self._ensure_identity_fields(track)

#         if update is None:
#             return

#         confidence = float(update.confidence or 0.0)
#         if confidence < CLASS_VOTE_MIN_CONFIDENCE:
#             return

#         detected_class = getattr(update, "class_name", None) or self._detected_class(track)

#         track.class_votes.append(
#             {
#                 "class": detected_class,
#                 "confidence": confidence,
#                 "timestamp_ms": float(update.timestamp_ms),
#                 "roi_weight": self._roi_weight_for_vote(update),
#                 "in_stable_roi": bool(update.in_stable_roi),
#                 "in_safe_roi": bool(update.in_safe_roi),
#                 "in_outer_roi": bool(update.in_outer_roi),
#             }
#         )

#     def _compute_weighted_class_vote(
#         self,
#         track: GlobalTrack,
#         *,
#         stable_only: bool = False,
#         include_safe: bool = True,
#     ) -> dict:
#         self._ensure_identity_fields(track)

#         votes = list(track.class_votes)

#         if stable_only:
#             votes = [v for v in votes if v.get("in_stable_roi")]
#         elif include_safe:
#             votes = [
#                 v
#                 for v in votes
#                 if v.get("in_stable_roi") or v.get("in_safe_roi") or v.get("in_outer_roi")
#             ]

#         if not votes:
#             return {
#                 "class": UNKNOWN_UNSTABLE_CLASS,
#                 "score": 0.0,
#                 "total_score": 0.0,
#                 "ratio": 0.0,
#                 "scores": {},
#             }

#         scores: Dict[str, float] = defaultdict(float)
#         n = len(votes)

#         for idx, vote in enumerate(votes):
#             age_from_latest = n - 1 - idx
#             recency_weight = CLASS_VOTE_RECENCY_DECAY ** age_from_latest

#             cls = vote["class"]
#             score = (
#                 float(vote.get("confidence", 0.0))
#                 * float(vote.get("roi_weight", 1.0))
#                 * recency_weight
#             )
#             scores[cls] += score

#         total_score = sum(scores.values())

#         if total_score <= 0:
#             return {
#                 "class": UNKNOWN_UNSTABLE_CLASS,
#                 "score": 0.0,
#                 "total_score": 0.0,
#                 "ratio": 0.0,
#                 "scores": dict(scores),
#             }

#         best_class, best_score = max(scores.items(), key=lambda kv: kv[1])
#         ratio = best_score / total_score

#         if total_score < CLASS_LOCK_MIN_TOTAL_SCORE or ratio < CLASS_LOCK_RATIO:
#             selected_class = UNKNOWN_UNSTABLE_CLASS
#         else:
#             selected_class = best_class

#         return {
#             "class": selected_class,
#             "score": best_score,
#             "total_score": total_score,
#             "ratio": ratio,
#             "scores": dict(scores),
#         }

#     def _lock_class_if_possible(self, track: GlobalTrack, *, reason: str) -> None:
#         self._ensure_identity_fields(track)

#         if track.locked_class_name and track.locked_class_name != UNKNOWN_UNSTABLE_CLASS:
#             return

#         # Prefer stable ROI votes.
#         result = self._compute_weighted_class_vote(track, stable_only=True)

#         # Fallback to safe/outer votes if stable-only voting is not enough.
#         if result["class"] == UNKNOWN_UNSTABLE_CLASS:
#             result = self._compute_weighted_class_vote(
#                 track,
#                 stable_only=False,
#                 include_safe=True,
#             )

#         locked_class = result["class"]

#         if locked_class == UNKNOWN_UNSTABLE_CLASS:
#             track.locked_class_name = UNKNOWN_UNSTABLE_CLASS
#             track.resolved_class_name = UNKNOWN_UNSTABLE_CLASS

#             self._emit_debug(
#                 f"[CLASS-LOCK-DELAYED] track={track.global_id} "
#                 f"reason={reason} ratio={result['ratio']:.2f} "
#                 f"scores={result['scores']}",
#                 level="info",
#             )
#             return

#         track.locked_class_name = locked_class
#         track.resolved_class_name = locked_class

#         detected_class = self._detected_class(track)
#         if detected_class != locked_class:
#             track.class_drift_flagged = True

#         self._emit_debug(
#             f"[CLASS-LOCKED] track={track.global_id} "
#             f"locked={locked_class} detected={detected_class} "
#             f"reason={reason} ratio={result['ratio']:.2f} "
#             f"scores={result['scores']}",
#             level="info",
#         )

#     # ─────────────────────────────────────────────────────────────────────────
#     # Confirmation helpers
#     # ─────────────────────────────────────────────────────────────────────────

#     def _append_confirmation(self, track: GlobalTrack, confidence: float) -> None:
#         self._ensure_identity_fields(track)

#         if track.confirmation_confidences.maxlen != CONFIRM_FRAMES:
#             track.confirmation_confidences = deque(
#                 track.confirmation_confidences,
#                 maxlen=CONFIRM_FRAMES,
#             )

#         if track.confirmation_class_votes.maxlen != CONFIRM_FRAMES:
#             track.confirmation_class_votes = deque(
#                 track.confirmation_class_votes,
#                 maxlen=CONFIRM_FRAMES,
#             )

#         track.confirmation_confidences.append(float(confidence))

#         update = track.current_update
#         if update is not None:
#             detected_class = getattr(update, "class_name", None) or self._detected_class(track)
#             track.confirmation_class_votes.append(
#                 {
#                     "class": detected_class,
#                     "confidence": float(confidence),
#                     "timestamp_ms": float(update.timestamp_ms),
#                 }
#             )

#     def _seed_confirmation_from_history(self, track: GlobalTrack) -> None:
#         self._ensure_identity_fields(track)

#         history = list(track.confidence_history)[-CONFIRM_FRAMES:]
#         track.confirmation_confidences = deque(history, maxlen=CONFIRM_FRAMES)

#         class_history = list(track.class_votes)[-CONFIRM_FRAMES:]
#         track.confirmation_class_votes = deque(
#             [
#                 {
#                     "class": item["class"],
#                     "confidence": item["confidence"],
#                     "timestamp_ms": item["timestamp_ms"],
#                 }
#                 for item in class_history
#             ],
#             maxlen=CONFIRM_FRAMES,
#         )

#     def _seed_class_return_confirmation(self, track: GlobalTrack) -> None:
#         self._ensure_identity_fields(track)

#         history = list(track.confidence_history)[-CONFIRM_FRAMES:]
#         if not history:
#             history = [track.last_confidence]

#         while len(history) < CONFIRM_FRAMES:
#             history.append(history[-1])

#         track.confirmation_confidences = deque(history, maxlen=CONFIRM_FRAMES)

#         class_history = list(track.class_votes)[-CONFIRM_FRAMES:]
#         if not class_history:
#             detected_class = self._detected_class(track)
#             class_history = [
#                 {
#                     "class": detected_class,
#                     "confidence": track.last_confidence,
#                     "timestamp_ms": track.last_seen_ms,
#                 }
#             ]

#         while len(class_history) < CONFIRM_FRAMES:
#             class_history.append(class_history[-1])

#         track.confirmation_class_votes = deque(
#             [
#                 {
#                     "class": item["class"],
#                     "confidence": item["confidence"],
#                     "timestamp_ms": item["timestamp_ms"],
#                 }
#                 for item in class_history[-CONFIRM_FRAMES:]
#             ],
#             maxlen=CONFIRM_FRAMES,
#         )

#     def _pending_timeout_elapsed(self, track: GlobalTrack, timestamp_ms: float) -> bool:
#         if track.pending_since_ms is None:
#             return False
#         return (timestamp_ms - track.pending_since_ms) >= MISSING_PENDING_CONFIRM_MS

#     def _confidence_ok(self, track: GlobalTrack) -> bool:
#         if len(track.confirmation_confidences) < CONFIRM_FRAMES:
#             return False

#         avg_conf = sum(track.confirmation_confidences) / len(track.confirmation_confidences)
#         min_conf = min(track.confirmation_confidences)

#         return avg_conf >= CONF_THRESHOLD and min_conf >= CONF_MIN_FRAME

#     def _class_consistency_ok(self, track: GlobalTrack) -> bool:
#         self._ensure_identity_fields(track)

#         event_class = self._event_class(track)
#         if not event_class or event_class == UNKNOWN_UNSTABLE_CLASS:
#             return False

#         votes = list(track.confirmation_class_votes)
#         if not votes:
#             return True

#         total_weight = 0.0
#         matched_weight = 0.0

#         for vote in votes:
#             cls = vote.get("class")
#             conf = float(vote.get("confidence", 0.0) or 0.0)

#             total_weight += conf
#             if cls == event_class:
#                 matched_weight += conf

#         if total_weight <= 0:
#             return False

#         ratio = matched_weight / total_weight

#         if ratio < CLASS_CONSISTENCY_RATIO:
#             track.class_drift_flagged = True
#             self._emit_debug(
#                 f"[CLASS-DRIFT] track={track.global_id} "
#                 f"event_class={event_class} votes={votes} consistency={ratio:.2f}",
#                 level="info",
#             )

#         return ratio >= CLASS_CONSISTENCY_RATIO

#     def _confirmation_ok(
#         self,
#         track: GlobalTrack,
#         *,
#         require_class_consistency: bool = True,
#     ) -> bool:
#         if not self._confidence_ok(track):
#             return False

#         if not require_class_consistency:
#             return True

#         return self._class_consistency_ok(track)

#     # ─────────────────────────────────────────────────────────────────────────
#     # Motion helpers
#     # ─────────────────────────────────────────────────────────────────────────

#     def _required_stability_frames(self, track: GlobalTrack, timestamp_ms: float) -> int:
#         return (
#             LONG_BOUNDARY_STABILITY_FRAMES
#             if track.boundary_linger_flagged
#             else STABILITY_FRAMES
#         )

#     def _has_any_recent_outward_motion(self, track: GlobalTrack, look_back: int = 3) -> bool:
#         if not track.motion_direction_history:
#             return False
#         recent = list(track.motion_direction_history)[-look_back:]
#         return any(d is True for d in recent)

#     def _has_any_recent_inward_motion(self, track: GlobalTrack, look_back: int = 3) -> bool:
#         if not track.motion_direction_history:
#             return False
#         recent = list(track.motion_direction_history)[-look_back:]
#         return any(d is False for d in recent)

#     def _has_sustained_outward_motion(self, track: GlobalTrack, min_consecutive: int = 2) -> bool:
#         if len(track.motion_direction_history) < min_consecutive:
#             return False
#         return all(d is True for d in list(track.motion_direction_history)[-min_consecutive:])

#     def _has_sustained_inward_motion(self, track: GlobalTrack, min_consecutive: int = 2) -> bool:
#         if len(track.motion_direction_history) < min_consecutive:
#             return False
#         return all(d is False for d in list(track.motion_direction_history)[-min_consecutive:])

#     # ─────────────────────────────────────────────────────────────────────────
#     # ROI helpers
#     # ─────────────────────────────────────────────────────────────────────────

#     @staticmethod
#     def _roi_exit_occurred(track: GlobalTrack, update, prev_stable: Optional[bool]) -> bool:
#         return (
#             track.was_stable
#             and prev_stable is True
#             and not update.in_stable_roi
#         )

#     @staticmethod
#     def _roi_reentry_occurred(update, prev_safe: Optional[bool], prev_outer: Optional[bool]) -> bool:
#         was_fully_absent = (prev_safe is False) and (prev_outer is False)
#         now_in_roi = update.in_safe_roi or update.in_outer_roi
#         return was_fully_absent and now_in_roi

#     # ─────────────────────────────────────────────────────────────────────────
#     # Ledger helpers
#     # ─────────────────────────────────────────────────────────────────────────

#     def _has_outstanding_pickup(self, class_name: str) -> bool:
#         return (self.pickup_count[class_name] - self.putback_count[class_name]) > 0

#     def _latest_pickup_timestamp(self, class_name: str) -> Optional[float]:
#         history = self.class_pickup_timestamps[class_name]
#         return history[-1] if history else None

#     def _active_pending_pickups(self, timestamp_ms: float) -> List[dict]:
#         active: List[dict] = []

#         for item in self.pending_pickups:
#             if item.get("closed"):
#                 continue

#             age = timestamp_ms - float(item["pickup_time_ms"])
#             if age <= PENDING_LEDGER_MAX_AGE_MS:
#                 active.append(item)

#         return active

#     def _class_similarity_score(self, a: Optional[str], b: Optional[str]) -> float:
#         if not a or not b:
#             return LEDGER_DIFFERENT_CLASS_SCORE

#         if a == b:
#             return LEDGER_SAME_CLASS_SCORE

#         return LEDGER_DIFFERENT_CLASS_SCORE

#     def _ledger_class_candidate_score(self, track: GlobalTrack, pending_item: dict) -> float:
#         current_vote = self._compute_weighted_class_vote(
#             track,
#             stable_only=False,
#             include_safe=True,
#         )
#         current_scores = current_vote.get("scores", {}) or {}
#         pickup_scores = pending_item.get("class_candidates", {}) or {}

#         if not current_scores or not pickup_scores:
#             return 0.0

#         current_total = sum(current_scores.values()) or 1.0
#         pickup_total = sum(pickup_scores.values()) or 1.0

#         overlap = set(current_scores.keys()) & set(pickup_scores.keys())
#         if not overlap:
#             return 0.0

#         score = 0.0
#         for cls in overlap:
#             score += min(
#                 current_scores.get(cls, 0.0) / current_total,
#                 pickup_scores.get(cls, 0.0) / pickup_total,
#             )

#         return min(1.0, score)

#     def _time_match_score(self, pickup_time_ms: float, return_time_ms: float) -> float:
#         gap = max(0.0, return_time_ms - pickup_time_ms)

#         if gap >= PENDING_LEDGER_MAX_AGE_MS:
#             return 0.0

#         return 1.0 - (gap / PENDING_LEDGER_MAX_AGE_MS)

#     def _track_match_score(self, track: GlobalTrack, pending_item: dict) -> float:
#         return 1.0 if pending_item.get("track_id") == track.global_id else 0.0

#     def _motion_match_score_for_putback(self, track: GlobalTrack) -> float:
#         if self._has_any_recent_inward_motion(track, look_back=5):
#             return 1.0
#         return 0.3

#     def _track_embedding(self, track: GlobalTrack) -> Optional[np.ndarray]:
#         if getattr(track, "visual_embedding", None) is not None:
#             return np.asarray(track.visual_embedding, dtype=np.float32)

#         if hasattr(track, "mean_embedding"):
#             emb = track.mean_embedding()
#             if emb is not None:
#                 return np.asarray(emb, dtype=np.float32)

#         return None

#     def _embedding_similarity_score(self, track: GlobalTrack, pending_item: dict) -> float:
#         emb_a = self._track_embedding(track)
#         emb_b = pending_item.get("visual_embedding")

#         if emb_a is None or emb_b is None:
#             return 0.0

#         emb_b = np.asarray(emb_b, dtype=np.float32)

#         denom = float(np.linalg.norm(emb_a) * np.linalg.norm(emb_b))
#         if denom <= 1e-6:
#             return 0.0

#         cosine = float(np.dot(emb_a, emb_b) / denom)
#         cosine = max(-1.0, min(1.0, cosine))

#         # Convert [-1, 1] to [0, 1].
#         normalized = (cosine + 1.0) / 2.0

#         if normalized < LEDGER_EMBEDDING_MIN_SIMILARITY:
#             return 0.0

#         return normalized

#     def _spatial_match_score(self, track: GlobalTrack, pending_item: dict) -> float:
#         update = track.current_update
#         pickup_centroid = pending_item.get("pickup_centroid")

#         if update is None or pickup_centroid is None:
#             return 0.0

#         try:
#             return_centroid = np.asarray(update.centroid, dtype=np.float32)
#             pickup_centroid_np = np.asarray(pickup_centroid, dtype=np.float32)
#         except Exception:
#             return 0.0

#         dist = float(np.linalg.norm(return_centroid - pickup_centroid_np))

#         if dist >= LEDGER_SPATIAL_MAX_DISTANCE_PX:
#             return 0.0

#         return 1.0 - (dist / LEDGER_SPATIAL_MAX_DISTANCE_PX)

#     def _match_putback_to_pending_pickup(
#         self,
#         track: GlobalTrack,
#         timestamp_ms: float,
#     ) -> Optional[dict]:
#         """
#         Matches return candidate to an open pickup ledger item.

#         This allows:
#             pickup event class = Lays Salted
#             return detected class = Lays Masala
#             final putback class = Lays Salted
#         """
#         self._ensure_identity_fields(track)

#         candidates = self._active_pending_pickups(timestamp_ms)
#         if not candidates:
#             return None

#         detected_return_class = self._detected_class(track)
#         resolved_return_class = self._event_class(track)

#         best_item = None
#         best_score = -1.0
#         best_breakdown = {}

#         for item in candidates:
#             picked_class = item["event_class"]

#             class_score = max(
#                 self._class_similarity_score(detected_return_class, picked_class),
#                 self._class_similarity_score(resolved_return_class, picked_class),
#             )
#             time_score = self._time_match_score(item["pickup_time_ms"], timestamp_ms)
#             track_score = self._track_match_score(track, item)
#             candidate_score = self._ledger_class_candidate_score(track, item)
#             motion_score = self._motion_match_score_for_putback(track)
#             embedding_score = self._embedding_similarity_score(track, item)
#             spatial_score = self._spatial_match_score(track, item)

#             total_score = (
#                 LEDGER_MATCH_WEIGHT_CLASS * class_score
#                 + LEDGER_MATCH_WEIGHT_TIME * time_score
#                 + LEDGER_MATCH_WEIGHT_TRACK * track_score
#                 + LEDGER_MATCH_WEIGHT_CLASS_CANDIDATE * candidate_score
#                 + LEDGER_MATCH_WEIGHT_MOTION * motion_score
#                 + LEDGER_MATCH_WEIGHT_EMBEDDING * embedding_score
#                 + LEDGER_MATCH_WEIGHT_SPATIAL * spatial_score
#             )

#             breakdown = {
#                 "total": total_score,
#                 "picked_class": picked_class,
#                 "detected_return_class": detected_return_class,
#                 "resolved_return_class": resolved_return_class,
#                 "class_score": class_score,
#                 "time_score": time_score,
#                 "track_score": track_score,
#                 "class_candidate_score": candidate_score,
#                 "motion_score": motion_score,
#                 "embedding_score": embedding_score,
#                 "spatial_score": spatial_score,
#             }

#             if total_score > best_score:
#                 best_score = total_score
#                 best_item = item
#                 best_breakdown = breakdown

#         if best_item is not None and best_score >= PUTBACK_LEDGER_MATCH_THRESHOLD:
#             best_item["_last_match_score"] = best_score
#             best_item["_last_match_breakdown"] = best_breakdown
#             return best_item

#         self._emit_debug(
#             f"[LEDGER-NO-MATCH] track={track.global_id} "
#             f"detected={detected_return_class} best_score={best_score:.2f} "
#             f"threshold={PUTBACK_LEDGER_MATCH_THRESHOLD} breakdown={best_breakdown}",
#             level="info",
#         )
#         return None

#     def _is_class_return_candidate(self, track: GlobalTrack) -> bool:
#         self._ensure_identity_fields(track)

#         update = track.current_update
#         if update is None:
#             return False

#         event_class = self._event_class(track)

#         exact_class_return = self._has_outstanding_pickup(event_class)
#         ledger_match = self._match_putback_to_pending_pickup(track, update.timestamp_ms)

#         if not exact_class_return and ledger_match is None:
#             return False

#         if exact_class_return:
#             latest_pickup_ts = self._latest_pickup_timestamp(event_class)
#         else:
#             latest_pickup_ts = ledger_match["pickup_time_ms"] if ledger_match else None

#         if latest_pickup_ts is None:
#             return False

#         within_gap = (update.timestamp_ms - latest_pickup_ts) <= CLASS_RETURN_MAX_GAP_MS
#         born_after_pickup = track.created_ms >= latest_pickup_ts
#         not_previously_stable = not track.was_stable
#         has_inward = update.inward_motion or self._has_any_recent_inward_motion(track, look_back=5)

#         if ledger_match is not None:
#             track.resolved_class_name = ledger_match["event_class"]

#         return within_gap and born_after_pickup and not_previously_stable and has_inward

#     # ─────────────────────────────────────────────────────────────────────────
#     # Neighbor stability boost
#     # ─────────────────────────────────────────────────────────────────────────

#     def _check_neighbor_stability_boost(
#         self,
#         track: GlobalTrack,
#         all_tracks: List[GlobalTrack],
#     ) -> None:
#         update = track.current_update
#         if update is None:
#             return

#         for neighbor in all_tracks:
#             self._ensure_identity_fields(neighbor)

#             if neighbor.global_id == track.global_id:
#                 continue

#             if neighbor.event_state not in {"INSIDE", "STABLE_INSIDE"}:
#                 continue

#             if neighbor.stability_boosted:
#                 continue

#             neighbor_update = neighbor.current_update
#             if neighbor_update is None:
#                 continue

#             same_zone = (
#                 (update.in_stable_roi and neighbor_update.in_stable_roi)
#                 or (update.in_outer_roi and neighbor_update.in_outer_roi)
#             )

#             half_stability = neighbor.inside_counter >= max(1, STABILITY_FRAMES // 2)

#             if same_zone and half_stability:
#                 required = self._required_stability_frames(neighbor, update.timestamp_ms)
#                 neighbor.inside_counter = required
#                 neighbor.event_state = "STABLE_INSIDE"
#                 neighbor.stability_boosted = True
#                 neighbor.was_stable = True

#                 self._lock_class_if_possible(neighbor, reason="neighbor stability boost")

#                 self._emit_debug(
#                     f"[BOOST] neighbor track={neighbor.global_id} -> STABLE_INSIDE "
#                     f"by track={track.global_id}",
#                     level="debug",
#                 )

#     # ─────────────────────────────────────────────────────────────────────────
#     # Event recording
#     # ─────────────────────────────────────────────────────────────────────────

#     def _pickup_ledger_item_from_event(self, track: GlobalTrack, event: dict) -> dict:
#         update = track.current_update

#         vote_result = self._compute_weighted_class_vote(
#             track,
#             stable_only=False,
#             include_safe=True,
#         )

#         pickup_centroid = None
#         pickup_bbox = None

#         if update is not None:
#             try:
#                 pickup_centroid = np.asarray(update.centroid, dtype=np.float32).tolist()
#             except Exception:
#                 pickup_centroid = None
#             pickup_bbox = tuple(update.bbox) if getattr(update, "bbox", None) is not None else None

#         return {
#             "ledger_id": str(uuid.uuid4()),
#             "pickup_event_id": event["event_id"],
#             "event_class": event["class"],
#             "locked_class": event.get("locked_class"),
#             "detected_class_at_pickup": event.get("detected_class"),
#             "track_id": track.global_id,
#             "pickup_time_ms": event["timestamp_ms"],
#             "camera_id": event["camera_id"],
#             "frame_index": event["frame_index"],
#             "class_candidates": vote_result.get("scores", {}),
#             "visual_embedding": self._track_embedding(track),
#             "pickup_centroid": pickup_centroid,
#             "pickup_bbox": pickup_bbox,
#             "closed": False,
#             "closed_by_event_id": None,
#         }

#     def _record_pickup(self, track: GlobalTrack) -> dict:
#         self._ensure_identity_fields(track)

#         update = track.current_update

#         event_class = self._event_class(track)

#         if event_class == UNKNOWN_UNSTABLE_CLASS:
#             self._lock_class_if_possible(track, reason="pickup final lock attempt")
#             event_class = self._event_class(track)

#         detected_class = self._detected_class(track)

#         self.pickup_count[event_class] += 1

#         event_timestamp = track.last_seen_ms if update is None else update.timestamp_ms
#         event_camera_id = track.last_camera_id if update is None else update.camera_id
#         event_frame_index = track.last_frame_index if update is None else update.frame_index

#         self.class_pickup_timestamps[event_class].append(event_timestamp)
#         track.pickup_confirmed_ms = event_timestamp
#         track.resolved_class_name = event_class

#         class_drift = detected_class != event_class

#         event: dict = {
#             "event_id": str(uuid.uuid4()),
#             "type": "pickup",
#             "class": event_class,
#             "event_class": event_class,
#             "detected_class": detected_class,
#             "locked_class": track.locked_class_name,
#             "resolved_class": event_class,
#             "class_drift": bool(class_drift or track.class_drift_flagged),
#             "global_id": track.global_id,
#             "camera_id": event_camera_id,
#             "frame_index": event_frame_index,
#             "timestamp_ms": event_timestamp,
#             "confidence": float(
#                 sum(track.confirmation_confidences)
#                 / max(len(track.confirmation_confidences), 1)
#             ),
#             "confidence_window": list(track.confirmation_confidences),
#             "confirmation_class_votes": list(track.confirmation_class_votes),
#         }

#         if ENABLE_CLASS_DRIFT_WARNINGS and event["class_drift"]:
#             event["warnings"] = [
#                 f"class drift observed on pickup: detected={detected_class}, resolved={event_class}"
#             ]

#         self.session.events.append(event)
#         self.session.pickup_records.append(event)

#         ledger_item = self._pickup_ledger_item_from_event(track, event)
#         self.pending_pickups.append(ledger_item)

#         overlay = f"PICKUP {event_class} G{track.global_id}"
#         if event["class_drift"]:
#             overlay += f" [detected:{detected_class}]"

#         self._push_overlay_event(event_timestamp, overlay)

#         self._emit_debug(
#             f"[EVENT] {overlay} | cam={event_camera_id} frame={event_frame_index} "
#             f"conf={list(track.confirmation_confidences)} "
#             f"ledger_id={ledger_item['ledger_id']}",
#             level="info",
#         )

#         return event

#     def _record_putback(self, track: GlobalTrack) -> Optional[dict]:
#         self._ensure_identity_fields(track)

#         update = track.current_update

#         event_timestamp = track.last_seen_ms if update is None else update.timestamp_ms
#         event_camera_id = track.last_camera_id if update is None else update.camera_id
#         event_frame_index = track.last_frame_index if update is None else update.frame_index

#         detected_class = self._detected_class(track)

#         matched_item = self._match_putback_to_pending_pickup(track, event_timestamp)

#         if matched_item is not None:
#             event_class = matched_item["event_class"]
#             track.resolved_class_name = event_class
#             ledger_match_score = matched_item.get("_last_match_score")
#             ledger_match_breakdown = matched_item.get("_last_match_breakdown")
#         else:
#             event_class = self._event_class(track)
#             ledger_match_score = None
#             ledger_match_breakdown = None

#         current_net = (
#             self._frame_net_snapshot.get(event_class, 0)
#             - self._frame_putback_used.get(event_class, 0)
#         )

#         if current_net <= 0:
#             warning = (
#                 f"Putback ignored class={event_class} detected={detected_class} "
#                 f"track={track.global_id} — no outstanding pickup to offset."
#             )
#             self.session.warnings.append(warning)
#             LOGGER.warning(warning)
#             return None

#         self.putback_count[event_class] += 1
#         self._frame_putback_used[event_class] += 1

#         track.last_putback_confirmed_ms = event_timestamp
#         track.resolved_class_name = event_class

#         if matched_item is not None:
#             matched_item["closed"] = True

#         is_long_hold = (
#             track.pickup_confirmed_ms is not None
#             and (event_timestamp - track.pickup_confirmed_ms) > CLASS_RETURN_MAX_GAP_MS
#         )

#         class_drift = detected_class != event_class

#         event: dict = {
#             "event_id": str(uuid.uuid4()),
#             "type": "putback",
#             "class": event_class,
#             "event_class": event_class,
#             "detected_class": detected_class,
#             "locked_class": track.locked_class_name,
#             "resolved_class": event_class,
#             "class_drift": bool(class_drift or track.class_drift_flagged),
#             "global_id": track.global_id,
#             "camera_id": event_camera_id,
#             "frame_index": event_frame_index,
#             "timestamp_ms": event_timestamp,
#             "confidence": float(
#                 sum(track.confirmation_confidences)
#                 / max(len(track.confirmation_confidences), 1)
#             ),
#             "confidence_window": list(track.confirmation_confidences),
#             "confirmation_class_votes": list(track.confirmation_class_votes),
#         }

#         if matched_item is not None:
#             event["matched_pickup_ledger_id"] = matched_item["ledger_id"]
#             event["matched_pickup_event_id"] = matched_item["pickup_event_id"]
#             event["ledger_match_score"] = ledger_match_score
#             event["ledger_match_breakdown"] = ledger_match_breakdown

#             matched_item["closed_by_event_id"] = event["event_id"]

#         if is_long_hold:
#             event["long_hold_return"] = True

#         if ENABLE_CLASS_DRIFT_WARNINGS and event["class_drift"]:
#             event["warnings"] = [
#                 f"class drift observed on putback: detected={detected_class}, resolved={event_class}"
#             ]

#         self.session.events.append(event)
#         self.session.putback_records.append(event)

#         overlay = f"PUTBACK {event_class} G{track.global_id}"
#         if event["class_drift"]:
#             overlay += f" [detected:{detected_class}]"
#         if is_long_hold:
#             overlay += " [long-hold]"

#         self._push_overlay_event(event_timestamp, overlay)

#         self._emit_debug(
#             f"[EVENT] {overlay} | cam={event_camera_id} frame={event_frame_index} "
#             f"conf={list(track.confirmation_confidences)} "
#             f"matched_ledger={matched_item['ledger_id'] if matched_item else None}",
#             level="info",
#         )

#         return event

#     # ─────────────────────────────────────────────────────────────────────────
#     # Overlay
#     # ─────────────────────────────────────────────────────────────────────────

#     def overlay_event_lines(self, timestamp_ms: float) -> List[str]:
#         while (
#             self.recent_overlay_events
#             and (timestamp_ms - self.recent_overlay_events[0][0]) > OVERLAY_EVENT_TTL_MS
#         ):
#             self.recent_overlay_events.popleft()

#         return [
#             item[1]
#             for item in list(self.recent_overlay_events)[-OVERLAY_MAX_EVENT_LINES:]
#         ]

#     # ─────────────────────────────────────────────────────────────────────────
#     # Core FSM
#     # ─────────────────────────────────────────────────────────────────────────

#     def _process_track(
#         self,
#         track: GlobalTrack,
#         timestamp_ms: float,
#         all_tracks: Optional[List[GlobalTrack]] = None,
#     ) -> List[dict]:
#         self._ensure_identity_fields(track)

#         update = track.current_update

#         # =====================================================================
#         # BRANCH A: no detection this frame
#         # =====================================================================
#         if update is None:
#             if track.event_state == "PICKUP_PENDING":
#                 if self._pending_timeout_elapsed(track, timestamp_ms):

#                     if self._has_any_recent_outward_motion(track, look_back=5):
#                         self._transition(
#                             track,
#                             "PICKED_UP",
#                             "pickup confirmed: lost outward motion",
#                         )
#                         return [self._record_pickup(track)]

#                     elif track.was_stable:
#                         self._transition(
#                             track,
#                             "PICKED_UP",
#                             "pickup confirmed: lost was stable occlusion/vertical",
#                         )
#                         event = self._record_pickup(track)
#                         event["occlusion_pickup"] = True
#                         self._replace_last_overlay(
#                             f"PICKUP {event['class']} G{track.global_id} [occluded]"
#                         )
#                         return [event]

#                     else:
#                         self._emit_debug(
#                             f"[SILENT-CANCEL] track={track.global_id} "
#                             "never reached stability — suppressed",
#                             level="debug",
#                         )
#                         self._transition(
#                             track,
#                             "INSIDE",
#                             "pickup cancelled: never reached stability",
#                         )
#                         return []

#             elif track.event_state == "PUTBACK_PENDING":
#                 if self._pending_timeout_elapsed(track, timestamp_ms):
#                     if (
#                         track.last_seen_in_stable_roi is True
#                         and self._has_any_recent_inward_motion(track, look_back=5)
#                     ):
#                         self._transition(
#                             track,
#                             "STABLE_INSIDE",
#                             "putback confirmed: lost in stable ROI",
#                         )
#                         event = self._record_putback(track)
#                         return [] if event is None else [event]

#                     self._emit_debug(
#                         f"[TIMEOUT] putback absent track={track.global_id} "
#                         f"last_stable={track.last_seen_in_stable_roi}",
#                         level="info",
#                     )
#                     self._transition(
#                         track,
#                         "PICKED_UP",
#                         "putback cancelled: lost without entering stable ROI",
#                     )

#             return []

#         # =====================================================================
#         # BRANCH B: detection exists
#         # =====================================================================

#         self._append_class_vote(track, update)

#         prev_safe: Optional[bool] = track.last_seen_in_safe_roi
#         prev_outer: Optional[bool] = track.last_seen_in_outer_roi
#         prev_stable: Optional[bool] = track.last_seen_in_stable_roi

#         track.last_seen_in_safe_roi = update.in_safe_roi
#         track.last_seen_in_outer_roi = update.in_outer_roi
#         track.last_seen_in_stable_roi = update.in_stable_roi

#         if update.displacement_magnitude > 0:
#             if update.outward_motion:
#                 motion_direction: Optional[bool] = True
#             elif update.inward_motion:
#                 motion_direction = False
#             else:
#                 motion_direction = None
#         else:
#             motion_direction = None

#         track.motion_direction_history.append(motion_direction)

#         if update.in_outer_roi and not update.in_stable_roi:
#             if track.boundary_linger_start_ms is None:
#                 track.boundary_linger_start_ms = update.timestamp_ms
#             elif (update.timestamp_ms - track.boundary_linger_start_ms) > 1000.0:
#                 track.boundary_linger_flagged = True
#         else:
#             track.boundary_linger_start_ms = None

#         required_stability = self._required_stability_frames(track, update.timestamp_ms)

#         # =====================================================================
#         # STATE: INSIDE
#         # =====================================================================
#         if track.event_state == "INSIDE":
#             if self._is_class_return_candidate(track):
#                 self._transition(
#                     track,
#                     "PUTBACK_PENDING",
#                     "class/ledger inward re-entry candidate",
#                 )
#                 track.pending_since_ms = update.timestamp_ms
#                 self._seed_class_return_confirmation(track)

#                 self._emit_debug(
#                     f"[PENDING] putback track={track.global_id} "
#                     f"event_class={self._event_class(track)} "
#                     f"detected={self._detected_class(track)}",
#                     level="info",
#                 )
#                 return []

#             if update.in_stable_roi and update.confidence >= CONF_THRESHOLD:
#                 track.inside_counter += 1

#                 if track.inside_counter >= required_stability:
#                     self._transition(track, "STABLE_INSIDE", "stable in shelf ROI")
#                     track.boundary_linger_flagged = False

#                     if all_tracks is not None:
#                         self._check_neighbor_stability_boost(track, all_tracks)
#             else:
#                 track.inside_counter = 0

#             return []

#         # =====================================================================
#         # STATE: STABLE_INSIDE
#         # =====================================================================
#         if track.event_state == "STABLE_INSIDE":
#             if track.locked_class_name == UNKNOWN_UNSTABLE_CLASS:
#                 self._lock_class_if_possible(track, reason="additional stable evidence")

#             if self._is_class_return_candidate(track):
#                 self._transition(
#                     track,
#                     "PUTBACK_PENDING",
#                     "class/ledger inward re-entry candidate",
#                 )
#                 track.pending_since_ms = update.timestamp_ms
#                 self._seed_class_return_confirmation(track)

#                 self._emit_debug(
#                     f"[PENDING] putback track={track.global_id} "
#                     f"event_class={self._event_class(track)} "
#                     f"detected={self._detected_class(track)}",
#                     level="info",
#                 )
#                 return []

#             if not update.in_safe_roi:
#                 has_motion = self._has_any_recent_outward_motion(track, look_back=3)
#                 has_roi_exit = self._roi_exit_occurred(track, update, prev_stable)

#                 if has_motion or has_roi_exit:
#                     if (
#                         track.last_putback_confirmed_ms is not None
#                         and (update.timestamp_ms - track.last_putback_confirmed_ms)
#                         < POST_PUTBACK_COOLDOWN_MS
#                     ):
#                         self._emit_debug(
#                             f"[COOLDOWN] pickup suppressed track={track.global_id} "
#                             f"elapsed={update.timestamp_ms - track.last_putback_confirmed_ms:.0f}ms "
#                             f"cooldown={POST_PUTBACK_COOLDOWN_MS}ms",
#                             level="debug",
#                         )
#                         return []

#                     self._lock_class_if_possible(track, reason="pickup trigger")

#                     trigger_reason = (
#                         "outward exit + motion"
#                         if has_motion
#                         else "ROI exit without motion vertical/edge/fast/low-tray pick"
#                     )

#                     self._transition(track, "PICKUP_PENDING", trigger_reason)
#                     track.pending_since_ms = update.timestamp_ms
#                     self._seed_confirmation_from_history(track)

#                     if all_tracks is not None:
#                         self._check_neighbor_stability_boost(track, all_tracks)

#                     self._emit_debug(
#                         f"[PENDING] pickup track={track.global_id} "
#                         f"event_class={self._event_class(track)} "
#                         f"detected={self._detected_class(track)} "
#                         f"motion={has_motion} roi_exit={has_roi_exit} "
#                         f"prev_safe={prev_safe} curr_safe={update.in_safe_roi} "
#                         f"prev_stable={prev_stable} curr_stable={update.in_stable_roi} "
#                         f"safe_dist={update.safe_roi_distance:.1f} "
#                         f"motion_history={list(track.motion_direction_history)}",
#                         level="info",
#                     )
#                     return []

#             if update.in_stable_roi and update.confidence >= CONF_THRESHOLD:
#                 track.inside_counter = required_stability
#             else:
#                 track.inside_counter = 0

#             return []

#         # =====================================================================
#         # STATE: PICKUP_PENDING
#         # =====================================================================
#         if track.event_state == "PICKUP_PENDING":

#             if update.in_stable_roi and not update.outward_motion:
#                 time_since_pending_ms = update.timestamp_ms - (
#                     track.pending_since_ms or update.timestamp_ms
#                 )

#                 if time_since_pending_ms < QUICK_RETURN_THRESHOLD_MS:
#                     self._transition(
#                         track,
#                         "PICKED_UP",
#                         "quick pickup: fast re-entry within threshold",
#                     )

#                     pickup_event = self._record_pickup(track)

#                     self._frame_net_snapshot[pickup_event["class"]] = (
#                         self._frame_net_snapshot.get(pickup_event["class"], 0) + 1
#                     )

#                     track.resolved_class_name = pickup_event["class"]

#                     self._transition(
#                         track,
#                         "PUTBACK_PENDING",
#                         "quick return: fast re-entry",
#                     )
#                     track.pending_since_ms = update.timestamp_ms
#                     self._seed_confirmation_from_history(track)

#                     self._emit_debug(
#                         f"[QUICK-RETURN] track={track.global_id} "
#                         f"time_ms={time_since_pending_ms:.0f} "
#                         f"event_class={pickup_event['class']} "
#                         f"detected={self._detected_class(track)}",
#                         level="info",
#                     )

#                     return [pickup_event]

#                 self._emit_debug(
#                     f"[CANCEL] pickup track={track.global_id} "
#                     "re-entered stable ROI after threshold",
#                     level="info",
#                 )
#                 self._transition(
#                     track,
#                     "STABLE_INSIDE",
#                     "pickup cancelled: returned to stable ROI",
#                 )
#                 return []

#             if not update.in_outer_roi:
#                 self._append_confirmation(track, update.confidence)

#                 confirmation_ok = self._confirmation_ok(
#                     track,
#                     require_class_consistency=True,
#                 )

#                 timeout_absent_ok = (
#                     track.was_stable
#                     and (update.timestamp_ms - (track.pending_since_ms or update.timestamp_ms)) > 100
#                 )

#                 if confirmation_ok or timeout_absent_ok:
#                     self._transition(
#                         track,
#                         "PICKED_UP",
#                         "pickup confirmed: SAFE -> OUTER -> ABSENT",
#                     )
#                     return [self._record_pickup(track)]

#                 return []

#             if update.in_outer_roi:
#                 self._append_confirmation(track, update.confidence)

#                 if len(track.confirmation_confidences) >= CONFIRM_FRAMES:
#                     conf_ok = self._confidence_ok(track)
#                     class_ok = self._class_consistency_ok(track)

#                     if not conf_ok:
#                         self._emit_debug(
#                             f"[CANCEL] pickup track={track.global_id} "
#                             f"confidence too low in OUTER conf={list(track.confirmation_confidences)}",
#                             level="info",
#                         )
#                         self._transition(
#                             track,
#                             "STABLE_INSIDE",
#                             "pickup cancelled: confidence too low",
#                         )
#                         return []

#                     if not class_ok:
#                         if track.was_stable and self._has_any_recent_outward_motion(track, look_back=5):
#                             self._emit_debug(
#                                 f"[CLASS-DRIFT-TOLERATED] pickup track={track.global_id} "
#                                 f"keeping event_class={self._event_class(track)} "
#                                 f"detected={self._detected_class(track)}",
#                                 level="info",
#                             )
#                             return []

#                         self._emit_debug(
#                             f"[CANCEL] pickup track={track.global_id} "
#                             "class consistency too low in OUTER without strong pickup evidence",
#                             level="info",
#                         )
#                         self._transition(
#                             track,
#                             "STABLE_INSIDE",
#                             "pickup cancelled: class consistency too low",
#                         )
#                         return []

#                 if self._has_any_recent_outward_motion(track, look_back=3):
#                     return []

#                 if self._pending_timeout_elapsed(track, update.timestamp_ms):
#                     self._emit_debug(
#                         f"[TIMEOUT] pickup OUTER track={track.global_id} timeout elapsed",
#                         level="info",
#                     )
#                     self._transition(
#                         track,
#                         "STABLE_INSIDE",
#                         "pickup cancelled: timeout in OUTER ROI",
#                     )
#                     return []

#                 return []

#             return []

#         # =====================================================================
#         # STATE: PICKED_UP
#         # =====================================================================
#         if track.event_state == "PICKED_UP":
#             track_pickup_ts: Optional[float] = track.pickup_confirmed_ms

#             is_long_hold = (
#                 track_pickup_ts is not None
#                 and (timestamp_ms - track_pickup_ts) > CLASS_RETURN_MAX_GAP_MS
#             )

#             if is_long_hold and (update.in_stable_roi or update.in_outer_roi):
#                 matched = self._match_putback_to_pending_pickup(track, update.timestamp_ms)
#                 if matched is not None:
#                     track.resolved_class_name = matched["event_class"]

#                 track.long_hold_pending = True
#                 self._transition(
#                     track,
#                     "PUTBACK_PENDING",
#                     "long-hold fallback: elapsed > CLASS_RETURN_MAX_GAP_MS",
#                 )
#                 track.pending_since_ms = update.timestamp_ms
#                 self._seed_confirmation_from_history(track)

#                 self._emit_debug(
#                     f"[LONG-HOLD] track={track.global_id} "
#                     f"event_class={self._event_class(track)} "
#                     f"detected={self._detected_class(track)} "
#                     f"elapsed_ms={timestamp_ms - track_pickup_ts:.0f}",
#                     level="info",
#                 )
#                 return []

#             if update.in_stable_roi or update.in_outer_roi:
#                 has_inward = self._has_any_recent_inward_motion(track, look_back=3)
#                 has_reentry = self._roi_reentry_occurred(update, prev_safe, prev_outer)

#                 matched = self._match_putback_to_pending_pickup(track, update.timestamp_ms)
#                 if matched is not None:
#                     track.resolved_class_name = matched["event_class"]

#                 trigger_reason = (
#                     "inward re-entry + motion"
#                     if has_inward
#                     else "ROI re-entry from absent vertical/edge/fast return"
#                     if has_reentry
#                     else "ROI presence after pickup product returned to zone"
#                 )

#                 self._transition(track, "PUTBACK_PENDING", trigger_reason)
#                 track.pending_since_ms = update.timestamp_ms
#                 self._seed_confirmation_from_history(track)

#                 self._emit_debug(
#                     f"[PENDING] putback track={track.global_id} "
#                     f"event_class={self._event_class(track)} "
#                     f"detected={self._detected_class(track)} "
#                     f"safe={update.in_safe_roi} stable={update.in_stable_roi} "
#                     f"outer={update.in_outer_roi} "
#                     f"motion={has_inward} reentry={has_reentry} "
#                     f"prev_safe={prev_safe} prev_outer={prev_outer}",
#                     level="info",
#                 )

#                 return []

#             return []

#         # =====================================================================
#         # STATE: PUTBACK_PENDING
#         # =====================================================================
#         if track.event_state == "PUTBACK_PENDING":

#             if update.in_stable_roi:
#                 self._append_confirmation(track, update.confidence)

#                 if len(track.confirmation_confidences) >= CONFIRM_FRAMES:
#                     matched = self._match_putback_to_pending_pickup(track, update.timestamp_ms)
#                     if matched is not None:
#                         track.resolved_class_name = matched["event_class"]

#                     # For putback, do not require class consistency.
#                     # A return may be classified as another product.
#                     if not self._confidence_ok(track):
#                         self._emit_debug(
#                             f"[CANCEL] putback confidence track={track.global_id} "
#                             f"conf={list(track.confirmation_confidences)}",
#                             level="info",
#                         )
#                         self._transition(
#                             track,
#                             "PICKED_UP",
#                             "putback cancelled by confidence window",
#                         )
#                         return []

#                     self._transition(
#                         track,
#                         "STABLE_INSIDE",
#                         "putback confirmed: stable in shelf ROI",
#                     )
#                     event = self._record_putback(track)
#                     return [] if event is None else [event]

#                 return []

#             if update.in_outer_roi:
#                 self._append_confirmation(track, update.confidence)

#                 if len(track.confirmation_confidences) >= CONFIRM_FRAMES:
#                     if not self._confidence_ok(track):
#                         self._emit_debug(
#                             f"[CANCEL] putback OUTER confidence track={track.global_id} "
#                             f"conf={list(track.confirmation_confidences)}",
#                             level="info",
#                         )
#                         self._transition(
#                             track,
#                             "PICKED_UP",
#                             "putback cancelled: confidence too low in OUTER",
#                         )
#                         return []

#                 if self._pending_timeout_elapsed(track, update.timestamp_ms):
#                     if not self._has_any_recent_inward_motion(track, look_back=5):
#                         self._emit_debug(
#                             f"[TIMEOUT] putback OUTER track={track.global_id} no inward motion",
#                             level="info",
#                         )
#                         self._transition(
#                             track,
#                             "PICKED_UP",
#                             "putback cancelled: timeout with no inward motion",
#                         )
#                         return []

#                     track.pending_since_ms = update.timestamp_ms

#                 return []

#             if self._pending_timeout_elapsed(track, update.timestamp_ms):
#                 self._emit_debug(
#                     f"[TIMEOUT] putback absent track={track.global_id}",
#                     level="info",
#                 )
#                 self._transition(
#                     track,
#                     "PICKED_UP",
#                     "putback cancelled: timeout while absent",
#                 )
#                 return []

#             return []

#         return []

#     # ─────────────────────────────────────────────────────────────────────────
#     # Public API
#     # ─────────────────────────────────────────────────────────────────────────

#     def process_frame(
#         self,
#         tracks: List[GlobalTrack],
#         frame_avg_confidence: float,
#         timestamp_ms: float,
#     ) -> List[dict]:
#         for track in tracks:
#             self._ensure_identity_fields(track)

#         self._frame_net_snapshot = {
#             class_name: self.pickup_count[class_name] - self.putback_count[class_name]
#             for class_name in set(self.pickup_count) | set(self.putback_count)
#         }

#         self._frame_putback_used = defaultdict(int)

#         events: List[dict] = []

#         for track in tracks:
#             events.extend(self._process_track(track, timestamp_ms, all_tracks=tracks))

#         self._frame_net_snapshot = {}
#         self._frame_putback_used = defaultdict(int)

#         return events

#     def finalize(self) -> SessionSummary:
#         classes = set(self.pickup_count) | set(self.putback_count)

#         self.session.pickup_count = dict(self.pickup_count)
#         self.session.putback_count = dict(self.putback_count)
#         self.session.net_change = {
#             cls: self.pickup_count[cls] - self.putback_count[cls]
#             for cls in classes
#         }
#         self.session.total_pickups = sum(self.pickup_count.values())
#         self.session.total_putbacks = sum(self.putback_count.values())

#         self.session.pending_pickups = [
#             item for item in self.pending_pickups if not item.get("closed")
#         ]

#         return self.session


# from __future__ import annotations

# import logging
# import uuid
# from collections import defaultdict, deque
# from typing import Deque, Dict, List, Optional

# import numpy as np

# from config import (
#     CLASS_RETURN_MAX_GAP_MS,
#     CONFIRM_FRAMES,
#     CONF_MIN_FRAME,
#     CONF_THRESHOLD,
#     LONG_BOUNDARY_STABILITY_FRAMES,
#     MISSING_PENDING_CONFIRM_MS,
#     OVERLAY_EVENT_TTL_MS,
#     OVERLAY_MAX_EVENT_LINES,
#     POST_PUTBACK_COOLDOWN_MS,
#     PRINT_DEBUG_EVENTS,
#     QUICK_RETURN_THRESHOLD_MS,
#     STABILITY_FRAMES,

#     CLASS_VOTE_HISTORY,
#     CLASS_LOCK_RATIO,
#     CLASS_LOCK_MIN_TOTAL_SCORE,
#     CLASS_VOTE_MIN_CONFIDENCE,
#     CLASS_VOTE_WEIGHT_STABLE,
#     CLASS_VOTE_WEIGHT_SAFE,
#     CLASS_VOTE_WEIGHT_OUTER,
#     CLASS_VOTE_WEIGHT_OUTSIDE,
#     CLASS_VOTE_RECENCY_DECAY,
#     CLASS_CONSISTENCY_RATIO,
#     UNKNOWN_UNSTABLE_CLASS,
#     ENABLE_CLASS_DRIFT_WARNINGS,

#     PENDING_LEDGER_MAX_AGE_MS,
#     PUTBACK_LEDGER_MATCH_THRESHOLD,
#     LEDGER_MATCH_WEIGHT_CLASS,
#     LEDGER_MATCH_WEIGHT_TIME,
#     LEDGER_MATCH_WEIGHT_TRACK,
#     LEDGER_MATCH_WEIGHT_CLASS_CANDIDATE,
#     LEDGER_MATCH_WEIGHT_MOTION,
#     LEDGER_MATCH_WEIGHT_EMBEDDING,
#     LEDGER_MATCH_WEIGHT_SPATIAL,
#     LEDGER_SAME_CLASS_SCORE,
#     LEDGER_DIFFERENT_CLASS_SCORE,
#     LEDGER_SPATIAL_MAX_DISTANCE_PX,
#     EMBEDDING_SIM_THRESHOLDS_BY_CLASS,      # New per‑class thresholds
# )
# from utils.types import GlobalTrack, SessionSummary

# LOGGER = logging.getLogger("vending_pipeline.events")


# class EventManager:
#     """
#     Pickup / putback FSM with:
#     - class-drift protection
#     - stable class locking
#     - pending pickup ledger
#     - cross-class putback matching
#     - conservative inward-motion blocking
#     - minimum hold time before putback

#     This version avoids return type hints on methods to prevent copy/paste
#     corruption like Optionalhistory, Listactive, Optionalself, Listwhile, etc.
#     """

#     def __init__(self, session_id: str):
#         self.session = SessionSummary(session_id=session_id)

#         self.pickup_count: Dict[str, int] = defaultdict(int)
#         self.putback_count: Dict[str, int] = defaultdict(int)
#         self.class_pickup_timestamps: Dict[str, Deque[float]] = defaultdict(deque)

#         self.pending_pickups: List[dict] = []

#         self._frame_net_snapshot: Dict[str, int] = {}
#         self._frame_putback_used: Dict[str, int] = defaultdict(int)

#         self.last_putback_by_class_ms: Dict[str, float] = {}

#         # Prevent instant fake putback after a pickup.
#         # Real putbacks normally take more than this unless quick-return path is used.
#         self.min_valid_putback_hold_ms = 500.0

#         self.last_event_text = ""
#         self.recent_overlay_events: Deque[tuple[float, str]] = deque(maxlen=20)

#     # ─────────────────────────────────────────────────────────────────────────
#     # Compatibility / identity helpers
#     # ─────────────────────────────────────────────────────────────────────────

#     def _ensure_identity_fields(self, track):
#         if not hasattr(track, "class_votes") or track.class_votes is None:
#             track.class_votes = deque(maxlen=CLASS_VOTE_HISTORY)

#         if not hasattr(track, "confirmation_class_votes") or track.confirmation_class_votes is None:
#             track.confirmation_class_votes = deque(maxlen=CONFIRM_FRAMES)

#         if not hasattr(track, "locked_class_name"):
#             track.locked_class_name = None

#         if not hasattr(track, "resolved_class_name"):
#             track.resolved_class_name = None

#         if not hasattr(track, "class_drift_flagged"):
#             track.class_drift_flagged = False

#         if not hasattr(track, "visual_embedding"):
#             track.visual_embedding = None

#     def _is_valid_class(self, class_name):
#         return bool(class_name) and class_name != UNKNOWN_UNSTABLE_CLASS

#     def _family_base_class(self, class_name):
#         if not class_name:
#             return UNKNOWN_UNSTABLE_CLASS
            
#         lower = class_name.lower()
#         if lower.startswith("barebells"):
#             return "Barebells"
#         elif lower.startswith("quest chips"):
#             return "Quest Chips"
#         elif lower.startswith("one bar"):
#             return "One Bar"
#         elif lower.startswith("legendary tasty pastry"):
#             return "Legendary Tasty Pastry"
#         elif lower.startswith("aquafina"):
#             return "Aquafina"
#         elif lower.startswith("skullcandy"):
#             return "Skullcandy DIME 3"
            
#         return class_name

#     def _detected_class(self, track):
#         update = track.current_update
#         if update is not None and getattr(update, "class_name", None):
#             return update.class_name
#         return getattr(track, "class_name", None) or UNKNOWN_UNSTABLE_CLASS

#     def _voted_event_class(self, track):
#         vote_result = self._compute_weighted_class_vote(
#             track,
#             stable_only=False,
#             include_safe=True,
#         )
#         voted_class = vote_result.get("class")

#         if self._is_valid_class(voted_class):
#             return voted_class

#         return self._detected_class(track)

#     def _pickup_event_class(self, track):
#         """
#         Pickup identity must come from stable shelf identity.
#         Do not allow stale ledger-resolved class to override locked class.
#         """
#         self._ensure_identity_fields(track)

#         if self._is_valid_class(track.locked_class_name):
#             return track.locked_class_name

#         return self._voted_event_class(track)

#     def _putback_event_class(self, track):
#         """
#         Putback identity may come from ledger/resolved identity because
#         return detections can be misclassified.
#         """
#         self._ensure_identity_fields(track)

#         if self._is_valid_class(track.resolved_class_name):
#             return track.resolved_class_name

#         if self._is_valid_class(track.locked_class_name):
#             return track.locked_class_name

#         return self._voted_event_class(track)

#     def _event_class(self, track, purpose="generic"):
#         self._ensure_identity_fields(track)

#         if purpose == "pickup":
#             return self._pickup_event_class(track)

#         if purpose == "putback":
#             return self._putback_event_class(track)

#         if self._is_valid_class(track.locked_class_name):
#             return track.locked_class_name

#         if self._is_valid_class(track.resolved_class_name):
#             return track.resolved_class_name

#         return self._voted_event_class(track)

#     # ─────────────────────────────────────────────────────────────────────────
#     # Logging / overlay
#     # ─────────────────────────────────────────────────────────────────────────

#     def _emit_debug(self, message, level="debug"):
#         getattr(LOGGER, level)(message)
#         if PRINT_DEBUG_EVENTS:
#             print(message)

#     def _push_overlay_event(self, timestamp_ms, message):
#         self.recent_overlay_events.append((timestamp_ms, message))
#         self.last_event_text = message

#     def _replace_last_overlay(self, message):
#         if self.recent_overlay_events:
#             ts, _ = self.recent_overlay_events[-1]
#             self.recent_overlay_events[-1] = (ts, message)
#             self.last_event_text = message

#     def _log_transition(self, track, old_state, new_state, reason):
#         message = (
#             f"[TRACK {track.global_id}] {old_state} -> {new_state} | "
#             f"event_class={self._event_class(track)} | "
#             f"detected_class={self._detected_class(track)} | "
#             f"locked_class={getattr(track, 'locked_class_name', None)} | "
#             f"resolved_class={getattr(track, 'resolved_class_name', None)} | "
#             f"reason={reason}"
#         )
#         track.last_debug_reason = reason
#         self._emit_debug(message, level="debug")

#     def _transition(self, track, new_state, reason):
#         self._ensure_identity_fields(track)

#         if track.event_state == new_state:
#             return

#         old_state = track.event_state
#         track.event_state = new_state

#         if new_state == "STABLE_INSIDE":
#             track.was_stable = True
#             self._lock_class_if_possible(track, reason="entered STABLE_INSIDE")

#         if new_state in {"PICKUP_PENDING", "PUTBACK_PENDING"}:
#             track.confirmation_confidences.clear()
#             track.confirmation_class_votes.clear()
#         else:
#             track.pending_since_ms = None

#         self._log_transition(track, old_state, new_state, reason)

#     # ─────────────────────────────────────────────────────────────────────────
#     # Class voting / class locking
#     # ─────────────────────────────────────────────────────────────────────────

#     def _roi_weight_for_vote(self, update):
#         if update is None:
#             return CLASS_VOTE_WEIGHT_OUTSIDE

#         if update.in_stable_roi:
#             return CLASS_VOTE_WEIGHT_STABLE

#         if update.in_safe_roi:
#             return CLASS_VOTE_WEIGHT_SAFE

#         if update.in_outer_roi:
#             return CLASS_VOTE_WEIGHT_OUTER

#         return CLASS_VOTE_WEIGHT_OUTSIDE

#     def _append_class_vote(self, track, update):
#         self._ensure_identity_fields(track)

#         if update is None:
#             return

#         confidence = float(update.confidence or 0.0)
#         if confidence < CLASS_VOTE_MIN_CONFIDENCE:
#             return

#         detected_class = getattr(update, "class_name", None) or self._detected_class(track)

#         track.class_votes.append(
#             {
#                 "class": detected_class,
#                 "confidence": confidence,
#                 "timestamp_ms": float(update.timestamp_ms),
#                 "roi_weight": self._roi_weight_for_vote(update),
#                 "in_stable_roi": bool(update.in_stable_roi),
#                 "in_safe_roi": bool(update.in_safe_roi),
#                 "in_outer_roi": bool(update.in_outer_roi),
#             }
#         )

#     def _compute_weighted_class_vote(self, track, stable_only=False, include_safe=True):
#         self._ensure_identity_fields(track)

#         votes = list(track.class_votes)

#         if stable_only:
#             votes = [v for v in votes if v.get("in_stable_roi")]
#         elif include_safe:
#             votes = [
#                 v
#                 for v in votes
#                 if v.get("in_stable_roi") or v.get("in_safe_roi") or v.get("in_outer_roi")
#             ]

#         if not votes:
#             return {
#                 "class": UNKNOWN_UNSTABLE_CLASS,
#                 "score": 0.0,
#                 "total_score": 0.0,
#                 "ratio": 0.0,
#                 "scores": {},
#             }

#         scores = defaultdict(float)
#         n = len(votes)

#         for idx, vote in enumerate(votes):
#             age_from_latest = n - 1 - idx
#             recency_weight = CLASS_VOTE_RECENCY_DECAY ** age_from_latest

#             cls = vote["class"]
#             score = (
#                 float(vote.get("confidence", 0.0))
#                 * float(vote.get("roi_weight", 1.0))
#                 * recency_weight
#             )
#             scores[cls] += score

#         total_score = sum(scores.values())

#         if total_score <= 0:
#             return {
#                 "class": UNKNOWN_UNSTABLE_CLASS,
#                 "score": 0.0,
#                 "total_score": 0.0,
#                 "ratio": 0.0,
#                 "scores": dict(scores),
#             }

#         sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
#         best_class, best_score = sorted_scores[0]
#         ratio = best_score / total_score

#         # Tie-breaker / Family Fallback (e.g. 80% vs 80% split)
#         if len(sorted_scores) >= 2:
#             runner_up_class, runner_up_score = sorted_scores[1]
#             runner_up_ratio = runner_up_score / total_score
            
#             # If both are very competitive and cause a tie that breaks the lock ratio
#             if ratio >= 0.35 and runner_up_ratio >= 0.35:
#                 base_a = self._family_base_class(best_class)
#                 base_b = self._family_base_class(runner_up_class)
#                 if base_a == base_b and base_a != UNKNOWN_UNSTABLE_CLASS:
#                     best_class = base_a
#                     best_score = best_score + runner_up_score
#                     ratio = best_score / total_score

#         selected_class = (
#             best_class
#             if total_score >= CLASS_LOCK_MIN_TOTAL_SCORE and ratio >= CLASS_LOCK_RATIO
#             else UNKNOWN_UNSTABLE_CLASS
#         )

#         return {
#             "class": selected_class,
#             "score": best_score,
#             "total_score": total_score,
#             "ratio": ratio,
#             "scores": dict(scores),
#         }

#     def _lock_class_if_possible(self, track, reason):
#         self._ensure_identity_fields(track)

#         if self._is_valid_class(track.locked_class_name):
#             return

#         result = self._compute_weighted_class_vote(track, stable_only=True)

#         if result["class"] == UNKNOWN_UNSTABLE_CLASS:
#             result = self._compute_weighted_class_vote(
#                 track,
#                 stable_only=False,
#                 include_safe=True,
#             )

#         locked_class = result["class"]

#         if locked_class == UNKNOWN_UNSTABLE_CLASS:
#             track.locked_class_name = UNKNOWN_UNSTABLE_CLASS
#             track.resolved_class_name = UNKNOWN_UNSTABLE_CLASS

#             self._emit_debug(
#                 f"[CLASS-LOCK-DELAYED] track={track.global_id} "
#                 f"reason={reason} ratio={result['ratio']:.2f} "
#                 f"scores={result['scores']}",
#                 level="info",
#             )
#             return

#         track.locked_class_name = locked_class
#         track.resolved_class_name = locked_class

#         detected_class = self._detected_class(track)
#         if detected_class != locked_class:
#             track.class_drift_flagged = True

#         self._emit_debug(
#             f"[CLASS-LOCKED] track={track.global_id} "
#             f"locked={locked_class} detected={detected_class} "
#             f"reason={reason} ratio={result['ratio']:.2f} "
#             f"scores={result['scores']}",
#             level="info",
#         )

#     # ─────────────────────────────────────────────────────────────────────────
#     # Confirmation helpers
#     # ─────────────────────────────────────────────────────────────────────────

#     def _append_confirmation(self, track, confidence):
#         self._ensure_identity_fields(track)

#         if track.confirmation_confidences.maxlen != CONFIRM_FRAMES:
#             track.confirmation_confidences = deque(
#                 track.confirmation_confidences,
#                 maxlen=CONFIRM_FRAMES,
#             )

#         if track.confirmation_class_votes.maxlen != CONFIRM_FRAMES:
#             track.confirmation_class_votes = deque(
#                 track.confirmation_class_votes,
#                 maxlen=CONFIRM_FRAMES,
#             )

#         track.confirmation_confidences.append(float(confidence))

#         update = track.current_update
#         if update is not None:
#             detected_class = getattr(update, "class_name", None) or self._detected_class(track)
#             track.confirmation_class_votes.append(
#                 {
#                     "class": detected_class,
#                     "confidence": float(confidence),
#                     "timestamp_ms": float(update.timestamp_ms),
#                 }
#             )

#     def _seed_confirmation_from_history(self, track):
#         self._ensure_identity_fields(track)

#         history = list(track.confidence_history)[-CONFIRM_FRAMES:]
#         track.confirmation_confidences = deque(history, maxlen=CONFIRM_FRAMES)

#         class_history = list(track.class_votes)[-CONFIRM_FRAMES:]
#         track.confirmation_class_votes = deque(
#             [
#                 {
#                     "class": item["class"],
#                     "confidence": item["confidence"],
#                     "timestamp_ms": item["timestamp_ms"],
#                 }
#                 for item in class_history
#             ],
#             maxlen=CONFIRM_FRAMES,
#         )

#     def _seed_class_return_confirmation(self, track):
#         self._ensure_identity_fields(track)

#         history = list(track.confidence_history)[-CONFIRM_FRAMES:]
#         if not history:
#             history = [track.last_confidence]

#         while len(history) < CONFIRM_FRAMES:
#             history.append(history[-1])

#         track.confirmation_confidences = deque(history, maxlen=CONFIRM_FRAMES)

#         class_history = list(track.class_votes)[-CONFIRM_FRAMES:]
#         if not class_history:
#             detected_class = self._detected_class(track)
#             class_history = [
#                 {
#                     "class": detected_class,
#                     "confidence": track.last_confidence,
#                     "timestamp_ms": track.last_seen_ms,
#                 }
#             ]

#         while len(class_history) < CONFIRM_FRAMES:
#             class_history.append(class_history[-1])

#         track.confirmation_class_votes = deque(
#             [
#                 {
#                     "class": item["class"],
#                     "confidence": item["confidence"],
#                     "timestamp_ms": item["timestamp_ms"],
#                 }
#                 for item in class_history[-CONFIRM_FRAMES:]
#             ],
#             maxlen=CONFIRM_FRAMES,
#         )

#     def _pending_timeout_elapsed(self, track, timestamp_ms):
#         if track.pending_since_ms is None:
#             return False
#         return (timestamp_ms - track.pending_since_ms) >= MISSING_PENDING_CONFIRM_MS

#     def _confidence_ok(self, track):
#         if len(track.confirmation_confidences) < CONFIRM_FRAMES:
#             return False

#         avg_conf = sum(track.confirmation_confidences) / len(track.confirmation_confidences)
#         min_conf = min(track.confirmation_confidences)

#         return avg_conf >= CONF_THRESHOLD and min_conf >= CONF_MIN_FRAME

#     def _class_consistency_ok(self, track):
#         self._ensure_identity_fields(track)

#         event_class = self._event_class(track, purpose="pickup")
#         if not event_class or event_class == UNKNOWN_UNSTABLE_CLASS:
#             return False

#         votes = list(track.confirmation_class_votes)
#         if not votes:
#             return True

#         total_weight = 0.0
#         matched_weight = 0.0

#         for vote in votes:
#             cls = vote.get("class")
#             conf = float(vote.get("confidence", 0.0) or 0.0)

#             total_weight += conf
#             if cls == event_class:
#                 matched_weight += conf

#         if total_weight <= 0:
#             return False

#         ratio = matched_weight / total_weight

#         if ratio < CLASS_CONSISTENCY_RATIO:
#             track.class_drift_flagged = True
#             self._emit_debug(
#                 f"[CLASS-DRIFT] track={track.global_id} "
#                 f"event_class={event_class} votes={votes} consistency={ratio:.2f}",
#                 level="info",
#             )

#         return ratio >= CLASS_CONSISTENCY_RATIO

#     def _confirmation_ok(self, track, require_class_consistency=True):
#         if not self._confidence_ok(track):
#             return False

#         if not require_class_consistency:
#             return True

#         return self._class_consistency_ok(track)

#     # ─────────────────────────────────────────────────────────────────────────
#     # Motion helpers
#     # ─────────────────────────────────────────────────────────────────────────

#     def _required_stability_frames(self, track, timestamp_ms):
#         return (
#             LONG_BOUNDARY_STABILITY_FRAMES
#             if track.boundary_linger_flagged
#             else STABILITY_FRAMES
#         )

#     def _has_any_recent_outward_motion(self, track, look_back=3):
#         if not track.motion_direction_history:
#             return False
#         recent = list(track.motion_direction_history)[-look_back:]
#         return any(d is True for d in recent)

#     def _has_any_recent_inward_motion(self, track, look_back=3):
#         if not track.motion_direction_history:
#             return False
#         recent = list(track.motion_direction_history)[-look_back:]
#         return any(d is False for d in recent)

#     def _has_sustained_outward_motion(self, track, min_consecutive=2):
#         if len(track.motion_direction_history) < min_consecutive:
#             return False
#         return all(d is True for d in list(track.motion_direction_history)[-min_consecutive:])

#     def _has_sustained_inward_motion(self, track, min_consecutive=2):
#         if len(track.motion_direction_history) < min_consecutive:
#             return False
#         return all(d is False for d in list(track.motion_direction_history)[-min_consecutive:])

#     def _is_inward_or_return_motion(self, track, update, prev_safe, prev_outer, prev_stable):
#         """
#         Conservative inward/return detector.
#         Blocks only strong return/inward evidence.
#         """
#         if update is None:
#             return False

#         current_outward = bool(update.outward_motion)
#         current_inward = bool(update.inward_motion)

#         sustained_outward = self._has_sustained_outward_motion(track, min_consecutive=2)
#         sustained_inward = self._has_sustained_inward_motion(track, min_consecutive=2)

#         if current_outward or sustained_outward:
#             return False

#         reentered_roi = self._roi_reentry_occurred(update, prev_safe, prev_outer)

#         if current_inward:
#             return True

#         if sustained_inward:
#             return True

#         if reentered_roi and (current_inward or sustained_inward):
#             return True

#         return False

#     # ─────────────────────────────────────────────────────────────────────────
#     # ROI helpers
#     # ─────────────────────────────────────────────────────────────────────────

#     @staticmethod
#     def _roi_exit_occurred(track, update, prev_stable):
#         return (
#             track.was_stable
#             and prev_stable is True
#             and not update.in_stable_roi
#         )

#     @staticmethod
#     def _roi_reentry_occurred(update, prev_safe, prev_outer):
#         was_fully_absent = (prev_safe is False) and (prev_outer is False)
#         now_in_roi = update.in_safe_roi or update.in_outer_roi
#         return was_fully_absent and now_in_roi

#     # ─────────────────────────────────────────────────────────────────────────
#     # Ledger helpers
#     # ─────────────────────────────────────────────────────────────────────────

#     def _has_outstanding_pickup(self, class_name):
#         return (self.pickup_count[class_name] - self.putback_count[class_name]) > 0

#     def _latest_pickup_timestamp(self, class_name):
#         history = self.class_pickup_timestamps[class_name]
#         return history[-1] if history else None

#     def _active_pending_pickups(self, timestamp_ms):
#         active = []

#         for item in self.pending_pickups:
#             if item.get("closed"):
#                 continue

#             age = timestamp_ms - float(item["pickup_time_ms"])
#             if age <= PENDING_LEDGER_MAX_AGE_MS:
#                 active.append(item)

#         return active

#     def _class_similarity_score(self, a, b):
#         if not a or not b:
#             return LEDGER_DIFFERENT_CLASS_SCORE

#         if a == b:
#             return LEDGER_SAME_CLASS_SCORE

#         return LEDGER_DIFFERENT_CLASS_SCORE

#     def _ledger_class_candidate_score(self, track, pending_item):
#         current_vote = self._compute_weighted_class_vote(
#             track,
#             stable_only=False,
#             include_safe=True,
#         )
#         current_scores = current_vote.get("scores", {}) or {}
#         pickup_scores = pending_item.get("class_candidates", {}) or {}

#         if not current_scores or not pickup_scores:
#             return 0.0

#         current_total = sum(current_scores.values()) or 1.0
#         pickup_total = sum(pickup_scores.values()) or 1.0

#         overlap = set(current_scores.keys()) & set(pickup_scores.keys())
#         if not overlap:
#             return 0.0

#         score = 0.0
#         for cls in overlap:
#             score += min(
#                 current_scores.get(cls, 0.0) / current_total,
#                 pickup_scores.get(cls, 0.0) / pickup_total,
#             )

#         return min(1.0, score)

#     def _time_match_score(self, pickup_time_ms, return_time_ms):
#         gap = max(0.0, return_time_ms - pickup_time_ms)

#         if gap >= PENDING_LEDGER_MAX_AGE_MS:
#             return 0.0

#         return 1.0 - (gap / PENDING_LEDGER_MAX_AGE_MS)

#     def _track_match_score(self, track, pending_item):
#         return 1.0 if pending_item.get("track_id") == track.global_id else 0.0

#     def _motion_match_score_for_putback(self, track):
#         if self._has_any_recent_inward_motion(track, look_back=5):
#             return 1.0
#         return 0.3

#     def _track_embedding(self, track):
#         if getattr(track, "visual_embedding", None) is not None:
#             return np.asarray(track.visual_embedding, dtype=np.float32)

#         if hasattr(track, "mean_embedding"):
#             emb = track.mean_embedding()
#             if emb is not None:
#                 return np.asarray(emb, dtype=np.float32)

#         return None

#     def _embedding_similarity_score(self, track, pending_item, expected_class=None):
#         """
#         Compute normalized cosine similarity between current track embedding
#         and the embedding stored in the pending pickup item.

#         Args:
#             track: GlobalTrack with current embedding.
#             pending_item: Pending pickup ledger entry.
#             expected_class: The class of the pending pickup (for threshold lookup).

#         Returns:
#             Similarity score (0.0 to 1.0), or 0.0 if below class‑specific threshold.
#         """
#         emb_a = self._track_embedding(track)
#         emb_b = pending_item.get("visual_embedding")

#         if emb_a is None or emb_b is None:
#             return 0.0

#         emb_b = np.asarray(emb_b, dtype=np.float32)

#         denom = float(np.linalg.norm(emb_a) * np.linalg.norm(emb_b))
#         if denom <= 1e-6:
#             return 0.0

#         cosine = float(np.dot(emb_a, emb_b) / denom)
#         cosine = max(-1.0, min(1.0, cosine))
#         normalized = (cosine + 1.0) / 2.0

#         # ---- PER-CLASS THRESHOLD LOOKUP ----
#         if expected_class and expected_class in EMBEDDING_SIM_THRESHOLDS_BY_CLASS:
#             min_sim = EMBEDDING_SIM_THRESHOLDS_BY_CLASS[expected_class]
#         else:
#             min_sim = EMBEDDING_SIM_THRESHOLDS_BY_CLASS.get("default", 0.65)
#         # -------------------------------------

#         if normalized < min_sim:
#             return 0.0

#         return normalized

#     def _spatial_match_score(self, track, pending_item):
#         update = track.current_update
#         pickup_centroid = pending_item.get("pickup_centroid")

#         if update is None or pickup_centroid is None:
#             return 0.0

#         try:
#             return_centroid = np.asarray(update.centroid, dtype=np.float32)
#             pickup_centroid_np = np.asarray(pickup_centroid, dtype=np.float32)
#         except Exception:
#             return 0.0

#         dist = float(np.linalg.norm(return_centroid - pickup_centroid_np))

#         if dist >= LEDGER_SPATIAL_MAX_DISTANCE_PX:
#             return 0.0

#         return 1.0 - (dist / LEDGER_SPATIAL_MAX_DISTANCE_PX)

#     def _match_putback_to_pending_pickup(self, track, timestamp_ms):
#         self._ensure_identity_fields(track)

#         candidates = self._active_pending_pickups(timestamp_ms)
#         if not candidates:
#             return None

#         detected_return_class = self._detected_class(track)
#         resolved_return_class = self._event_class(track, purpose="putback")

#         best_item = None
#         best_score = -1.0
#         best_breakdown = {}

#         for item in candidates:
#             picked_class = item["event_class"]

#             class_score = max(
#                 self._class_similarity_score(detected_return_class, picked_class),
#                 self._class_similarity_score(resolved_return_class, picked_class),
#             )
#             time_score = self._time_match_score(item["pickup_time_ms"], timestamp_ms)
#             track_score = self._track_match_score(track, item)
#             candidate_score = self._ledger_class_candidate_score(track, item)
#             motion_score = self._motion_match_score_for_putback(track)
#             embedding_score = self._embedding_similarity_score(track, item, expected_class=picked_class)  # <-- updated
#             spatial_score = self._spatial_match_score(track, item)

#             total_score = (
#                 LEDGER_MATCH_WEIGHT_CLASS * class_score
#                 + LEDGER_MATCH_WEIGHT_TIME * time_score
#                 + LEDGER_MATCH_WEIGHT_TRACK * track_score
#                 + LEDGER_MATCH_WEIGHT_CLASS_CANDIDATE * candidate_score
#                 + LEDGER_MATCH_WEIGHT_MOTION * motion_score
#                 + LEDGER_MATCH_WEIGHT_EMBEDDING * embedding_score
#                 + LEDGER_MATCH_WEIGHT_SPATIAL * spatial_score
#             )

#             breakdown = {
#                 "total": total_score,
#                 "picked_class": picked_class,
#                 "detected_return_class": detected_return_class,
#                 "resolved_return_class": resolved_return_class,
#                 "class_score": class_score,
#                 "time_score": time_score,
#                 "track_score": track_score,
#                 "class_candidate_score": candidate_score,
#                 "motion_score": motion_score,
#                 "embedding_score": embedding_score,
#                 "spatial_score": spatial_score,
#             }

#             if total_score > best_score:
#                 best_score = total_score
#                 best_item = item
#                 best_breakdown = breakdown

#         if best_item is not None and best_score >= PUTBACK_LEDGER_MATCH_THRESHOLD:
#             best_item["_last_match_score"] = best_score
#             best_item["_last_match_breakdown"] = best_breakdown
#             return best_item

#         self._emit_debug(
#             f"[LEDGER-NO-MATCH] track={track.global_id} "
#             f"detected={detected_return_class} best_score={best_score:.2f} "
#             f"threshold={PUTBACK_LEDGER_MATCH_THRESHOLD} breakdown={best_breakdown}",
#             level="info",
#         )
#         return None

#     def _is_class_return_candidate(self, track):
#         """
#         Side-effect-safe return candidate checker.

#         Tracks that were already stable should not become class-level return
#         candidates. That prevents stale resolved_class contamination.
#         """
#         self._ensure_identity_fields(track)

#         update = track.current_update
#         if update is None:
#             return False

#         if track.was_stable:
#             return False

#         has_inward = update.inward_motion or self._has_any_recent_inward_motion(track, look_back=5)
#         if not has_inward:
#             return False

#         candidate_class = (
#             track.locked_class_name
#             if self._is_valid_class(track.locked_class_name)
#             else self._voted_event_class(track)
#         )

#         exact_class_return = self._has_outstanding_pickup(candidate_class)

#         latest_pickup_ts = None
#         ledger_match = None

#         if exact_class_return:
#             latest_pickup_ts = self._latest_pickup_timestamp(candidate_class)
#         else:
#             ledger_match = self._match_putback_to_pending_pickup(track, update.timestamp_ms)
#             if ledger_match is not None:
#                 latest_pickup_ts = ledger_match["pickup_time_ms"]

#         if latest_pickup_ts is None:
#             return False

#         within_gap = (update.timestamp_ms - latest_pickup_ts) <= CLASS_RETURN_MAX_GAP_MS
#         born_after_pickup = track.created_ms >= latest_pickup_ts
#         not_previously_stable = not track.was_stable

#         is_candidate = (
#             within_gap
#             and born_after_pickup
#             and not_previously_stable
#             and has_inward
#         )

#         if is_candidate and ledger_match is not None:
#             track.resolved_class_name = ledger_match["event_class"]

#         return is_candidate

#     # ─────────────────────────────────────────────────────────────────────────
#     # Neighbor stability boost
#     # ─────────────────────────────────────────────────────────────────────────

#     def _check_neighbor_stability_boost(self, track, all_tracks):
#         update = track.current_update
#         if update is None:
#             return

#         for neighbor in all_tracks:
#             self._ensure_identity_fields(neighbor)

#             if neighbor.global_id == track.global_id:
#                 continue

#             if neighbor.event_state not in {"INSIDE", "STABLE_INSIDE"}:
#                 continue

#             if neighbor.stability_boosted:
#                 continue

#             neighbor_update = neighbor.current_update
#             if neighbor_update is None:
#                 continue

#             same_zone = (
#                 (update.in_stable_roi and neighbor_update.in_stable_roi)
#                 or (update.in_outer_roi and neighbor_update.in_outer_roi)
#             )

#             half_stability = neighbor.inside_counter >= max(1, STABILITY_FRAMES // 2)

#             if same_zone and half_stability:
#                 required = self._required_stability_frames(neighbor, update.timestamp_ms)
#                 neighbor.inside_counter = required
#                 neighbor.event_state = "STABLE_INSIDE"
#                 neighbor.stability_boosted = True
#                 neighbor.was_stable = True

#                 self._lock_class_if_possible(neighbor, reason="neighbor stability boost")

#                 self._emit_debug(
#                     f"[BOOST] neighbor track={neighbor.global_id} -> STABLE_INSIDE "
#                     f"by track={track.global_id}",
#                     level="debug",
#                 )

#     # ─────────────────────────────────────────────────────────────────────────
#     # Event recording
#     # ─────────────────────────────────────────────────────────────────────────

#     def _pickup_ledger_item_from_event(self, track, event):
#         update = track.current_update

#         vote_result = self._compute_weighted_class_vote(
#             track,
#             stable_only=False,
#             include_safe=True,
#         )

#         pickup_centroid = None
#         pickup_bbox = None

#         if update is not None:
#             try:
#                 pickup_centroid = np.asarray(update.centroid, dtype=np.float32).tolist()
#             except Exception:
#                 pickup_centroid = None

#             pickup_bbox = tuple(update.bbox) if getattr(update, "bbox", None) is not None else None

#         return {
#             "ledger_id": str(uuid.uuid4()),
#             "pickup_event_id": event["event_id"],
#             "event_class": event["class"],
#             "locked_class": event.get("locked_class"),
#             "detected_class_at_pickup": event.get("detected_class"),
#             "track_id": track.global_id,
#             "pickup_time_ms": event["timestamp_ms"],
#             "camera_id": event["camera_id"],
#             "frame_index": event["frame_index"],
#             "class_candidates": vote_result.get("scores", {}),
#             "visual_embedding": self._track_embedding(track),
#             "pickup_centroid": pickup_centroid,
#             "pickup_bbox": pickup_bbox,
#             "closed": False,
#             "closed_by_event_id": None,
#         }

#     def _record_pickup(self, track):
#         self._ensure_identity_fields(track)

#         update = track.current_update

#         event_class = self._event_class(track, purpose="pickup")

#         if event_class == UNKNOWN_UNSTABLE_CLASS:
#             self._lock_class_if_possible(track, reason="pickup final lock attempt")
#             event_class = self._event_class(track, purpose="pickup")

#         detected_class = self._detected_class(track)

#         if (
#             self._is_valid_class(track.locked_class_name)
#             and event_class != track.locked_class_name
#         ):
#             self._emit_debug(
#                 f"[PICKUP-CLASS-CORRECTED] track={track.global_id} "
#                 f"event_class={event_class} -> locked_class={track.locked_class_name} "
#                 f"detected={detected_class}",
#                 level="warning",
#             )
#             event_class = track.locked_class_name
#             track.resolved_class_name = event_class

#         self.pickup_count[event_class] += 1

#         event_timestamp = track.last_seen_ms if update is None else update.timestamp_ms
#         event_camera_id = track.last_camera_id if update is None else update.camera_id
#         event_frame_index = track.last_frame_index if update is None else update.frame_index

#         self.class_pickup_timestamps[event_class].append(event_timestamp)
#         track.pickup_confirmed_ms = event_timestamp
#         track.resolved_class_name = event_class

#         class_drift = detected_class != event_class

#         event = {
#             "event_id": str(uuid.uuid4()),
#             "type": "pickup",
#             "class": event_class,
#             "event_class": event_class,
#             "detected_class": detected_class,
#             "locked_class": track.locked_class_name,
#             "resolved_class": event_class,
#             "class_drift": bool(class_drift or track.class_drift_flagged),
#             "global_id": track.global_id,
#             "camera_id": event_camera_id,
#             "frame_index": event_frame_index,
#             "timestamp_ms": event_timestamp,
#             "confidence": float(
#                 sum(track.confirmation_confidences)
#                 / max(len(track.confirmation_confidences), 1)
#             ),
#             "confidence_window": list(track.confirmation_confidences),
#             "confirmation_class_votes": list(track.confirmation_class_votes),
#         }

#         if ENABLE_CLASS_DRIFT_WARNINGS and event["class_drift"]:
#             event["warnings"] = [
#                 f"class drift observed on pickup: detected={detected_class}, resolved={event_class}"
#             ]

#         self.session.events.append(event)

#         if hasattr(self.session, "pickup_records"):
#             self.session.pickup_records.append(event)

#         ledger_item = self._pickup_ledger_item_from_event(track, event)
#         self.pending_pickups.append(ledger_item)

#         overlay = f"PICKUP {event_class} G{track.global_id}"
#         if event["class_drift"]:
#             overlay += f" [detected:{detected_class}]"

#         self._push_overlay_event(event_timestamp, overlay)

#         self._emit_debug(
#             f"[EVENT] {overlay} | cam={event_camera_id} frame={event_frame_index} "
#             f"conf={list(track.confirmation_confidences)} "
#             f"ledger_id={ledger_item['ledger_id']}",
#             level="info",
#         )

#         return event

#     def _record_putback(self, track):
#         self._ensure_identity_fields(track)

#         update = track.current_update

#         event_timestamp = track.last_seen_ms if update is None else update.timestamp_ms
#         event_camera_id = track.last_camera_id if update is None else update.camera_id
#         event_frame_index = track.last_frame_index if update is None else update.frame_index

#         detected_class = self._detected_class(track)

#         matched_item = self._match_putback_to_pending_pickup(track, event_timestamp)

#         if matched_item is not None:
#             event_class = matched_item["event_class"]
#             track.resolved_class_name = event_class
#             ledger_match_score = matched_item.get("_last_match_score")
#             ledger_match_breakdown = matched_item.get("_last_match_breakdown")
#         else:
#             event_class = self._event_class(track, purpose="putback")
#             ledger_match_score = None
#             ledger_match_breakdown = None

#         current_net = (
#             self._frame_net_snapshot.get(event_class, 0)
#             - self._frame_putback_used.get(event_class, 0)
#         )

#         if current_net <= 0:
#             warning = (
#                 f"Putback ignored class={event_class} detected={detected_class} "
#                 f"track={track.global_id} — no outstanding pickup to offset."
#             )
#             self.session.warnings.append(warning)
#             LOGGER.warning(warning)
#             return None

#         self.putback_count[event_class] += 1
#         self._frame_putback_used[event_class] += 1

#         track.last_putback_confirmed_ms = event_timestamp
#         track.resolved_class_name = event_class
#         self.last_putback_by_class_ms[event_class] = event_timestamp

#         if matched_item is not None:
#             matched_item["closed"] = True

#         is_long_hold = (
#             track.pickup_confirmed_ms is not None
#             and (event_timestamp - track.pickup_confirmed_ms) > CLASS_RETURN_MAX_GAP_MS
#         )

#         class_drift = detected_class != event_class

#         event = {
#             "event_id": str(uuid.uuid4()),
#             "type": "putback",
#             "class": event_class,
#             "event_class": event_class,
#             "detected_class": detected_class,
#             "locked_class": track.locked_class_name,
#             "resolved_class": event_class,
#             "class_drift": bool(class_drift or track.class_drift_flagged),
#             "global_id": track.global_id,
#             "camera_id": event_camera_id,
#             "frame_index": event_frame_index,
#             "timestamp_ms": event_timestamp,
#             "confidence": float(
#                 sum(track.confirmation_confidences)
#                 / max(len(track.confirmation_confidences), 1)
#             ),
#             "confidence_window": list(track.confirmation_confidences),
#             "confirmation_class_votes": list(track.confirmation_class_votes),
#         }

#         if matched_item is not None:
#             event["matched_pickup_ledger_id"] = matched_item["ledger_id"]
#             event["matched_pickup_event_id"] = matched_item["pickup_event_id"]
#             event["ledger_match_score"] = ledger_match_score
#             event["ledger_match_breakdown"] = ledger_match_breakdown
#             matched_item["closed_by_event_id"] = event["event_id"]

#         if is_long_hold:
#             event["long_hold_return"] = True

#         if ENABLE_CLASS_DRIFT_WARNINGS and event["class_drift"]:
#             event["warnings"] = [
#                 f"class drift observed on putback: detected={detected_class}, resolved={event_class}"
#             ]

#         self.session.events.append(event)

#         if hasattr(self.session, "putback_records"):
#             self.session.putback_records.append(event)

#         overlay = f"PUTBACK {event_class} G{track.global_id}"
#         if event["class_drift"]:
#             overlay += f" [detected:{detected_class}]"
#         if is_long_hold:
#             overlay += " [long-hold]"

#         self._push_overlay_event(event_timestamp, overlay)

#         self._emit_debug(
#             f"[EVENT] {overlay} | cam={event_camera_id} frame={event_frame_index} "
#             f"conf={list(track.confirmation_confidences)} "
#             f"matched_ledger={matched_item['ledger_id'] if matched_item else None}",
#             level="info",
#         )

#         return event

#     # ─────────────────────────────────────────────────────────────────────────
#     # Overlay
#     # ─────────────────────────────────────────────────────────────────────────

#     def overlay_event_lines(self, timestamp_ms):
#         while (
#             self.recent_overlay_events
#             and (timestamp_ms - self.recent_overlay_events[0][0]) > OVERLAY_EVENT_TTL_MS
#         ):
#             self.recent_overlay_events.popleft()

#         return [
#             item[1]
#             for item in list(self.recent_overlay_events)[-OVERLAY_MAX_EVENT_LINES:]
#         ]

#     # ─────────────────────────────────────────────────────────────────────────
#     # Core FSM
#     # ─────────────────────────────────────────────────────────────────────────

#     def _process_track(self, track, timestamp_ms, all_tracks=None):
#         self._ensure_identity_fields(track)

#         update = track.current_update

#         # =====================================================================
#         # BRANCH A: no detection this frame
#         # =====================================================================
#         if update is None:
#             if track.event_state == "PICKUP_PENDING":
#                 if self._pending_timeout_elapsed(track, timestamp_ms):
#                     if self._has_any_recent_outward_motion(track, look_back=5):
#                         self._transition(
#                             track,
#                             "PICKED_UP",
#                             "pickup confirmed: lost outward motion",
#                         )
#                         return [self._record_pickup(track)]

#                     if track.was_stable:
#                         self._transition(
#                             track,
#                             "PICKED_UP",
#                             "pickup confirmed: lost was stable occlusion/vertical",
#                         )
#                         event = self._record_pickup(track)
#                         event["occlusion_pickup"] = True
#                         self._replace_last_overlay(
#                             f"PICKUP {event['class']} G{track.global_id} [occluded]"
#                         )
#                         return [event]

#                     self._emit_debug(
#                         f"[SILENT-CANCEL] track={track.global_id} "
#                         "never reached stability — suppressed",
#                         level="debug",
#                     )
#                     self._transition(
#                         track,
#                         "INSIDE",
#                         "pickup cancelled: never reached stability",
#                     )
#                     return []

#             if track.event_state == "PUTBACK_PENDING":
#                 if self._pending_timeout_elapsed(track, timestamp_ms):
#                     if (
#                         track.last_seen_in_stable_roi is True
#                         and self._has_any_recent_inward_motion(track, look_back=5)
#                     ):
#                         self._transition(
#                             track,
#                             "STABLE_INSIDE",
#                             "putback confirmed: lost in stable ROI",
#                         )
#                         event = self._record_putback(track)
#                         return [] if event is None else [event]

#                     self._emit_debug(
#                         f"[TIMEOUT] putback absent track={track.global_id} "
#                         f"last_stable={track.last_seen_in_stable_roi}",
#                         level="info",
#                     )
#                     self._transition(
#                         track,
#                         "PICKED_UP",
#                         "putback cancelled: lost without entering stable ROI",
#                     )

#             return []

#         # =====================================================================
#         # BRANCH B: detection exists
#         # =====================================================================

#         self._append_class_vote(track, update)

#         prev_safe = track.last_seen_in_safe_roi
#         prev_outer = track.last_seen_in_outer_roi
#         prev_stable = track.last_seen_in_stable_roi

#         track.last_seen_in_safe_roi = update.in_safe_roi
#         track.last_seen_in_outer_roi = update.in_outer_roi
#         track.last_seen_in_stable_roi = update.in_stable_roi

#         if update.displacement_magnitude > 0:
#             if update.outward_motion:
#                 motion_direction = True
#             elif update.inward_motion:
#                 motion_direction = False
#             else:
#                 motion_direction = None
#         else:
#             motion_direction = None

#         track.motion_direction_history.append(motion_direction)

#         if update.in_outer_roi and not update.in_stable_roi:
#             if track.boundary_linger_start_ms is None:
#                 track.boundary_linger_start_ms = update.timestamp_ms
#             elif (update.timestamp_ms - track.boundary_linger_start_ms) > 1000.0:
#                 track.boundary_linger_flagged = True
#         else:
#             track.boundary_linger_start_ms = None

#         required_stability = self._required_stability_frames(track, update.timestamp_ms)

#         # =====================================================================
#         # STATE: INSIDE
#         # =====================================================================
#         if track.event_state == "INSIDE":
#             if self._is_class_return_candidate(track):
#                 self._transition(
#                     track,
#                     "PUTBACK_PENDING",
#                     "class/ledger inward re-entry candidate",
#                 )
#                 track.pending_since_ms = update.timestamp_ms
#                 self._seed_class_return_confirmation(track)

#                 self._emit_debug(
#                     f"[PENDING] putback track={track.global_id} "
#                     f"event_class={self._event_class(track, purpose='putback')} "
#                     f"detected={self._detected_class(track)}",
#                     level="info",
#                 )
#                 return []

#             if update.in_stable_roi and update.confidence >= CONF_THRESHOLD:
#                 track.inside_counter += 1

#                 if track.inside_counter >= required_stability:
#                     self._transition(track, "STABLE_INSIDE", "stable in shelf ROI")
#                     track.boundary_linger_flagged = False

#                     if all_tracks is not None:
#                         self._check_neighbor_stability_boost(track, all_tracks)
#             else:
#                 track.inside_counter = 0

#             return []

#         # =====================================================================
#         # STATE: STABLE_INSIDE
#         # =====================================================================
#         if track.event_state == "STABLE_INSIDE":
#             if track.locked_class_name == UNKNOWN_UNSTABLE_CLASS:
#                 self._lock_class_if_possible(track, reason="additional stable evidence")

#             if self._is_class_return_candidate(track):
#                 self._transition(
#                     track,
#                     "PUTBACK_PENDING",
#                     "class/ledger inward re-entry candidate",
#                 )
#                 track.pending_since_ms = update.timestamp_ms
#                 self._seed_class_return_confirmation(track)

#                 self._emit_debug(
#                     f"[PENDING] putback track={track.global_id} "
#                     f"event_class={self._event_class(track, purpose='putback')} "
#                     f"detected={self._detected_class(track)}",
#                     level="info",
#                 )
#                 return []

#             # ── PICKUP TRIGGER WITH CONSERVATIVE INWARD BLOCK ─────────────────
#             if not update.in_safe_roi:
#                 has_outward = self._has_any_recent_outward_motion(track, look_back=3)
#                 has_sustained_outward = self._has_sustained_outward_motion(
#                     track,
#                     min_consecutive=2,
#                 )
#                 has_roi_exit = self._roi_exit_occurred(track, update, prev_stable)

#                 is_inward_or_return = self._is_inward_or_return_motion(
#                     track,
#                     update,
#                     prev_safe,
#                     prev_outer,
#                     prev_stable,
#                 )

#                 # Only block if clearly inward/return AND no pickup evidence.
#                 if is_inward_or_return and not has_outward and not has_roi_exit:
#                     self._emit_debug(
#                         f"[PICKUP-BLOCKED-INWARD] track={track.global_id} "
#                         f"event_class={self._event_class(track, purpose='pickup')} "
#                         f"detected={self._detected_class(track)} "
#                         f"prev_safe={prev_safe} curr_safe={update.in_safe_roi} "
#                         f"prev_outer={prev_outer} curr_outer={update.in_outer_roi} "
#                         f"prev_stable={prev_stable} curr_stable={update.in_stable_roi} "
#                         f"outward={update.outward_motion} inward={update.inward_motion} "
#                         f"motion_history={list(track.motion_direction_history)}",
#                         level="info",
#                     )
#                     return []

#                 pickup_like_exit = (
#                     has_sustained_outward
#                     or has_outward
#                     or has_roi_exit
#                 )

#                 if pickup_like_exit:
#                     if (
#                         track.last_putback_confirmed_ms is not None
#                         and (update.timestamp_ms - track.last_putback_confirmed_ms)
#                         < POST_PUTBACK_COOLDOWN_MS
#                     ):
#                         self._emit_debug(
#                             f"[COOLDOWN] pickup suppressed track={track.global_id} "
#                             f"elapsed={update.timestamp_ms - track.last_putback_confirmed_ms:.0f}ms "
#                             f"cooldown={POST_PUTBACK_COOLDOWN_MS}ms",
#                             level="debug",
#                         )
#                         return []

#                     self._lock_class_if_possible(track, reason="pickup trigger")

#                     pickup_class = self._event_class(track, purpose="pickup")
#                     last_class_putback_ms = self.last_putback_by_class_ms.get(pickup_class)

#                     if (
#                         last_class_putback_ms is not None
#                         and (update.timestamp_ms - last_class_putback_ms)
#                         < POST_PUTBACK_COOLDOWN_MS
#                     ):
#                         self._emit_debug(
#                             f"[CLASS-COOLDOWN] pickup suppressed track={track.global_id} "
#                             f"class={pickup_class} "
#                             f"elapsed={update.timestamp_ms - last_class_putback_ms:.0f}ms "
#                             f"cooldown={POST_PUTBACK_COOLDOWN_MS}ms",
#                             level="debug",
#                         )
#                         return []

#                     if (
#                         self._is_valid_class(track.locked_class_name)
#                         and self._is_valid_class(track.resolved_class_name)
#                         and track.resolved_class_name != track.locked_class_name
#                     ):
#                         self._emit_debug(
#                             f"[CLEAR-STALE-RESOLVED-BEFORE-PICKUP] track={track.global_id} "
#                             f"resolved={track.resolved_class_name} "
#                             f"locked={track.locked_class_name}",
#                             level="warning",
#                         )
#                         track.resolved_class_name = track.locked_class_name

#                     trigger_reason = (
#                         "outward exit + motion"
#                         if has_outward or has_sustained_outward
#                         else "ROI exit without motion vertical/edge/fast/low-tray pick"
#                     )

#                     self._transition(track, "PICKUP_PENDING", trigger_reason)
#                     track.pending_since_ms = update.timestamp_ms
#                     self._seed_confirmation_from_history(track)

#                     if all_tracks is not None:
#                         self._check_neighbor_stability_boost(track, all_tracks)

#                     self._emit_debug(
#                         f"[PENDING] pickup track={track.global_id} "
#                         f"event_class={self._event_class(track, purpose='pickup')} "
#                         f"detected={self._detected_class(track)} "
#                         f"outward={has_outward} sustained_outward={has_sustained_outward} "
#                         f"roi_exit={has_roi_exit} inward_or_return={is_inward_or_return} "
#                         f"prev_safe={prev_safe} curr_safe={update.in_safe_roi} "
#                         f"prev_stable={prev_stable} curr_stable={update.in_stable_roi} "
#                         f"safe_dist={update.safe_roi_distance:.1f} "
#                         f"motion_history={list(track.motion_direction_history)}",
#                         level="info",
#                     )
#                     return []

#             if update.in_stable_roi and update.confidence >= CONF_THRESHOLD:
#                 track.inside_counter = required_stability
#             else:
#                 track.inside_counter = 0

#             return []

#         # =====================================================================
#         # STATE: PICKUP_PENDING
#         # =====================================================================
#         if track.event_state == "PICKUP_PENDING":

#             # Quick return must be checked before inward-cancel.
#             if update.in_stable_roi and not update.outward_motion:
#                 time_since_pending_ms = update.timestamp_ms - (
#                     track.pending_since_ms or update.timestamp_ms
#                 )

#                 if time_since_pending_ms < QUICK_RETURN_THRESHOLD_MS:
#                     self._transition(
#                         track,
#                         "PICKED_UP",
#                         "quick pickup: fast re-entry within threshold",
#                     )

#                     pickup_event = self._record_pickup(track)

#                     self._frame_net_snapshot[pickup_event["class"]] = (
#                         self._frame_net_snapshot.get(pickup_event["class"], 0) + 1
#                     )

#                     track.resolved_class_name = pickup_event["class"]

#                     self._transition(
#                         track,
#                         "PUTBACK_PENDING",
#                         "quick return: fast re-entry",
#                     )
#                     track.pending_since_ms = update.timestamp_ms
#                     self._seed_confirmation_from_history(track)

#                     self._emit_debug(
#                         f"[QUICK-RETURN] track={track.global_id} "
#                         f"time_ms={time_since_pending_ms:.0f} "
#                         f"event_class={pickup_event['class']} "
#                         f"detected={self._detected_class(track)}",
#                         level="info",
#                     )

#                     return [pickup_event]

#                 self._emit_debug(
#                     f"[CANCEL] pickup track={track.global_id} "
#                     "re-entered stable ROI after threshold",
#                     level="info",
#                 )
#                 self._transition(
#                     track,
#                     "STABLE_INSIDE",
#                     "pickup cancelled: returned to stable ROI",
#                 )
#                 return []

#             # Cancel false pending pickup only after quick-return check.
#             if (
#                 self._has_sustained_inward_motion(track, min_consecutive=2)
#                 and (update.in_safe_roi or update.in_stable_roi)
#             ):
#                 self._emit_debug(
#                     f"[CANCEL] pickup track={track.global_id} "
#                     f"cancelled by sustained inward return before confirmation "
#                     f"event_class={self._event_class(track, purpose='pickup')} "
#                     f"detected={self._detected_class(track)}",
#                     level="info",
#                 )
#                 self._transition(
#                     track,
#                     "STABLE_INSIDE",
#                     "pickup cancelled: inward return before confirmation",
#                 )
#                 return []

#             if not update.in_outer_roi:
#                 self._append_confirmation(track, update.confidence)

#                 # Do not require class consistency for fast / occlusion pickup.
#                 # Event class is protected by locked class.
#                 confirmation_ok = self._confirmation_ok(
#                     track,
#                     require_class_consistency=False,
#                 )

#                 timeout_absent_ok = (
#                     track.was_stable
#                     and (update.timestamp_ms - (track.pending_since_ms or update.timestamp_ms)) > 100
#                     and not self._has_any_recent_inward_motion(track, look_back=5)
#                 )

#                 if confirmation_ok or timeout_absent_ok:
#                     self._transition(
#                         track,
#                         "PICKED_UP",
#                         "pickup confirmed: SAFE -> OUTER -> ABSENT",
#                     )
#                     return [self._record_pickup(track)]

#                 return []

#             if update.in_outer_roi:
#                 self._append_confirmation(track, update.confidence)

#                 if len(track.confirmation_confidences) >= CONFIRM_FRAMES:
#                     conf_ok = self._confidence_ok(track)
#                     class_ok = self._class_consistency_ok(track)

#                     if not conf_ok:
#                         self._emit_debug(
#                             f"[CANCEL] pickup track={track.global_id} "
#                             f"confidence too low in OUTER conf={list(track.confirmation_confidences)}",
#                             level="info",
#                         )
#                         self._transition(
#                             track,
#                             "STABLE_INSIDE",
#                             "pickup cancelled: confidence too low",
#                         )
#                         return []

#                     if not class_ok:
#                         if track.was_stable and self._has_any_recent_outward_motion(track, look_back=5):
#                             self._emit_debug(
#                                 f"[CLASS-DRIFT-TOLERATED] pickup track={track.global_id} "
#                                 f"keeping event_class={self._event_class(track, purpose='pickup')} "
#                                 f"detected={self._detected_class(track)}",
#                                 level="info",
#                             )
#                             return []

#                         self._emit_debug(
#                             f"[CANCEL] pickup track={track.global_id} "
#                             "class consistency too low in OUTER without strong pickup evidence",
#                             level="info",
#                         )
#                         self._transition(
#                             track,
#                             "STABLE_INSIDE",
#                             "pickup cancelled: class consistency too low",
#                         )
#                         return []

#                 if self._has_any_recent_outward_motion(track, look_back=3):
#                     return []

#                 if self._pending_timeout_elapsed(track, update.timestamp_ms):
#                     self._emit_debug(
#                         f"[TIMEOUT] pickup OUTER track={track.global_id} timeout elapsed",
#                         level="info",
#                     )
#                     self._transition(
#                         track,
#                         "STABLE_INSIDE",
#                         "pickup cancelled: timeout in OUTER ROI",
#                     )
#                     return []

#                 return []

#             return []

#         # =====================================================================
#         # STATE: PICKED_UP
#         # =====================================================================
#         if track.event_state == "PICKED_UP":
#             track_pickup_ts = track.pickup_confirmed_ms

#             time_since_pickup_ms = (
#                 update.timestamp_ms - track_pickup_ts
#                 if track_pickup_ts is not None
#                 else 0.0
#             )

#             is_long_hold = (
#                 track_pickup_ts is not None
#                 and time_since_pickup_ms > CLASS_RETURN_MAX_GAP_MS
#             )

#             # Long-hold fallback remains.
#             if is_long_hold and (update.in_stable_roi or update.in_outer_roi):
#                 matched = self._match_putback_to_pending_pickup(track, update.timestamp_ms)
#                 if matched is not None:
#                     track.resolved_class_name = matched["event_class"]

#                 track.long_hold_pending = True
#                 self._transition(
#                     track,
#                     "PUTBACK_PENDING",
#                     "long-hold fallback: elapsed > CLASS_RETURN_MAX_GAP_MS",
#                 )
#                 track.pending_since_ms = update.timestamp_ms
#                 self._seed_confirmation_from_history(track)

#                 self._emit_debug(
#                     f"[LONG-HOLD] track={track.global_id} "
#                     f"event_class={self._event_class(track, purpose='putback')} "
#                     f"detected={self._detected_class(track)} "
#                     f"elapsed_ms={time_since_pickup_ms:.0f}",
#                     level="info",
#                 )
#                 return []

#             # Minimum hold prevents fake 1-frame putback after pickup.
#             if time_since_pickup_ms < self.min_valid_putback_hold_ms:
#                 self._emit_debug(
#                     f"[PUTBACK-HOLD] suppressed early putback track={track.global_id} "
#                     f"elapsed={time_since_pickup_ms:.0f}ms "
#                     f"min_hold={self.min_valid_putback_hold_ms:.0f}ms",
#                     level="debug",
#                 )
#                 return []

#             has_inward = self._has_any_recent_inward_motion(track, look_back=5)
#             has_reentry = self._roi_reentry_occurred(update, prev_safe, prev_outer)

#             # Real putback should be either inward/reentry or stable presence after hold.
#             putback_like = (
#                 (
#                     has_inward
#                     and (update.in_stable_roi or update.in_outer_roi or update.in_safe_roi)
#                 )
#                 or has_reentry
#                 or (
#                     update.in_stable_roi
#                     and time_since_pickup_ms >= self.min_valid_putback_hold_ms
#                 )
#             )

#             if putback_like:
#                 matched = self._match_putback_to_pending_pickup(track, update.timestamp_ms)
#                 if matched is not None:
#                     track.resolved_class_name = matched["event_class"]

#                 trigger_reason = (
#                     "inward re-entry + motion"
#                     if has_inward
#                     else "ROI re-entry from absent vertical/edge/fast return"
#                     if has_reentry
#                     else "stable ROI presence after pickup hold"
#                 )

#                 self._transition(track, "PUTBACK_PENDING", trigger_reason)
#                 track.pending_since_ms = update.timestamp_ms
#                 self._seed_confirmation_from_history(track)

#                 self._emit_debug(
#                     f"[PENDING] putback track={track.global_id} "
#                     f"event_class={self._event_class(track, purpose='putback')} "
#                     f"detected={self._detected_class(track)} "
#                     f"safe={update.in_safe_roi} stable={update.in_stable_roi} "
#                     f"outer={update.in_outer_roi} "
#                     f"motion={has_inward} reentry={has_reentry} "
#                     f"elapsed={time_since_pickup_ms:.0f} "
#                     f"prev_safe={prev_safe} prev_outer={prev_outer}",
#                     level="info",
#                 )

#                 return []

#             return []

#         # =====================================================================
#         # STATE: PUTBACK_PENDING
#         # =====================================================================
#         if track.event_state == "PUTBACK_PENDING":

#             if update.in_stable_roi:
#                 self._append_confirmation(track, update.confidence)

#                 if len(track.confirmation_confidences) >= CONFIRM_FRAMES:
#                     matched = self._match_putback_to_pending_pickup(track, update.timestamp_ms)
#                     if matched is not None:
#                         track.resolved_class_name = matched["event_class"]

#                     if not self._confidence_ok(track):
#                         self._emit_debug(
#                             f"[CANCEL] putback confidence track={track.global_id} "
#                             f"conf={list(track.confirmation_confidences)}",
#                             level="info",
#                         )
#                         self._transition(
#                             track,
#                             "PICKED_UP",
#                             "putback cancelled by confidence window",
#                         )
#                         return []

#                     self._transition(
#                         track,
#                         "STABLE_INSIDE",
#                         "putback confirmed: stable in shelf ROI",
#                     )
#                     event = self._record_putback(track)
#                     return [] if event is None else [event]

#                 return []

#             if update.in_outer_roi:
#                 self._append_confirmation(track, update.confidence)

#                 if len(track.confirmation_confidences) >= CONFIRM_FRAMES:
#                     if not self._confidence_ok(track):
#                         self._emit_debug(
#                             f"[CANCEL] putback OUTER confidence track={track.global_id} "
#                             f"conf={list(track.confirmation_confidences)}",
#                             level="info",
#                         )
#                         self._transition(
#                             track,
#                             "PICKED_UP",
#                             "putback cancelled: confidence too low in OUTER",
#                         )
#                         return []

#                 if self._pending_timeout_elapsed(track, update.timestamp_ms):
#                     if not self._has_any_recent_inward_motion(track, look_back=5):
#                         self._emit_debug(
#                             f"[TIMEOUT] putback OUTER track={track.global_id} no inward motion",
#                             level="info",
#                         )
#                         self._transition(
#                             track,
#                             "PICKED_UP",
#                             "putback cancelled: timeout with no inward motion",
#                         )
#                         return []

#                     track.pending_since_ms = update.timestamp_ms

#                 return []

#             if self._pending_timeout_elapsed(track, update.timestamp_ms):
#                 self._emit_debug(
#                     f"[TIMEOUT] putback absent track={track.global_id}",
#                     level="info",
#                 )
#                 self._transition(
#                     track,
#                     "PICKED_UP",
#                     "putback cancelled: timeout while absent",
#                 )
#                 return []

#             return []

#         return []

#     # ─────────────────────────────────────────────────────────────────────────
#     # Public API
#     # ─────────────────────────────────────────────────────────────────────────

#     def process_frame(self, tracks, frame_avg_confidence, timestamp_ms):
#         for track in tracks:
#             self._ensure_identity_fields(track)

#         self._frame_net_snapshot = {
#             class_name: self.pickup_count[class_name] - self.putback_count[class_name]
#             for class_name in set(self.pickup_count) | set(self.putback_count)
#         }

#         self._frame_putback_used = defaultdict(int)

#         events = []

#         for track in tracks:
#             events.extend(self._process_track(track, timestamp_ms, all_tracks=tracks))

#         self._frame_net_snapshot = {}
#         self._frame_putback_used = defaultdict(int)

#         return events

#     def finalize(self):
#         classes = set(self.pickup_count) | set(self.putback_count)

#         self.session.pickup_count = dict(self.pickup_count)
#         self.session.putback_count = dict(self.putback_count)
#         self.session.net_change = {
#             cls: self.pickup_count[cls] - self.putback_count[cls]
#             for cls in classes
#         }
#         self.session.total_pickups = sum(self.pickup_count.values())
#         self.session.total_putbacks = sum(self.putback_count.values())

#         if hasattr(self.session, "pending_pickups"):
#             self.session.pending_pickups = [
#                 item for item in self.pending_pickups if not item.get("closed")
#             ]

#         return self.session


from __future__ import annotations

import logging
import uuid
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Any

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
)
from utils.types import GlobalTrack, SessionSummary

# [DEBUG] Import DebugLogger (optional, may be None if not installed)
try:
    from debug_logger import DebugLogger
except ImportError:
    DebugLogger = None

LOGGER = logging.getLogger("vending_pipeline.events")


class EventManager:
    """
    Pickup/putback FSM with class‑drift protection, ledger matching,
    and robust quick‑return detection using consecutive frames outside stable ROI.
    """

    def __init__(self, session_id: str, debug_logger: Optional[Any] = None):   # [DEBUG] debug_logger param
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
        self.recent_overlay_events: Deque[tuple[float, str]] = deque(maxlen=20)

        # [DEBUG] store the debug logger
        self.debug_logger = debug_logger

    # -------------------------------------------------------------------------
    # Identity helpers
    # -------------------------------------------------------------------------

    def _ensure_identity_fields(self, track):
        if not hasattr(track, "class_votes") or track.class_votes is None:
            track.class_votes = deque(maxlen=CLASS_VOTE_HISTORY)
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
        if not hasattr(track, "prev_update"):
            track.prev_update = None

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

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def _emit_debug(self, message, level="debug"):
        getattr(LOGGER, level)(message)
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
            self._lock_class_if_possible(track, reason="entered STABLE_INSIDE")

        if new_state in {"PICKUP_PENDING", "PUTBACK_PENDING"}:
            track.confirmation_confidences.clear()
            track.confirmation_class_votes.clear()
        else:
            track.pending_since_ms = None

        self._log_transition(track, old_state, new_state, reason)

        # [DEBUG] Log state transition via debug logger
        if self.debug_logger is not None:
            timestamp_ms = track.current_update.timestamp_ms if track.current_update else track.last_seen_ms
            self.debug_logger.log_event_manager_state(
                track=track,
                old_state=old_state,
                new_state=new_state,
                reason=reason,
                timestamp_ms=timestamp_ms,
            )

    # -------------------------------------------------------------------------
    # Class voting / locking
    # -------------------------------------------------------------------------

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
            votes = [v for v in votes if v.get("in_stable_roi") or v.get("in_safe_roi") or v.get("in_outer_roi")]
        if not votes:
            return {"class": UNKNOWN_UNSTABLE_CLASS, "score": 0.0, "total_score": 0.0, "ratio": 0.0, "scores": {}}
        scores = defaultdict(float)
        n = len(votes)
        for idx, vote in enumerate(votes):
            age_from_latest = n - 1 - idx
            recency_weight = CLASS_VOTE_RECENCY_DECAY ** age_from_latest
            cls = vote["class"]
            score = float(vote.get("confidence", 0.0)) * float(vote.get("roi_weight", 1.0)) * recency_weight
            scores[cls] += score
        total_score = sum(scores.values())
        if total_score <= 0:
            return {"class": UNKNOWN_UNSTABLE_CLASS, "score": 0.0, "total_score": 0.0, "ratio": 0.0, "scores": dict(scores)}
        sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_class, best_score = sorted_scores[0]
        ratio = best_score / total_score
        if len(sorted_scores) >= 2:
            runner_up_class, runner_up_score = sorted_scores[1]
            runner_up_ratio = runner_up_score / total_score
            if ratio >= 0.35 and runner_up_ratio >= 0.35:
                base_a = self._family_base_class(best_class)
                base_b = self._family_base_class(runner_up_class)
                if base_a == base_b and base_a != UNKNOWN_UNSTABLE_CLASS:
                    best_class = base_a
                    best_score += runner_up_score
                    ratio = best_score / total_score
        selected_class = best_class if (total_score >= CLASS_LOCK_MIN_TOTAL_SCORE and ratio >= CLASS_LOCK_RATIO) else UNKNOWN_UNSTABLE_CLASS
        return {"class": selected_class, "score": best_score, "total_score": total_score, "ratio": ratio, "scores": dict(scores)}

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
            self._emit_debug(f"[CLASS-LOCK-DELAYED] track={track.global_id} reason={reason} ratio={result['ratio']:.2f} scores={result['scores']}", level="info")
            return
        track.locked_class_name = locked_class
        track.resolved_class_name = locked_class
        detected_class = self._detected_class(track)
        if detected_class != locked_class:
            track.class_drift_flagged = True
        self._emit_debug(f"[CLASS-LOCKED] track={track.global_id} locked={locked_class} detected={detected_class} reason={reason} ratio={result['ratio']:.2f} scores={result['scores']}", level="info")

    # -------------------------------------------------------------------------
    # Confirmation helpers
    # -------------------------------------------------------------------------

    def _append_confirmation(self, track, confidence):
        self._ensure_identity_fields(track)
        if track.confirmation_confidences.maxlen != CONFIRM_FRAMES:
            track.confirmation_confidences = deque(track.confirmation_confidences, maxlen=CONFIRM_FRAMES)
        if track.confirmation_class_votes.maxlen != CONFIRM_FRAMES:
            track.confirmation_class_votes = deque(track.confirmation_class_votes, maxlen=CONFIRM_FRAMES)
        track.confirmation_confidences.append(float(confidence))
        update = track.current_update
        if update is not None:
            detected_class = getattr(update, "class_name", None) or self._detected_class(track)
            track.confirmation_class_votes.append({"class": detected_class, "confidence": float(confidence), "timestamp_ms": float(update.timestamp_ms)})

    def _seed_confirmation_from_history(self, track):
        self._ensure_identity_fields(track)
        history = list(track.confidence_history)[-CONFIRM_FRAMES:]
        track.confirmation_confidences = deque(history, maxlen=CONFIRM_FRAMES)
        class_history = list(track.class_votes)[-CONFIRM_FRAMES:]
        track.confirmation_class_votes = deque([{"class": item["class"], "confidence": item["confidence"], "timestamp_ms": item["timestamp_ms"]} for item in class_history], maxlen=CONFIRM_FRAMES)

    def _seed_class_return_confirmation(self, track):
        self._ensure_identity_fields(track)
        history = list(track.confidence_history)[-CONFIRM_FRAMES:]
        if not history:
            history = [track.last_confidence]
        while len(history) < CONFIRM_FRAMES:
            history.append(history[-1])
        track.confirmation_confidences = deque(history, maxlen=CONFIRM_FRAMES)
        class_history = list(track.class_votes)[-CONFIRM_FRAMES:]
        if not class_history:
            detected_class = self._detected_class(track)
            class_history = [{"class": detected_class, "confidence": track.last_confidence, "timestamp_ms": track.last_seen_ms}]
        while len(class_history) < CONFIRM_FRAMES:
            class_history.append(class_history[-1])
        track.confirmation_class_votes = deque([{"class": item["class"], "confidence": item["confidence"], "timestamp_ms": item["timestamp_ms"]} for item in class_history[-CONFIRM_FRAMES:]], maxlen=CONFIRM_FRAMES)

    def _pending_timeout_elapsed(self, track, timestamp_ms):
        if track.pending_since_ms is None:
            return False
        return (timestamp_ms - track.pending_since_ms) >= MISSING_PENDING_CONFIRM_MS

    def _confidence_ok(self, track):
        if len(track.confirmation_confidences) < CONFIRM_FRAMES:
            return False
        avg_conf = sum(track.confirmation_confidences) / len(track.confirmation_confidences)
        min_conf = min(track.confirmation_confidences)
        return avg_conf >= CONF_THRESHOLD and min_conf >= CONF_MIN_FRAME

    def _class_consistency_ok(self, track):
        self._ensure_identity_fields(track)
        event_class = self._event_class(track, purpose="pickup")
        if not event_class or event_class == UNKNOWN_UNSTABLE_CLASS:
            return False
        votes = list(track.confirmation_class_votes)
        if not votes:
            return True
        total_weight = 0.0
        matched_weight = 0.0
        for vote in votes:
            cls = vote.get("class")
            conf = float(vote.get("confidence", 0.0) or 0.0)
            total_weight += conf
            if cls == event_class:
                matched_weight += conf
        if total_weight <= 0:
            return False
        ratio = matched_weight / total_weight
        if ratio < CLASS_CONSISTENCY_RATIO:
            track.class_drift_flagged = True
            self._emit_debug(f"[CLASS-DRIFT] track={track.global_id} event_class={event_class} votes={votes} consistency={ratio:.2f}", level="info")
        return ratio >= CLASS_CONSISTENCY_RATIO

    def _confirmation_ok(self, track, require_class_consistency=True):
        if not self._confidence_ok(track):
            return False
        if not require_class_consistency:
            return True
        return self._class_consistency_ok(track)

    # -------------------------------------------------------------------------
    # Motion helpers
    # -------------------------------------------------------------------------

    def _required_stability_frames(self, track, timestamp_ms):
        return LONG_BOUNDARY_STABILITY_FRAMES if track.boundary_linger_flagged else STABILITY_FRAMES

    def _has_any_recent_outward_motion(self, track, look_back=3):
        if not track.motion_direction_history:
            return False
        recent = list(track.motion_direction_history)[-look_back:]
        return any(d is True for d in recent)

    def _has_any_recent_inward_motion(self, track, look_back=3):
        if not track.motion_direction_history:
            return False
        recent = list(track.motion_direction_history)[-look_back:]
        return any(d is False for d in recent)

    def _has_sustained_outward_motion(self, track, min_consecutive=2):
        if len(track.motion_direction_history) < min_consecutive:
            return False
        return all(d is True for d in list(track.motion_direction_history)[-min_consecutive:])

    def _has_sustained_inward_motion(self, track, min_consecutive=2):
        if len(track.motion_direction_history) < min_consecutive:
            return False
        return all(d is False for d in list(track.motion_direction_history)[-min_consecutive:])

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
        if current_inward or sustained_inward or (reentered_roi and (current_inward or sustained_inward)):
            return True
        return False

    # -------------------------------------------------------------------------
    # ROI helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _roi_exit_occurred(track, update, prev_stable):
        return track.was_stable and prev_stable is True and not update.in_stable_roi

    @staticmethod
    def _roi_reentry_occurred(update, prev_safe, prev_outer):
        was_fully_absent = (prev_safe is False) and (prev_outer is False)
        now_in_roi = update.in_safe_roi or update.in_outer_roi
        return was_fully_absent and now_in_roi

    # -------------------------------------------------------------------------
    # Ledger helpers
    # -------------------------------------------------------------------------

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

    def _class_similarity_score(self, a, b):
        if not a or not b:
            return LEDGER_DIFFERENT_CLASS_SCORE
        return LEDGER_SAME_CLASS_SCORE if a == b else LEDGER_DIFFERENT_CLASS_SCORE

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
            score += min(current_scores.get(cls, 0.0) / current_total, pickup_scores.get(cls, 0.0) / pickup_total)
        return min(1.0, score)

    def _time_match_score(self, pickup_time_ms, return_time_ms):
        gap = max(0.0, return_time_ms - pickup_time_ms)
        if gap >= PENDING_LEDGER_MAX_AGE_MS:
            return 0.0
        return 1.0 - (gap / PENDING_LEDGER_MAX_AGE_MS)

    def _track_match_score(self, track, pending_item):
        return 1.0 if pending_item.get("track_id") == track.global_id else 0.0

    def _motion_match_score_for_putback(self, track):
        return 1.0 if self._has_any_recent_inward_motion(track, look_back=5) else 0.3

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
            return 0.0
        return 1.0 - (dist / LEDGER_SPATIAL_MAX_DISTANCE_PX)

    def _match_putback_to_pending_pickup(self, track, timestamp_ms):
        self._ensure_identity_fields(track)
        candidates = self._active_pending_pickups(timestamp_ms)
        if not candidates:
            return None
        detected_return_class = self._detected_class(track)
        resolved_return_class = self._event_class(track, purpose="putback")
        best_item = None
        best_score = -1.0
        best_breakdown = {}
        for item in candidates:
            picked_class = item["event_class"]
            class_score = max(self._class_similarity_score(detected_return_class, picked_class), self._class_similarity_score(resolved_return_class, picked_class))
            time_score = self._time_match_score(item["pickup_time_ms"], timestamp_ms)
            track_score = self._track_match_score(track, item)
            candidate_score = self._ledger_class_candidate_score(track, item)
            motion_score = self._motion_match_score_for_putback(track)
            embedding_score = self._embedding_similarity_score(track, item, expected_class=picked_class)
            spatial_score = self._spatial_match_score(track, item)
            total_score = (LEDGER_MATCH_WEIGHT_CLASS * class_score + LEDGER_MATCH_WEIGHT_TIME * time_score + LEDGER_MATCH_WEIGHT_TRACK * track_score + LEDGER_MATCH_WEIGHT_CLASS_CANDIDATE * candidate_score + LEDGER_MATCH_WEIGHT_MOTION * motion_score + LEDGER_MATCH_WEIGHT_EMBEDDING * embedding_score + LEDGER_MATCH_WEIGHT_SPATIAL * spatial_score)
            breakdown = {"total": total_score, "picked_class": picked_class, "detected_return_class": detected_return_class, "resolved_return_class": resolved_return_class, "class_score": class_score, "time_score": time_score, "track_score": track_score, "class_candidate_score": candidate_score, "motion_score": motion_score, "embedding_score": embedding_score, "spatial_score": spatial_score}
            if total_score > best_score:
                best_score = total_score
                best_item = item
                best_breakdown = breakdown
        if best_item is not None and best_score >= PUTBACK_LEDGER_MATCH_THRESHOLD:
            best_item["_last_match_score"] = best_score
            best_item["_last_match_breakdown"] = best_breakdown
            return best_item
        self._emit_debug(f"[LEDGER-NO-MATCH] track={track.global_id} detected={detected_return_class} best_score={best_score:.2f} threshold={PUTBACK_LEDGER_MATCH_THRESHOLD} breakdown={best_breakdown}", level="info")
        return None

    def _is_class_return_candidate(self, track):
        self._ensure_identity_fields(track)
        update = track.current_update
        if update is None:
            return False
        if track.was_stable:
            return False
        has_inward = update.inward_motion or self._has_any_recent_inward_motion(track, look_back=5)
        if not has_inward:
            return False
        candidate_class = track.locked_class_name if self._is_valid_class(track.locked_class_name) else self._voted_event_class(track)
        exact_class_return = self._has_outstanding_pickup(candidate_class)
        latest_pickup_ts = None
        ledger_match = None
        if exact_class_return:
            latest_pickup_ts = self._latest_pickup_timestamp(candidate_class)
        else:
            ledger_match = self._match_putback_to_pending_pickup(track, update.timestamp_ms)
            if ledger_match is not None:
                latest_pickup_ts = ledger_match["pickup_time_ms"]
        if latest_pickup_ts is None:
            return False
        within_gap = (update.timestamp_ms - latest_pickup_ts) <= CLASS_RETURN_MAX_GAP_MS
        born_after_pickup = track.created_ms >= latest_pickup_ts
        not_previously_stable = not track.was_stable
        is_candidate = within_gap and born_after_pickup and not_previously_stable and has_inward
        if is_candidate and ledger_match is not None:
            track.resolved_class_name = ledger_match["event_class"]
        return is_candidate

    # -------------------------------------------------------------------------
    # Neighbor stability boost
    # -------------------------------------------------------------------------

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
            if neighbor.stability_boosted:
                continue
            neighbor_update = neighbor.current_update
            if neighbor_update is None:
                continue
            same_zone = (update.in_stable_roi and neighbor_update.in_stable_roi) or (update.in_outer_roi and neighbor_update.in_outer_roi)
            half_stability = neighbor.inside_counter >= max(1, STABILITY_FRAMES // 2)
            if same_zone and half_stability:
                required = self._required_stability_frames(neighbor, update.timestamp_ms)
                neighbor.inside_counter = required
                neighbor.event_state = "STABLE_INSIDE"
                neighbor.stability_boosted = True
                neighbor.was_stable = True
                self._lock_class_if_possible(neighbor, reason="neighbor stability boost")
                self._emit_debug(f"[BOOST] neighbor track={neighbor.global_id} -> STABLE_INSIDE by track={track.global_id}", level="debug")

    # -------------------------------------------------------------------------
    # Event recording
    # -------------------------------------------------------------------------

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
        update = track.current_update
        event_class = self._event_class(track, purpose="pickup")
        if event_class == UNKNOWN_UNSTABLE_CLASS:
            self._lock_class_if_possible(track, reason="pickup final lock attempt")
            event_class = self._event_class(track, purpose="pickup")
        detected_class = self._detected_class(track)
        if self._is_valid_class(track.locked_class_name) and event_class != track.locked_class_name:
            self._emit_debug(f"[PICKUP-CLASS-CORRECTED] track={track.global_id} event_class={event_class} -> locked_class={track.locked_class_name} detected={detected_class}", level="warning")
            event_class = track.locked_class_name
            track.resolved_class_name = event_class
        self.pickup_count[event_class] += 1
        event_timestamp = track.last_seen_ms if update is None else update.timestamp_ms
        event_camera_id = track.last_camera_id if update is None else update.camera_id
        event_frame_index = track.last_frame_index if update is None else update.frame_index
        self.class_pickup_timestamps[event_class].append(event_timestamp)
        track.pickup_confirmed_ms = event_timestamp
        track.resolved_class_name = event_class
        class_drift = detected_class != event_class
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
            "confidence": float(sum(track.confirmation_confidences) / max(len(track.confirmation_confidences), 1)),
            "confidence_window": list(track.confirmation_confidences),
            "confirmation_class_votes": list(track.confirmation_class_votes),
        }
        if ENABLE_CLASS_DRIFT_WARNINGS and event["class_drift"]:
            event["warnings"] = [f"class drift observed on pickup: detected={detected_class}, resolved={event_class}"]
        self.session.events.append(event)
        if hasattr(self.session, "pickup_records"):
            self.session.pickup_records.append(event)
        ledger_item = self._pickup_ledger_item_from_event(track, event)
        self.pending_pickups.append(ledger_item)
        overlay = f"PICKUP {event_class} G{track.global_id}"
        if event["class_drift"]:
            overlay += f" [detected:{detected_class}]"
        self._push_overlay_event(event_timestamp, overlay)
        self._emit_debug(f"[EVENT] {overlay} | cam={event_camera_id} frame={event_frame_index} conf={list(track.confirmation_confidences)} ledger_id={ledger_item['ledger_id']}", level="info")

        # [DEBUG] Log event via debug logger
        if self.debug_logger is not None:
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
        current_net = self._frame_net_snapshot.get(event_class, 0) - self._frame_putback_used.get(event_class, 0)
        if current_net <= 0:
            warning = f"Putback ignored class={event_class} detected={detected_class} track={track.global_id} — no outstanding pickup to offset."
            self.session.warnings.append(warning)
            LOGGER.warning(warning)
            return None
        self.putback_count[event_class] += 1
        self._frame_putback_used[event_class] += 1
        track.last_putback_confirmed_ms = event_timestamp
        track.resolved_class_name = event_class
        self.last_putback_by_class_ms[event_class] = event_timestamp
        if matched_item is not None:
            matched_item["closed"] = True
        is_long_hold = track.pickup_confirmed_ms is not None and (event_timestamp - track.pickup_confirmed_ms) > CLASS_RETURN_MAX_GAP_MS
        class_drift = detected_class != event_class
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
            "confidence": float(sum(track.confirmation_confidences) / max(len(track.confirmation_confidences), 1)),
            "confidence_window": list(track.confirmation_confidences),
            "confirmation_class_votes": list(track.confirmation_class_votes),
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
            event["warnings"] = [f"class drift observed on putback: detected={detected_class}, resolved={event_class}"]
        self.session.events.append(event)
        if hasattr(self.session, "putback_records"):
            self.session.putback_records.append(event)
        overlay = f"PUTBACK {event_class} G{track.global_id}"
        if event["class_drift"]:
            overlay += f" [detected:{detected_class}]"
        if is_long_hold:
            overlay += " [long-hold]"
        self._push_overlay_event(event_timestamp, overlay)
        self._emit_debug(f"[EVENT] {overlay} | cam={event_camera_id} frame={event_frame_index} conf={list(track.confirmation_confidences)} matched_ledger={matched_item['ledger_id'] if matched_item else None}", level="info")

        # [DEBUG] Log event via debug logger
        if self.debug_logger is not None:
            self.debug_logger.log_event_fired(
                track=track,
                event_type="putback",
                event=event,
                timestamp_ms=event_timestamp,
                suppressed=False,
                suppression_reason=None,
            )

        return event

    # -------------------------------------------------------------------------
    # Overlay (fixed: method was missing)
    # -------------------------------------------------------------------------

    def overlay_event_lines(self, timestamp_ms):
        """Return a list of recent overlay event strings (max OVERLAY_MAX_EVENT_LINES)."""
        while self.recent_overlay_events and (timestamp_ms - self.recent_overlay_events[0][0]) > OVERLAY_EVENT_TTL_MS:
            self.recent_overlay_events.popleft()
        return [item[1] for item in list(self.recent_overlay_events)[-OVERLAY_MAX_EVENT_LINES:]]

    # -------------------------------------------------------------------------
    # Core FSM – with debug logging for blocking and cooldowns
    # -------------------------------------------------------------------------

    def _process_track(self, track, timestamp_ms, all_tracks=None):
        self._ensure_identity_fields(track)
        update = track.current_update

        # Store the previous frame's outside-stable count before updating it for this frame
        prev_outside_frames = track.frames_outside_stable

        # Update frames_outside_stable for the *next* frame (do not use it for this frame's decisions yet)
        if update is not None:
            if update.in_stable_roi:
                track.frames_outside_stable = 0
            else:
                track.frames_outside_stable += 1
        else:
            track.frames_outside_stable += 1

        # =====================================================================
        # BRANCH A: no detection this frame
        # =====================================================================
        if update is None:
            if track.event_state == "PICKUP_PENDING":
                if self._pending_timeout_elapsed(track, timestamp_ms):
                    if self._has_any_recent_outward_motion(track, look_back=5):
                        self._transition(track, "PICKED_UP", "pickup confirmed: lost outward motion")
                        return [self._record_pickup(track)]
                    if track.was_stable:
                        self._transition(track, "PICKED_UP", "pickup confirmed: lost was stable occlusion/vertical")
                        event = self._record_pickup(track)
                        event["occlusion_pickup"] = True
                        self._replace_last_overlay(f"PICKUP {event['class']} G{track.global_id} [occluded]")
                        return [event]
                    self._emit_debug(f"[SILENT-CANCEL] track={track.global_id} never reached stability — suppressed", level="debug")
                    self._transition(track, "INSIDE", "pickup cancelled: never reached stability")
                    return []
            if track.event_state == "PUTBACK_PENDING":
                if self._pending_timeout_elapsed(track, timestamp_ms):
                    if track.last_seen_in_stable_roi is True and self._has_any_recent_inward_motion(track, look_back=5):
                        self._transition(track, "STABLE_INSIDE", "putback confirmed: lost in stable ROI")
                        event = self._record_putback(track)
                        return [] if event is None else [event]
                    self._emit_debug(f"[TIMEOUT] putback absent track={track.global_id} last_stable={track.last_seen_in_stable_roi}", level="info")
                    self._transition(track, "PICKED_UP", "putback cancelled: lost without entering stable ROI")
            return []

        # =====================================================================
        # BRANCH B: detection exists
        # =====================================================================
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

        # =====================================================================
        # STATE: INSIDE
        # =====================================================================
        if track.event_state == "INSIDE":
            if self._is_class_return_candidate(track):
                self._transition(track, "PUTBACK_PENDING", "class/ledger inward re-entry candidate")
                track.pending_since_ms = update.timestamp_ms
                self._seed_class_return_confirmation(track)
                self._emit_debug(f"[PENDING] putback track={track.global_id} event_class={self._event_class(track, purpose='putback')} detected={self._detected_class(track)}", level="info")
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
            if track.locked_class_name == UNKNOWN_UNSTABLE_CLASS:
                self._lock_class_if_possible(track, reason="additional stable evidence")
            if self._is_class_return_candidate(track):
                self._transition(track, "PUTBACK_PENDING", "class/ledger inward re-entry candidate")
                track.pending_since_ms = update.timestamp_ms
                self._seed_class_return_confirmation(track)
                self._emit_debug(f"[PENDING] putback track={track.global_id} event_class={self._event_class(track, purpose='putback')} detected={self._detected_class(track)}", level="info")
                return []

            # Pickup trigger
            if not update.in_safe_roi:
                has_outward = self._has_any_recent_outward_motion(track, look_back=3)
                has_sustained_outward = self._has_sustained_outward_motion(track, min_consecutive=2)
                has_roi_exit = self._roi_exit_occurred(track, update, prev_stable)
                is_inward_or_return = self._is_inward_or_return_motion(track, update, prev_safe, prev_outer, prev_stable)
                if is_inward_or_return and not has_outward and not has_roi_exit:
                    self._emit_debug(f"[PICKUP-BLOCKED-INWARD] track={track.global_id} event_class={self._event_class(track, purpose='pickup')} detected={self._detected_class(track)} prev_safe={prev_safe} curr_safe={update.in_safe_roi} prev_outer={prev_outer} curr_outer={update.in_outer_roi} prev_stable={prev_stable} curr_stable={update.in_stable_roi} outward={update.outward_motion} inward={update.inward_motion} motion_history={list(track.motion_direction_history)}", level="info")
                    # [DEBUG] Log blocked inward
                    if self.debug_logger is not None:
                        self.debug_logger.log_blocked_inward(
                            track=track,
                            reason="inward motion while in stable inside, blocking pickup",
                            timestamp_ms=update.timestamp_ms,
                            prev_safe=prev_safe,
                            curr_safe=update.in_safe_roi,
                            prev_outer=prev_outer,
                            curr_outer=update.in_outer_roi,
                            prev_stable=prev_stable,
                            curr_stable=update.in_stable_roi,
                            motion_history=list(track.motion_direction_history),
                        )
                    return []
                pickup_like_exit = has_sustained_outward or has_outward or has_roi_exit
                if pickup_like_exit:
                    if track.last_putback_confirmed_ms is not None and (update.timestamp_ms - track.last_putback_confirmed_ms) < POST_PUTBACK_COOLDOWN_MS:
                        self._emit_debug(f"[COOLDOWN] pickup suppressed track={track.global_id} elapsed={update.timestamp_ms - track.last_putback_confirmed_ms:.0f}ms cooldown={POST_PUTBACK_COOLDOWN_MS}ms", level="debug")
                        # [DEBUG] Log cooldown suppression
                        if self.debug_logger is not None:
                            self.debug_logger.log_cooldown_suppression(
                                track=track,
                                cooldown_ms=POST_PUTBACK_COOLDOWN_MS,
                                elapsed_ms=update.timestamp_ms - track.last_putback_confirmed_ms,
                                timestamp_ms=update.timestamp_ms,
                            )
                        return []
                    self._lock_class_if_possible(track, reason="pickup trigger")
                    pickup_class = self._event_class(track, purpose="pickup")
                    last_class_putback_ms = self.last_putback_by_class_ms.get(pickup_class)
                    if last_class_putback_ms is not None and (update.timestamp_ms - last_class_putback_ms) < POST_PUTBACK_COOLDOWN_MS:
                        self._emit_debug(f"[CLASS-COOLDOWN] pickup suppressed track={track.global_id} class={pickup_class} elapsed={update.timestamp_ms - last_class_putback_ms:.0f}ms cooldown={POST_PUTBACK_COOLDOWN_MS}ms", level="debug")
                        # [DEBUG] Log class cooldown suppression
                        if self.debug_logger is not None:
                            self.debug_logger.log_cooldown_suppression(
                                track=track,
                                cooldown_ms=POST_PUTBACK_COOLDOWN_MS,
                                elapsed_ms=update.timestamp_ms - last_class_putback_ms,
                                timestamp_ms=update.timestamp_ms,
                            )
                        return []
                    if self._is_valid_class(track.locked_class_name) and self._is_valid_class(track.resolved_class_name) and track.resolved_class_name != track.locked_class_name:
                        self._emit_debug(f"[CLEAR-STALE-RESOLVED-BEFORE-PICKUP] track={track.global_id} resolved={track.resolved_class_name} locked={track.locked_class_name}", level="warning")
                        track.resolved_class_name = track.locked_class_name
                    trigger_reason = "outward exit + motion" if (has_outward or has_sustained_outward) else "ROI exit without motion vertical/edge/fast/low-tray pick"
                    self._transition(track, "PICKUP_PENDING", trigger_reason)
                    track.pending_since_ms = update.timestamp_ms
                    self._seed_confirmation_from_history(track)
                    if all_tracks is not None:
                        self._check_neighbor_stability_boost(track, all_tracks)
                    self._emit_debug(f"[PENDING] pickup track={track.global_id} event_class={self._event_class(track, purpose='pickup')} detected={self._detected_class(track)} outward={has_outward} sustained_outward={has_sustained_outward} roi_exit={has_roi_exit} inward_or_return={is_inward_or_return} prev_safe={prev_safe} curr_safe={update.in_safe_roi} prev_stable={prev_stable} curr_stable={update.in_stable_roi} safe_dist={update.safe_roi_distance:.1f} motion_history={list(track.motion_direction_history)}", level="info")
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
            # Quick return: check using previous frame's outside count (before reset)
            if update.in_stable_roi and not update.outward_motion:
                time_since_pending_ms = update.timestamp_ms - (track.pending_since_ms or update.timestamp_ms)
                was_outside_stable = prev_outside_frames >= QUICK_RETURN_MIN_OUTSIDE_FRAMES
                if was_outside_stable and time_since_pending_ms < QUICK_RETURN_THRESHOLD_MS:
                    self._transition(track, "PICKED_UP", "quick pickup: stayed outside stable ROI for enough frames")
                    pickup_event = self._record_pickup(track)
                    self._frame_net_snapshot[pickup_event["class"]] = self._frame_net_snapshot.get(pickup_event["class"], 0) + 1
                    track.resolved_class_name = pickup_event["class"]
                    self._transition(track, "PUTBACK_PENDING", "quick return: fast re-entry")
                    track.pending_since_ms = update.timestamp_ms
                    self._seed_confirmation_from_history(track)
                    self._emit_debug(f"[QUICK-RETURN] track={track.global_id} time_ms={time_since_pending_ms:.0f} outside_frames={prev_outside_frames} event_class={pickup_event['class']} detected={self._detected_class(track)}", level="info")
                    return [pickup_event]
                # Cancel – not enough evidence
                self._emit_debug(f"[CANCEL] pickup track={track.global_id} re-entered stable ROI without enough outside frames (outside_frames={prev_outside_frames})", level="info")
                self._transition(track, "STABLE_INSIDE", "pickup cancelled: not enough time outside stable ROI")
                return []

            # Cancel if sustained inward motion returns before confirmation
            if self._has_sustained_inward_motion(track, min_consecutive=2) and (update.in_safe_roi or update.in_stable_roi):
                self._emit_debug(f"[CANCEL] pickup track={track.global_id} cancelled by sustained inward return event_class={self._event_class(track, purpose='pickup')} detected={self._detected_class(track)}", level="info")
                self._transition(track, "STABLE_INSIDE", "pickup cancelled: inward return before confirmation")
                return []

            if not update.in_outer_roi:
                self._append_confirmation(track, update.confidence)
                confirmation_ok = self._confirmation_ok(track, require_class_consistency=False)
                timeout_absent_ok = track.was_stable and (update.timestamp_ms - (track.pending_since_ms or update.timestamp_ms)) > 100 and not self._has_any_recent_inward_motion(track, look_back=5)
                if confirmation_ok or timeout_absent_ok:
                    self._transition(track, "PICKED_UP", "pickup confirmed: SAFE -> OUTER -> ABSENT")
                    return [self._record_pickup(track)]
                return []

            if update.in_outer_roi:
                self._append_confirmation(track, update.confidence)
                if len(track.confirmation_confidences) >= CONFIRM_FRAMES:
                    conf_ok = self._confidence_ok(track)
                    class_ok = self._class_consistency_ok(track)
                    if not conf_ok:
                        self._emit_debug(f"[CANCEL] pickup track={track.global_id} confidence too low in OUTER conf={list(track.confirmation_confidences)}", level="info")
                        self._transition(track, "STABLE_INSIDE", "pickup cancelled: confidence too low")
                        return []
                    if not class_ok:
                        if track.was_stable and self._has_any_recent_outward_motion(track, look_back=5):
                            self._emit_debug(f"[CLASS-DRIFT-TOLERATED] pickup track={track.global_id} keeping event_class={self._event_class(track, purpose='pickup')} detected={self._detected_class(track)}", level="info")
                            return []
                        self._emit_debug(f"[CANCEL] pickup track={track.global_id} class consistency too low in OUTER", level="info")
                        self._transition(track, "STABLE_INSIDE", "pickup cancelled: class consistency too low")
                        return []
                if self._has_any_recent_outward_motion(track, look_back=3):
                    return []
                if self._pending_timeout_elapsed(track, update.timestamp_ms):
                    self._emit_debug(f"[TIMEOUT] pickup OUTER track={track.global_id} timeout elapsed", level="info")
                    self._transition(track, "STABLE_INSIDE", "pickup cancelled: timeout in OUTER ROI")
                    return []
                return []
            return []

        # =====================================================================
        # STATE: PICKED_UP
        # =====================================================================
        if track.event_state == "PICKED_UP":
            track_pickup_ts = track.pickup_confirmed_ms
            time_since_pickup_ms = (update.timestamp_ms - track_pickup_ts) if track_pickup_ts is not None else 0.0
            is_long_hold = track_pickup_ts is not None and time_since_pickup_ms > CLASS_RETURN_MAX_GAP_MS
            if is_long_hold and (update.in_stable_roi or update.in_outer_roi):
                matched = self._match_putback_to_pending_pickup(track, update.timestamp_ms)
                if matched is not None:
                    track.resolved_class_name = matched["event_class"]
                track.long_hold_pending = True
                self._transition(track, "PUTBACK_PENDING", "long-hold fallback: elapsed > CLASS_RETURN_MAX_GAP_MS")
                track.pending_since_ms = update.timestamp_ms
                self._seed_confirmation_from_history(track)
                self._emit_debug(f"[LONG-HOLD] track={track.global_id} event_class={self._event_class(track, purpose='putback')} detected={self._detected_class(track)} elapsed_ms={time_since_pickup_ms:.0f}", level="info")
                return []
            if time_since_pickup_ms < self.min_valid_putback_hold_ms:
                self._emit_debug(f"[PUTBACK-HOLD] suppressed early putback track={track.global_id} elapsed={time_since_pickup_ms:.0f}ms min_hold={self.min_valid_putback_hold_ms:.0f}ms", level="debug")
                return []
            has_inward = self._has_any_recent_inward_motion(track, look_back=5)
            has_reentry = self._roi_reentry_occurred(update, prev_safe, prev_outer)
            putback_like = (has_inward and (update.in_stable_roi or update.in_outer_roi or update.in_safe_roi)) or has_reentry or (update.in_stable_roi and time_since_pickup_ms >= self.min_valid_putback_hold_ms)
            if putback_like:
                matched = self._match_putback_to_pending_pickup(track, update.timestamp_ms)
                if matched is not None:
                    track.resolved_class_name = matched["event_class"]
                trigger_reason = "inward re-entry + motion" if has_inward else ("ROI re-entry from absent vertical/edge/fast return" if has_reentry else "stable ROI presence after pickup hold")
                self._transition(track, "PUTBACK_PENDING", trigger_reason)
                track.pending_since_ms = update.timestamp_ms
                self._seed_confirmation_from_history(track)
                self._emit_debug(f"[PENDING] putback track={track.global_id} event_class={self._event_class(track, purpose='putback')} detected={self._detected_class(track)} safe={update.in_safe_roi} stable={update.in_stable_roi} outer={update.in_outer_roi} motion={has_inward} reentry={has_reentry} elapsed={time_since_pickup_ms:.0f} prev_safe={prev_safe} prev_outer={prev_outer}", level="info")
                return []
            return []

        # =====================================================================
        # STATE: PUTBACK_PENDING
        # =====================================================================
        if track.event_state == "PUTBACK_PENDING":
            if update.in_stable_roi:
                self._append_confirmation(track, update.confidence)
                if len(track.confirmation_confidences) >= CONFIRM_FRAMES:
                    matched = self._match_putback_to_pending_pickup(track, update.timestamp_ms)
                    if matched is not None:
                        track.resolved_class_name = matched["event_class"]
                    if not self._confidence_ok(track):
                        self._emit_debug(f"[CANCEL] putback confidence track={track.global_id} conf={list(track.confirmation_confidences)}", level="info")
                        self._transition(track, "PICKED_UP", "putback cancelled by confidence window")
                        return []
                    self._transition(track, "STABLE_INSIDE", "putback confirmed: stable in shelf ROI")
                    event = self._record_putback(track)
                    return [] if event is None else [event]
                return []
            if update.in_outer_roi:
                self._append_confirmation(track, update.confidence)
                if len(track.confirmation_confidences) >= CONFIRM_FRAMES:
                    if not self._confidence_ok(track):
                        self._emit_debug(f"[CANCEL] putback OUTER confidence track={track.global_id} conf={list(track.confirmation_confidences)}", level="info")
                        self._transition(track, "PICKED_UP", "putback cancelled: confidence too low in OUTER")
                        return []
                if self._pending_timeout_elapsed(track, update.timestamp_ms):
                    if not self._has_any_recent_inward_motion(track, look_back=5):
                        self._emit_debug(f"[TIMEOUT] putback OUTER track={track.global_id} no inward motion", level="info")
                        self._transition(track, "PICKED_UP", "putback cancelled: timeout with no inward motion")
                        return []
                    track.pending_since_ms = update.timestamp_ms
                return []
            if self._pending_timeout_elapsed(track, update.timestamp_ms):
                self._emit_debug(f"[TIMEOUT] putback absent track={track.global_id}", level="info")
                self._transition(track, "PICKED_UP", "putback cancelled: timeout while absent")
                return []
            return []

        return []

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def process_frame(self, tracks, frame_avg_confidence, timestamp_ms):
        for track in tracks:
            self._ensure_identity_fields(track)
        self._frame_net_snapshot = {
            class_name: self.pickup_count[class_name] - self.putback_count[class_name]
            for class_name in set(self.pickup_count) | set(self.putback_count)
        }
        self._frame_putback_used = defaultdict(int)
        events = []
        for track in tracks:
            events.extend(self._process_track(track, timestamp_ms, all_tracks=tracks))
        self._frame_net_snapshot = {}
        self._frame_putback_used = defaultdict(int)
        return events

    def finalize(self):
        classes = set(self.pickup_count) | set(self.putback_count)
        self.session.pickup_count = dict(self.pickup_count)
        self.session.putback_count = dict(self.putback_count)
        self.session.net_change = {cls: self.pickup_count[cls] - self.putback_count[cls] for cls in classes}
        self.session.total_pickups = sum(self.pickup_count.values())
        self.session.total_putbacks = sum(self.putback_count.values())
        if hasattr(self.session, "pending_pickups"):
            self.session.pending_pickups = [item for item in self.pending_pickups if not item.get("closed")]
        return self.session