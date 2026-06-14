"""Fusion state machine: when does sound become an alert?

Normal mode (DoA + classifier):
IDLE ──DoA stable for N hops──► CANDIDATE ──vehicle_prob ≥ θ──► TRIGGERED
TRIGGERED ──vision confirms──► CONFIRMED (highest-trust alert, localized)
TRIGGERED ──vision sees nothing & front hyp──► votes "back", stays TRIGGERED*
TRIGGERED ──vision timeout──► TRIGGERED* (audio-only alert, lower trust)
any state ──track lost──► IDLE (with hysteresis: CONFIRMED coasts max_coast_s)

classifier_only mode (phone video / single mic / no reliable DoA):
IDLE ──vehicle_prob ≥ θ──► TRIGGERED ──vehicle_prob < θ/2──► IDLE
DoA tracks are ignored; direction info unavailable.

Stationary-source suppression (both modes):
If in TRIGGERED and confidence history is FLAT (std < min_cls_std over cls_history
classifier calls) OR alert has been sustained > max_sustained_alert_s seconds,
the source is classified as a stationary machine (museum equipment, HVAC, etc.)
and the FSM enters a "suppressed-IDLE" sub-state.

In suppressed-IDLE, triggering is blocked until confidence drops below θ/2
(proving the source stopped), preventing the constant-hum sources from causing
rapid TRIGGERED→IDLE→TRIGGERED oscillation.

Key distinction:
  Cars     = transient event (approaching→passing→receding): confidence rises,
             peaks, falls.  std over a 5-call window is HIGH.  DoA angle shifts.
  Machines = steady-state constant drone: confidence stays flat indefinitely.
             std over a 5-call window is LOW.

The asymmetry in alert design is deliberate: an unconfirmed rear-hypothesis
vehicle is the *most* dangerous case for a blind-spot system, so absence of
visual confirmation escalates rather than suppresses the alert.
"""
import collections
import enum
import time
from dataclasses import dataclass
from typing import Optional

from .tracker import TrackState
from ..inference.classifier import ClassifierResult


class FusionState(enum.Enum):
    IDLE = "idle"
    CANDIDATE = "candidate"
    TRIGGERED = "triggered"
    CONFIRMED = "confirmed"


@dataclass
class FusionEvent:
    state: FusionState
    track: Optional[TrackState]
    sound_class: str
    confidence: float
    vision_confirmed: bool
    alert: bool                 # should downstream consumers care
    request_vision: bool        # ask the high-power path to look NOW


class FusionStateMachine:
    def __init__(self, candidate_hold_hops: int = 8,
                 doa_stability_deg: float = 15.0,
                 trigger_threshold: float = 0.6,
                 vision_timeout_s: float = 2.0,
                 classifier_only: bool = False,
                 max_sustained_alert_s: float = 20.0,
                 cls_history: int = 10,
                 min_cls_std: float = 0.08):
        self._hold = candidate_hold_hops
        self._stab = doa_stability_deg
        self._thr = trigger_threshold
        self._vto = vision_timeout_s
        self._cls_only = classifier_only
        self._max_sustained_s = max_sustained_alert_s
        self._min_cls_std = min_cls_std

        self.state = FusionState.IDLE
        self._stable_hops = 0
        self._last_angle: Optional[float] = None
        self._last_cls: Optional[ClassifierResult] = None
        self._vision_requested_at: Optional[float] = None
        self._vision_confirmed = False
        self._triggered_since: Optional[float] = None
        # Rolling window of vehicle_prob scores (accumulates regardless of state)
        self._cls_window: collections.deque = collections.deque(maxlen=cls_history)
        # When True: confidence must drop below θ/2 before we re-arm.
        # Prevents constant-hum sources from rapid TRIGGERED↔IDLE oscillation.
        self._suppressed: bool = False

    def on_classifier(self, result: ClassifierResult) -> None:
        self._last_cls = result
        self._cls_window.append(result.vehicle_prob)

    def on_vision(self, confirmed: bool) -> None:
        if self.state == FusionState.TRIGGERED:
            self._vision_confirmed = confirmed
            if confirmed:
                self.state = FusionState.CONFIRMED

    def on_track(self, track: Optional[TrackState],
                 now: Optional[float] = None) -> FusionEvent:
        now = time.monotonic() if now is None else now

        # classifier_only: DoA not available (phone video, single mic).
        if self._cls_only:
            veh_p = self._last_cls.vehicle_prob if self._last_cls else 0.0

            if self.state == FusionState.IDLE:
                # Suppression gate: clear once confidence actually drops
                if self._suppressed:
                    if veh_p < self._thr * 0.5:
                        self._suppressed = False
                    # still suppressed — stay IDLE regardless of veh_p
                elif veh_p >= self._thr:
                    self.state = FusionState.TRIGGERED
                    self._triggered_since = now

            elif self.state == FusionState.TRIGGERED:
                if veh_p < self._thr * 0.5:
                    self._suppress()
                elif self._stationary_detected(now):
                    self._suppress()

            return self._event(None, request_vision=False)

        # Normal mode (DoA + classifier)
        if track is None:
            self._reset()
            return self._event(track, request_vision=False)

        # DoA stability bookkeeping
        if self._last_angle is not None and \
                abs(track.angle_deg - self._last_angle) <= self._stab:
            self._stable_hops += 1
        else:
            self._stable_hops = 0
        self._last_angle = track.angle_deg

        request_vision = False
        veh_p = self._last_cls.vehicle_prob if self._last_cls else 0.0

        if self.state == FusionState.IDLE:
            if self._suppressed:
                if veh_p < self._thr * 0.5:
                    self._suppressed = False
            elif self._stable_hops >= self._hold:
                self.state = FusionState.CANDIDATE
        elif self.state == FusionState.CANDIDATE:
            if self._stable_hops < self._hold // 2:
                self.state = FusionState.IDLE
            elif veh_p >= self._thr:
                self.state = FusionState.TRIGGERED
                self._triggered_since = now
                self._vision_requested_at = now
                self._vision_confirmed = False
                request_vision = True
        elif self.state == FusionState.TRIGGERED:
            if self._stationary_detected(now):
                self._suppress()
            else:
                if self._vision_requested_at is not None and \
                        now - self._vision_requested_at > self._vto:
                    self._vision_requested_at = now
                    request_vision = True
                if veh_p < self._thr * 0.5:
                    self.state = FusionState.CANDIDATE
                    self._triggered_since = None
        elif self.state == FusionState.CONFIRMED:
            if veh_p < self._thr * 0.5:
                self._reset()

        return self._event(track, request_vision)

    def _stationary_detected(self, now: float) -> bool:
        """True if current alert looks like a stationary machine, not a passing vehicle."""
        # Sustained: alert has been on too long to be a passing car
        if self._triggered_since is not None and \
                now - self._triggered_since > self._max_sustained_s:
            return True
        # Flat confidence: low variance over recent history → constant source
        if len(self._cls_window) == self._cls_window.maxlen:
            vals = list(self._cls_window)
            mean = sum(vals) / len(vals)
            if mean >= self._thr:
                variance = sum((v - mean) ** 2 for v in vals) / len(vals)
                if variance ** 0.5 < self._min_cls_std:
                    return True
        return False

    def _reset(self) -> None:
        self.state = FusionState.IDLE
        self._stable_hops = 0
        self._last_angle = None
        self._vision_requested_at = None
        self._vision_confirmed = False
        self._triggered_since = None
        self._suppressed = False

    def _suppress(self) -> None:
        """Mark as stationary source; arm suppression gate.

        cls_window is intentionally NOT cleared: the flat-confidence evidence
        stays in the window so any immediate re-trigger attempt is caught by
        the variance check on the very next on_track call.
        The suppression gate (_suppressed=True) then blocks further triggers
        until confidence genuinely drops, preventing rapid oscillation.
        """
        self._reset()
        self._suppressed = True

    def _event(self, track, request_vision: bool) -> FusionEvent:
        cls = self._last_cls
        return FusionEvent(
            state=self.state, track=track,
            sound_class=cls.top_class if cls else "unknown",
            confidence=cls.vehicle_prob if cls else 0.0,
            vision_confirmed=self._vision_confirmed,
            alert=self.state in (FusionState.TRIGGERED, FusionState.CONFIRMED),
            request_vision=request_vision)
