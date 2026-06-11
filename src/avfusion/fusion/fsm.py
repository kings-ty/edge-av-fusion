"""Fusion state machine: when does sound become an alert?

IDLE ──DoA stable for N hops──► CANDIDATE ──vehicle_prob ≥ θ──► TRIGGERED
TRIGGERED ──vision confirms──► CONFIRMED (highest-trust alert, localized)
TRIGGERED ──vision sees nothing & front hyp──► votes "back", stays TRIGGERED*
TRIGGERED ──vision timeout──► TRIGGERED* (audio-only alert, lower trust)
any state ──track lost──► IDLE (with hysteresis: CONFIRMED coasts max_coast_s)

The asymmetry is deliberate: an unconfirmed rear-hypothesis vehicle is the
*most* dangerous case for a blind-spot system, so absence of visual
confirmation escalates rather than suppresses the alert.
"""
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
                 vision_timeout_s: float = 2.0):
        self._hold = candidate_hold_hops
        self._stab = doa_stability_deg
        self._thr = trigger_threshold
        self._vto = vision_timeout_s

        self.state = FusionState.IDLE
        self._stable_hops = 0
        self._last_angle: Optional[float] = None
        self._last_cls: Optional[ClassifierResult] = None
        self._vision_requested_at: Optional[float] = None
        self._vision_confirmed = False

    def on_classifier(self, result: ClassifierResult) -> None:
        self._last_cls = result

    def on_vision(self, confirmed: bool) -> None:
        if self.state == FusionState.TRIGGERED:
            self._vision_confirmed = confirmed
            if confirmed:
                self.state = FusionState.CONFIRMED

    def on_track(self, track: Optional[TrackState],
                 now: Optional[float] = None) -> FusionEvent:
        now = time.monotonic() if now is None else now

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
            if self._stable_hops >= self._hold:
                self.state = FusionState.CANDIDATE
        elif self.state == FusionState.CANDIDATE:
            if self._stable_hops < self._hold // 2:
                self.state = FusionState.IDLE
            elif veh_p >= self._thr:
                self.state = FusionState.TRIGGERED
                self._vision_requested_at = now
                self._vision_confirmed = False
                request_vision = True
        elif self.state == FusionState.TRIGGERED:
            if self._vision_requested_at is not None and \
                    now - self._vision_requested_at > self._vto:
                # timeout: keep alerting on audio alone; re-request periodically
                self._vision_requested_at = now
                request_vision = True
            if veh_p < self._thr * 0.5:   # hysteresis on the way down
                self.state = FusionState.CANDIDATE
        elif self.state == FusionState.CONFIRMED:
            if veh_p < self._thr * 0.5:
                self.state = FusionState.IDLE
                self._reset()

        return self._event(track, request_vision)

    def _reset(self) -> None:
        self.state = FusionState.IDLE
        self._stable_hops = 0
        self._last_angle = None
        self._vision_requested_at = None
        self._vision_confirmed = False

    def _event(self, track, request_vision: bool) -> FusionEvent:
        cls = self._last_cls
        return FusionEvent(
            state=self.state, track=track,
            sound_class=cls.top_class if cls else "unknown",
            confidence=cls.vehicle_prob if cls else 0.0,
            vision_confirmed=self._vision_confirmed,
            alert=self.state in (FusionState.TRIGGERED, FusionState.CONFIRMED),
            request_vision=request_vision)
