"""DoA temporal tracking + front/back ambiguity resolution (ARCHITECTURE §5).

A 2-mic linear array measures one angle theta in [-90, +90] that is consistent
with TWO world hypotheses: source in front at theta, or behind at (180-theta)
mirrored. We maintain BOTH hypotheses explicitly and let evidence vote:

1. Yaw fusion: when the robot yaws by dpsi, a front source moves by -dpsi in
   array frame, a back source by +dpsi. Each hypothesis predicts the next
   measurement; prediction error updates its log-likelihood.
2. Motion parity: a passing vehicle sweeps theta monotonically; front/back
   hypotheses imply opposite world-frame motion. Falls out of (1)'s residuals.
3. Vision veto (applied by the FSM): camera sees nothing at a confident front
   angle -> back hypothesis gets a strong vote.

Smoothing: alpha-beta filter — appropriate order for "angle + angular rate"
with 10 ms updates; a full Kalman adds tuning surface, not accuracy, here.
"""
import math
from dataclasses import dataclass, field
from typing import Optional

from ..dsp.gcc_phat import DoaEstimate


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


@dataclass
class TrackState:
    track_id: int
    angle_deg: float            # smoothed array-frame angle [-90, +90]
    rate_dps: float             # smoothed angular rate
    confidence: float
    front_back_ambiguous: bool
    front_log_odds: float       # >0: front more likely
    age_s: float
    coasting: bool


@dataclass
class _Hypothesis:
    world_angle: float          # robot-frame azimuth at track birth heading
    log_lik: float = 0.0


class DoaTracker:
    def __init__(self, alpha: float = 0.35, beta: float = 0.05,
                 max_coast_s: float = 1.5, ambiguity_margin: float = 2.0):
        self._alpha = alpha
        self._beta = beta
        self._max_coast = max_coast_s
        self._margin = ambiguity_margin   # |log-odds| to declare disambiguated
        self._next_id = 1

        self._track: Optional[TrackState] = None
        self._front: Optional[_Hypothesis] = None
        self._back: Optional[_Hypothesis] = None
        self._yaw_ref: Optional[float] = None   # robot yaw at last update
        self._yaw: float = 0.0
        self._last_t: Optional[float] = None
        self._birth_t: float = 0.0

    # ------------------------------------------------------------ inputs
    def update_yaw(self, yaw_deg: float) -> None:
        """Feed robot heading from /odom or IMU (world frame, degrees)."""
        self._yaw = yaw_deg

    def vision_vote(self, saw_something: bool, weight: float = 3.0) -> None:
        """FSM reports the vision check at the front-hypothesis angle."""
        if self._front is None:
            return
        self._front.log_lik += weight if saw_something else -weight

    # ------------------------------------------------------------ update
    def update(self, est: DoaEstimate, t: float) -> Optional[TrackState]:
        if not est.valid:
            return self._coast(t)

        if self._track is None:
            self._birth(est, t)
        else:
            dt = max(t - self._last_t, 1e-3) if self._last_t else 0.01
            tr = self._track

            # gating: a jump farther than physically plausible starts a new track
            pred = tr.angle_deg + tr.rate_dps * dt
            if abs(_wrap(est.angle_deg - pred)) > 40.0:
                self._birth(est, t)
            else:
                resid = est.angle_deg - pred
                tr.angle_deg = float(min(90.0, max(-90.0,
                    pred + self._alpha * resid)))
                tr.rate_dps += self._beta * resid / dt
                tr.confidence = 0.7 * tr.confidence + 0.3 * est.confidence
                tr.age_s = t - (self._birth_t)
                tr.coasting = False
                self._score_hypotheses(est, dt)
        self._last_t = t
        self._yaw_ref = self._yaw
        return self._publish()

    # ---------------------------------------------------------- internals
    def _birth(self, est: DoaEstimate, t: float) -> None:
        self._track = TrackState(
            track_id=self._next_id, angle_deg=est.angle_deg, rate_dps=0.0,
            confidence=est.confidence, front_back_ambiguous=True,
            front_log_odds=0.0, age_s=0.0, coasting=False)
        self._next_id += 1
        self._birth_t = t
        # world-frame hypotheses frozen at birth heading
        self._front = _Hypothesis(world_angle=_wrap(self._yaw + est.angle_deg))
        self._back = _Hypothesis(world_angle=_wrap(self._yaw + 180.0 - est.angle_deg))
        self._yaw_ref = self._yaw

    def _score_hypotheses(self, est: DoaEstimate, dt: float) -> None:
        """Each hypothesis predicts the array-frame angle from its (assumed
        static) world angle and the current yaw; smaller residual wins votes.
        For a static source this is exact; for a moving one, hypothesis world
        angles drift slowly while the wrong hypothesis' residual grows ~2x the
        yaw change — the parity argument."""
        if self._front is None or self._back is None:
            return
        sigma = 8.0  # deg, measurement noise scale
        for hyp, sign in ((self._front, 1.0), (self._back, -1.0)):
            rel = _wrap(hyp.world_angle - self._yaw)
            if sign < 0:
                rel = _wrap(180.0 - rel)
            pred = max(-90.0, min(90.0, rel))
            r = est.angle_deg - pred
            hyp.log_lik += -(r * r) / (2 * sigma * sigma) * est.confidence
            # slow world-angle adaptation lets hypotheses follow a moving source
            hyp.world_angle = _wrap(hyp.world_angle + 0.1 * r * (1.0 if sign > 0 else -1.0))

    def _coast(self, t: float) -> Optional[TrackState]:
        if self._track is None or self._last_t is None:
            return None
        if t - self._last_t > self._max_coast:
            self._track = None
            self._front = self._back = None
            return None
        self._track.coasting = True
        return self._publish()

    def _publish(self) -> Optional[TrackState]:
        tr = self._track
        if tr is None:
            return None
        if self._front is not None and self._back is not None:
            tr.front_log_odds = self._front.log_lik - self._back.log_lik
            tr.front_back_ambiguous = abs(tr.front_log_odds) < self._margin
        return tr
