"""T4.2 AC: state transitions including vision timeout and hysteresis paths."""
import numpy as np

from avfusion.fusion.fsm import FusionState, FusionStateMachine
from avfusion.fusion.tracker import TrackState
from avfusion.inference.classifier import ClassifierResult


def track(angle=20.0, conf=0.9):
    return TrackState(track_id=1, angle_deg=angle, rate_dps=0.0,
                      confidence=conf, front_back_ambiguous=True,
                      front_log_odds=0.0, age_s=1.0, coasting=False)


def cls(vehicle_prob):
    probs = np.array([vehicle_prob, 0, 0, 0, 1 - vehicle_prob])
    return ClassifierResult(probs=probs, top_class="vehicle_engine",
                            top_prob=vehicle_prob, vehicle_prob=vehicle_prob,
                            degraded=False)


def make_fsm():
    return FusionStateMachine(candidate_hold_hops=4, doa_stability_deg=15.0,
                              trigger_threshold=0.6, vision_timeout_s=2.0)


def drive_to_candidate(fsm, t0=0.0):
    for i in range(6):
        ev = fsm.on_track(track(), now=t0 + i * 0.01)
    assert fsm.state == FusionState.CANDIDATE
    return ev


def test_idle_to_candidate_needs_stability():
    fsm = make_fsm()
    # unstable DoA (jumps > stability band) never leaves IDLE
    for i in range(10):
        fsm.on_track(track(angle=(-40.0) ** (i % 2)), now=i * 0.01)
    assert fsm.state == FusionState.IDLE
    drive_to_candidate(fsm, t0=1.0)


def test_candidate_to_triggered_requests_vision():
    fsm = make_fsm()
    drive_to_candidate(fsm)
    fsm.on_classifier(cls(0.8))
    ev = fsm.on_track(track(), now=0.1)
    assert fsm.state == FusionState.TRIGGERED
    assert ev.request_vision and ev.alert


def test_low_vehicle_prob_never_triggers():
    fsm = make_fsm()
    drive_to_candidate(fsm)
    fsm.on_classifier(cls(0.4))
    for i in range(20):
        ev = fsm.on_track(track(), now=0.1 + i * 0.01)
    assert fsm.state == FusionState.CANDIDATE
    assert not ev.alert


def test_vision_confirmation():
    fsm = make_fsm()
    drive_to_candidate(fsm)
    fsm.on_classifier(cls(0.8))
    fsm.on_track(track(), now=0.1)
    fsm.on_vision(confirmed=True)
    ev = fsm.on_track(track(), now=0.2)
    assert fsm.state == FusionState.CONFIRMED
    assert ev.vision_confirmed and ev.alert


def test_vision_rejection_keeps_audio_alert():
    """No visual correlate does NOT clear the alarm — rear-hypothesis vehicle
    is the most dangerous case (ARCHITECTURE §5)."""
    fsm = make_fsm()
    drive_to_candidate(fsm)
    fsm.on_classifier(cls(0.8))
    fsm.on_track(track(), now=0.1)
    fsm.on_vision(confirmed=False)
    ev = fsm.on_track(track(), now=0.2)
    assert fsm.state == FusionState.TRIGGERED
    assert ev.alert and not ev.vision_confirmed


def test_vision_timeout_rerequests():
    fsm = make_fsm()
    drive_to_candidate(fsm)
    fsm.on_classifier(cls(0.8))
    ev = fsm.on_track(track(), now=0.1)
    assert ev.request_vision
    ev = fsm.on_track(track(), now=0.5)        # within timeout: no re-request
    assert not ev.request_vision
    ev = fsm.on_track(track(), now=3.0)        # past 2 s timeout
    assert fsm.state == FusionState.TRIGGERED  # still alerting on audio alone
    assert ev.request_vision


def test_downward_hysteresis():
    fsm = make_fsm()
    drive_to_candidate(fsm)
    fsm.on_classifier(cls(0.8))
    fsm.on_track(track(), now=0.1)
    fsm.on_classifier(cls(0.4))                # above thr/2: hold TRIGGERED
    fsm.on_track(track(), now=0.2)
    assert fsm.state == FusionState.TRIGGERED
    fsm.on_classifier(cls(0.2))                # below thr/2: release
    fsm.on_track(track(), now=0.3)
    assert fsm.state == FusionState.CANDIDATE


def test_track_loss_resets_to_idle():
    fsm = make_fsm()
    drive_to_candidate(fsm)
    fsm.on_classifier(cls(0.8))
    fsm.on_track(track(), now=0.1)
    ev = fsm.on_track(None, now=0.2)
    assert fsm.state == FusionState.IDLE
    assert not ev.alert and ev.track is None
