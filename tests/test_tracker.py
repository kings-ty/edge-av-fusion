"""T4.1 AC: hypothesis bookkeeping, yaw-fusion disambiguation within 2 s,
pass-by motion parity, coast/drop lifecycle."""
import numpy as np

from avfusion.dsp.gcc_phat import DoaEstimate
from avfusion.fusion.tracker import DoaTracker, _wrap


def est(angle, conf=1.0, valid=True):
    return DoaEstimate(angle_deg=angle, tau_s=0.0, confidence=conf,
                       valid=valid, peak_prominence=10.0, peak_ratio=3.0)


def make_tracker(**kw):
    return DoaTracker(alpha=0.35, beta=0.05, max_coast_s=1.5, **kw)


def test_birth_smoothing_and_stable_id():
    tr = make_tracker()
    track = None
    for i in range(50):
        track = tr.update(est(20.0 + np.random.default_rng(i).normal(0, 1)),
                          t=i * 0.01)
    assert track is not None
    assert track.track_id == 1
    assert abs(track.angle_deg - 20.0) < 3.0
    assert track.front_back_ambiguous          # no yaw change: cannot know


def test_large_jump_births_new_track():
    tr = make_tracker()
    for i in range(20):
        tr.update(est(10.0), t=i * 0.01)
    track = tr.update(est(80.0), t=0.21)       # > 40 deg gate
    assert track.track_id == 2
    assert track.front_back_ambiguous          # ambiguity resets with the track


def test_yaw_fusion_resolves_front_source():
    """Static source truly in FRONT at +30 deg world; robot yaws 0 -> +12 deg.
    Array-frame measurement for a front source is (world - yaw)."""
    tr = make_tracker()
    track = None
    for i in range(100):                       # 1 s at 100 Hz, yaw ramps to 12
        yaw = 12.0 * i / 99
        tr.update_yaw(yaw)
        track = tr.update(est(_wrap(30.0 - yaw)), t=i * 0.01)
    assert not track.front_back_ambiguous
    assert track.front_log_odds > 0            # front hypothesis won


def test_yaw_fusion_resolves_back_source():
    """Static source BEHIND: array angle mirrors as 180 - (world - yaw)."""
    tr = make_tracker()
    track = None
    world = 150.0                              # behind-left
    for i in range(100):
        yaw = 12.0 * i / 99
        tr.update_yaw(yaw)
        meas = _wrap(180.0 - (world - yaw))
        track = tr.update(est(meas), t=i * 0.01)
    assert not track.front_back_ambiguous
    assert track.front_log_odds < 0            # back hypothesis won


def test_vision_veto_votes_back():
    tr = make_tracker()
    for i in range(30):
        tr.update(est(25.0), t=i * 0.01)
    before = tr._front.log_lik
    tr.vision_vote(saw_something=False)
    assert tr._front.log_lik < before


def test_coast_then_drop():
    tr = make_tracker()
    for i in range(30):
        tr.update(est(10.0), t=i * 0.01)
    track = tr.update(est(0.0, valid=False), t=0.5)
    assert track is not None and track.coasting
    assert tr.update(est(0.0, valid=False), t=0.5 + 2.0) is None  # > max_coast
    assert tr._front is None and tr._back is None                 # bookkeeping


def test_invalid_before_birth_returns_none():
    tr = make_tracker()
    assert tr.update(est(0.0, valid=False), t=0.0) is None
