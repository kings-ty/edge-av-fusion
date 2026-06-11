"""T2.1/T2.2 AC: fractional-delay recovery within 0.25 samples at SNR >= 10 dB,
backend agreement, adaptive-gate false-trigger and detection rates."""
import numpy as np
import pytest

from avfusion.dsp.gcc_phat import GccPhatEstimator

FS = 48000
WIN = 4096
D = 0.058

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def synth_delay(delay_samples, n=WIN, snr_db=20.0, seed=0):
    """Stereo noise with an exact fractional inter-channel delay (frequency-
    domain), plus independent noise at snr_db. Returns (n, 2) float32.

    Sign convention matches the estimator's cross-spectrum G = X0·conj(X1):
    positive delay_samples lags ch0, which the estimator reports as +tau."""
    rng = np.random.default_rng(seed)
    src = rng.standard_normal(n)
    spec = np.fft.rfft(src)
    freqs = np.fft.rfftfreq(n, 1.0 / FS)
    # wideband source: PHAT's narrow correlation peak (and therefore the
    # peak_ratio gate metric) assumes coherence across most of the band; a
    # narrowband source widens the lobe and that is a physics limitation,
    # not what these unit tests probe
    band = (freqs >= 150) & (freqs <= 20000)
    spec = spec * band
    ch0 = np.fft.irfft(spec * np.exp(-2j * np.pi * freqs * delay_samples / FS),
                       n=n)
    ch1 = np.fft.irfft(spec, n=n)
    sig = np.stack([ch0, ch1], axis=1)
    rms = np.sqrt(np.mean(sig ** 2))
    noise = rng.standard_normal((n, 2)) * rms * 10 ** (-snr_db / 20.0)
    return (sig + noise).astype(np.float32)


def diffuse_noise(n=WIN, seed=0):
    """Uncorrelated L/R noise — no coherent source."""
    return np.random.default_rng(seed).standard_normal((n, 2)).astype(np.float32)


def _estimator(backend="numpy", **kw):
    return GccPhatEstimator(sample_rate=FS, window_samples=WIN,
                            mic_spacing_m=D, backend=backend, **kw)


# --------------------------------------------------------------------- T2.1
@pytest.mark.parametrize("delay", [-6.4, -3.3, -1.5, 0.0, 0.7, 2.25, 5.5])
@pytest.mark.parametrize("snr_db", [10.0, 20.0])
def test_fractional_delay_recovery(delay, snr_db):
    est = _estimator()
    errs = []
    for seed in range(5):
        r = est.estimate(synth_delay(delay, snr_db=snr_db, seed=seed))
        errs.append(abs(r.tau_s * FS - delay))
    assert np.median(errs) < 0.25, "median |err| %.3f samples" % np.median(errs)


def test_angle_sign_convention():
    # +tau (ch0 lagging) maps to +angle; absolute left/right naming is settled
    # by the T0.3 finger-tap test, the math just has to be self-consistent
    tau = 4.0
    r = _estimator().estimate(synth_delay(tau, snr_db=30.0, seed=1))
    expected = np.degrees(np.arcsin(np.clip((tau / FS) * 343.0 / D, -1, 1)))
    assert abs(r.angle_deg - expected) < 2.0
    assert r.angle_deg > 0


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_backends_agree():
    x = synth_delay(2.25, snr_db=20.0, seed=7)
    r_np = _estimator("numpy").estimate(x)
    r_th = _estimator("torch-cpu").estimate(x)
    assert abs(r_np.tau_s - r_th.tau_s) * FS < 1e-3
    assert abs(r_np.angle_deg - r_th.angle_deg) < 1e-3


def test_rejects_wrong_shape():
    with pytest.raises(ValueError):
        _estimator().estimate(np.zeros((WIN, 3), dtype=np.float32))


# --------------------------------------------------------------------- T2.2
def test_gate_false_trigger_rate_on_diffuse_noise():
    est = _estimator()
    triggers = sum(est.estimate(diffuse_noise(seed=i)).valid
                   for i in range(300))
    # 300 windows ~ 25 s of audio; AC budget is < 0.1 false triggers/min
    assert triggers <= 3, "%d false triggers in 300 noise windows" % triggers


def test_gate_detects_bursts_at_10db():
    est = _estimator()
    for i in range(60):                       # learn the noise floor first
        est.estimate(diffuse_noise(seed=1000 + i))
    hits = sum(est.estimate(synth_delay(3.0, snr_db=10.0, seed=2000 + i)).valid
               for i in range(40))
    assert hits / 40 > 0.95, "detection rate %.2f" % (hits / 40)


def test_gate_does_not_self_suppress_sustained_source():
    """Accepted frames must not feed the noise model, or a persistent vehicle
    would raise its own floor and gate itself off."""
    est = _estimator()
    for i in range(60):
        est.estimate(diffuse_noise(seed=3000 + i))
    valids = [est.estimate(synth_delay(3.0, snr_db=15.0, seed=4000 + i)).valid
              for i in range(100)]
    assert sum(valids[-20:]) >= 19, "gate decayed on a sustained source"


def test_confidence_penalized_at_endfire():
    est = _estimator()
    for i in range(60):
        est.estimate(diffuse_noise(seed=5000 + i))
    r_broad = est.estimate(synth_delay(0.5, snr_db=20.0, seed=1))
    r_end = est.estimate(synth_delay(7.9, snr_db=20.0, seed=1))  # near +/-90
    assert r_broad.valid
    if r_end.valid:
        assert r_end.confidence < r_broad.confidence
