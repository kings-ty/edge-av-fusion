"""GCC-PHAT direction-of-arrival estimation for a 2-mic pair.

Numerics:
- Hann window + 2x zero-padded FFT (linear, not circular, correlation).
- beta-regularized PHAT weighting: G / (|G|^beta + eps). beta=1 is classic PHAT
  (pure phase, robust to reverb); beta<1 re-admits some magnitude weighting,
  useful in very low SNR.
- Lag search restricted to physically possible TDOAs (|tau| <= d/c), with
  parabolic sub-sample interpolation — essential at d=5.8 cm where tau_max is
  only ~8 samples even at 48 kHz (ARCHITECTURE §2.1).
- Adaptive noise-floor gate: running median/MAD of peak prominence; a peak is
  accepted only if it is a statistical outlier vs the recent diffuse-noise
  background AND dominates the secondary peak. This is what rejects wind,
  ego-noise and reverb ghosts.

Backends: "numpy" (default — wins at this problem size, see ARCHITECTURE §2.3),
"torch-cpu", "torch-cuda" (kept so bench/latency.py can prove that claim).

Sign convention: positive angle = source toward mic channel 0 (left).
Swap `mic_order` if your wiring disagrees with the finger-tap test (T0.3).
"""
import collections
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:  # numpy backend works without torch
    torch = None
    _HAS_TORCH = False


@dataclass
class DoaEstimate:
    angle_deg: float        # [-90, +90], + = left
    tau_s: float            # signed TDOA
    confidence: float       # [0,1]: gate z-score x endfire geometry penalty
    valid: bool             # passed the adaptive gate
    peak_prominence: float  # main peak / noise floor
    peak_ratio: float       # main peak / secondary peak


class _AdaptiveGate:
    """Median/MAD outlier gate over recent peak prominences."""

    def __init__(self, history: int, z_threshold: float, min_peak_ratio: float):
        self._hist = collections.deque(maxlen=history)
        self._z_thr = z_threshold
        self._ratio_thr = min_peak_ratio

    def evaluate(self, prominence: float, peak_ratio: float):
        if len(self._hist) >= 16:
            arr = np.asarray(self._hist)
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med)))
            z = (prominence - med) / (1.4826 * mad + 1e-9)
        else:
            z = 0.0  # not enough history: refuse to trigger, keep learning
        accepted = z > self._z_thr and peak_ratio > self._ratio_thr
        # only non-accepted frames update the noise model, so a sustained
        # source does not inflate its own floor and gate itself off
        if not accepted:
            self._hist.append(prominence)
        return accepted, z


class GccPhatEstimator:
    def __init__(self, sample_rate: int, window_samples: int,
                 mic_spacing_m: float, speed_of_sound: float = 343.0,
                 backend: str = "numpy", phat_beta: float = 1.0,
                 gate_history: int = 200, gate_z: float = 4.0,
                 gate_min_ratio: float = 1.8):
        self.fs = sample_rate
        self.n = window_samples
        self.d = mic_spacing_m
        self.c = speed_of_sound
        self.beta = phat_beta
        self.nfft = 2 * window_samples
        self.max_lag = int(math.ceil(mic_spacing_m / speed_of_sound * sample_rate)) + 2
        self._gate = _AdaptiveGate(gate_history, gate_z, gate_min_ratio)

        self.backend = backend
        if backend == "numpy":
            self._win = np.hanning(window_samples).astype(np.float32)
        elif backend in ("torch-cpu", "torch-cuda"):
            if not _HAS_TORCH:
                raise RuntimeError("torch backend requested but torch not installed")
            self._dev = "cuda" if backend == "torch-cuda" else "cpu"
            self._win_t = torch.hann_window(window_samples, periodic=False,
                                            dtype=torch.float32, device=self._dev)
        else:
            raise ValueError("unknown backend %r" % backend)

    # ---------------------------------------------------------------- core
    def _cross_correlate_numpy(self, x: np.ndarray) -> np.ndarray:
        xw = x.astype(np.float32) * self._win[:, None]
        X = np.fft.rfft(xw, n=self.nfft, axis=0)
        G = X[:, 0] * np.conj(X[:, 1])
        G /= np.abs(G) ** self.beta + 1e-12
        cc = np.fft.irfft(G, n=self.nfft)
        return np.concatenate([cc[-self.max_lag:], cc[:self.max_lag + 1]])

    def _cross_correlate_torch(self, x: np.ndarray) -> np.ndarray:
        xt = torch.as_tensor(x, dtype=torch.float32).to(self._dev, non_blocking=True)
        xw = xt * self._win_t[:, None]
        X = torch.fft.rfft(xw, n=self.nfft, dim=0)
        G = X[:, 0] * torch.conj(X[:, 1])
        G = G / (torch.abs(G) ** self.beta + 1e-12)
        cc = torch.fft.irfft(G, n=self.nfft)
        cc = torch.cat([cc[-self.max_lag:], cc[:self.max_lag + 1]])
        return cc.cpu().numpy()  # 2*max_lag+1 floats: copy cost is noise

    # ----------------------------------------------------------------- api
    def estimate(self, stereo_window: np.ndarray) -> DoaEstimate:
        """stereo_window: (window_samples, 2) int16/float."""
        if stereo_window.shape != (self.n, 2):
            raise ValueError("expected (%d,2), got %s" % (self.n, stereo_window.shape))
        if self.backend == "numpy":
            cc = self._cross_correlate_numpy(stereo_window)
        else:
            cc = self._cross_correlate_torch(stereo_window)

        k = int(np.argmax(cc))
        peak = float(cc[k])

        # parabolic sub-sample interpolation around the peak
        if 0 < k < len(cc) - 1:
            y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
            denom = y0 - 2 * y1 + y2
            delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
            delta = float(np.clip(delta, -0.5, 0.5))
        else:
            delta = 0.0
        tau = (k + delta - self.max_lag) / self.fs

        # peak quality metrics
        noise_floor = float(np.median(np.abs(cc))) + 1e-12
        prominence = peak / noise_floor
        guard = np.ones(len(cc), dtype=bool)
        guard[max(0, k - 2):k + 3] = False
        secondary = float(np.max(cc[guard])) if guard.any() else 1e-12
        peak_ratio = peak / max(secondary, 1e-12)

        sin_arg = float(np.clip(self.c * tau / self.d, -1.0, 1.0))
        angle = math.degrees(math.asin(sin_arg))

        accepted, z = self._gate.evaluate(prominence, peak_ratio)
        # endfire penalty: dtheta/dtau ~ 1/cos(theta) -> shrink confidence there
        geom = math.cos(math.radians(angle))
        confidence = float(np.clip(z / (2 * self._gate._z_thr), 0.0, 1.0) * max(geom, 0.1)) \
            if accepted else 0.0

        return DoaEstimate(angle_deg=angle, tau_s=tau, confidence=confidence,
                           valid=accepted, peak_prominence=prominence,
                           peak_ratio=peak_ratio)
