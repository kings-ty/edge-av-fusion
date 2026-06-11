"""GPU log-mel extraction for the classifier branch.

Flow: 48 kHz stereo hops → CPU mono ring (cheap numpy, no GPU touch) →
at classify time (1 Hz): one upload → polyphase decimation to 16 kHz →
log-mel patch [1,1,64,96], device-resident for the TensorRT hand-off.

Why hop accumulation is CPU-side: an earlier revision resampled every 10.7 ms
hop on the GPU; at 30 W those ~94 small launches/s blew the DSP thread's hop
budget and dropped ~24% of real-time audio (caught by tools/analyze_clip.py
on real street footage, not by the unpaced synthetic bench). Batching the GPU
work to one call per classification leaves per-hop cost at a numpy memmove.

Honest note (ARCHITECTURE §3.3): these tensors are ~100 KB — the GPU win here
is one fused cuFFT/cuBLAS mel graph per second, not bandwidth.
"""
from typing import Optional

import numpy as np

try:
    import torch
    import torchaudio
    _HAS_TORCH = True
except ImportError:
    torch = None
    torchaudio = None
    _HAS_TORCH = False


class MelPatchExtractor:
    def __init__(self, capture_rate: int = 48000, classifier_rate: int = 16000,
                 n_mels: int = 64, win_ms: float = 25.0, hop_ms: float = 10.0,
                 patch_frames: int = 96, device: Optional[str] = None):
        if not _HAS_TORCH:
            raise RuntimeError("MelPatchExtractor requires torch + torchaudio "
                               "(scripts/setup_env.sh installs the JetPack wheel)")
        if capture_rate % classifier_rate:
            raise ValueError("capture_rate must be an integer multiple of "
                             "classifier_rate (%d %% %d != 0)"
                             % (capture_rate, classifier_rate))
        self.capture_rate = capture_rate
        self.rate = classifier_rate
        self.patch_frames = patch_frames
        self.hop = int(classifier_rate * hop_ms / 1000)      # 160
        self.win = int(classifier_rate * win_ms / 1000)      # 400
        self.n_fft = 512
        # center=False frame count is governed by n_fft (> win here), not win
        self.patch_samples = self.hop * (patch_frames - 1) + max(self.win, self.n_fft)
        self._decim = capture_rate // classifier_rate
        self._cap_samples = self.patch_samples * self._decim

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._resampler = torchaudio.transforms.Resample(
            capture_rate, classifier_rate).to(self.device)
        self._melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=classifier_rate, n_fft=self.n_fft, win_length=self.win,
            hop_length=self.hop, n_mels=n_mels, center=False, power=2.0,
        ).to(self.device)

        # rolling mono buffer at CAPTURE rate, host-side: per-hop cost is a
        # memmove, the GPU is touched once per patch() call only
        self._buf = np.zeros(self._cap_samples, dtype=np.float32)
        self._filled = 0

    def push_hop(self, stereo_hop: np.ndarray) -> None:
        """Feed one capture hop (hop_samples, 2) int16 at capture_rate."""
        mono = stereo_hop.astype(np.float32).mean(axis=1) / 32768.0
        n = mono.shape[0]
        if n >= self._cap_samples:
            self._buf[:] = mono[-self._cap_samples:]
        else:
            self._buf[:-n] = self._buf[n:]
            self._buf[-n:] = mono
        self._filled = min(self._filled + n, self._cap_samples)

    @property
    def ready(self) -> bool:
        return self._filled >= self._cap_samples

    def warmup(self) -> "torch.Tensor":
        """Run one throwaway patch to absorb first-call costs (CUDA context,
        cuFFT plans) BEFORE realtime audio starts: at low power modes these
        take seconds and would overflow the ring buffer mid-stream. Returns
        the dummy patch so the caller can warm its consumer too."""
        filled = self._filled
        self._filled = self._cap_samples
        try:
            return self.patch()
        finally:
            self._filled = filled

    @torch.no_grad()
    def patch(self) -> "torch.Tensor":
        """Return log-mel patch [1, 1, n_mels, patch_frames] on self.device."""
        mono = torch.from_numpy(self._buf).to(self.device, non_blocking=True)
        mono = self._resampler(mono)[: self.patch_samples]
        mel = self._melspec(mono)                          # [n_mels, frames]
        mel = torch.log(mel + 1e-6)
        mel = mel[:, : self.patch_frames]
        # per-patch standardization: classifier was trained on normalized mels
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        return mel.unsqueeze(0).unsqueeze(0).contiguous()
