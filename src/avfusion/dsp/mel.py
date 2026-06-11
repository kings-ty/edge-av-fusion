"""GPU log-mel extraction for the classifier branch.

Flow: 48 kHz stereo hops → mono mix → polyphase decimation to 16 kHz →
rolling 0.96 s buffer → log-mel patch [1,1,64,96], device-resident so the
tensor handed to TensorRT never round-trips through host memory.

Honest note (ARCHITECTURE §3.3): these tensors are ~24 KB — the GPU win here
is avoiding Python-side STFT cost at the 1 Hz classify rate, not bandwidth.
torchaudio's MelSpectrogram keeps everything in one cuFFT/cuBLAS graph.
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
        self.capture_rate = capture_rate
        self.rate = classifier_rate
        self.patch_frames = patch_frames
        self.hop = int(classifier_rate * hop_ms / 1000)      # 160
        self.win = int(classifier_rate * win_ms / 1000)      # 400
        self.n_fft = 512
        # center=False frame count is governed by n_fft (> win here), not win
        self.patch_samples = self.hop * (patch_frames - 1) + max(self.win, self.n_fft)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._resampler = torchaudio.transforms.Resample(
            capture_rate, classifier_rate).to(self.device)
        self._melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=classifier_rate, n_fft=self.n_fft, win_length=self.win,
            hop_length=self.hop, n_mels=n_mels, center=False, power=2.0,
        ).to(self.device)

        # rolling mono buffer at classifier rate, device-resident
        self._buf = torch.zeros(self.patch_samples, device=self.device)
        self._filled = 0

    @torch.no_grad()
    def push_hop(self, stereo_hop: np.ndarray) -> None:
        """Feed one capture hop (hop_samples, 2) int16 at capture_rate."""
        mono = torch.as_tensor(stereo_hop, dtype=torch.float32).mean(dim=1) / 32768.0
        mono = mono.to(self.device, non_blocking=True)
        mono = self._resampler(mono)
        n = mono.shape[0]
        if n >= self.patch_samples:
            self._buf = mono[-self.patch_samples:]
        else:
            self._buf = torch.cat([self._buf[n:], mono])
        self._filled = min(self._filled + n, self.patch_samples)

    @property
    def ready(self) -> bool:
        return self._filled >= self.patch_samples

    @torch.no_grad()
    def patch(self) -> "torch.Tensor":
        """Return log-mel patch [1, 1, n_mels, patch_frames] on self.device."""
        mel = self._melspec(self._buf)                     # [n_mels, frames]
        mel = torch.log(mel + 1e-6)
        mel = mel[:, : self.patch_frames]
        # per-patch standardization: classifier was trained on normalized mels
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        return mel.unsqueeze(0).unsqueeze(0).contiguous()
