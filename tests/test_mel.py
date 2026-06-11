"""T2.3 AC: patch shape [1,1,64,96], device-resident, standardized.

Guards the center=False frame-count regression: with n_fft (512) > win (400),
torchaudio derives the frame count from n_fft, so the rolling buffer must be
sized by n_fft or the patch comes out one frame short (64x95)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")

from avfusion.dsp.mel import MelPatchExtractor


def make_extractor():
    # device=None -> same auto-selection as production (cuda when available).
    # Forcing cpu here breaks on Jetson: the JP5 wheel's CPU GEMM is broken
    # on Carmel cores (NaNs), and production never runs this path on CPU.
    return MelPatchExtractor(capture_rate=48000, classifier_rate=16000,
                             n_mels=64, win_ms=25.0, hop_ms=10.0,
                             patch_frames=96, device=None)


def test_patch_shape_and_stats():
    ex = make_extractor()
    rng = np.random.default_rng(0)
    hop = 512
    while not ex.ready:
        ex.push_hop((rng.standard_normal((hop, 2)) * 3000).astype(np.int16))
    p = ex.patch()
    assert tuple(p.shape) == (1, 1, 64, 96)
    assert torch.isfinite(p).all()
    assert abs(float(p.mean())) < 1e-3          # standardized
    assert abs(float(p.std()) - 1.0) < 1e-2


def test_not_ready_until_full_patch():
    ex = make_extractor()
    ex.push_hop(np.zeros((512, 2), dtype=np.int16))
    assert not ex.ready


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_patch_stays_on_cuda():
    ex = MelPatchExtractor(device="cuda")
    rng = np.random.default_rng(1)
    while not ex.ready:
        ex.push_hop((rng.standard_normal((512, 2)) * 3000).astype(np.int16))
    assert ex.patch().device.type == "cuda"
