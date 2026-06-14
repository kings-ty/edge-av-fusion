#!/usr/bin/env python3
"""Estimate microphone array spacing from a hand-clap recording.

Usage:
  # Hold device, clap sharply from exactly 90° to one side
  python tools/calibrate_mic.py Edge-materials/Left.m4a --side left

Output: mic_spacing_m → paste into config/pipeline.yaml
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def load_stereo(path: str, target_rate: int = 48000) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-ar", str(target_rate), "-ac", "2",
         "-f", "wav", tmp_path],
        check=True, capture_output=True)
    import wave
    with wave.open(tmp_path, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
        data = (np.frombuffer(raw, dtype=np.int16)
                .reshape(-1, 2).astype(np.float32) / 32768.0)
    Path(tmp_path).unlink()
    return data


def gcc_phat(ch0: np.ndarray, ch1: np.ndarray, max_lag: int):
    n = len(ch0)
    nfft = 1 << (2 * n - 1).bit_length()
    X0 = np.fft.rfft(ch0, n=nfft)
    X1 = np.fft.rfft(ch1, n=nfft)
    G = X0 * np.conj(X1)
    G /= np.abs(G) + 1e-10
    corr = np.fft.irfft(G, n=nfft)
    corr = np.concatenate([corr[-max_lag:], corr[:max_lag + 1]])
    peak_idx = int(np.argmax(corr))
    lag = peak_idx - max_lag
    return lag, float(corr[peak_idx])


def find_clap(stereo: np.ndarray, fs: int, window_ms: float = 3.0) -> int:
    win = max(1, int(fs * window_ms / 1000))
    mono = np.abs(stereo).sum(axis=1)
    energy = np.convolve(mono ** 2, np.ones(win) / win, mode="same")
    return int(np.argmax(energy))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", help="stereo recording (m4a / mp4 / wav)")
    ap.add_argument("--side", choices=["left", "right"], default="left",
                    help="side the clap came from (default: left)")
    ap.add_argument("--rate", type=int, default=48000)
    ap.add_argument("--speed", type=float, default=343.0, help="speed of sound m/s")
    ap.add_argument("--window-ms", type=float, default=20.0,
                    help="GCC-PHAT window size around clap (ms)")
    ap.add_argument("--max-spacing-m", type=float, default=0.30,
                    help="upper bound for mic spacing search (m)")
    args = ap.parse_args()

    print("Loading %s ..." % args.audio)
    stereo = load_stereo(args.audio, args.rate)
    duration = len(stereo) / args.rate
    print("  %.2f s  |  %d samples  |  2 ch @ %d Hz" % (duration, len(stereo), args.rate))

    clap_idx = find_clap(stereo, args.rate)
    print("  Loudest impulse at %.1f ms (sample %d)" % (clap_idx / args.rate * 1000, clap_idx))

    half = int(args.rate * args.window_ms / 1000)
    seg = stereo[max(0, clap_idx - half): min(len(stereo), clap_idx + half)]
    ch0, ch1 = seg[:, 0], seg[:, 1]

    max_lag = int(args.max_spacing_m / args.speed * args.rate) + 2
    lag, peak = gcc_phat(ch0, ch1, max_lag)
    tdoa_s = lag / args.rate
    spacing = abs(tdoa_s) * args.speed

    # Print full correlation profile so mic axis can be diagnosed
    nfft = 1 << (2 * len(ch0) - 1).bit_length()
    X0 = np.fft.rfft(ch0, n=nfft)
    X1 = np.fft.rfft(ch1, n=nfft)
    G = X0 * np.conj(X1)
    G /= np.abs(G) + 1e-10
    corr_full = np.fft.irfft(G, n=nfft)
    corr_slice = np.concatenate([corr_full[-max_lag:], corr_full[:max_lag + 1]])

    print("\n  Cross-correlation profile (lag → value):")
    bar_max = max(abs(corr_slice))
    for i, v in enumerate(corr_slice):
        l = i - max_lag
        bar = int(abs(v) / bar_max * 30)
        sign = "+" if v >= 0 else "-"
        if abs(v) >= bar_max * 0.15:   # print only notable lags
            print("    lag %+3d  %s%s  %.3f" % (l, sign, "#" * bar, abs(v)))

    print("\n  GCC-PHAT  lag = %+d samples (%.1f µs)  peak = %.3f" % (
        lag, tdoa_s * 1e6, peak))

    if lag == 0:
        print("\n  lag=0: 두 마이크에 소리가 동시에 도달했습니다.")
        print("  → 소리가 마이크 배치 축의 수직 방향에서 왔거나,")
        print("    폰 마이크가 상하 배치라면 앞/뒤로 소리를 내보세요.")
        print("\n  시도해볼 것:")
        print("    1) 폰을 가로(landscape)로 들고 왼쪽에서 손뼉")
        print("    2) 폰 정면 30cm 앞에서 손뼉 (앞/뒤 축 테스트)")
        print("    3) 폰 뒤에서 손뼉 (앞 vs 뒤)")
        sys.exit(1)

    # Channel orientation
    # G = X0*conj(X1): positive lag → ch0 leads → source closer to ch0
    # For left clap: if ch0 is left mic, lag < 0 (ch1 leads? no...)
    # Let's just report what we observe
    leading = "ch0" if lag > 0 else "ch1"
    print("  Leading channel: %s  (source is closer to that mic)" % leading)
    expected_lag_sign = +1 if args.side == "left" else -1
    if np.sign(lag) == expected_lag_sign:
        print("  Sign consistent with '%s' clap → ch0 is the LEFT mic" % args.side)
    else:
        print("  Sign inverted for '%s' clap → ch0 is the RIGHT mic" % args.side)

    print("\n  ✓  Estimated mic spacing: %.4f m  (%.1f cm)" % (spacing, spacing * 100))

    print("""
─── Copy into config/pipeline.yaml ─────────────────────────────────
doa:
  mic_spacing_m: %.4f   # calibrated from %s
─────────────────────────────────────────────────────────────────────
""" % (spacing, Path(args.audio).name))


if __name__ == "__main__":
    main()
