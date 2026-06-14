"""T5.1 latency harness + T5.2 GCC backend duel.

Per-stage spans come from the *production* pipeline's StageTimes — benchmarks
that exercise a different code path than production measure nothing. GPU spans
are correct by construction: TrtEngine.infer() synchronizes its stream before
returning, and the torch-cuda GCC backend ends in a .cpu() copy (an implicit
sync), so a closed span always means "GPU finished", not "kernel launched".

Synthetic source: fractional-delay stereo noise (a virtual source at a known
angle) generated on demand, no realtime pacing — we measure compute, not the
10.7 ms hop period we already know. Hardware modes (--source alsa/file) keep
pacing and additionally expose ring-buffer dwell.

Usage:
  python3 -m avfusion.bench.latency --hops 12000
  python3 -m avfusion.bench.latency --duel            # numpy vs torch-{cpu,cuda}
  python3 -m avfusion.bench.latency --gcc-backend torch-cuda --hops 4000
"""
import argparse
import time
from dataclasses import replace
from typing import Optional

import numpy as np

from . import markdown_table, percentiles, write_report
from ..audio.source_base import AudioChunk, AudioSource
from ..config import load_config
from ..dsp.gcc_phat import GccPhatEstimator
from ..pipeline import Pipeline

WARMUP_HOPS = 500


def synth_stereo(n: int, fs: int, angle_deg: float, mic_spacing_m: float,
                 snr_db: float = 20.0, seed: int = 0,
                 lowcut_hz: float = 150.0, highcut_hz: float = 16000.0) -> np.ndarray:
    """Band-limited noise from a virtual source at `angle_deg`, delayed across
    the pair in the frequency domain (exact fractional delay), plus diffuse
    (uncorrelated) noise at the given SNR. Returns (n, 2) int16."""
    rng = np.random.default_rng(seed)
    tau = mic_spacing_m * np.sin(np.radians(angle_deg)) / 343.0
    src = rng.standard_normal(n)
    spec = np.fft.rfft(src)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    band = (freqs >= lowcut_hz) & (freqs <= highcut_hz)
    spec *= band
    # delay ch0: estimator's G = X0·conj(X1) reports a lagging ch0 as +tau,
    # so a virtual source at +angle comes back as +angle
    ch0 = np.fft.irfft(spec * np.exp(-2j * np.pi * freqs * tau), n=n)
    ch1 = np.fft.irfft(spec, n=n)
    sig = np.stack([ch0, ch1], axis=1)
    sig /= np.abs(sig).max() + 1e-12
    noise = rng.standard_normal((n, 2))
    noise *= np.sqrt(np.mean(sig ** 2)) / np.sqrt(np.mean(noise ** 2)) \
        * 10 ** (-snr_db / 20.0)
    out = sig + noise
    return (out / (np.abs(out).max() + 1e-12) * 0.5 * 32767).astype(np.int16)


class SyntheticSource(AudioSource):
    """Serves pre-generated hops with no pacing; `finished` after n_hops."""

    def __init__(self, sample_rate: int, channels: int, hop_samples: int,
                 n_hops: int, angle_deg: float = 30.0, snr_db: float = 15.0):
        super().__init__(hop_samples, channels, capacity_hops=4)
        seconds = 4
        self._data = synth_stereo(sample_rate * seconds, sample_rate,
                                  angle_deg, 0.058, snr_db)
        self._hop = hop_samples
        self._fs = sample_rate
        self._n_hops = n_hops
        self._i = 0
        self.finished = False

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def read(self, timeout: Optional[float] = 1.0) -> Optional[AudioChunk]:
        if self._i >= self._n_hops:
            self.finished = True
            return None
        off = (self._i * self._hop) % (len(self._data) - self._hop)
        self._i += 1
        return AudioChunk(samples=self._data[off:off + self._hop],
                          pts=self._i * self._hop / self._fs)


# ----------------------------------------------------------------- pipeline
def run_pipeline_bench(args) -> None:
    cfg = load_config(args.config)
    if args.gcc_backend:
        cfg = replace(cfg, doa=replace(cfg.doa, backend=args.gcc_backend))

    if args.source == "synthetic":
        src = SyntheticSource(cfg.audio.sample_rate, cfg.audio.channels,
                              cfg.audio.hop_samples, args.hops + WARMUP_HOPS)
    elif args.source == "file":
        from ..audio.gst_file_source import GstFileSource
        media = args.media
        if not media:
            import os
            if os.path.isdir("Edge-materials"):
                clips = sorted([f for f in os.listdir("Edge-materials") if f.endswith(".mp4")])
                if clips:
                    media = clips[0]
                    print("No media specified, defaulting to: %s" % media)
        if not media:
            raise SystemExit("Error: --source file requires --media or a non-empty Edge-materials/")

        src = GstFileSource(media, cfg.audio.sample_rate,
                            cfg.audio.channels, cfg.audio.hop_samples,
                            cfg.audio.ring_capacity_hops)
    else:
        from ..audio.gst_alsa_source import GstAlsaSource
        src = GstAlsaSource(cfg.audio.alsa_device, cfg.audio.sample_rate,
                            cfg.audio.channels, cfg.audio.hop_samples,
                            cfg.audio.ring_capacity_hops)

    pipe = Pipeline(cfg, src, enable_classifier=not args.no_classifier,
                    keep_stage_times=True)
    t0 = time.monotonic()
    pipe.start()
    try:
        while pipe._thread.is_alive() and pipe.stats.hops < args.hops + WARMUP_HOPS:
            pipe._thread.join(timeout=0.5)
            if args.source != "synthetic" and time.monotonic() - t0 > args.max_seconds:
                break
    finally:
        pipe.stop()

    times = pipe.stats.stage_times[WARMUP_HOPS:]
    if not times:
        raise SystemExit("no samples collected after warmup (%d hops total)"
                         % pipe.stats.hops)

    stages = [
        ("capture→read (dwell)", [t.capture_to_read_ns for t in times]),
        ("GCC-PHAT (%s)" % cfg.doa.backend, [t.gcc_ns for t in times]),
        ("tracker", [t.track_ns for t in times]),
        ("mel (GPU)", [t.mel_ns for t in times if t.mel_ns]),
        ("TRT classifier", [t.trt_ns for t in times if t.trt_ns]),
        ("fusion FSM", [t.fusion_ns for t in times]),
        ("end-to-end hop", [t.e2e_ns for t in times]),
    ]
    payload = {"config": {"source": args.source, "backend": cfg.doa.backend,
                          "classifier": not args.no_classifier,
                          "degraded": pipe.stats.degraded,
                          "hops": len(times), "warmup": WARMUP_HOPS,
                          "dropped_hops": pipe.stats.dropped_hops},
               "stages_ms": {}}
    rows = []
    for name, ns in stages:
        p = percentiles([v / 1e6 for v in ns])
        payload["stages_ms"][name] = p
        rows.append([name, p["n"], "%.3f" % p["mean"], "%.3f" % p["p50"],
                     "%.3f" % p["p95"], "%.3f" % p["p99"]])
    md = "# Per-stage latency (ms) — source=%s, %d hops%s\n\n%s" % (
        args.source, len(times),
        " (DEGRADED: no classifier engine)" if pipe.stats.degraded else "",
        markdown_table(["stage", "n", "mean", "p50", "p95", "p99"], rows))
    path = write_report("latency", payload, md, tag=args.tag)
    print(md)
    print("\nwrote", path)


# --------------------------------------------------------------------- duel
def run_gcc_duel(args) -> None:
    """numpy vs torch-cpu vs torch-cuda over a window-size sweep (T5.2).
    This is the measured proof of ARCHITECTURE §2.3: the GPU loses on small
    FFTs because of launch overhead, and the table shows where the crossover
    would be (if it exists on this board at all)."""
    cfg = load_config(args.config)
    backends = ["numpy"]
    try:
        import torch
        backends.append("torch-cpu")
        if torch.cuda.is_available():
            backends.append("torch-cuda")
    except ImportError:
        print("torch not installed: duel reduced to numpy only")

    windows = [1024, 2048, 4096, 8192, 16384]
    reps = args.duel_reps
    rows, payload = [], {"reps": reps, "results_ms": {}}
    for win in windows:
        x = synth_stereo(win, cfg.audio.sample_rate, 30.0, cfg.doa.mic_spacing_m)
        row = [str(win)]
        for backend in backends:
            est = GccPhatEstimator(
                sample_rate=cfg.audio.sample_rate, window_samples=win,
                mic_spacing_m=cfg.doa.mic_spacing_m, backend=backend)
            for _ in range(20):           # warmup: allocator, cuFFT plans
                est.estimate(x)
            t0 = time.perf_counter_ns()
            for _ in range(reps):
                est.estimate(x)           # torch-cuda path ends in .cpu(): syncs
            ms = (time.perf_counter_ns() - t0) / reps / 1e6
            payload["results_ms"]["%s/%d" % (backend, win)] = ms
            row.append("%.3f" % ms)
        rows.append(row)

    md = ("# GCC-PHAT backend duel — mean ms/call (%d reps)\n\n%s\n\n"
          "Reading: if torch-cuda never beats numpy at production window size "
          "(4096), the GPU port is not worth its complexity. That conclusion "
          "is the point of this table." % (
              reps, markdown_table(["window"] + backends, rows)))
    path = write_report("gcc_duel", payload, md, tag=args.tag)
    print(md)
    print("\nwrote", path)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hops", type=int, default=10000,
                    help="measured hops after %d warmup" % WARMUP_HOPS)
    ap.add_argument("--source", choices=["synthetic", "alsa", "file"],
                    default="synthetic")
    ap.add_argument("--media", default="", help="mp4 for --source file")
    ap.add_argument("--config", default=None)
    ap.add_argument("--gcc-backend", default="",
                    choices=["", "numpy", "torch-cpu", "torch-cuda"])
    ap.add_argument("--no-classifier", action="store_true")
    ap.add_argument("--duel", action="store_true", help="run T5.2 backend duel")
    ap.add_argument("--duel-reps", type=int, default=300)
    ap.add_argument("--max-seconds", type=float, default=600.0)
    ap.add_argument("--tag", default="")
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.duel:
        run_gcc_duel(args)
    else:
        run_pipeline_bench(args)


if __name__ == "__main__":
    main()
