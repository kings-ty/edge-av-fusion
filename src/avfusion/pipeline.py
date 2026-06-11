"""Pipeline orchestrator: source → DoA → (mel → TRT) → fusion.

Single process, two compute threads (ARCHITECTURE §4): the heavy work happens
inside numpy/torch/TRT C extensions that release the GIL, so threads give real
concurrency without multiprocessing's serialization tax.

Every stage is instrumented with perf_counter_ns spans; the benchmark harness
consumes `StageTimes` records, the ROS node consumes `FusionEvent`s. One code
path serves both — benchmarks that exercise a different code path than
production measure nothing.
"""
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from .audio.source_base import AudioSource
from .config import PipelineConfig
from .dsp.gcc_phat import GccPhatEstimator
from .fusion.fsm import FusionEvent, FusionStateMachine
from .fusion.tracker import DoaTracker

log = logging.getLogger(__name__)


@dataclass
class StageTimes:
    """Nanosecond spans for one processed hop (classifier fields only on
    hops where a classification ran)."""
    t_capture_pts: float            # source PTS (its own clock domain)
    capture_to_read_ns: int         # ring-buffer dwell (Mode A only meaningful)
    gcc_ns: int
    track_ns: int
    mel_ns: int = 0
    trt_ns: int = 0
    fusion_ns: int = 0
    e2e_ns: int = 0


@dataclass
class PipelineStats:
    hops: int = 0
    classifications: int = 0
    dropped_hops: int = 0
    degraded: bool = False
    stage_times: List[StageTimes] = field(default_factory=list)


class Pipeline:
    def __init__(self, cfg: PipelineConfig, source: AudioSource,
                 event_cb: Optional[Callable[[FusionEvent, StageTimes], None]] = None,
                 enable_classifier: bool = True,
                 keep_stage_times: bool = False):
        self.cfg = cfg
        self.source = source
        self._event_cb = event_cb
        self._keep_times = keep_stage_times
        self.stats = PipelineStats()

        d = cfg.doa
        self.gcc = GccPhatEstimator(
            sample_rate=cfg.audio.sample_rate, window_samples=d.window_samples,
            mic_spacing_m=d.mic_spacing_m, speed_of_sound=d.speed_of_sound,
            backend=d.backend, phat_beta=d.phat_beta,
            gate_history=d.gate.history, gate_z=d.gate.z_threshold,
            gate_min_ratio=d.gate.min_peak_ratio)
        self.tracker = DoaTracker(alpha=cfg.fusion.alpha, beta=cfg.fusion.beta,
                                  max_coast_s=cfg.fusion.track_max_coast_s)
        self.fsm = FusionStateMachine(
            candidate_hold_hops=cfg.fusion.candidate_hold_hops,
            doa_stability_deg=cfg.fusion.doa_stability_deg,
            trigger_threshold=cfg.classifier.trigger_threshold,
            vision_timeout_s=cfg.fusion.vision_timeout_s)

        self.mel = None
        self.classifier = None
        if enable_classifier:
            try:
                from .dsp.mel import MelPatchExtractor
                from .inference.classifier import VehicleSoundClassifier
                c = cfg.classifier
                self.mel = MelPatchExtractor(
                    capture_rate=cfg.audio.sample_rate,
                    classifier_rate=c.classifier_sample_rate, n_mels=c.n_mels,
                    win_ms=c.win_ms, hop_ms=c.hop_ms, patch_frames=c.patch_frames)
                self.classifier = VehicleSoundClassifier(
                    c.engine_path, list(c.classes), list(c.vehicle_classes))
                self.stats.degraded = self.classifier.degraded
                # absorb lazy CUDA/cuFFT/TRT first-call costs before the
                # stream starts (seconds at 30W = a ring overflow mid-clip)
                self.classifier.classify(self.mel.warmup())
            except Exception as exc:  # noqa: BLE001
                log.warning("classifier branch disabled (%s); DoA-only mode", exc)
                self.stats.degraded = True

        # analysis window assembled from hops
        self._window = np.zeros((cfg.doa.window_samples, cfg.audio.channels),
                                dtype=np.int16)
        self._window_fill = 0
        self._classify_period = 1.0 / cfg.classifier.target_rate_hz
        self._last_classify = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------- control
    def start(self) -> None:
        self.source.start()
        self._thread = threading.Thread(target=self._loop, name="dsp", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self.source.stop()

    def update_yaw(self, yaw_deg: float) -> None:
        self.tracker.update_yaw(yaw_deg)

    def on_vision(self, confirmed: bool) -> None:
        self.fsm.on_vision(confirmed)
        self.tracker.vision_vote(confirmed)

    # ---------------------------------------------------------------- loop
    def _loop(self) -> None:
        hop = self.cfg.audio.hop_samples
        win = self.cfg.doa.window_samples
        while not self._stop.is_set():
            t0 = time.perf_counter_ns()
            chunk = self.source.read(timeout=1.0)
            if chunk is None:
                if getattr(self.source, "finished", False):
                    break
                continue
            t_read = time.perf_counter_ns()

            # slide hop into analysis window
            self._window[:-hop] = self._window[hop:]
            self._window[-hop:] = chunk.samples
            self._window_fill = min(self._window_fill + hop, win)
            if self._window_fill < win:
                continue

            est = self.gcc.estimate(self._window)
            t_gcc = time.perf_counter_ns()

            track = self.tracker.update(est, chunk.pts)
            t_track = time.perf_counter_ns()

            st = StageTimes(
                t_capture_pts=chunk.pts, capture_to_read_ns=t_read - t0,
                gcc_ns=t_gcc - t_read, track_ns=t_track - t_gcc)

            # classifier branch at its own (slower) cadence
            if self.mel is not None:
                self.mel.push_hop(chunk.samples)
                now = time.monotonic()
                if self.mel.ready and now - self._last_classify >= self._classify_period:
                    self._last_classify = now
                    m0 = time.perf_counter_ns()
                    patch = self.mel.patch()
                    m1 = time.perf_counter_ns()
                    result = self.classifier.classify(patch)
                    m2 = time.perf_counter_ns()
                    self.fsm.on_classifier(result)
                    st.mel_ns, st.trt_ns = m1 - m0, m2 - m1
                    self.stats.classifications += 1

            f0 = time.perf_counter_ns()
            event = self.fsm.on_track(track)
            st.fusion_ns = time.perf_counter_ns() - f0
            st.e2e_ns = time.perf_counter_ns() - t0

            self.stats.hops += 1
            self.stats.dropped_hops = self.source.dropped_hops
            if self._keep_times:
                self.stats.stage_times.append(st)
            if self._event_cb is not None:
                self._event_cb(event, st)
