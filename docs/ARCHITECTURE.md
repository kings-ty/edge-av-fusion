# edge-av-fusion — Architecture & Design Review

Multimodal Edge-AI Perception Pipeline: acoustic early-warning + vision confirmation
for autonomous-robot blind spots, on Jetson Xavier AGX 32GB (JetPack 5.1.x, L4T R35.6,
CUDA 11.4, TensorRT 8.5.2, ROS 2 Foxy, Python 3.8).

---

## 1. System Overview

```
                 ┌────────────────────────── LOW-POWER ALWAYS-ON PATH ──────────────────────────┐
                 │                                                                              │
 Mode A: ReSpeaker 2-Mic HAT (I2S, 48 kHz S32→S16 stereo)                                      │
 Mode B: .mp4 ──qtdemux──► audio branch (clock-paced)                                          │
                 │                                                                              │
                 ▼                                                                              │
        ┌────────────────┐    48 kHz stereo     ┌─────────────────┐  τ (sub-sample)  ┌────────┐ │
        │ GStreamer      │ ───────────────────► │ GCC-PHAT (CPU,  │ ───────────────► │ DoA    │ │
        │ appsink →      │                      │ numpy rFFT) +   │                  │ Track  │ │
        │ SPSC ring buf  │                      │ adaptive gate   │                  │ -er    │ │
        └────────────────┘                      └─────────────────┘                  └───┬────┘ │
                 │ 16 kHz mono (resampled)                                               │      │
                 ▼                                                                       ▼      │
        ┌────────────────┐  log-mel 64×96  ┌──────────────────────┐  vehicle p>θ  ┌──────────┐  │
        │ GPU Mel-Spec   │ ──────────────► │ TensorRT FP16        │ ────────────► │ Fusion   │  │
        │ (torchaudio)   │                 │ vehicle classifier   │               │ FSM      │  │
        └────────────────┘                 └──────────────────────┘               └────┬─────┘  │
                                                                                       │ trigger │
                 ┌──────────────────────── HIGH-POWER ON-DEMAND PATH ──────────────────┤        │
                 ▼                                                                     ▼        │
        ┌────────────────┐   confirm/locate   ┌──────────────┐    ROS 2 topic  /av_fusion/      │
        │ Camera+detector│ ─────────────────► │ AvDetection  │ ──────────────► detections       │
        │ (NVMM, gated)  │                    │ publisher    │                                  │
        └────────────────┘                    └──────────────┘                                  │
```

Design intent: **audio is the cheap tripwire, vision is the expensive judge.** Audio path
runs continuously at < 1 W incremental cost; the vision path is woken only by a fused
acoustic trigger (vehicle class confidence × DoA stability).

---

## 2. Challenging the Baseline (PyAudio + GCC-PHAT@16 kHz + torchaudio)

Your baseline is functional but has three real problems. Two of them are not where you
thought the problems were.

### 2.1 The biggest flaw nobody mentioned: 16 kHz kills your DoA resolution

The ReSpeaker 2-Mic HAT spacing is **d ≈ 0.058 m**. Maximum TDOA is

```
τ_max = d / c = 0.058 / 343 ≈ 169 µs
@16 kHz → ±2.7 samples   → ~6 distinguishable integer-lag angles. Useless.
@48 kHz → ±8.1 samples   → ~17 integer lags, and sub-sample interpolation works far better.
```

**Decision: capture at 48 kHz (native I2S rate), run GCC-PHAT at 48 kHz, and
decimate to 16 kHz mono only for the classifier branch.** Combined with parabolic
sub-sample peak interpolation this takes angular RMSE from ~15–20° to a realistic
~5–8° broadside. This single change matters more than any GPU optimization in the
audio path.

### 2.2 PyAudio → GStreamer

PyAudio (PortAudio) works, but it gives you: a callback thread you don't control,
no clock/PTS, no demux story for Mode B, and a second, completely different code
path for file playback. GStreamer gives one abstraction for both modes (§4).

### 2.3 GCC-PHAT on GPU is (mostly) a trap — measure before you port

A 4096-point stereo rFFT + cross-spectrum + irFFT is ~10⁶ FLOPs. On Carmel cores
with numpy/MKL-class FFT this is **tens of microseconds**. A CUDA round trip costs
5–15 µs *per kernel launch* and torch dispatches ~8 kernels for this op chain.
The GPU does not lose because it is slow; it loses because the problem is too small.

**Decision: GCC-PHAT runs on CPU (numpy) by default; a torch/cuFFT backend is kept
behind the same interface purely so the benchmark harness can prove this claim with
numbers** (`bench/latency.py --gcc-backend {numpy,torch-cpu,torch-cuda}`). That
measured table is a portfolio asset: knowing what *not* to put on the GPU is the
senior skill.

---

## 3. Jetson-Specific Optimization — what applies where

### 3.1 Why TensorRT does NOT apply to GCC-PHAT

TensorRT is a **neural-network graph compiler**: it ingests an ONNX graph of
learned-weight ops (conv/gemm/attention), then does layer fusion, precision
calibration (FP16/INT8), kernel autotuning, and memory planning. GCC-PHAT has:

- **no weights** — nothing to quantize or calibrate;
- **no fusible op graph** — it is one FFT → elementwise → iFFT chain, already served
  by a single vendor-optimal library (cuFFT, which `torch.fft` calls directly);
- **no TRT FFT op** — ONNX/TRT have no DFT kernel coverage worth using (ONNX `DFT`
  exists since opset 17 but TRT 8.5 does not implement it).

Running signal processing "through TensorRT" would mean expressing an FFT as
matmuls — strictly slower than cuFFT. TRT's value-add is zero here.

### 3.2 Where TensorRT DOES apply: the audio classifier

The vehicle-sound classifier (CNN over log-mel patches) is exactly TRT's target:

```
PyTorch model ──torch.onnx.export (opset 13, static [1,1,64,96])──► model.onnx
   └─ tools/export_onnx.py
model.onnx ──trtexec --fp16 --saveEngine──► model_fp16.plan
   └─ tools/build_trt_engine.py   (wraps /usr/src/tensorrt/bin/trtexec, records build log)
model_fp16.plan ──tensorrt.Runtime → ICudaEngine → IExecutionContext──► inference
   └─ src/avfusion/inference/trt_engine.py
```

Classifier choice: a YAMNet-class model. YAMNet itself is TensorFlow; on a torch
stack the pragmatic equivalents are **PANNs CNN10/CNN14** (AudioSet-pretrained,
PyTorch, exports to ONNX cleanly) or the bundled `VehicleSoundNet` (MobileNetV2-style,
~0.6 M params) fine-tuned on vehicle classes. FP16 on Xavier's Volta Tensor Cores
gives ~2× over FP32 with negligible accuracy loss for this task; INT8 needs a
calibration set and is only worth it if you need < 1 ms inference (you don't —
the audio frame period is 32 ms, your real-time budget is enormous).

Engine build is done **once, offline, on-device** (TRT engines are not portable
across GPU architectures or TRT versions).

### 3.3 Unified memory: where it matters and where it doesn't

Xavier's CPU and GPU share one LPDDR4x pool (137 GB/s). Three transfer strategies:

| Strategy | API | Behavior on Tegra |
|---|---|---|
| Pageable copy | `tensor.to('cuda')` | staging copy through pinned bounce buffer |
| Pinned + async | `pin_memory()` + `to(non_blocking=True)` | DMA copy, overlappable |
| Zero-copy mapped | `cudaHostAllocMapped` / `torch` via custom allocator | **no copy at all**; GPU reads host memory through SMMU (uncached for GPU) |

**Honest sizing:** an audio hop is 16 KB (512 samples × 2 ch × int16 @48 kHz); a
1-second mel patch input is 24 KB. At 137 GB/s a copy is **sub-microsecond**; even
the launch overhead of the copy dwarfs the copy. Unified-memory optimization is
**irrelevant for the audio path** and the benchmark (`bench/membw.py`) exists to
show that honestly.

Where it *does* matter: **video**. A 1080p RGBA frame is 8.3 MB; at 30 fps a naïve
decode→CPU→GPU flow burns ~500 MB/s of bandwidth and milliseconds of latency. That
is why the Mode-B/vision path keeps frames in **NVMM** (`nvv4l2decoder` →
`nvvidconv` stays in NVMM until the consumer needs system memory). The
before/after EMC-utilization measurement in §6 is run on the video branch, where
the effect is real — not on audio, where it would be theater.

---

## 4. Streaming Architecture: multiprocessing ring buffer vs GStreamer

### Option (a): Python multiprocessing + shared-memory ring buffer

The popular claim "lock-free SPSC ring buffer in Python" is misleading:

- Across **threads**, the GIL serializes bytecode but a read-modify-write like
  `self.head += 1` is *still* multiple bytecodes; you get atomicity by accident of
  GIL scheduling, not by design — and numpy slice writes release the GIL mid-write.
- Across **processes** (shared `multiprocessing.shared_memory`), there is no GIL
  to hide behind at all: Python exposes **no atomic store/load with memory-order
  guarantees** on a shared buffer. A correct lock-free SPSC queue requires
  acquire/release semantics on head/tail indices (C++ `std::atomic`); in pure
  Python you must either take a lock (so it's not lock-free) or accept torn
  index reads on weakly-ordered ARM (Carmel is ARMv8 — it *will* reorder).
- Real cost of multiprocessing here: every consumer hop pays pickling or careful
  manual buffer layout, process startup/supervision, and debugging across pids.

### Option (b): GStreamer (Jetson-native)

- **Mode A:** `alsasrc device=hw:seeed2micvoicec ! audioconvert !
  audio/x-raw,format=S16LE,rate=48000,channels=2 ! appsink` — kernel-driven ALSA
  ring buffer, hardware timestamps as buffer PTS.
- **Mode B:** `filesrc ! qtdemux` → audio branch (`avdec_aac ! audioconvert !
  audioresample ! appsink sync=true`) + video branch (`h264parse ! nvv4l2decoder
  ! video/x-raw(memory:NVMM) ! nvvidconv ! appsink`). `sync=true` paces buffers
  against the pipeline clock → **faithful live-robot simulation**, and audio/video
  share one clock so A/V sync is free (PTS from the same demuxer).
- All the threading, clocking, and format negotiation is C code that never holds
  the GIL.

### Decision: GStreamer for transport, single process + threads for compute

GStreamer owns capture/demux/pacing; `appsink` hands numpy arrays to one Python
process. Inside that process, the DSP/inference threads spend their time in
numpy/torch/TRT C extensions which **release the GIL**, so threads give real
concurrency without multiprocessing's serialization tax. A small
**locked** SPSC ring buffer (we name it what it is) decouples the appsink callback
from the compute thread; the lock is held for ~1 µs per 32 ms hop — contention is
unmeasurable, and correctness is provable.

Multiprocessing is kept in reserve for exactly one scenario: if a future vision
model's *Python-side* pre/post-processing becomes GIL-bound. Not before.

---

## 5. The 2-Mic Linear Array: physics, limits, mitigation

A single mic pair measures one TDOA τ, giving `θ = arcsin(c·τ/d)` — a **cone of
confusion** that a horizontal linear array collapses to a left/right angle in
[−90°, +90°] with two fundamental limits:

1. **Front–back ambiguity.** A source at +40° front and +140° (i.e., 40° behind)
   produce identical τ. Unresolvable from one static measurement, period.
2. **Endfire compression.** `dθ/dτ ∝ 1/cos θ` → angular resolution degrades toward
   ±90°; near endfire, one sample of TDOA error is tens of degrees. We therefore
   publish a per-estimate `doa_confidence` derived from both peak quality *and*
   `cos θ` geometry.

### Mitigations (implemented in `fusion/tracker.py`)

- **Temporal motion parity:** track θ(t) with an α-β filter. For a vehicle passing
  the robot, front and back hypotheses produce **opposite signs of dθ/dt** as it
  crosses broadside; sustained trajectory consistency votes down one hypothesis.
- **IMU/odometry heading fusion (active disambiguation):** when the robot yaws by
  Δψ (from `/odom` or IMU), a front source shifts by −Δψ and a back source by +Δψ
  in array coordinates. Even incidental rotations of a few degrees disambiguate
  within ~2 s. The tracker consumes yaw and scores both hypotheses; a deliberate
  5–10° "glance" can be commanded when stakes are high.
- **Vision as the tie-breaker (the whole point of this system):** the camera FOV
  covers the front hemisphude; an acoustic track at +30° with *no* visual
  correlate after the high-power check is evidence for the back hypothesis — which
  for blind-spot warning is the *more* dangerous case and is flagged as such
  (`front_back_ambiguous=true`, `vision_confirmed=false` → highest alert tier).
- **Level/Doppler trend** as weak priors (rising intensity = approaching).

What we do **not** pretend to do: elevation, multi-source separation, or true
360° instantaneous DoA. Those need ≥4 mics in a non-collinear layout
(documented as the v2 hardware path: ReSpeaker 4-Mic square → full azimuth via
SRP-PHAT, same code structure).

---

## 6. Benchmark Harness (portfolio core)

All benchmarks emit JSON to `bench_results/` + a rendered markdown table.

| Benchmark | Tool | Method |
|---|---|---|
| Per-stage latency | `bench/latency.py` | `time.perf_counter_ns()` spans around capture→DoA→mel→TRT→fusion; `cudaStreamSynchronize` before closing GPU spans (async launch ≠ done) |
| E2E distribution | same | p50/p95/p99 over ≥10 000 frames, warmup 500 excluded; reports per-stage and end-to-end |
| GCC backend duel | `--gcc-backend` sweep | numpy vs torch-cpu vs torch-cuda, same input tensors — proves §2.3 |
| Power modes | `bench/power_sweep.py` | wraps `nvpmodel -m {0,3,2}` (MAXN/30W/15W) + `jetson_clocks --show`, reruns latency bench per mode; needs sudo, restores prior mode |
| Memory bandwidth | `bench/membw.py` | spawns `tegrastats --interval 100`, parses `EMC_FREQ x%@yMHz` → effective GB/s during workload; run video branch with/without NVMM to show the delta where it actually exists |
| DoA accuracy | `bench/doa_validation.py` | protocol: speaker at measured angles {0, ±30, ±60, ±90}°, 1 m and 3 m, chirp + pink noise + recorded engine; reports per-angle bias/RMSE + confusion at endfire. Synthetic mode (fractional-delay generated stereo) runs in CI without hardware |

Latency budget (targets, MAXN):

| Stage | Budget | Expected |
|---|---|---|
| Capture hop (48 kHz, 512) | 10.7 ms (inherent) | pipeline adds < 1 ms |
| GCC-PHAT (numpy, 4096 win) | 1 ms | ~0.1–0.3 ms |
| Mel (GPU, 1 s patch, 1 Hz) | 5 ms | ~1–2 ms |
| TRT classifier FP16 | 5 ms | ~1–3 ms |
| Fusion + publish | 1 ms | ~0.2 ms |
| **E2E (sound→topic)** | **< 50 ms p95** | dominated by windowing, not compute |

**Honest non-goals:** INT8 quantization (no latency need), GPU GCC-PHAT (loses),
DLA offload (classifier is too small to amortize DLA submission overhead; DLA
becomes interesting only for the vision detector), CUDA Graphs (worth it only if
classifier rate rises above ~100 Hz).

---

## 7. Module Map

```
src/avfusion/
  config.py                  dataclass config + YAML loader (config/pipeline.yaml)
  audio/ring_buffer.py       SPSC ring buffer (locked, honest), numpy-backed
  audio/source_base.py       AudioSource ABC: blocking read of (hop,2) int16 + PTS
  audio/gst_alsa_source.py   Mode A — GStreamer alsasrc
  audio/gst_file_source.py   Mode B — qtdemux, clock-paced audio + optional video appsink
  audio/pyaudio_source.py    fallback (kept for A/B comparison vs baseline)
  dsp/gcc_phat.py            GCC-PHAT, numpy + torch backends, sub-sample interp,
                             adaptive noise-floor gate (median/MAD on peak prominence)
  dsp/mel.py                 GPU log-mel (torchaudio), 64 mels × 96 frames patches
  inference/trt_engine.py    TRT 8.5 runtime wrapper (engine load, pinned I/O, ctx)
  inference/classifier.py    VehicleSoundClassifier: mel patch → class probs
  fusion/tracker.py          DoA α-β tracker, front/back hypothesis scoring, yaw fusion
  fusion/fsm.py              IDLE→CANDIDATE→TRIGGERED→CONFIRMED state machine
  bench/                     latency.py, power_sweep.py, membw.py, doa_validation.py
tools/export_onnx.py         torch → ONNX (VehicleSoundNet or PANNs)
tools/build_trt_engine.py    trtexec wrapper → .plan + build report
ros2_ws/src/av_fusion_interfaces/   AvDetection.msg
ros2_ws/src/av_fusion_node/         rclpy node: owns pipeline, pubs /av_fusion/detections,
                                    subs vision confirm + /odom yaw
```

Threading model (single process):

```
[GStreamer streaming thread] --appsink--> RingBuffer --> [DSP thread: GCC-PHAT @ every hop]
                                                  \--> [Classifier thread: mel+TRT @ 1 Hz]
both --> [Fusion (runs in DSP thread)] --> rclpy publisher (executor thread)
```

## 8. Failure-mode notes

- ALSA xrun → GStreamer signals; we log, reset ring buffer, increment a dropped-
  frames counter exposed on a diagnostics topic.
- TRT engine/version mismatch → classifier degrades to "energy-only trigger" mode
  rather than killing the node (acoustic tripwire stays alive).
- Clock: Mode B uses pipeline PTS as the timestamp domain; Mode A uses
  `CLOCK_MONOTONIC` mapped to ROS time at source. Never mix domains in latency math.
