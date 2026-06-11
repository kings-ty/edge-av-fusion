# Implementation Tasks

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` blocked

Each task lists **acceptance criteria (AC)** — a task is done when its AC pass on
the Xavier, not when the code exists.

---

## Phase 0 — Environment & Hardware Bring-up (blocking everything)

- [x] **T0.1 Repo scaffold** — package layout, config loader, docs.
- [x] **T0.2 Python env** — `scripts/setup_env.sh`: venv with `--system-site-packages`
      on python3.8 (keeps system `tensorrt`, `gi`, ROS 2); install NVIDIA torch wheel
      for JP5 (torch 2.x cp38 aarch64) + torchaudio built against it.
      AC: `python3.8 -c "import torch,tensorrt,gi; print(torch.cuda.is_available())"` → True.
      *(Done 2026-06-11: torch 2.1.0a0+nv23.06, torchaudio 2.1.0 source build,
      mel-on-GPU verified; full pytest suite 42/42 with torch backends.)*
- [ ] **T0.3 ReSpeaker driver** — install seeed-voicecard DKMS for L4T R35 kernel 5.10
      (needs device-tree overlay for 40-pin I2S on AGX; document the dtb step).
      AC: `arecord -l` shows seeed card; `arecord -D hw:seeed... -r 48000 -c 2 -f S16_LE`
      records both channels; channel L/R identity verified by finger-tap test.
- [x] **T0.4 ROS 2 Foxy sanity** — AC: `ros2 topic list` works from the venv shell.
      *(Done 2026-06-11; full Mode-B node run verified at ~91 Hz on /av_fusion/detections.)*

## Phase 1 — Audio Transport (Mode A & B)

- [x] **T1.1 SPSC ring buffer** — numpy-backed, condition-variable blocking read.
      AC: `pytest tests/test_ring_buffer.py` — wraparound, overflow policy, blocking
      timeout, 10⁶-frame soak with producer/consumer threads, zero data corruption.
- [x] **T1.2 GStreamer ALSA source (Mode A)** — 48 kHz S16LE stereo appsink → ring buffer,
      PTS propagation. AC: 60 s capture, zero discontinuities in PTS deltas (±5%),
      xrun counter exposed.
- [x] **T1.3 GStreamer file source (Mode B)** — qtdemux, `sync=true` realtime pacing,
      audio→ring buffer; optional video appsink (NVMM→BGRx) with PTS.
      AC: smartphone .mp4 streams at 1.0× realtime (wall-clock check ±2%);
      audio/video PTS skew < 20 ms over full clip.
- [x] **T1.4 PyAudio fallback source** — same ABC, for baseline comparison only.

## Phase 2 — DSP Core

- [x] **T2.1 GCC-PHAT estimator** — numpy + torch backends behind one interface;
      Hann window, β-regularized PHAT weighting, lag-restricted peak search,
      parabolic sub-sample interpolation, arcsin angle map.
      AC: `pytest tests/test_gcc_phat.py` — synthetic fractional delays recovered
      within 0.25 samples across SNR ≥ 10 dB; both backends agree within 1e-3.
- [x] **T2.2 Adaptive noise-floor gate** — running median/MAD of peak prominence;
      emit `(angle, confidence, valid)`.
      AC: on silence/diffuse-noise input, false-trigger rate < 0.1/min; on tone
      bursts at 10 dB SNR, detection rate > 95% (synthetic test).
- [x] **T2.3 GPU log-mel extractor** — 16 kHz mono decimation + torchaudio
      MelSpectrogram(64 mels, 25 ms/10 ms), 0.96 s patches, device-resident output.
      AC: shape [1,1,64,96]; output matches librosa reference within 1e-3 rel.

## Phase 3 — Classifier & TensorRT

- [x] **T3.1 VehicleSoundNet (torch)** — MobileNetV2-style small CNN, classes
      {vehicle_engine, horn, siren, tire_road, background}; PANNs transfer-learning
      hook documented. AC: forward pass on dummy input; checkpoint load/save.
- [ ] **T3.2 Fine-tune on data** — AudioSet vehicle subset + ESC-50 + self-recorded.
      AC: held-out F1(vehicle) ≥ 0.85. *(Data work — runs on a desktop GPU, not Xavier.)*
- [x] **T3.3 ONNX export** — `tools/export_onnx.py`, opset 13, static [1,1,64,96],
      onnxruntime parity check. AC: max |torch−ort| < 1e-4.
- [x] **T3.4 TRT engine build** — `tools/build_trt_engine.py` (trtexec wrapper),
      FP16, saves build log + layer profile. AC: .plan loads in TRT 8.5; FP16 vs
      FP32 output cosine sim > 0.999 on 100 random mels.
- [x] **T3.5 TRT runtime wrapper** — engine load, pinned host I/O buffers, single
      CUDA stream, sync semantics. AC: 1000-run latency p95 < 5 ms @MAXN.

## Phase 4 — Fusion & ROS 2

- [x] **T4.1 DoA α-β tracker + ambiguity logic** — front/back hypothesis pair,
      dθ/dt parity vote, yaw-fusion update from /odom. AC: synthetic pass-by
      trajectory disambiguates within 2 s; unit-tested hypothesis bookkeeping.
- [x] **T4.2 Fusion FSM** — IDLE→CANDIDATE (DoA stable) →TRIGGERED (classifier
      vehicle ≥ θ) →CONFIRMED/REJECTED (vision result, timeout fallback).
      AC: state-transition unit tests incl. timeout and hysteresis paths.
- [x] **T4.3 `av_fusion_interfaces`** — AvDetection.msg (header, class, confidence,
      doa_deg, doa_confidence, front_back_ambiguous, vision_confirmed, track_id,
      latency_ms). AC: `colcon build` + `ros2 interface show` on Foxy.
- [x] **T4.4 `av_fusion_node` (rclpy)** — owns pipeline threads; pubs
      `/av_fusion/detections` (SensorDataQoS) + `/av_fusion/diagnostics`; subs
      `/av_fusion/vision_confirmation`, `/odom`; params for mode A/B, paths,
      thresholds. AC: `ros2 launch av_fusion_node pipeline.launch.py mode:=file
      media:=test.mp4` publishes real messages; `ros2 topic hz` ≈ hop rate.
- [ ] **T4.5 Vision confirmer** — gated camera/detector node consuming TRIGGERED
      events (separate package; stub interface defined). *(v1.1 scope.)*

## Phase 5 — Benchmarks (portfolio core)

- [x] **T5.1 Latency harness** — per-stage spans, CUDA-sync-correct, p50/95/99,
      JSON + markdown report, ≥10 k frames. AC: report generated in synthetic mode
      on dev machine; on Xavier with hardware.
- [x] **T5.2 GCC backend duel** — numpy vs torch-cpu vs torch-cuda table. AC: report
      includes the crossover analysis (window size sweep 1k–16k).
- [x] **T5.3 Power-mode sweep** — nvpmodel 0/3/2 (MAXN/30W/15W), restores previous
      mode, captures `jetson_clocks --show`. AC: one command → 3-mode comparison table.
- [x] **T5.4 Memory-bandwidth probe** — tegrastats EMC parser; NVMM vs system-memory
      video branch comparison; audio-path measurement included to *show* it's noise.
      AC: report with GB/s estimates + delta.
- [x] **T5.5 DoA validation protocol** — synthetic mode (CI) + field protocol doc
      (angles {0,±30,±60,±90}°, 1 m/3 m, chirp/pink/engine source); per-angle
      bias/RMSE table. AC: synthetic RMSE < 3° broadside, < 12° at ±60°.

## Phase 6 — Hardening

- [ ] **T6.1 Soak test** — 24 h Mode A run; zero leaks (RSS slope ~0), xrun recovery.
- [ ] **T6.2 systemd unit + launch on boot**, log rotation.
- [ ] **T6.3 README demo assets** — bench tables, DoA polar plots, rqt screenshot.

### Dependency graph

```
T0.2 ──► T2.3, T3.*, T5.*        T0.3 ──► T1.2, T5.5(field)
T1.1 ──► T1.2/T1.3 ──► T5.1      T2.1 ──► T2.2 ──► T4.1 ──► T4.2 ──► T4.4
T3.1 ──► T3.3 ──► T3.4 ──► T3.5 ──► T4.4      T4.3 ──► T4.4 ──► T6.*
```
