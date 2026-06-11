"""Typed configuration loaded from config/pipeline.yaml.

Dataclasses (not raw dicts) so that a typo in a config key fails at load time,
not 40 minutes into a field test.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass(frozen=True)
class GateConfig:
    history: int = 200
    z_threshold: float = 4.0
    min_peak_ratio: float = 1.8


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 48000
    channels: int = 2
    hop_samples: int = 512
    ring_capacity_hops: int = 256
    alsa_device: str = "hw:seeed2micvoicec"


@dataclass(frozen=True)
class DoaConfig:
    window_samples: int = 4096
    mic_spacing_m: float = 0.058
    speed_of_sound: float = 343.0
    backend: str = "numpy"
    phat_beta: float = 1.0
    gate: GateConfig = field(default_factory=GateConfig)


@dataclass(frozen=True)
class ClassifierConfig:
    target_rate_hz: float = 1.0
    classifier_sample_rate: int = 16000
    n_mels: int = 64
    win_ms: float = 25.0
    hop_ms: float = 10.0
    patch_frames: int = 96
    engine_path: str = "models/vehicle_fp16.plan"
    onnx_path: str = "models/vehicle.onnx"
    classes: List[str] = field(default_factory=lambda: [
        "vehicle_engine", "horn", "siren", "tire_road", "background"])
    vehicle_classes: List[str] = field(default_factory=lambda: [
        "vehicle_engine", "horn", "siren", "tire_road"])
    trigger_threshold: float = 0.6


@dataclass(frozen=True)
class FusionConfig:
    candidate_hold_hops: int = 8
    doa_stability_deg: float = 15.0
    vision_timeout_s: float = 2.0
    track_max_coast_s: float = 1.5
    alpha: float = 0.35
    beta: float = 0.05


@dataclass(frozen=True)
class RosConfig:
    detections_topic: str = "/av_fusion/detections"
    diagnostics_topic: str = "/av_fusion/diagnostics"
    vision_topic: str = "/av_fusion/vision_confirmation"
    odom_topic: str = "/odom"


@dataclass(frozen=True)
class PipelineConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    doa: DoaConfig = field(default_factory=DoaConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    ros: RosConfig = field(default_factory=RosConfig)


def _build(cls, data: dict):
    fields = cls.__dataclass_fields__  # type: ignore[attr-defined]
    unknown = set(data) - set(fields)
    if unknown:
        raise KeyError("unknown config keys for %s: %s" % (cls.__name__, sorted(unknown)))
    kwargs = {}
    for name, f in fields.items():
        if name not in data:
            continue
        value = data[name]
        if hasattr(f.type, "__dataclass_fields__") and isinstance(value, dict):
            value = _build(f.type, value)
        kwargs[name] = value
    return cls(**kwargs)


def load_config(path: Optional[str] = None) -> PipelineConfig:
    if path is None:
        path = str(Path(__file__).resolve().parents[2] / "config" / "pipeline.yaml")
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    return PipelineConfig(
        audio=_build(AudioConfig, raw.get("audio", {})),
        doa=_build(DoaConfig, raw.get("doa", {})),
        classifier=_build(ClassifierConfig, raw.get("classifier", {})),
        fusion=_build(FusionConfig, raw.get("fusion", {})),
        ros=_build(RosConfig, raw.get("ros", {})),
    )
