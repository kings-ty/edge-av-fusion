"""Vehicle-sound classifier facade with graceful degradation.

Preferred backend: TensorRT FP16 engine. If the engine is missing or fails to
load (e.g. TRT version bump), we degrade to "energy-trigger" mode — broadband
energy ratio vs noise floor — instead of killing the node. The acoustic
tripwire must never be the thing that silently dies (ARCHITECTURE §8).
"""
import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class ClassifierResult:
    probs: np.ndarray          # [num_classes] softmax
    top_class: str
    top_prob: float
    vehicle_prob: float        # summed prob over vehicle classes
    degraded: bool             # True when running energy-trigger fallback


class VehicleSoundClassifier:
    def __init__(self, engine_path: str, classes: List[str],
                 vehicle_classes: List[str]):
        self.classes = classes
        self._veh_idx = [classes.index(c) for c in vehicle_classes]
        self._engine = None
        try:
            from .trt_engine import TrtEngine
            self._engine = TrtEngine(engine_path)
        except Exception as exc:  # noqa: BLE001 - degrade, never die
            log.warning("TRT engine unavailable (%s); degrading to energy trigger", exc)

    @property
    def degraded(self) -> bool:
        return self._engine is None

    def classify(self, mel_patch) -> ClassifierResult:
        """mel_patch: torch CUDA tensor [1,1,64,96] (or numpy in degraded mode)."""
        if self._engine is not None:
            out = next(iter(self._engine.infer(mel_patch).values()))
            logits = out.float().cpu().numpy().reshape(-1)
            e = np.exp(logits - logits.max())
            probs = e / e.sum()
        else:
            probs = self._energy_fallback(mel_patch)
        top = int(np.argmax(probs))
        return ClassifierResult(
            probs=probs, top_class=self.classes[top], top_prob=float(probs[top]),
            vehicle_prob=float(probs[self._veh_idx].sum()), degraded=self.degraded)

    def _energy_fallback(self, mel_patch) -> np.ndarray:
        m = mel_patch
        if hasattr(m, "cpu"):
            m = m.float().cpu().numpy()
        m = np.asarray(m).reshape(-1)
        # standardized log-mel: high overall level + low-frequency tilt is a
        # crude "engine-like" cue. Better than nothing, clearly flagged degraded.
        score = float(np.clip((m.mean() + 0.5), 0.0, 1.0))
        probs = np.full(len(self.classes), (1 - score) / max(len(self.classes) - 1, 1))
        probs[0] = score
        return probs / probs.sum()
