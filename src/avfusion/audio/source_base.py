"""AudioSource abstraction shared by Mode A (live) and Mode B (file).

A source owns its transport (GStreamer pipeline / PortAudio stream) and feeds a
RingBuffer. Consumers only ever see (hop, pts) tuples, so the rest of the
pipeline is mode-agnostic.
"""
import abc
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .ring_buffer import RingBuffer


@dataclass
class AudioChunk:
    samples: np.ndarray   # (hop_samples, channels) int16
    pts: float            # seconds, source clock domain (monotonic / pipeline PTS)


class AudioSource(abc.ABC):
    """Lifecycle: construct -> start() -> read() loop -> stop()."""

    def __init__(self, hop_samples: int, channels: int, capacity_hops: int):
        self.ring = RingBuffer(hop_samples, channels, capacity_hops)
        self.hop_samples = hop_samples
        self.channels = channels

    @abc.abstractmethod
    def start(self) -> None: ...

    @abc.abstractmethod
    def stop(self) -> None: ...

    def read(self, timeout: Optional[float] = 1.0) -> Optional[AudioChunk]:
        item = self.ring.pop(timeout)
        if item is None:
            return None
        samples, pts = item
        return AudioChunk(samples=samples, pts=pts)

    @property
    def dropped_hops(self) -> int:
        return self.ring.dropped

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
