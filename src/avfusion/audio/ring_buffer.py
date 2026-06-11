"""Single-producer / single-consumer ring buffer for audio hops.

Deliberately *locked*, not "lock-free": pure Python cannot express the
acquire/release atomics a correct lock-free SPSC queue needs on weakly-ordered
ARM (see ARCHITECTURE §4). The lock is held for ~1 µs per 10.7 ms hop, so
contention is unmeasurable — correctness wins.

Overflow policy is drop-oldest: for a live perception system, fresh audio is
worth more than stale audio. Drops are counted, never silent.
"""
import threading
from typing import Optional, Tuple

import numpy as np


class RingBuffer:
    def __init__(self, hop_samples: int, channels: int, capacity_hops: int,
                 dtype=np.int16):
        self._hop = hop_samples
        self._buf = np.zeros((capacity_hops, hop_samples, channels), dtype=dtype)
        self._pts = np.zeros(capacity_hops, dtype=np.float64)
        self._cap = capacity_hops
        self._head = 0          # next write slot
        self._tail = 0          # next read slot
        self._size = 0
        self._dropped = 0
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._closed = False

    @property
    def dropped(self) -> int:
        return self._dropped

    def push(self, hop: np.ndarray, pts: float) -> None:
        """Producer side. `hop` must be shape (hop_samples, channels)."""
        if hop.shape != self._buf.shape[1:]:
            raise ValueError("hop shape %s != %s" % (hop.shape, self._buf.shape[1:]))
        with self._not_empty:
            if self._size == self._cap:           # drop-oldest
                self._tail = (self._tail + 1) % self._cap
                self._size -= 1
                self._dropped += 1
            self._buf[self._head] = hop
            self._pts[self._head] = pts
            self._head = (self._head + 1) % self._cap
            self._size += 1
            self._not_empty.notify()

    def pop(self, timeout: Optional[float] = None
            ) -> Optional[Tuple[np.ndarray, float]]:
        """Consumer side. Blocks until a hop is available; returns a copy
        (the slot may be overwritten after return). None on timeout/close."""
        with self._not_empty:
            if not self._not_empty.wait_for(
                    lambda: self._size > 0 or self._closed, timeout):
                return None
            if self._size == 0:                   # closed and drained
                return None
            hop = self._buf[self._tail].copy()
            pts = float(self._pts[self._tail])
            self._tail = (self._tail + 1) % self._cap
            self._size -= 1
            return hop, pts

    def close(self) -> None:
        with self._not_empty:
            self._closed = True
            self._not_empty.notify_all()

    def reset(self) -> None:
        """Drop everything buffered (e.g. after an ALSA xrun)."""
        with self._not_empty:
            self._head = self._tail = self._size = 0
