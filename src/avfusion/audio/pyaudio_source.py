"""PortAudio fallback source — kept only as the baseline comparator.

Limitations vs GstAlsaSource (why this is not the production path):
no hardware PTS (we stamp with time.monotonic() in the callback, adding
scheduler jitter), no xrun introspection, no shared clock with any video path.
"""
import logging
import time

import numpy as np

from .source_base import AudioSource

log = logging.getLogger(__name__)


class PyAudioSource(AudioSource):
    def __init__(self, sample_rate: int, channels: int, hop_samples: int,
                 capacity_hops: int, device_index=None):
        super().__init__(hop_samples, channels, capacity_hops)
        import pyaudio  # lazy: optional dependency
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16, channels=channels, rate=sample_rate,
            input=True, frames_per_buffer=hop_samples,
            input_device_index=device_index,
            stream_callback=self._callback, start=False)

    def _callback(self, in_data, frame_count, time_info, status):
        import pyaudio
        hop = np.frombuffer(in_data, dtype=np.int16).reshape(
            -1, self.channels)
        self.ring.push(hop, time.monotonic())
        return None, pyaudio.paContinue

    def start(self) -> None:
        self._stream.start_stream()

    def stop(self) -> None:
        self._stream.stop_stream()
        self._stream.close()
        self._pa.terminate()
        self.ring.close()
