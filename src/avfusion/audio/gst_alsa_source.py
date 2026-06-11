"""Mode A: live capture from ReSpeaker 2-Mic HAT via GStreamer alsasrc.

The kernel ALSA ring buffer absorbs scheduling jitter; appsink hands us
hardware-timestamped buffers (PTS from the pipeline clock, CLOCK_MONOTONIC
domain). On xrun we reset our ring so DoA never correlates across a gap.
"""
import logging

from .gst_base import AppsinkRechunker, GstPipelineOwner
from .source_base import AudioSource

log = logging.getLogger(__name__)


class GstAlsaSource(AudioSource):
    def __init__(self, device: str, sample_rate: int, channels: int,
                 hop_samples: int, capacity_hops: int):
        super().__init__(hop_samples, channels, capacity_hops)
        desc = (
            "alsasrc device={dev} ! audioconvert ! audioresample ! "
            "audio/x-raw,format=S16LE,rate={rate},channels={ch},layout=interleaved ! "
            "queue max-size-buffers=8 leaky=downstream ! "
            "appsink name=asink emit-signals=true sync=false max-buffers=16 drop=true"
        ).format(dev=device, rate=sample_rate, ch=channels)
        self._owner = GstPipelineOwner(desc, "alsa-src")
        self._rechunk = AppsinkRechunker(self.ring, hop_samples, channels, sample_rate)
        sink = self._owner.pipeline.get_by_name("asink")
        sink.connect("new-sample", self._rechunk.on_sample)

    def start(self) -> None:
        self._owner.start()
        log.info("Mode A live capture started")

    def stop(self) -> None:
        self._owner.stop()
        self.ring.close()

    @property
    def xruns(self) -> int:
        return self._owner.xruns
