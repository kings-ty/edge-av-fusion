"""Mode B: demux a smartphone .mp4 into clock-paced audio hops (+ optional
video frames), simulating a live robot POV.

Key choices (ARCHITECTURE §4):
- `sync=true` on both appsinks paces buffers against the shared pipeline clock
  → the file plays at 1.0× realtime and audio/video stay mutually synced via
  demuxer PTS, with zero bookkeeping on our side.
- Video is decoded by NVDEC (`nvv4l2decoder`) and stays in NVMM until
  `nvvidconv` converts to BGRx for the appsink. If/when the vision consumer is
  DeepStream/VPI, drop the conversion and pass NVMM caps straight through —
  that is the zero-copy path measured in bench/membw.py.
"""
import logging
import threading
from typing import Callable, Optional

import numpy as np

from .gst_base import AppsinkRechunker, Gst, GstPipelineOwner
from .source_base import AudioSource

log = logging.getLogger(__name__)

VideoCallback = Callable[[np.ndarray, float], None]  # (HxWx4 BGRx, pts_seconds)


class GstFileSource(AudioSource):
    def __init__(self, media_path: str, sample_rate: int, channels: int,
                 hop_samples: int, capacity_hops: int,
                 with_video: bool = False,
                 video_callback: Optional[VideoCallback] = None,
                 use_nvdec: bool = True):
        super().__init__(hop_samples, channels, capacity_hops)
        self._video_cb = video_callback

        audio_branch = (
            "demux.audio_0 ! queue ! decodebin ! audioconvert ! audioresample ! "
            "audio/x-raw,format=S16LE,rate={rate},channels={ch},layout=interleaved ! "
            "appsink name=asink emit-signals=true sync=true max-buffers=32"
        ).format(rate=sample_rate, ch=channels)

        if with_video:
            if use_nvdec:
                # parsebin: codec-agnostic (phones record H.264 or HEVC);
                # nvv4l2decoder handles both in hardware
                video_branch = (
                    " demux.video_0 ! queue ! parsebin ! nvv4l2decoder ! "
                    "nvvidconv ! video/x-raw,format=BGRx ! "
                    "appsink name=vsink emit-signals=true sync=true max-buffers=4 drop=true")
            else:  # system-memory decode path, kept for the membw A/B benchmark
                video_branch = (
                    " demux.video_0 ! queue ! decodebin ! videoconvert ! "
                    "video/x-raw,format=BGRx ! "
                    "appsink name=vsink emit-signals=true sync=true max-buffers=4 drop=true")
        else:
            video_branch = ""

        import os
        path = media_path
        if not os.path.isabs(path) and not os.path.exists(path):
            # Try relative to repo root (assuming this file is in src/avfusion/audio/)
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            alt = os.path.join(repo_root, "Edge-materials", path)
            if os.path.exists(alt):
                path = alt
            elif os.path.exists(os.path.join("Edge-materials", path)):
                path = os.path.abspath(os.path.join("Edge-materials", path))
        
        path = os.path.abspath(path)
        if not os.path.exists(path):
            log.error("Media path does not exist: %s", path)

        desc = 'filesrc location="%s" ! qtdemux name=demux %s%s' % (
            path, audio_branch, video_branch)
        self._owner = GstPipelineOwner(desc, "file-src")

        self._rechunk = AppsinkRechunker(self.ring, hop_samples, channels, sample_rate)
        asink = self._owner.pipeline.get_by_name("asink")
        asink.connect("new-sample", self._rechunk.on_sample)

        if with_video:
            vsink = self._owner.pipeline.get_by_name("vsink")
            vsink.connect("new-sample", self._on_video_sample)
        self._video_frames = 0
        self._lock = threading.Lock()

    def _on_video_sample(self, sink) -> "Gst.FlowReturn":
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.EOS
        buf = sample.get_buffer()
        caps = sample.get_caps().get_structure(0)
        w, h = caps.get_value("width"), caps.get_value("height")
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.ERROR
        try:
            frame = np.frombuffer(mapinfo.data, dtype=np.uint8)[: h * w * 4]
            frame = frame.reshape(h, w, 4).copy()
        finally:
            buf.unmap(mapinfo)
        with self._lock:
            self._video_frames += 1
        if self._video_cb is not None:
            pts = buf.pts / Gst.SECOND if buf.pts != Gst.CLOCK_TIME_NONE else 0.0
            self._video_cb(frame, pts)
        return Gst.FlowReturn.OK

    def start(self) -> None:
        self._owner.start()
        log.info("Mode B file streaming started (realtime-paced)")

    def stop(self) -> None:
        self._owner.stop()
        self.ring.close()

    @property
    def finished(self) -> bool:
        return self._owner.eos.is_set()

    @property
    def video_frames(self) -> int:
        with self._lock:
            return self._video_frames
