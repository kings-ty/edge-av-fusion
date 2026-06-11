"""Shared GStreamer plumbing: appsink → hop-aligned ring-buffer writes.

GStreamer buffers arrive in arbitrary sizes; we re-chunk into exact hops here
(in the streaming thread, which holds no Python state besides this remainder)
so every downstream consumer sees fixed-size frames.
"""
import logging
import threading
from typing import Optional

import numpy as np

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GstApp  # noqa: E402

from .ring_buffer import RingBuffer  # noqa: E402

log = logging.getLogger(__name__)

Gst.init(None)


class AppsinkRechunker:
    """Accumulates raw S16LE interleaved audio from an appsink and emits
    exact (hop_samples, channels) hops into a RingBuffer with interpolated PTS."""

    def __init__(self, ring: RingBuffer, hop_samples: int, channels: int,
                 sample_rate: int):
        self._ring = ring
        self._hop = hop_samples
        self._ch = channels
        self._rate = sample_rate
        self._rem = np.zeros((0, channels), dtype=np.int16)
        self._rem_pts: Optional[float] = None

    def on_sample(self, sink) -> Gst.FlowReturn:
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.EOS
        buf = sample.get_buffer()
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.ERROR
        try:
            data = np.frombuffer(mapinfo.data, dtype=np.int16)
        finally:
            buf.unmap(mapinfo)
        frames = data.reshape(-1, self._ch)
        pts = buf.pts / Gst.SECOND if buf.pts != Gst.CLOCK_TIME_NONE else None

        if self._rem.shape[0] == 0:
            self._rem_pts = pts
        self._rem = np.concatenate([self._rem, frames]) if self._rem.size else frames.copy()

        base_pts = self._rem_pts if self._rem_pts is not None else 0.0
        emitted = 0
        while self._rem.shape[0] >= self._hop:
            hop = self._rem[:self._hop]
            hop_pts = base_pts + emitted * self._hop / self._rate
            self._ring.push(np.ascontiguousarray(hop), hop_pts)
            self._rem = self._rem[self._hop:]
            emitted += 1
        if emitted:
            self._rem_pts = base_pts + emitted * self._hop / self._rate
        return Gst.FlowReturn.OK


class GstPipelineOwner:
    """Owns a Gst pipeline + a GLib-free polling bus watcher thread."""

    def __init__(self, description: str, name: str):
        log.info("gst-launch %s", description)
        self.pipeline = Gst.parse_launch(description)
        self._name = name
        self._bus_thread: Optional[threading.Thread] = None
        self._running = False
        self.eos = threading.Event()
        self.error: Optional[str] = None
        self.xruns = 0

    def start(self) -> None:
        self._running = True
        self._bus_thread = threading.Thread(
            target=self._bus_loop, name="%s-bus" % self._name, daemon=True)
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("pipeline %s failed to start" % self._name)
        self._bus_thread.start()

    def stop(self) -> None:
        self._running = False
        self.pipeline.set_state(Gst.State.NULL)
        if self._bus_thread is not None:
            self._bus_thread.join(timeout=2.0)

    def _bus_loop(self) -> None:
        bus = self.pipeline.get_bus()
        while self._running:
            msg = bus.timed_pop_filtered(
                200 * Gst.MSECOND,
                Gst.MessageType.ERROR | Gst.MessageType.EOS
                | Gst.MessageType.WARNING | Gst.MessageType.ELEMENT)
            if msg is None:
                continue
            if msg.type == Gst.MessageType.ERROR:
                err, dbg = msg.parse_error()
                self.error = "%s (%s)" % (err.message, dbg)
                log.error("[%s] %s", self._name, self.error)
                self.eos.set()
            elif msg.type == Gst.MessageType.EOS:
                log.info("[%s] EOS", self._name)
                self.eos.set()
            elif msg.type == Gst.MessageType.WARNING:
                warn, _ = msg.parse_warning()
                if "xrun" in warn.message.lower() or "lost" in warn.message.lower():
                    self.xruns += 1
                log.warning("[%s] %s", self._name, warn.message)
