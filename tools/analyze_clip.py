#!/usr/bin/env python3.8
"""Run a real-world clip through the full pipeline (Mode B) and summarize
what the acoustic stack did with it: gate accept rate, DoA activity, FSM
state occupancy, classifier calls.

This is the "real noise" companion to the synthetic benches: synthetic input
proves the math against known ground truth; a street clip shows how the
adaptive gate and FSM behave against wind/footsteps/speech they were never
told about. There is no ground-truth angle here — read DoA columns as
left/right activity, not calibrated bearings (phone mic geometry != 5.8 cm).

Usage: python tools/analyze_clip.py ~/street_test.mp4 [--no-classifier]
"""
import argparse
import collections
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from avfusion.audio.gst_file_source import GstFileSource  # noqa: E402
from avfusion.config import load_config                    # noqa: E402
from avfusion.pipeline import Pipeline                     # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("media")
    ap.add_argument("--no-classifier", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    src = GstFileSource(args.media, cfg.audio.sample_rate, cfg.audio.channels,
                        cfg.audio.hop_samples, cfg.audio.ring_capacity_hops)

    states = collections.Counter()
    angles = []
    alerts = 0
    vision_requests = 0
    track_ids = set()

    def on_event(event, st):
        nonlocal alerts, vision_requests
        states[event.state.value] += 1
        if event.track is not None and not event.track.coasting:
            angles.append(event.track.angle_deg)
            track_ids.add(event.track.track_id)
        alerts += event.alert
        vision_requests += event.request_vision

    pipe = Pipeline(cfg, src, event_cb=on_event,
                    enable_classifier=not args.no_classifier)
    t0 = time.monotonic()
    pipe.start()
    try:
        while pipe._thread.is_alive() and not src.finished:
            pipe._thread.join(timeout=0.5)
        pipe._thread.join(timeout=2.0)
    finally:
        pipe.stop()
    wall = time.monotonic() - t0

    s = pipe.stats
    total = sum(states.values()) or 1
    print("\n=== clip analysis: %s ===" % args.media)
    print("wall %.1f s | hops %d | dropped %d | classifier calls %d%s"
          % (wall, s.hops, s.dropped_hops, s.classifications,
             " (DEGRADED)" if s.degraded else ""))
    print("gate/tracker: %d hops with an active track (%.1f%%), %d track(s)"
          % (len(angles), 100.0 * len(angles) / total, len(track_ids)))
    if angles:
        import numpy as np
        a = np.asarray(angles)
        hist, edges = np.histogram(a, bins=[-90, -60, -30, -10, 10, 30, 60, 90])
        print("DoA activity (deg buckets):")
        for n, lo, hi in zip(hist, edges[:-1], edges[1:]):
            print("  [%+4.0f, %+4.0f): %s" % (lo, hi, "#" * int(40 * n / max(hist.max(), 1))))
    print("FSM occupancy: " + ", ".join(
        "%s %.1f%%" % (k, 100.0 * v / total) for k, v in states.most_common()))
    print("alert hops: %d | vision requests: %d" % (alerts, vision_requests))


if __name__ == "__main__":
    main()
