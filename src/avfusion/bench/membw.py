"""T5.4 memory-bandwidth probe via tegrastats EMC counters.

tegrastats reports `EMC_FREQ <util>%@<MHz>`; on AGX Xavier the 256-bit
LPDDR4x bus moves 64 B/clock, so effective bandwidth is
    GB/s = util/100 * freq_MHz * 1e6 * 64 / 1e9      (peak 137 GB/s @ 2133 MHz)

Workloads measured:
- idle baseline (everything else is reported as delta over this)
- audio pipeline, synthetic source — included to *show* the audio path is
  bandwidth noise (~KB/hop, ARCHITECTURE §3.3), not to optimize it
- video decode of --media with NVMM kept end-to-end vs forced system-memory
  conversion — the comparison where unified-memory discipline actually pays

Usage: python3 -m avfusion.bench.membw --media test.mp4
"""
import argparse
import os
import pty
import re
import subprocess
import threading
import time
from typing import List, Optional

import numpy as np

from . import markdown_table, percentiles, write_report

_EMC_RE = re.compile(r"EMC_FREQ (\d+)%@(\d+)")
BYTES_PER_CLOCK = 64  # 256-bit LPDDR4x


class TegrastatsMonitor:
    """Background tegrastats reader; collects effective-GB/s samples."""

    def __init__(self, interval_ms: int = 100):
        # EMC counters need root on L4T R35; /etc/sudoers.d/avfusion-bench
        # grants passwordless tegrastats, so prefer sudo when it works
        # probe with `sudo -l <cmd>`: checks the tegrastats whitelist entry
        # itself — `sudo -n true` would need a broader grant and fails
        cmd = ["tegrastats", "--interval", str(interval_ms)]
        self._sudo = False
        try:
            if subprocess.run(["sudo", "-n", "-l", "tegrastats"],
                              capture_output=True, timeout=5).returncode == 0:
                cmd = ["sudo", "-n"] + cmd
                self._sudo = True
        except (OSError, subprocess.SubprocessError):
            pass
        # tegrastats block-buffers stdout when it is not a tty, so a plain
        # PIPE never delivers a line; a pty forces line buffering
        master, slave = pty.openpty()
        self._proc = subprocess.Popen(cmd, stdout=slave,
                                      stderr=subprocess.DEVNULL)
        os.close(slave)
        self._stdout = os.fdopen(master, "r", errors="replace")
        self.samples: List[float] = []
        self._collect = False
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        try:
            for line in self._stdout:
                m = _EMC_RE.search(line)
                if m and self._collect:
                    util, mhz = int(m.group(1)), int(m.group(2))
                    self.samples.append(
                        util / 100.0 * mhz * 1e6 * BYTES_PER_CLOCK / 1e9)
        except OSError:   # pty closed on shutdown
            pass

    def measure(self, seconds: float) -> List[float]:
        self.samples = []
        self._collect = True
        time.sleep(seconds)
        self._collect = False
        return list(self.samples)

    def close(self) -> None:
        if self._sudo:
            # the daemon runs as root: SIGTERM from the user is EPERM, but
            # `tegrastats --stop` is covered by the same sudoers entry
            subprocess.run(["sudo", "-n", "tegrastats", "--stop"],
                           capture_output=True, timeout=10)
        else:
            self._proc.terminate()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        self._stdout.close()


def run_audio_pipeline(seconds: float) -> None:
    from ..bench.latency import SyntheticSource
    from ..config import load_config
    from ..pipeline import Pipeline
    cfg = load_config(None)
    n_hops = int(seconds * cfg.audio.sample_rate / cfg.audio.hop_samples) * 50
    src = SyntheticSource(cfg.audio.sample_rate, cfg.audio.channels,
                          cfg.audio.hop_samples, n_hops)
    pipe = Pipeline(cfg, src, keep_stage_times=False)
    pipe.start()
    time.sleep(seconds)
    pipe.stop()


def gst_video_cmd(media: str, nvmm: bool) -> List[str]:
    if nvmm:
        chain = ("parsebin ! nvv4l2decoder ! "
                 "'video/x-raw(memory:NVMM)' ! fakesink sync=false")
    else:
        chain = ("parsebin ! nvv4l2decoder ! nvvidconv ! "
                 "video/x-raw,format=BGRx ! videoconvert ! "
                 "video/x-raw,format=RGB ! fakesink sync=false")
    return ["bash", "-c",
            "gst-launch-1.0 -q filesrc location='%s' ! qtdemux ! %s" % (media, chain)]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--media", default="", help="mp4 for the video A/B test")
    ap.add_argument("--seconds", type=float, default=10.0, help="per workload")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    mon = TegrastatsMonitor()
    time.sleep(1.0)  # let tegrastats start emitting
    results = {}
    try:
        print("measuring idle baseline ...")
        results["idle"] = mon.measure(args.seconds)

        print("measuring audio pipeline (synthetic) ...")
        worker = threading.Thread(target=run_audio_pipeline,
                                  args=(args.seconds + 1.0,), daemon=True)
        worker.start()
        time.sleep(0.5)
        results["audio_pipeline"] = mon.measure(args.seconds)
        worker.join()

        if args.media:
            for name, nvmm in (("video_nvmm", True), ("video_system_mem", False)):
                print("measuring %s ..." % name)
                proc = subprocess.Popen(gst_video_cmd(args.media, nvmm),
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL)
                time.sleep(1.0)
                results[name] = mon.measure(args.seconds)
                proc.terminate()
                proc.wait()
    finally:
        mon.close()

    if not any(results.values()):
        raise SystemExit("no EMC samples — tegrastats may need sudo on this "
                         "L4T build (try: sudo python3 -m avfusion.bench.membw)")

    idle_mean = float(np.mean(results["idle"])) if results["idle"] else 0.0
    rows, payload = [], {"bytes_per_clock": BYTES_PER_CLOCK, "workloads": {}}
    for name, samples in results.items():
        p = percentiles(samples)
        payload["workloads"][name] = p
        rows.append([name, p["n"], "%.2f" % p["mean"], "%.2f" % p["p95"],
                     "%+.2f" % (p["mean"] - idle_mean)])
    delta_note = ""
    if "video_nvmm" in results and results["video_nvmm"]:
        d = (np.mean(results["video_system_mem"]) - np.mean(results["video_nvmm"]))
        delta_note = ("\n\nNVMM saves **%.2f GB/s** of EMC traffic on the video "
                      "branch; the audio row's delta vs idle is the honest "
                      "evidence that unified-memory work on the audio path "
                      "would be theater." % d)
    md = ("# EMC bandwidth by workload (GB/s, %.0fs each)\n\n%s%s" % (
        args.seconds,
        markdown_table(["workload", "samples", "mean", "p95", "Δ vs idle"], rows),
        delta_note))
    path = write_report("membw", payload, md, tag=args.tag)
    print(md)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
