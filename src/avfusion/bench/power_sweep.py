"""T5.3 power-mode sweep: rerun the latency bench under MAXN / 30W / 15W.

Each mode's bench runs in a fresh subprocess (clean allocators, cold caches —
the same conditions a deployment would see after boot in that mode). The
previously active mode is restored afterwards, even on Ctrl-C.

Needs passwordless sudo for `nvpmodel -m` (or run the whole sweep under sudo):
  sudo visudo -f /etc/sudoers.d/nvpmodel   ->
  <user> ALL=(root) NOPASSWD: /usr/sbin/nvpmodel

Usage: python3 -m avfusion.bench.power_sweep --hops 6000
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from . import RESULTS_DIR, markdown_table, write_report

# AGX Xavier JetPack 5 mode table (nvpmodel.conf): 0=MAXN, 3=30W(all), 2=15W
MODES = [(0, "MAXN"), (3, "30W"), (2, "15W")]


def _nvpmodel(*args: str) -> str:
    for cmd in (["nvpmodel", *args], ["sudo", "-n", "nvpmodel", *args]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if r.returncode == 0:
                return r.stdout
        except (OSError, subprocess.SubprocessError):
            continue
    raise SystemExit("nvpmodel %s failed — configure passwordless sudo "
                     "(see module docstring)" % " ".join(args))


def current_mode_id() -> int:
    m = re.search(r"^(\d+)\s*$", _nvpmodel("-q"), re.MULTILINE)
    return int(m.group(1)) if m else 0


def jetson_clocks_show() -> str:
    try:
        return subprocess.run(["sudo", "-n", "jetson_clocks", "--show"],
                              capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def latest_latency_json(tag: str) -> dict:
    files = sorted(RESULTS_DIR.glob("latency_*_%s.json" % tag))
    return json.loads(files[-1].read_text()) if files else {}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hops", type=int, default=6000)
    ap.add_argument("--gcc-backend", default="")
    ap.add_argument("--settle-seconds", type=float, default=5.0,
                    help="DVFS settle time after a mode switch")
    args = ap.parse_args(argv)

    prev = current_mode_id()
    print("active nvpmodel mode: %d (will be restored)" % prev)
    per_mode, clocks = {}, {}
    try:
        for mode_id, name in MODES:
            print("\n=== nvpmodel -m %d (%s) ===" % (mode_id, name))
            _nvpmodel("-m", str(mode_id))
            time.sleep(args.settle_seconds)
            clocks[name] = jetson_clocks_show()

            tag = "mode%d" % mode_id
            cmd = [sys.executable, "-m", "avfusion.bench.latency",
                   "--hops", str(args.hops), "--tag", tag]
            if args.gcc_backend:
                cmd += ["--gcc-backend", args.gcc_backend]
            subprocess.run(cmd, check=True,
                           cwd=str(Path(__file__).resolve().parents[2]))
            per_mode[name] = latest_latency_json(tag)
    finally:
        print("\nrestoring nvpmodel mode %d" % prev)
        _nvpmodel("-m", str(prev))

    stage_names = list(next(iter(per_mode.values()))["stages_ms"].keys())
    rows = []
    for stage in stage_names:
        row = [stage]
        for _, name in MODES:
            p = per_mode.get(name, {}).get("stages_ms", {}).get(stage, {})
            row.append("%.3f / %.3f" % (p.get("p50", 0), p.get("p95", 0)))
        rows.append(row)
    md = ("# Power-mode sweep — p50 / p95 ms per stage\n\n%s\n\n"
          "Per-mode jetson_clocks snapshots are embedded in the JSON record."
          % markdown_table(["stage"] + [n for _, n in MODES], rows))
    path = write_report("power_sweep", {"modes": per_mode, "clocks": clocks}, md)
    print("\n" + md)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
