#!/usr/bin/env python3.8
"""Build a TensorRT engine from ONNX via trtexec, with a recorded build report.

Why trtexec instead of the Python builder API: identical output engine, but we
get layer-precision and per-layer profile dumps for free, and the exact
reproducible command line lands in the build report — reviewers can rerun it.

Usage:
  python3.8 tools/build_trt_engine.py --onnx models/vehicle.onnx \
      --out models/vehicle_fp16.plan [--fp32] [--int8 --calib-dir data/calib]

Engines are device+TRT-version specific: always build on the Xavier itself.
"""
import argparse
import datetime
import subprocess
import sys
from pathlib import Path

TRTEXEC = "/usr/src/tensorrt/bin/trtexec"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fp32", action="store_true", help="disable FP16 (debug)")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--calib-dir", default="")
    ap.add_argument("--workspace-mb", type=int, default=1024)
    args = ap.parse_args()

    if not Path(TRTEXEC).exists():
        sys.exit("trtexec not found at %s (JetPack TensorRT missing?)" % TRTEXEC)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        TRTEXEC,
        "--onnx=%s" % args.onnx,
        "--saveEngine=%s" % args.out,
        "--memPoolSize=workspace:%dM" % args.workspace_mb,
        "--dumpLayerInfo", "--dumpProfile", "--separateProfileRun",
        "--iterations=200", "--warmUp=500",
    ]
    if not args.fp32:
        cmd.append("--fp16")
    if args.int8:
        if not args.calib_dir:
            sys.exit("--int8 requires --calib-dir with representative mel patches")
        cmd.append("--int8")  # calibration cache wiring left to the data phase (T3.2)

    report = Path(args.out).with_suffix(".buildlog.txt")
    print(" ".join(cmd))
    with open(report, "w") as fh:
        fh.write("# built %s\n# %s\n\n" % (datetime.datetime.now().isoformat(),
                                           " ".join(cmd)))
        fh.flush()
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        sys.exit("trtexec failed (see %s)" % report)
    print("engine: %s\nbuild report: %s" % (args.out, report))


if __name__ == "__main__":
    main()
