"""Benchmark harness (ARCHITECTURE §6). Shared report/percentile helpers.

Every benchmark writes a JSON record plus a rendered markdown table into
bench_results/, tagged with timestamp and the active nvpmodel mode, so results
from different power modes / code versions stay comparable.
"""
import datetime
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parents[3] / "bench_results"


def percentiles(samples: Sequence[float]) -> Dict[str, float]:
    """p50/p95/p99 + mean over a sample list (ms in, ms out)."""
    if not len(samples):
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    a = np.asarray(samples, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
    }


def nvpmodel_mode() -> str:
    """Active power-mode name, or 'unknown' off-Jetson / without permission."""
    try:
        out = subprocess.run(["nvpmodel", "-q"], capture_output=True,
                             text=True, timeout=5).stdout
        for line in out.splitlines():
            if "NV Power Mode" in line:
                return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def write_report(name: str, payload: dict, markdown: str,
                 tag: Optional[str] = None) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = "%s_%s%s" % (name, stamp, ("_" + tag) if tag else "")
    payload = dict(payload, timestamp=stamp, nvpmodel=nvpmodel_mode())
    (RESULTS_DIR / (base + ".json")).write_text(
        json.dumps(payload, indent=2, default=float))
    md_path = RESULTS_DIR / (base + ".md")
    md_path.write_text(markdown + "\n")
    return md_path
