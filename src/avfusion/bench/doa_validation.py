"""T5.5 DoA accuracy validation.

Two modes:
- synthetic (default, runs in CI / on a dev machine): stereo generated with
  exact frequency-domain fractional delays at known angles, SNR-swept; reports
  per-angle bias/RMSE and checks the acceptance thresholds (broadside RMSE
  < 3 deg, +/-60 deg RMSE < 12 deg). Exit code 1 on failure so CI can gate.
- --field prints the hardware measurement protocol (T5.5 AC) for use with the
  real ReSpeaker; analysis of recorded field WAVs reuses the same code path.

Endfire (+/-90 deg) rows are reported but not gated: arcsin saturation makes
the estimate one-sided there by geometry, not by bug (ARCHITECTURE §5).
"""
import argparse
import sys

import numpy as np

from . import markdown_table, write_report
from ..config import load_config
from ..dsp.gcc_phat import GccPhatEstimator
from .latency import synth_stereo

ANGLES = [0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0]

FIELD_PROTOCOL = """\
# DoA field validation protocol (hardware)

Setup: robot static on tripod-marked floor grid; ReSpeaker axis aligned with
the 0-deg mark via laser line. Powered speaker on a stand at mic height.

1. Positions: angles {0, +/-30, +/-60, +/-90} deg at BOTH 1 m and 3 m
   (14 positions). Mark each with tape; measure angle with a protractor
   template printed from docs/, not by eye (+/-1 deg matters at d=5.8 cm).
2. Stimuli per position, 10 s each, calibrated to ~70 dB SPL at the array:
   a) logarithmic chirp 100 Hz - 8 kHz   (broadband, best case)
   b) pink noise                          (diffuse-spectrum reference)
   c) recorded idling engine + a pass-by clip (the actual target class)
3. Record raw 48 kHz stereo (arecord -D hw:seeed2micvoicec -r 48000 -c 2
   -f S16_LE pos_<angle>_<dist>_<stim>.wav) and the ambient floor (60 s,
   no source) for the gate's noise model.
4. Between positions inject 10 s of silence -> verifies false-trigger rate.
5. Analyze with this module's estimator (same config as production); report
   per-angle bias, RMSE, valid-rate, and the endfire confusion table.
Repeat the 1 m row once with the robot's own motors running for ego-noise.
"""


def run_synthetic(args) -> int:
    cfg = load_config(args.config)
    d = cfg.doa
    rows, payload, failures = [], {"angles": {}}, []

    for angle in ANGLES:
        errors = []
        for trial in range(args.trials):
            for snr in (20.0, 10.0):
                est = GccPhatEstimator(
                    sample_rate=cfg.audio.sample_rate,
                    window_samples=d.window_samples,
                    mic_spacing_m=d.mic_spacing_m, backend="numpy",
                    gate_history=d.gate.history, gate_z=d.gate.z_threshold,
                    gate_min_ratio=d.gate.min_peak_ratio)
                x = synth_stereo(d.window_samples, cfg.audio.sample_rate,
                                 angle, d.mic_spacing_m, snr_db=snr,
                                 seed=hash((trial, snr, angle)) % 2**31)
                r = est.estimate(x)
                errors.append(r.angle_deg - angle)
        e = np.asarray(errors)
        bias, rmse = float(e.mean()), float(np.sqrt(np.mean(e ** 2)))
        payload["angles"][str(angle)] = {"bias": bias, "rmse": rmse,
                                         "n": int(e.size)}
        gate = "-"
        if angle == 0.0:
            gate = "< 3"
            if rmse >= 3.0:
                failures.append("broadside RMSE %.2f >= 3" % rmse)
        elif abs(angle) == 60.0:
            gate = "< 12"
            if rmse >= 12.0:
                failures.append("%+.0f deg RMSE %.2f >= 12" % (angle, rmse))
        rows.append(["%+.0f" % angle, len(e), "%.2f" % bias, "%.2f" % rmse, gate])

    md = ("# DoA synthetic validation — bias/RMSE (deg), SNR {20,10} dB\n\n%s"
          "\n\nResult: %s" % (
              markdown_table(["true angle", "n", "bias", "RMSE", "AC gate"], rows),
              "FAIL — " + "; ".join(failures) if failures else "PASS"))
    payload["pass"] = not failures
    path = write_report("doa_validation", payload, md, tag=args.tag)
    print(md)
    print("\nwrote", path)
    return 1 if failures else 0


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=30, help="per angle per SNR")
    ap.add_argument("--config", default=None)
    ap.add_argument("--field", action="store_true",
                    help="print the hardware measurement protocol and exit")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    if args.field:
        print(FIELD_PROTOCOL)
        return
    sys.exit(run_synthetic(args))


if __name__ == "__main__":
    main()
