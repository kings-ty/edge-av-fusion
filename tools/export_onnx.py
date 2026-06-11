#!/usr/bin/env python3.8
"""Export the vehicle-sound classifier to ONNX (TRT-ready).

Usage:
  python3.8 tools/export_onnx.py --checkpoint models/vehicle.pt --out models/vehicle.onnx
  python3.8 tools/export_onnx.py --arch panns --out models/vehicle.onnx   # PANNs CNN10 head

Static shape [1,1,64,96], opset 13 (well inside TRT 8.5 coverage). Parity vs
torch is checked with onnxruntime when available.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from avfusion.inference.model import VehicleSoundNet  # noqa: E402

INPUT_SHAPE = (1, 1, 64, 96)


def build_model(arch: str, num_classes: int, checkpoint: str):
    if arch == "vehiclesoundnet":
        model = VehicleSoundNet(num_classes=num_classes)
    elif arch == "panns":
        # transfer-learning path: PANNs CNN10 trunk + linear head
        from panns_inference.models import Cnn10  # pip install panns-inference
        trunk = Cnn10(sample_rate=16000, window_size=400, hop_size=160,
                      mel_bins=64, fmin=0, fmax=8000, classes_num=num_classes)
        model = trunk
    else:
        raise SystemExit("unknown arch %s" % arch)
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state.get("model", state))
        print("loaded checkpoint:", checkpoint)
    else:
        print("WARNING: exporting with random weights (pipeline bring-up only)")
    return model.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="vehiclesoundnet",
                    choices=["vehiclesoundnet", "panns"])
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--num-classes", type=int, default=5)
    ap.add_argument("--out", default="models/vehicle.onnx")
    args = ap.parse_args()

    model = build_model(args.arch, args.num_classes, args.checkpoint)
    dummy = torch.randn(*INPUT_SHAPE)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, args.out, opset_version=13,
        input_names=["mel"], output_names=["logits"],
        do_constant_folding=True)
    print("exported:", args.out)

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        # torch reference on CUDA: the JP5 NVIDIA wheel's CPU GEMM emits NaNs
        # on Carmel cores (1x1 conv), and production never runs torch on CPU
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.no_grad():
            ref = model.to(dev)(dummy.to(dev)).cpu().numpy()
        out = sess.run(None, {"mel": dummy.numpy()})[0]
        err = float(np.abs(ref - out).max())
        print("onnxruntime parity: max|err| = %.2e %s"
              % (err, "OK" if err < 1e-4 else "FAIL"))
        if err >= 1e-4:
            raise SystemExit(1)
    except ImportError:
        print("onnxruntime not installed; skipping parity check")


if __name__ == "__main__":
    main()
