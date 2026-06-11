#!/usr/bin/env bash
# T0.2 — Python env for edge-av-fusion on Jetson Xavier AGX (JetPack 5, L4T R35).
#
# Creates .venv with --system-site-packages on /usr/bin/python3.8 (keeps system
# tensorrt / gi / rclpy visible), installs the NVIDIA JetPack torch wheel
# (CUDA 11.4, cp38 aarch64), then builds torchaudio from source against it
# (NVIDIA ships no torchaudio wheel for JP5 — the build takes 30–60 min).
#
# Run apt deps ONCE interactively first (sudo needs a tty):
#   sudo apt-get update && sudo apt-get install -y \
#     libopenblas-base libopenmpi-dev git pkg-config
#
# Then run this script detached:
#   nohup bash scripts/setup_env.sh > ~/setup_env.log 2>&1 &
#   tail -f ~/setup_env.log
#
# AC (printed at the end):
#   python3.8 -c "import torch,torchaudio,tensorrt,gi; print(torch.cuda.is_available())" -> True
set -euo pipefail

# conda hijacks PATH on this machine and breaks everything ROS/Jetson; go clean
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO/.venv"
BUILD="$REPO/.build"
PY=/usr/bin/python3.8

# JetPack 5.1.x / CUDA 11.4 wheel (v512 redist works on L4T R35.x)
TORCH_WHL="https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl"
TORCHAUDIO_TAG="v2.1.0"   # must match the torch minor version

echo "=== [1/5] venv (${VENV}) ==="
test -x "$PY" || { echo "FATAL: $PY not found"; exit 1; }
"$PY" -m venv --system-site-packages "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip wheel setuptools

echo "=== [2/5] torch (NVIDIA JetPack wheel) ==="
python -c "import torch" 2>/dev/null && echo "torch already present, skipping" || \
    pip install --no-cache-dir "$TORCH_WHL"
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "torch installed but CUDA unavailable"
print("torch %s, CUDA OK, device: %s" % (torch.__version__,
                                         torch.cuda.get_device_name(0)))
EOF

echo "=== [3/5] torchaudio build deps ==="
# packaging>=22: the system packaging 20.3 leaks through --system-site-packages
# and breaks setuptools>=71 metadata (canonicalize_version kwarg).
# pytest pinned to 7.x: Foxy's launch_testing pytest plugin (auto-loaded via
# the user's sourced ROS env) is incompatible with pytest >= 8.1.
pip install --no-cache-dir "cmake>=3.18" ninja "packaging>=22" "pytest==7.4.4"

echo "=== [4/5] torchaudio ${TORCHAUDIO_TAG} from source ==="
if python -c "import torchaudio" 2>/dev/null; then
    echo "torchaudio already present, skipping"
else
    mkdir -p "$BUILD"
    if [ ! -d "$BUILD/audio" ]; then
        git clone --depth 1 -b "$TORCHAUDIO_TAG" \
            https://github.com/pytorch/audio.git "$BUILD/audio"
    fi
    cd "$BUILD/audio"
    # core transforms only: no sox/ffmpeg backends, CUDA on
    USE_CUDA=1 BUILD_SOX=0 USE_FFMPEG=0 USE_ROCM=0 \
    MAX_JOBS="$(nproc)" \
        pip install --no-cache-dir --no-build-isolation -v .
fi

echo "=== [5/5] acceptance check (T0.2 AC) ==="
python - <<'EOF'
import gi
import tensorrt
import torch
import torchaudio
print("tensorrt", tensorrt.__version__)
print("torchaudio", torchaudio.__version__)
m = torchaudio.transforms.MelSpectrogram(16000, n_mels=64).to("cuda")
x = torch.randn(1, 16000, device="cuda")
print("mel on GPU:", tuple(m(x).shape))
print("torch.cuda.is_available():", torch.cuda.is_available())
EOF
echo "=== T0.2 DONE — activate with: source $REPO/.venv/bin/activate ==="
