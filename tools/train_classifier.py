#!/usr/bin/env python3
"""T3.2 — fine-tune VehicleSoundNet with PyTorch DDP + Weights & Biases.

Written DDP-first: the same script runs single-GPU on the Xavier (smoke test)
and multi-GPU on a desktop / Kaggle 2xT4 without modification — launch decides:

  # Xavier smoke test (1 process, synthetic data, offline W&B):
  WANDB_MODE=offline torchrun --nproc_per_node=1 tools/train_classifier.py --synthetic

  # real run, 2 GPUs (Kaggle / desktop):
  torchrun --nproc_per_node=2 tools/train_classifier.py --data-root data/ESC-50-master

DDP notes (the parts interviewers ask about):
- torchrun sets RANK/LOCAL_RANK/WORLD_SIZE; we read env, never hardcode.
- DistributedSampler shards the dataset per rank; set_epoch() reshuffles.
- Gradients sync inside DDP's backward via bucketed all-reduce; the LR is
  scaled by world_size (linear scaling rule) since the effective batch grows.
- Backend: NCCL on x86 GPUs; gloo on Jetson (no NCCL on Tegra).
- W&B logs from rank 0 only — N ranks logging the same step is noise.

Dataset: ESC-50 (https://github.com/karolpiczak/ESC-50, CC BY-NC) mapped onto
our classes; folds 1-4 train, fold 5 validation (the dataset's own protocol).
Preprocessing MUST mirror dsp/mel.py (16 kHz, 64 mels, 25/10 ms, log,
per-patch standardize) — train/infer skew here silently destroys deployment
accuracy. AC: held-out F1(vehicle) >= 0.85.
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from avfusion.inference.model import VehicleSoundNet  # noqa: E402

CLASSES = ["vehicle_engine", "horn", "siren", "tire_road", "background"]
# ESC-50 categories -> our taxonomy. tire_road has no ESC-50 equivalent;
# "train" is the closest broadband-rolling proxy and is clearly marked so.
ESC50_MAP = {
    "engine": "vehicle_engine", "helicopter": "vehicle_engine",
    "car_horn": "horn",
    "siren": "siren",
    "train": "tire_road",
    # everything else becomes background (sampled, not exhaustive)
}
SAMPLE_RATE = 16000
PATCH_FRAMES = 96


def make_mel():
    """Identical params to dsp/mel.py — keep in sync (see module docstring)."""
    import torchaudio
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=512, win_length=400, hop_length=160,
        n_mels=64, center=False, power=2.0)


def standardize(mel: torch.Tensor) -> torch.Tensor:
    mel = torch.log(mel + 1e-6)[:, :PATCH_FRAMES]
    return (mel - mel.mean()) / (mel.std() + 1e-6)


class Esc50Vehicles(Dataset):
    """ESC-50 wavs -> standardized log-mel patches with our class mapping."""

    def __init__(self, root: str, train: bool, background_per_fold: int = 60):
        import torchaudio  # noqa: F401  (backend check at construction)
        self.root = Path(root)
        self.items = []
        rng = np.random.default_rng(0)
        background = []
        with open(self.root / "meta" / "esc50.csv") as fh:
            for row in csv.DictReader(fh):
                fold = int(row["fold"])
                if train != (fold != 5):          # folds 1-4 train, 5 val
                    continue
                label = ESC50_MAP.get(row["category"])
                entry = (str(self.root / "audio" / row["filename"]), label)
                if label is None:
                    background.append((entry[0], "background"))
                else:
                    self.items.append(entry)
        # cap background so it does not drown the vehicle classes 9:1
        rng.shuffle(background)
        self.items += background[: background_per_fold * (4 if train else 1)]
        self.mel = make_mel()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import torchaudio
        path, label = self.items[i]
        wav, sr = torchaudio.load(path)
        wav = wav.mean(dim=0)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        # random 0.96 s crop (train) / center crop (val)
        need = 160 * (PATCH_FRAMES - 1) + 512
        if wav.numel() < need:
            wav = torch.nn.functional.pad(wav, (0, need - wav.numel()))
        off = (np.random.randint(0, wav.numel() - need + 1)
               if self.train_mode else (wav.numel() - need) // 2)
        patch = standardize(self.mel(wav[off:off + need]))
        return patch.unsqueeze(0), CLASSES.index(label)

    train_mode = True


class SyntheticPatches(Dataset):
    """Random patches: verifies DDP/W&B mechanics without any download."""

    def __init__(self, n=512):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        label = i % len(CLASSES)
        x = torch.randn(1, 64, PATCH_FRAMES, generator=g)
        x[:, label * 12:(label + 1) * 12, :] += 2.0   # learnable band cue
        return x, label


# ---------------------------------------------------------------- dist utils
def setup_dist():
    """Init DDP when the environment supports it; degrade to single-process
    otherwise. The NVIDIA Jetson wheel is built with USE_DISTRIBUTED=0 (no
    NCCL on Tegra), so on the Xavier this script always takes the fallback —
    the edge box smoke-tests the training logic, multi-GPU runs happen on
    x86 (desktop / Kaggle 2xT4) with the exact same file."""
    if not dist.is_available():
        return 0, 1, 0, "none (single-process fallback)"
    if "RANK" not in os.environ:                 # plain `python` launch
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ["RANK"] = os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
    try:
        nccl_ok = torch.cuda.is_available() and torch.cuda.nccl.version() is not None
    except Exception:  # noqa: BLE001 - wheel-dependent attribute
        nccl_ok = False
    backend = "nccl" if nccl_ok else "gloo"
    dist.init_process_group(backend)
    rank = dist.get_rank()
    local = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    return rank, dist.get_world_size(), local, backend


def reduce_cat(t: torch.Tensor, world: int) -> torch.Tensor:
    if world == 1:
        return t
    out = [torch.zeros_like(t) for _ in range(world)]
    dist.all_gather(out, t)
    return torch.cat(out)


# --------------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/ESC-50-master")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32, help="per-GPU batch")
    ap.add_argument("--lr", type=float, default=3e-4, help="base LR @ 1 GPU")
    ap.add_argument("--out", default="models/vehicle.pt")
    ap.add_argument("--wandb-project", default="edge-av-fusion")
    args = ap.parse_args()

    rank, world, local, backend = setup_dist()
    device = torch.device("cuda", local) if torch.cuda.is_available() else "cpu"
    is_main = rank == 0

    if args.synthetic:
        train_ds, val_ds = SyntheticPatches(512), SyntheticPatches(128)
    else:
        train_ds = Esc50Vehicles(args.data_root, train=True)
        val_ds = Esc50Vehicles(args.data_root, train=False)
        val_ds.train_mode = False
    use_ddp = dist.is_available() and dist.is_initialized() and world > 1
    train_sampler = DistributedSampler(train_ds, shuffle=True) if use_ddp else None
    train_dl = DataLoader(train_ds, batch_size=args.batch, num_workers=2,
                          sampler=train_sampler, shuffle=train_sampler is None,
                          pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, num_workers=2,
                        sampler=DistributedSampler(val_ds, shuffle=False)
                        if use_ddp else None)

    model = VehicleSoundNet(num_classes=len(CLASSES)).to(device)
    if use_ddp:
        model = DDP(model, device_ids=[local] if device != "cpu" else None)
    lr = args.lr * world                          # linear scaling rule
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    run = None
    if is_main:
        import wandb
        run = wandb.init(project=args.wandb_project, job_type="train",
                         config={**vars(args), "world_size": world,
                                 "backend": backend, "effective_lr": lr,
                                 "classes": CLASSES,
                                 "device": torch.cuda.get_device_name(local)
                                 if device != "cpu" else "cpu"})

    veh_idx = torch.tensor([0, 1, 2, 3])          # vehicle classes
    for epoch in range(args.epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)        # reshuffle shards
        t0, seen, loss_sum = time.time(), 0, 0.0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()                       # all-reduce happens here
            opt.step()
            loss_sum += loss.item() * y.numel()
            seen += y.numel()
        sched.step()

        # ---- validation (gathered across ranks so metrics are global)
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for x, y in val_dl:
                preds.append(model(x.to(device)).argmax(1).cpu())
                gts.append(y)
        preds = reduce_cat(torch.cat(preds).to(device), world).cpu()
        gts = reduce_cat(torch.cat(gts).to(device), world).cpu()
        acc = (preds == gts).float().mean().item()
        veh_p, veh_t = torch.isin(preds, veh_idx), torch.isin(gts, veh_idx)
        tp = (veh_p & veh_t).sum().item()
        prec = tp / max(veh_p.sum().item(), 1)
        rec = tp / max(veh_t.sum().item(), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)

        if is_main:
            run.log({"epoch": epoch, "loss": loss_sum / max(seen, 1),
                     "val_acc": acc, "val_f1_vehicle": f1,
                     "lr": sched.get_last_lr()[0],
                     "epoch_s": time.time() - t0})
            print("epoch %3d  loss %.4f  acc %.3f  F1(veh) %.3f" %
                  (epoch, loss_sum / max(seen, 1), acc, f1))

    if is_main:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        state = model.module if hasattr(model, "module") else model
        torch.save({"model": state.state_dict(),
                    "classes": CLASSES}, args.out)
        art = None
        if run.settings.mode != "offline":
            import wandb
            art = wandb.Artifact("vehicle-soundnet", type="model")
            art.add_file(args.out)
            run.log_artifact(art)
        run.summary["final_f1_vehicle"] = f1
        run.finish()
        print("saved:", args.out,
              "| next: tools/export_onnx.py --checkpoint", args.out)
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
