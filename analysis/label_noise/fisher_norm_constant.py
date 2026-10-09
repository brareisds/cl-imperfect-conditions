"""FISHER_NORM constant: median mean-Fisher over clean (p=0) EWC runs.
Usage: python analysis/label_noise/fisher_norm_constant.py --root results/label_noise --dataset organamnist"""
import argparse
import glob
import os
import statistics

import torch


def mean_of(path):
    try:
        x = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        x = torch.load(path, map_location="cpu")
    return sum(v.double().sum().item() for v in x.values()) / sum(v.numel() for v in x.values())


ap = argparse.ArgumentParser()
ap.add_argument("--root", default="results/label_noise")
ap.add_argument("--dataset", required=True)
ap.add_argument("--noise", default="0")
a = ap.parse_args()
files = sorted(glob.glob(os.path.join(a.root, f"{a.dataset}_ewc_p{a.noise}", "*_fixd_*", "fisher", "task*_raw.pt")))
vals = [mean_of(f) for f in files]
print(f"{len(vals)} files; median={statistics.median(vals):.6e}  min={min(vals):.3e}  max={max(vals):.3e}")
print(f"FISHER_NORM={statistics.median(vals):.6e}")
