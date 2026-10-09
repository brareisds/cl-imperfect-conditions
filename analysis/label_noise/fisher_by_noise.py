import csv
import glob
import os
import statistics
from collections import defaultdict

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
NOISE_ROOT = os.path.join(ROOT, "results/label_noise")
OUT_DIR = os.path.join(ROOT, "analysis", "label_noise")

DATASETS = ["organamnist", "bloodmnist"]
METHODS = ["ewc"]
NOISES = ["0.01", "0.1", "0.2", "0.3"]
SEEDS = range(5)
TASKS = range(4)


def find(d, m, p, s, t):
    pat = os.path.join(NOISE_ROOT, f"{d}_{m}_p{p}", f"*_fixd_{s}", "fisher", f"task{t}_raw.pt")
    hits = glob.glob(pat)
    return hits[0] if len(hits) == 1 else None


def load(path):
    try:
        x = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        x = torch.load(path, map_location="cpu")
    return torch.cat([x[k].flatten().double() for k in sorted(x)])


rows = []
for d in DATASETS:
    for m in METHODS:
        for s in SEEDS:
            for t in TASKS:
                cp = find(d, m, "0", s, t)
                if cp is None:
                    print(f"WARN missing clean Fisher: {d} {m} seed{s} task{t}")
                    continue
                c = load(cp)
                mc = c.mean().item()
                rows.append([d, m, "0", s, t, mc, 0.0, 0.0, c.max().item() / mc, 1.0])
                for p in NOISES:
                    npth = find(d, m, p, s, t)
                    if npth is None:
                        print(f"WARN missing noisy Fisher: {d} {m} p{p} seed{s} task{t}")
                        continue
                    n = load(npth)
                    assert n.numel() == c.numel(), "Fisher size mismatch"
                    mn = n.mean().item()
                    signed = 100 * (mn - mc) / mc
                    cos = (torch.dot(c, n) / (c.norm() * n.norm())).item()
                    rows.append([d, m, p, s, t, mn, signed, abs(signed), n.max().item() / mn, cos])

with open(os.path.join(OUT_DIR, "fisher_by_noise.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "method", "noise", "seed", "task", "mean_magnitude", "signed_change_pct",
                "abs_change_pct", "peak_ratio", "cosine_vs_clean"])
    w.writerows([r[:5] + [f"{x:.6g}" for x in r[5:]] for r in rows])

g = defaultdict(list)
for r in rows:
    g[(r[0], r[1], r[2])].append(r)
print(f"{'dataset':12s} {'method':6s} {'p':>5s} {'n':>3s} {'mean_mag':>10s} {'abs_chg%':>9s} {'signed%':>8s} {'up':>5s} {'peak':>8s} {'cos':>6s}")
for d in DATASETS:
    for m in METHODS:
        for p in ["0"] + NOISES:
            v = g.get((d, m, p), [])
            if not v:
                continue
            med = lambda i: statistics.median(x[i] for x in v)
            up = sum(1 for x in v if x[6] > 0)
            print(f"{d:12s} {m:6s} {p:>5s} {len(v):3d} {med(5):10.3e} {med(7):9.1f} {med(6):8.1f} {up:>3d}/{len(v):<2d} {med(8):8.1f} {med(9):6.3f}")
