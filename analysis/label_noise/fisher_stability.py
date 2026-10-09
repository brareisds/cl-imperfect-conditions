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
ITEMS = [(t, "raw") for t in range(4)] + [(3, "accumulated")]


def find(d, m, p, s, t, kind):
    pat = os.path.join(NOISE_ROOT, f"{d}_{m}_p{p}", f"*_fixd_{s}", "fisher", f"task{t}_{kind}.pt")
    hits = glob.glob(pat)
    return hits[0] if len(hits) == 1 else None


def load(path):
    try:
        x = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        x = torch.load(path, map_location="cpu")
    return torch.cat([x[k].flatten().double() for k in sorted(x)])


def compare(c, n):
    assert c.numel() == n.numel(), "Fisher size mismatch"
    mc, mn = c.mean().item(), n.mean().item()
    dev = 100 * abs(mn - mc) / mc
    cos = (torch.dot(c, n) / (c.norm() * n.norm())).item()
    return dev, cos, c.max().item() / mc, n.max().item() / mn


def sanity():
    mine = find("organamnist", "ewc", "0", 0, 0, "raw")
    old = glob.glob(os.path.join(ROOT, "results_organamnist_shuffle_fisher", "organamnist_ewc_seed0",
                                 "**", "fisher", "task0_raw.pt"), recursive=True)
    if mine and len(old) == 1:
        a, b = load(mine), load(old[0])
        print(f"sanity: p0 vs results_organamnist_shuffle_fisher (organamnist ewc seed0 task0 raw): "
              f"max|diff|={(a - b).abs().max().item():.3e}")
    else:
        print("sanity: files not found", mine, old)


def main():
    sanity()
    rows = []
    for d in DATASETS:
        for m in METHODS:
            for s in SEEDS:
                for t, kind in ITEMS:
                    cp = find(d, m, "0", s, t, kind)
                    if cp is None:
                        print(f"WARN missing clean: {d} {m} seed{s} task{t} {kind}")
                        continue
                    c = load(cp)
                    for p in NOISES:
                        npth = find(d, m, p, s, t, kind)
                        if npth is None:
                            print(f"WARN missing noisy: {d} {m} p{p} seed{s} task{t} {kind}")
                            continue
                        dev, cos, pk_c, pk_n = compare(c, load(npth))
                        rows.append([d, m, p, s, t, kind, dev, cos, pk_c, pk_n])
    with open(os.path.join(OUT_DIR, "fisher_stability.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "method", "noise", "seed", "task", "kind", "dev_pct", "cosine", "peak_clean", "peak_noisy"])
        w.writerows([r[:6] + [round(x, 6) for x in r[6:]] for r in rows])

    g = defaultdict(list)
    for r in rows:
        g[(r[0], r[1], r[2], r[5])].append(r)
    print(f"\n{'dataset':12s} {'method':6s} {'noise':5s} {'kind':11s} {'n':>3s} {'dev_mean%':>10s} "
          f"{'dev_med%':>10s} {'cos_mean':>9s} {'peak_noisy':>11s}")
    for k in sorted(g):
        v = g[k]
        devs = [x[6] for x in v]
        print(f"{k[0]:12s} {k[1]:6s} {k[2]:5s} {k[3]:11s} {len(v):3d} {statistics.mean(devs):10.1f} "
              f"{statistics.median(devs):10.1f} {statistics.mean(x[7] for x in v):9.3f} "
              f"{statistics.mean(x[9] for x in v):11.1f}")


if __name__ == "__main__":
    main()
