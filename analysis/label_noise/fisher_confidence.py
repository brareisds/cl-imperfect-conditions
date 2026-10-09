import argparse
import csv
import glob
import itertools
import os
import statistics
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ".")
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from confusion_matrix_check import build_model
from datasets.data_loader import get_loaders
import utils as project_utils

NOISE_ROOT = os.path.join(ROOT, "results/label_noise")
OUT_CSV = os.path.join(ROOT, "analysis", "label_noise", "fisher_confidence.csv")
TEMPS = [1.0, 1.5, 2.0, 3.0]
METHODS = ["ewc"]


def run_dir(d, m, p, s):
    hits = glob.glob(os.path.join(NOISE_ROOT, f"{d}_{m}_p{p}", f"*_fixd_{s}"))
    return hits[0] if len(hits) == 1 else None


def saved_mean(path):
    try:
        x = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        x = torch.load(path, map_location="cpu")
    return sum(v.double().sum().item() for v in x.values()) / sum(v.numel() for v in x.values())


def logit_stats(net, loader, nb, device):
    """Mean max-probability and mean squared gradient norm wrt the logits of the EWC loss."""
    net.train()
    n, conf, g_e = 0, 0.0, 0.0
    with torch.no_grad():
        for images, _ in itertools.islice(loader, nb):
            z = torch.cat(net(images.to(device)), dim=1)
            oh = F.one_hot(z.argmax(1), z.shape[1]).float()
            p = F.softmax(z, 1)
            conf += p.max(1).values.sum().item()
            g_e += ((p - oh) ** 2).sum().item()
            n += z.shape[0]
    return conf / n, g_e / n


def fisher_mean(net, loader, nb, T, device):
    """Mean Fisher magnitude (as in compute_fisher_matrix_diag) with the logits divided by a temperature T
    (the loss is multiplied by T so that only the confidence changes, not the scale of the gradient)."""
    params = [p for p in net.model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in params)
    acc, seen = 0.0, 0
    net.train()
    for images, _ in itertools.islice(loader, nb):
        z = torch.cat(net(images.to(device)), dim=1)
        yhat = z.argmax(1).detach()
        loss = T * F.cross_entropy(z / T, yhat)
        net.zero_grad()
        loss.backward()
        acc += sum(p.grad.pow(2).sum().item() for p in params if p.grad is not None) * images.shape[0]
        seen += images.shape[0]
    return acc / seen / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["organamnist", "bloodmnist"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--noises", nargs="+", default=["0", "0.1", "0.2", "0.3"])
    ap.add_argument("--temp-noises", nargs="+", default=["0", "0.3"], help="noise levels where the temperature test is run")
    ap.add_argument("--tasks", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--nb", type=int, default=60, help="batches of 32 images per measurement")
    ap.add_argument("--smoke", action="store_true", help="tiny run: organamnist, seed 0, p=0, tasks 0 and 3, 10 batches")
    ap.add_argument("--out", default=OUT_CSV)
    a = ap.parse_args()
    if a.smoke:
        a.datasets, a.seeds, a.noises, a.temp_noises, a.tasks, a.nb = ["organamnist"], [0], ["0"], ["0"], [0, 3], 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cols = ["dataset", "method", "noise", "seed", "task", "conf", "g_ewc", "saved_mean"] + [f"f_T{T}" for T in TEMPS]
    rows = []
    for d in a.datasets:
        for s in a.seeds:
            for p in a.noises:
                os.environ["LABEL_NOISE"], os.environ["LABEL_NOISE_SEED"] = p, str(s)
                project_utils.seed_everything(seed=s)
                trn, _, _, taskcla = get_loaders([d], 4, nc_first_task=None, batch_size=32, num_workers=2, pin_memory=False)
                for m in METHODS:
                    rd = run_dir(d, m, p, s)
                    if rd is None:
                        print(f"WARN no run dir: {d} {m} p{p} seed{s}")
                        continue
                    for t in a.tasks:
                        net = build_model("resnet18", taskcla[: t + 1])
                        net.load_state_dict(torch.load(os.path.join(rd, "models", f"task{t}.ckpt"), map_location=device))
                        net.to(device)
                        conf, ge = logit_stats(net, trn[t], a.nb, device)
                        row = dict(dataset=d, method=m, noise=p, seed=s, task=t, conf=conf, g_ewc=ge,
                                   saved_mean=saved_mean(os.path.join(rd, "fisher", f"task{t}_raw.pt")))
                        if p in a.temp_noises:
                            for T in TEMPS:
                                row[f"f_T{T}"] = fisher_mean(net, trn[t], a.nb, T, device)
                        rows.append(row)
                        print(f"{d} {m} p{p} seed{s} task{t}: conf={conf:.3f} g_ewc={ge:.4f} "
                              f"saved={row['saved_mean']:.3e} f_T1={row.get('f_T1.0', float('nan')):.3e}", flush=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    g = defaultdict(list)
    for r in rows:
        g[(r["dataset"], r["method"], r["noise"])].append(r)
    med = lambda v, k: statistics.median(x[k] for x in v if x.get(k) is not None)
    print("\n== confidence and logit-gradient scale (medians over seeds and tasks) ==")
    print(f"{'dataset':12s} {'method':6s} {'p':>4s} {'n':>3s} {'conf':>6s} {'g_ewc':>7s} {'saved_mean(e-6)':>16s}")
    for k in sorted(g):
        v = g[k]
        print(f"{k[0]:12s} {k[1]:6s} {k[2]:>4s} {len(v):3d} {med(v, 'conf'):6.3f} {med(v, 'g_ewc'):7.4f} {med(v, 'saved_mean') * 1e6:16.2f}")
    print("\n== temperature test: Fisher(T) / Fisher(T=1), and recomputed Fisher(T=1) / saved Fisher ==")
    print(f"{'dataset':12s} {'method':6s} {'p':>4s} {'T=1.5':>7s} {'T=2':>7s} {'T=3':>7s} {'T1/saved':>9s}")
    for k in sorted(g):
        v = [x for x in g[k] if x.get("f_T1.0")]
        if v:
            rr = lambda T: statistics.median(x[f"f_T{T}"] / x["f_T1.0"] for x in v)
            print(f"{k[0]:12s} {k[1]:6s} {k[2]:>4s} {rr(1.5):7.2f} {rr(2.0):7.2f} {rr(3.0):7.2f} "
                  f"{statistics.median(x['f_T1.0'] / x['saved_mean'] for x in v):9.2f}")


if __name__ == "__main__":
    main()
