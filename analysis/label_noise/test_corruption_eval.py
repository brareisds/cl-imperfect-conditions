import argparse
import csv
import glob
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ".")
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from confusion_matrix_check import build_model
from datasets.data_loader import get_loaders
import utils as project_utils

OUT_DIR = os.path.join(ROOT, "analysis", "label_noise")
TOL = 0.06
CORRUPTIONS = {
    "gauss_noise": [0.1, 0.2, 0.3, 0.5],
    "blur": [0.75, 1.0, 1.5, 2.0],
    "contrast": [0.8, 0.7, 0.6, 0.5],
    "brightness": [0.03, 0.06, 0.1, 0.15],
}


def gblur(x, s):
    t = torch.from_numpy(x)
    if t.ndim == 3:
        t = t.unsqueeze(-1)
    t = t.permute(0, 3, 1, 2).contiguous()
    r = int(math.ceil(3 * s))
    ax = torch.arange(-r, r + 1, dtype=t.dtype)
    k = torch.exp(-ax ** 2 / (2 * s * s))
    k = k / k.sum()
    c = t.shape[1]
    kh = k.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    kv = k.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
    t = F.pad(t, (r, r, r, r), mode="reflect")
    t = F.conv2d(t, kh, groups=c)
    t = F.conv2d(t, kv, groups=c)
    t = t.permute(0, 2, 3, 1)
    return (t.squeeze(-1) if x.ndim == 3 else t).numpy()


def corrupt(x, kind, par, rs):
    if kind == "gauss_noise":
        y = x + rs.normal(0.0, par, x.shape).astype(np.float32)
    elif kind == "blur":
        y = gblur(x, par)
    elif kind == "contrast":
        m = x.reshape(len(x), -1).mean(1).reshape((-1,) + (1,) * (x.ndim - 1))
        y = m + (x - m) * par
    elif kind == "brightness":
        y = x + par
    else:
        raise ValueError(kind)
    return (np.clip(y, 0, 1) * 255).round().astype(np.uint8)


def summarize(cm, taskcla):
    sizes = [nc for _, nc in taskcla]
    n = cm.shape[0]
    assert sum(sizes) == n
    starts = np.concatenate(([0], np.cumsum(sizes))).astype(int)
    accs = [100 * np.trace(cm[starts[u]:starts[u + 1], starts[u]:starts[u + 1]]) / cm[starts[u]:starts[u + 1], :].sum()
            for u in range(len(sizes))]
    k = sizes[-1]
    old = cm[: n - k, :].sum()
    share = cm[: n - k, n - k:].sum() / old if old > 0 else float("nan")
    col = cm.sum(axis=0)
    n_coll = int(sum(1 for i in range(n) if col[i] == 0 and cm[i].sum() > 0))
    return float(np.mean(accs)), n_coll, float(share)


@torch.no_grad()
def evaluate(net, loaders, device, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for ld in loaders:
        for images, targets in ld:
            pred = torch.cat(net(images.to(device)), dim=1).argmax(1).cpu().numpy()
            np.add.at(cm, (targets.numpy(), pred), 1)
    return cm


def ckpt_dir(d, m, p, s):
    base = (os.path.join(ROOT, "results/official_savemodel", f"{d}_{m}") if p == 0
            else os.path.join(ROOT, "results/label_noise", f"{d}_{m}_p{p}"))
    hits = glob.glob(os.path.join(base, f"*_fixd_{s}"))
    return hits[0] if len(hits) == 1 else None


def run_one(d, m, p, s, device, log_acc, calibrate):
    cdir = ckpt_dir(d, m, p, s)
    if cdir is None:
        print(f"WARN no checkpoint dir: {d} {m} p{p} seed{s}")
        return []
    project_utils.seed_everything(seed=s)
    _, _, tst_loader, taskcla = get_loaders([d], 4, nc_first_task=None, batch_size=32, num_workers=2, pin_memory=False)
    net = build_model("resnet18", taskcla)
    net.load_state_dict(torch.load(os.path.join(cdir, "models", "task3.ckpt"), map_location=device))
    net.to(device).eval()
    n_classes = int(net.task_cls.sum().item())
    dsets = [ld.dataset for ld in tst_loader]
    for ds in dsets:
        assert hasattr(ds, "images") and hasattr(ds, "transform"), "unexpected dataset class"
    orig = [np.asarray(list(ds.images)) for ds in dsets]
    assert all(o.dtype == np.uint8 for o in orig), "test images are not uint8"
    rows = []

    def one(kind, level, par):
        t0 = time.time()
        for u, ds in enumerate(dsets):
            if kind == "none":
                ds.images = orig[u]
            else:
                rs = np.random.RandomState(12345 + 100 * u + 10 * level)
                ds.images = corrupt(orig[u].astype(np.float32) / 255.0, kind, par, rs)
        acc, nc, sh = summarize(evaluate(net, tst_loader, device, n_classes), taskcla)
        rows.append([d, m, p, s, kind, level, par, round(acc, 3), nc, round(sh, 4)])
        if calibrate:
            print(f"{d:11s} {kind:11s} L{level} par={par:5.2f} TAg={acc:6.2f} collapsed={nc} share={sh:.2f} ({time.time() - t0:.1f}s)")
        return acc

    clean = one("none", 0, 0.0)
    ref = log_acc.get((d, m, float(p), s))
    ok = ref is not None and abs(clean - ref) <= TOL
    print(f"check vs log {d} {m} p{p} seed{s}: matrix={clean:.2f} log={ref} {'OK' if ok else 'MISMATCH'}")
    for kind, pars in CORRUPTIONS.items():
        for level, par in enumerate(pars, start=1):
            one(kind, level, par)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["organamnist", "bloodmnist"])
    ap.add_argument("--methods", nargs="+", default=["ewc"])
    ap.add_argument("--train-noise", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "test_corruption_results.csv"))
    a = ap.parse_args()
    assert float(os.environ.get("LABEL_NOISE", "0") or 0) == 0, "LABEL_NOISE must be unset when evaluating"
    if a.calibrate:
        a.methods, a.train_noise, a.seeds = ["ewc"], [0.0], [0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_acc = {}
    with open(os.path.join(OUT_DIR, "per_seed.csv")) as f:
        for r in csv.DictReader(f):
            log_acc[(r["dataset"], r["method"], float(r["noise"]), int(r["seed"]))] = float(r["tag_acc"])
    t0 = time.time()
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "method", "train_noise", "seed", "corruption", "level", "param", "tag_acc", "n_collapsed", "recency_share"])
        for d in a.datasets:
            for m in a.methods:
                for p in a.train_noise:
                    for s in a.seeds:
                        w.writerows(run_one(d, m, p, s, device, log_acc, a.calibrate))
                        f.flush()
    print(f"done in {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
