import argparse
import copy
import csv
import glob
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for sub in ("analysis/label_noise", "analysis", "src"):
    sys.path.insert(0, os.path.join(ROOT, sub))
os.environ.setdefault("SHUFFLE_CLASSES_FIX", "1")
from confusion_matrix_check import build_model
from datasets.data_loader import get_loaders
from parse_label_noise import load, mean, std
import utils as project_utils

OUT_DIR = os.path.join(ROOT, "analysis", "label_noise")
TOL = 0.15
PHASES = {
    "intra": {"icarl": "phase1_icarl", "ewc": "phase4_ewc_exemplars"},
    "cross": {"icarl": "phase2_cross", "ewc": "phase5_ewc_cross_exemplars"},
    "other": {"icarl": "phase6_other_datasets", "ewc": "phase6_other_datasets"},
}
NOISES = {"intra": [0.0, 0.1, 0.2, 0.3], "cross": [0.1, 0.3], "other": [0.0]}
DEFAULT_DATASETS = {"intra": ["organamnist", "bloodmnist"], "cross": ["organamnist", "bloodmnist"],
                    "other": ["pathmnist", "tissuemnist"]}


def run_parent(base, d, m, p):
    return os.path.join(base, f"{d}_{m}_p{p:g}")


def find_ckpt_dir(phase, d, m, p, s, legacy_root):
    bases = [os.path.join(ROOT, "results", phase)]
    if legacy_root:
        bases.append(os.path.join(legacy_root, f"results_{phase}"))
    for b in bases:
        for h in glob.glob(os.path.join(run_parent(b, d, m, p), f"*_fixd_{s}")):
            if all(os.path.exists(os.path.join(h, *x)) for x in (("models", "task3.ckpt"), ("exemplars", "task3.pt"))):
                return h
    return None


@torch.no_grad()
def forward_all(net, loader, device):
    outs, feats, ys = [], [], []
    for x, y in loader:
        o, f = net(x.to(device), return_features=True)
        outs.append(torch.cat(o, dim=1).cpu())
        feats.append(f.cpu())
        ys.append(torch.as_tensor(y))
    return torch.cat(outs), torch.cat(feats), torch.cat(ys)


def class_means(net, template_ds, ex, device, batch_size):
    ds = copy.copy(template_ds)
    assert hasattr(ds, "images") and hasattr(ds, "labels") and hasattr(ds, "transform"), "unexpected dataset class"
    ds.images, ds.labels = ex["images"], ex["labels"]
    _, f, y = forward_all(net, DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2), device)
    f = f / f.norm(dim=1, keepdim=True)
    y = y.numpy()
    classes = np.unique(y)
    means = torch.stack([f[y == c].mean(0) / f[y == c].mean(0).norm() for c in classes])
    return means, classes


def run_one(d, m, p, s, ckpt_dir, device):
    project_utils.seed_everything(seed=s)
    _, _, tst_loader, taskcla = get_loaders([d], 4, nc_first_task=None, batch_size=32, num_workers=2, pin_memory=False)
    net = build_model("resnet18", taskcla)
    net.load_state_dict(torch.load(os.path.join(ckpt_dir, "models", "task3.ckpt"), map_location=device))
    net.to(device).eval()
    n_cls = int(net.task_cls.sum().item())
    ex = torch.load(os.path.join(ckpt_dir, "exemplars", "task3.pt"), map_location="cpu", weights_only=False)
    means, classes = class_means(net, tst_loader[0].dataset, ex, device, 32)
    assert list(classes) == list(range(n_cls)), f"exemplar classes {list(classes)}"
    res = {"tag_nme": [], "taw_nme": [], "tag_logits": [], "taw_logits": []}
    for u, ld in enumerate(tst_loader):
        o, f, y = forward_all(net, ld, device)
        off, nc = int(net.task_offset[u]), int(net.task_cls[u])
        dist = torch.cdist(f / f.norm(dim=1, keepdim=True), means)
        res["tag_logits"].append((o.argmax(1) == y).float().mean().item())
        res["taw_logits"].append(((o[:, off:off + nc].argmax(1) + off) == y).float().mean().item())
        res["tag_nme"].append((dist.argmin(1) == y).float().mean().item())
        res["taw_nme"].append(((dist[:, off:off + nc].argmin(1) + off) == y).float().mean().item())
    return {k: 100 * float(np.mean(v)) for k, v in res.items()}, int(ex["labels"].__len__())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", nargs="+", default=["intra"], choices=list(PHASES))
    ap.add_argument("--methods", nargs="+", default=["icarl", "ewc"])
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--noise", nargs="+", type=float, default=None)
    ap.add_argument("--legacy-root", default=None)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "nme_eval.csv"))
    a = ap.parse_args()
    assert float(os.environ.get("LABEL_NOISE", "0") or 0) == 0, "LABEL_NOISE must be unset when evaluating"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, summ = [], []
    for mode in a.setup:
        for m in a.methods:
            phase = PHASES[mode][m]
            for d in (a.datasets or DEFAULT_DATASETS[mode]):
                for p in (a.noise if a.noise is not None else NOISES[mode]):
                    logs = load(run_parent(os.path.join(ROOT, "results", phase), d, m, p))
                    got = []
                    for s in a.seeds:
                        cdir = find_ckpt_dir(phase, d, m, p, s, a.legacy_root)
                        if cdir is None:
                            print(f"WARN no checkpoint+exemplars: {phase} {d} {m} p{p:g} seed{s}")
                            continue
                        r, n_ex = run_one(d, m, p, s, cdir, device)
                        key = "tag_nme" if m == "icarl" else "tag_logits"
                        ref = logs.get(s, {}).get("tag_acc")
                        ok = ref is not None and abs(r[key] - ref) <= TOL
                        print(f"{mode:5s} {d:11s} {m:5s} p={p:g} seed{s} n_ex={n_ex} nme={r['tag_nme']:.2f} "
                              f"logits={r['tag_logits']:.2f} log={ref} {'OK' if ok else 'MISMATCH'}")
                        rows.append([mode, phase, d, m, p, s, n_ex] + [round(r[k], 3) for k in ("tag_nme", "taw_nme", "tag_logits", "taw_logits")]
                                    + [ref, key, int(ok)])
                        got.append(r)
                    if got:
                        summ.append((mode, d, m, p, len(got), mean([g["tag_nme"] for g in got]), std([g["tag_nme"] for g in got]),
                                     mean([g["tag_logits"] for g in got]), std([g["tag_logits"] for g in got])))
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "phase", "dataset", "method", "noise", "seed", "n_exemplars", "tag_nme", "taw_nme",
                    "tag_logits", "taw_logits", "log_tag", "log_compared", "log_ok"])
        w.writerows(rows)
    print(f"\n{'mode':5s} {'dataset':12s} {'method':6s} {'noise':>5s} {'n':>2s} {'TAg NME':>16s} {'TAg logits':>16s}")
    for r in summ:
        print(f"{r[0]:5s} {r[1]:12s} {r[2]:6s} {r[3]:5.1f} {r[4]:2d} {r[5]:7.2f} +-{r[6]:5.2f} {r[7]:7.2f} +-{r[8]:5.2f}")
    print(f"\nwrote {a.out}; mismatches: {sum(1 for r in rows if not r[-1])}/{len(rows)}")


if __name__ == "__main__":
    main()
