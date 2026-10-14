import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ".")
sys.path.insert(0, os.path.join(ROOT, "analysis", "label_noise"))
import nme_eval as ne
from test_corruption_eval import CORRUPTIONS, corrupt, summarize
from datasets.data_loader import get_loaders
import utils as project_utils

OUT_DIR = os.path.join(ROOT, "analysis", "label_noise")
TOL = 0.15


def predict(net, loaders, means, device):
    n = means.shape[0]
    cms = {c: np.zeros((n, n), dtype=np.int64) for c in ("logits", "nme")}
    for ld in loaders:
        o, f, y = ne.forward_all(net, ld, device)
        pred = {"logits": o.argmax(1), "nme": torch.cdist(f / f.norm(dim=1, keepdim=True), means).argmin(1)}
        for c in cms:
            np.add.at(cms[c], (y.numpy(), pred[c].numpy()), 1)
    return cms


def run_one(d, m, p, s, cdir, device, ref):
    project_utils.seed_everything(seed=s)
    _, _, tst_loader, taskcla = get_loaders([d], 4, nc_first_task=None, batch_size=32, num_workers=2, pin_memory=False)
    net = ne.build_model("resnet18", taskcla)
    net.load_state_dict(torch.load(os.path.join(cdir, "models", "task3.ckpt"), map_location=device))
    net.to(device).eval()
    ex = torch.load(os.path.join(cdir, "exemplars", "task3.pt"), map_location="cpu", weights_only=False)
    means, _ = ne.class_means(net, tst_loader[0].dataset, ex, device, 32)
    dsets = [ld.dataset for ld in tst_loader]
    orig = [np.asarray(list(ds.images)) for ds in dsets]
    assert all(o.dtype == np.uint8 for o in orig), "test images are not uint8"
    rows = []

    def one(kind, level, par):
        for u, ds in enumerate(dsets):
            if kind == "none":
                ds.images = orig[u]
            else:
                rs = np.random.RandomState(12345 + 100 * u + 10 * level)
                ds.images = corrupt(orig[u].astype(np.float32) / 255.0, kind, par, rs)
        out = {}
        for c, cm in predict(net, tst_loader, means, device).items():
            acc, nc, sh = summarize(cm, taskcla)
            rows.append([d, m, p, s, c, kind, level, par, round(acc, 3), nc, round(sh, 4)])
            out[c] = acc
        return out

    clean = one("none", 0, 0.0)
    ok = ref is not None and all(abs(clean[c] - ref[c]) <= TOL for c in clean)
    print(f"check {d} {m} p{p:g} seed{s}: nme={clean['nme']:.3f} logits={clean['logits']:.3f} ref={ref} {'OK' if ok else 'MISMATCH'}", flush=True)
    for kind, pars in CORRUPTIONS.items():
        for level, par in enumerate(pars, start=1):
            one(kind, level, par)
    return rows, int(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["organamnist", "bloodmnist"])
    ap.add_argument("--methods", nargs="+", default=["icarl", "ewc"])
    ap.add_argument("--train-noise", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--legacy-root", default=None)
    ap.add_argument("--ref", default=os.path.join(OUT_DIR, "nme_eval.csv"))
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "test_corruption_nme.csv"))
    a = ap.parse_args()
    assert float(os.environ.get("LABEL_NOISE", "0") or 0) == 0, "LABEL_NOISE must be unset when evaluating"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref = {}
    with open(a.ref) as f:
        for r in csv.DictReader(f):
            if r["mode"] == "intra":
                ref[(r["dataset"], r["method"], float(r["noise"]), int(r["seed"]))] = {
                    "nme": float(r["tag_nme"]), "logits": float(r["tag_logits"])}
    t0, bad, tot = time.time(), 0, 0
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "method", "train_noise", "seed", "classifier", "corruption", "level", "param",
                    "tag_acc", "n_collapsed", "recency_share"])
        for d in a.datasets:
            for m in a.methods:
                phase = ne.PHASES["intra"][m]
                for p in a.train_noise:
                    for s in a.seeds:
                        cdir = ne.find_ckpt_dir(phase, d, m, p, s, a.legacy_root)
                        if cdir is None:
                            print(f"WARN no checkpoint+exemplars: {phase} {d} {m} p{p:g} seed{s}")
                            continue
                        rows, ok = run_one(d, m, p, s, cdir, device, ref.get((d, m, p, s)))
                        w.writerows(rows)
                        f.flush()
                        bad, tot = bad + 1 - ok, tot + 1
    print(f"done in {time.time() - t0:.0f}s -> {a.out}; mismatches: {bad}/{tot}")


if __name__ == "__main__":
    main()
