"""Phase 0 (CPU): LTB, error share, TAw/TAg, last-layer norms and Weight Aligning on saved final checkpoints.
Run from LifeLonger_default/src:  python ../analysis/label_noise/phase0_recency_wa.py [--smoke|--selftest]"""
import argparse
import csv
import glob
import os
import statistics
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ANALYSIS = os.path.join(ROOT, "analysis")
OUT_DIR = os.path.join(ANALYSIS, "label_noise")


def run_dir(d, m, p, s, base=None):
    if base:
        pat = os.path.join(ROOT, base, f"{d}_{m}_p{p}", f"*_fixd_{s}")
    elif p == 0.0:
        pat = os.path.join(ROOT, "results/official_savemodel", f"{d}_{m}", f"*_fixd_{s}")
    else:
        pat = os.path.join(ROOT, "results/label_noise", f"{d}_{m}_p{p}", f"*_fixd_{s}")
    hits = glob.glob(pat)
    return hits[0] if len(hits) == 1 else None


def conf_matrix(pred, target, n):
    return np.bincount(target * n + pred, minlength=n * n).reshape(n, n)


def metrics(cm, sizes):
    n, k = cm.shape[0], sizes[-1]
    starts = np.concatenate(([0], np.cumsum(sizes))).astype(int)
    tag = np.mean([100 * np.trace(cm[a:b, a:b]) / cm[a:b, :].sum() for a, b in zip(starts[:-1], starts[1:])])
    ltb = 100 * np.mean([cm[a:b, n - k:].sum() / cm[a:b, :].sum() for a, b in zip(starts[:-2], starts[1:-1])])
    old = cm[: n - k, :]
    pooled = 100 * old[:, n - k:].sum() / old.sum()
    err = old.sum() - np.trace(cm[: n - k, : n - k])
    err_last = 100 * old[:, n - k:].sum() / err if err > 0 else float("nan")
    return dict(tag=tag, ltb=ltb, pooled=pooled, err_last=err_last, uniform=100 * k / (n - 1))


def taw_acc(logits, target, sizes):
    starts = np.concatenate(([0], np.cumsum(sizes))).astype(int)
    accs = []
    for a, b in zip(starts[:-1], starts[1:]):
        m = (target >= a) & (target < b)
        accs.append(100 * np.mean(logits[m][:, a:b].argmax(1) + a == target[m]))
    return float(np.mean(accs))


def selftest():
    rs = np.random.RandomState(0)
    sizes = [3, 3, 3, 2]
    n = sum(sizes)
    cm = rs.randint(0, 50, size=(n, n)) + 400 * np.eye(n, dtype=int)
    pred, tgt = [], []
    for i in range(n):
        for j in range(n):
            pred += [j] * cm[i, j]
            tgt += [i] * cm[i, j]
    assert (conf_matrix(np.array(pred), np.array(tgt), n) == cm).all()
    r = metrics(cm, sizes)
    # reference: same formulas as recency_under_noise.py (accuracy and pooled share)
    starts = np.concatenate(([0], np.cumsum(sizes))).astype(int)
    accs = [100 * np.trace(cm[a:b, a:b]) / cm[a:b, :].sum() for a, b in zip(starts[:-1], starts[1:])]
    k = sizes[-1]
    share = cm[: n - k, n - k:].sum() / cm[: n - k, :].sum()
    assert abs(r["tag"] - np.mean(accs)) < 1e-9 and abs(r["pooled"] - 100 * share) < 1e-9
    # LTB by hand
    f = [cm[a:b, n - k:].sum() / cm[a:b, :].sum() for a, b in [(0, 3), (3, 6), (6, 9)]]
    assert abs(r["ltb"] - 100 * np.mean(f)) < 1e-9
    # perfect model -> 0
    assert metrics(np.eye(n, dtype=int) * 10, sizes)["ltb"] == 0
    # all old images predicted as last task -> 100
    cm2 = np.zeros((n, n), dtype=int)
    cm2[: n - k, n - k] = 10
    cm2[n - k:, n - k] = 10
    assert abs(metrics(cm2, sizes)["ltb"] - 100) < 1e-9
    # TAw: logits favouring the right class within the task -> 100
    lg = rs.randn(200, n)
    t = rs.randint(0, n, 200)
    lg[np.arange(200), t] += 10
    assert taw_acc(lg, t, sizes) == 100
    print("selftest OK:", {k_: round(float(v), 2) for k_, v in r.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["organamnist", "bloodmnist"])
    ap.add_argument("--methods", nargs="+", default=["ewc", "icarl"])
    ap.add_argument("--noises", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--base", default=None, help="results folder (relative to repo) laid out as <d>_<m>_p<p>/*_fixd_<s>")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.smoke:
        a.datasets, a.methods, a.noises, a.seeds = ["organamnist"], ["ewc"], [0.0], [0]

    os.environ["SHUFFLE_CLASSES_FIX"] = "1"
    os.environ.pop("LABEL_NOISE", None)
    import torch
    sys.path.insert(0, ".")
    sys.path.insert(0, ANALYSIS)
    from confusion_matrix_check import build_model
    from datasets.data_loader import get_loaders
    import utils as project_utils
    torch.set_num_threads(os.cpu_count() or 1)

    log_acc = {}
    if not a.base and os.path.exists(os.path.join(OUT_DIR, "per_seed.csv")):
        with open(os.path.join(OUT_DIR, "per_seed.csv")) as f:
            for r in csv.DictReader(f):
                log_acc[(r["dataset"], r["method"], float(r["noise"]), int(r["seed"]))] = float(r["tag_acc"])

    rows, norm_rows = [], []
    for d in a.datasets:
        for s in a.seeds:
            project_utils.seed_everything(seed=s)
            _, _, tst, taskcla = get_loaders([d], 4, nc_first_task=None, batch_size=256, num_workers=2, pin_memory=False)
            sizes = [nc for _, nc in taskcla]
            n, k = sum(sizes), sizes[-1]
            for m in a.methods:
                for p in a.noises:
                    rd = run_dir(d, m, p, s, a.base)
                    if rd is None:
                        print(f"WARN no run dir: {d} {m} p{p} seed{s}")
                        continue
                    net = build_model("resnet18", taskcla)
                    net.load_state_dict(torch.load(os.path.join(rd, "models", f"task{len(sizes) - 1}.ckpt"), map_location="cpu"))
                    net.eval()
                    feats, tgt = [], []
                    with torch.no_grad():
                        for u in range(len(sizes)):
                            for images, targets in tst[u]:
                                out, f = net(images, return_features=True)
                                if not feats:
                                    W0 = torch.cat([h.weight for h in net.heads]); b0 = torch.cat([h.bias for h in net.heads])
                                    assert torch.allclose(torch.cat(out, 1), f @ W0.T + b0, atol=1e-4), "head recompute mismatch"
                                feats.append(f.numpy()); tgt.append(targets.numpy())
                    F, T = np.concatenate(feats), np.concatenate(tgt)
                    W = torch.cat([h.weight for h in net.heads]).detach().numpy()
                    b = torch.cat([h.bias for h in net.heads]).detach().numpy()
                    norms = np.linalg.norm(W, axis=1)
                    gamma = norms[: n - k].mean() / norms[n - k:].mean()
                    W2 = W.copy(); W2[n - k:] *= gamma
                    L, L2 = F @ W.T + b, F @ W2.T + b
                    mt = metrics(conf_matrix(L.argmax(1), T, n), sizes)
                    mw = metrics(conf_matrix(L2.argmax(1), T, n), sizes)
                    ref = log_acc.get((d, m, p, s))
                    if ref is not None and m == "ewc" and abs(mt["tag"] - ref) > 0.06:
                        print(f"MISMATCH {d} {m} p{p} seed{s}: recomputed={mt['tag']:.2f} log={ref}")
                    rows.append(dict(dataset=d, method=m, noise=p, seed=s, tag=mt["tag"], taw=taw_acc(L, T, sizes),
                                     ltb=mt["ltb"], pooled=mt["pooled"], err_last=mt["err_last"], uniform=mt["uniform"],
                                     norm_old=norms[: n - k].mean(), norm_new=norms[n - k:].mean(), gamma=gamma,
                                     bias_old=float(b[: n - k].mean()), bias_new=float(b[n - k:].mean()),
                                     tag_wa=mw["tag"], ltb_wa=mw["ltb"], pooled_wa=mw["pooled"]))
                    for c in range(n):
                        norm_rows.append([d, m, p, s, c, int(np.searchsorted(np.cumsum(sizes), c, side="right")), round(float(norms[c]), 5)])
                    print(f"{d} {m} p{p} seed{s}: TAg={mt['tag']:.1f} LTB={mt['ltb']:.1f} gamma={gamma:.2f} "
                          f"-> WA TAg={mw['tag']:.1f} LTB={mw['ltb']:.1f}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    cols = list(rows[0].keys()) if rows else []
    with open(os.path.join(OUT_DIR, f"phase0_{a.tag}_runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT_DIR, f"phase0_{a.tag}_norms.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "method", "noise", "seed", "class", "task", "weight_norm"])
        w.writerows(norm_rows)

    g = defaultdict(list)
    for r in rows:
        g[(r["dataset"], r["method"], r["noise"])].append(r)
    print(f"\n{'dataset':12s} {'method':6s} {'p':>4s} {'n':>2s} {'TAg':>5s} {'TAw':>5s} {'LTB':>5s} {'errLast':>7s} {'unif':>5s} "
          f"{'gamma':>5s} {'TAg_WA':>6s} {'LTB_WA':>6s} {'b_old':>6s} {'b_new':>6s}")
    for key in sorted(g):
        v = g[key]
        mean = lambda c: statistics.mean(x[c] for x in v)
        print(f"{key[0]:12s} {key[1]:6s} {key[2]:4.1f} {len(v):2d} {mean('tag'):5.1f} {mean('taw'):5.1f} {mean('ltb'):5.1f} "
              f"{mean('err_last'):7.1f} {mean('uniform'):5.1f} {mean('gamma'):5.2f} {mean('tag_wa'):6.1f} {mean('ltb_wa'):6.1f} "
              f"{mean('bias_old'):6.3f} {mean('bias_new'):6.3f}")


if __name__ == "__main__":
    main()
