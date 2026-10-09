import ast
import csv
import glob
import os
import re
import statistics
from collections import defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ANALYSIS = os.path.join(ROOT, "analysis")
OUT_DIR = os.path.join(ANALYSIS, "label_noise")
CONF_DIR = os.path.join(OUT_DIR, "confusions")

DATASETS = ["organamnist", "bloodmnist"]
METHODS = ["ewc"]
NOISES = [0.0, 0.1, 0.2, 0.3]
SEEDS = range(5)
TOL = 0.06

SEED_RE = re.compile(r"\[class-order check\] seed=(\d+) dataset=\S+ taskcla=(\[[^\]]*\])")

log_acc = {}
with open(os.path.join(OUT_DIR, "per_seed.csv")) as f:
    for r in csv.DictReader(f):
        log_acc[(r["dataset"], r["method"], float(r["noise"]), int(r["seed"]))] = float(r["tag_acc"])


def csv_path(d, m, p, s):
    if p == 0.0:
        return os.path.join(ANALYSIS, f"{d}_{m}_seed{s}_confusion.csv")
    return os.path.join(CONF_DIR, f"{d}_{m}_p{p}_seed{s}_confusion.csv")


def run_dir(d, m, p, s):
    if p == 0.0:
        base = os.path.join(ROOT, "results/official_savemodel", f"{d}_{m}")
    else:
        base = os.path.join(ROOT, "results/label_noise", f"{d}_{m}_p{p}")
    hits = glob.glob(os.path.join(base, f"*_fixd_{s}"))
    return hits[0] if len(hits) == 1 else None


def read_taskcla(run, seed):
    files = sorted(glob.glob(os.path.join(run, "stdout-*.txt")))
    if not files:
        return None
    if len(files) > 1:
        print(f"WARN {len(files)} stdout files in {run}, using {os.path.basename(files[-1])}")
    m = SEED_RE.search(open(files[-1], errors="ignore").read())
    if not m or int(m.group(1)) != seed:
        return None
    return ast.literal_eval(m.group(2))


def analyze(cm, taskcla):
    sizes = [nc for _, nc in taskcla]
    n = cm.shape[0]
    assert sum(sizes) == n, f"taskcla {sizes} vs matrix {n}"
    starts = np.concatenate(([0], np.cumsum(sizes))).astype(int)
    accs = []
    for u in range(len(sizes)):
        a, b = starts[u], starts[u + 1]
        accs.append(100 * np.trace(cm[a:b, a:b]) / cm[a:b, :].sum())
    k = sizes[-1]
    old_total = cm[: n - k, :].sum()
    share = cm[: n - k, n - k:].sum() / old_total if old_total > 0 else float("nan")
    col = cm.sum(axis=0)
    coll = []
    for i in range(n):
        if col[i] == 0 and cm[i].sum() > 0:
            row = cm[i].copy()
            row[i] = -1
            j = int(row.argmax())
            coll.append((i, j, row[j] / cm[i].sum(), j >= n - k))
    return float(np.mean(accs)), coll, float(share)


runs, events, mismatches = [], [], 0
for d in DATASETS:
    for m in METHODS:
        for p in NOISES:
            for s in SEEDS:
                cpath = csv_path(d, m, p, s)
                if not os.path.exists(cpath):
                    print(f"WARN missing matrix: {cpath}")
                    continue
                rd = run_dir(d, m, p, s)
                taskcla = read_taskcla(rd, s) if rd else None
                if taskcla is None:
                    print(f"WARN no taskcla for {d} {m} p{p} seed{s}")
                    continue
                acc_cm, coll, share = analyze(np.loadtxt(cpath, delimiter=","), taskcla)
                ref = log_acc.get((d, m, p, s))
                if ref is None or abs(acc_cm - ref) > TOL:
                    mismatches += 1
                    print(f"MISMATCH {d} {m} p{p} seed{s}: matrix={acc_cm:.2f} log={ref}")
                runs.append(dict(dataset=d, method=m, noise=p, seed=s, acc_cm=round(acc_cm, 3), acc_log=ref,
                                 n_collapsed=len(coll), share=share))
                for i, j, frac, in_final in coll:
                    events.append([d, m, p, s, i, j, round(float(frac), 3), int(in_final)])

with open(os.path.join(OUT_DIR, "recency_by_run.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "method", "noise", "seed", "tag_acc_from_cm", "tag_acc_log", "n_collapsed", "recency_share"])
    w.writerows([[r["dataset"], r["method"], r["noise"], r["seed"], r["acc_cm"], r["acc_log"],
                  r["n_collapsed"], round(r["share"], 4)] for r in runs])
with open(os.path.join(OUT_DIR, "recency_collapses.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "method", "noise", "seed", "collapsed_pos", "target_pos", "target_frac", "target_in_final_task"])
    w.writerows(events)

g, ev = defaultdict(list), defaultdict(list)
for r in runs:
    g[(r["dataset"], r["method"], r["noise"])].append(r)
for e in events:
    ev[(e[0], e[1], e[2])].append(e)

print(f"\naccuracy check vs log: {len(runs) - mismatches}/{len(runs)} matrices match (tol {TOL})\n")
print(f"{'dataset':12s} {'method':6s} {'noise':>5s} {'n':>2s} {'collapsed/run':>13s} {'seeds>=1':>9s} "
      f"{'->final task':>13s} {'old->final %':>13s}")
for k in sorted(g):
    v, e = g[k], ev.get(k, [])
    seeds_c = len({x[3] for x in e})
    print(f"{k[0]:12s} {k[1]:6s} {k[2]:5.1f} {len(v):2d} {statistics.mean(x['n_collapsed'] for x in v):13.2f} "
          f"{seeds_c:>6d}/{len(v):<2d} {sum(x[7] for x in e):>9d}/{len(e):<3d} "
          f"{100 * statistics.mean(x['share'] for x in v):13.1f}")
