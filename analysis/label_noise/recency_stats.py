import csv
import math
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT_DIR = os.path.join(ROOT, "analysis", "label_noise")

try:
    from scipy import stats

    def tcrit(df):
        return float(stats.t.ppf(0.975, df))
except ImportError:
    _T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}

    def tcrit(df):
        return _T[df]

DATASETS = ["organamnist", "bloodmnist"]
METHODS = ["ewc"]
NAMES = {"ewc": "EWC", "icarl": "iCaRL"}
NOISES = [0.0, 0.1, 0.2, 0.3]
METRICS = ["n_collapsed", "recency_share"]

val = {}
with open(os.path.join(OUT_DIR, "recency_by_run.csv")) as f:
    for r in csv.DictReader(f):
        val[(r["dataset"], r["method"], float(r["noise"]), int(r["seed"]))] = {
            "n_collapsed": float(r["n_collapsed"]),
            "recency_share": float(r["recency_share"]),
            "acc": float(r["tag_acc_from_cm"]),
        }


def series(d, m, p, key):
    return {s: v[key] for (dd, mm, pp, s), v in val.items() if (dd, mm, pp) == (d, m, p)}


def paired(a, b):
    seeds = sorted(set(a) & set(b))
    diffs = [a[s] - b[s] for s in seeds]
    n = len(diffs)
    if n < 2:
        return None
    m = sum(diffs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in diffs) / (n - 1))
    h = tcrit(n - 1) * sd / math.sqrt(n)
    return n, m, m - h, m + h, sum(x > 0 for x in diffs)


rows = []


def add(kind, d, metric, p, label, res):
    if res is None:
        return
    n, m, lo, hi, w = res
    rows.append([kind, d, metric, p, label, n, round(m, 4), round(lo, 4), round(hi, 4), w, int(lo > 0 or hi < 0)])


for d in DATASETS:
    for metric in METRICS:
        for m in METHODS:
            for p in NOISES[1:]:
                add("increase_from_clean", d, metric, p, NAMES[m],
                    paired(series(d, m, p, metric), series(d, m, 0.0, metric)))

hdr = ["analysis", "dataset", "metric", "noise", "comparison", "n", "mean_diff", "ci95_lo", "ci95_hi", "wins", "excludes_0"]
with open(os.path.join(OUT_DIR, "recency_stats.csv"), "w", newline="") as f:
    csv.writer(f).writerows([hdr] + rows)

for r in rows:
    print(f"{r[0]:19s} {r[1]:11s} {r[2]:13s} {r[3]:3.1f} {r[4]:13s} n={r[5]} diff={r[6]:8.3f} "
          f"CI[{r[7]:8.3f},{r[8]:8.3f}] wins={r[9]}/{r[5]} {'*' if r[10] else ''}")

print("\nCorrelation between recency_share and TAg acc (matrix-derived), per dataset and method")
print(f"{'dataset':12s} {'method':6s} {'n':>3s} {'pooled_r':>9s} {'within_noise_r':>15s}")
for d in DATASETS:
    for m in METHODS:
        xs, ys, cx, cy = [], [], [], []
        for p in NOISES:
            sh, ac = series(d, m, p, "recency_share"), series(d, m, p, "acc")
            seeds = sorted(set(sh) & set(ac))
            if not seeds:
                continue
            a = np.array([sh[s] for s in seeds])
            b = np.array([ac[s] for s in seeds])
            xs += a.tolist()
            ys += b.tolist()
            cx += (a - a.mean()).tolist()
            cy += (b - b.mean()).tolist()
        print(f"{d:12s} {m:6s} {len(xs):3d} {np.corrcoef(xs, ys)[0, 1]:9.3f} {np.corrcoef(cx, cy)[0, 1]:15.3f}")
