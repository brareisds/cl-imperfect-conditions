import csv
import math
import os
import statistics
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT_DIR = os.path.join(ROOT, "analysis", "label_noise")

DATASETS = ["organamnist", "bloodmnist"]
METHODS = ["ewc"]
NAMES = {"ewc": "EWC", "icarl": "iCaRL"}
CORR = ["gauss_noise", "blur", "contrast", "brightness"]
LEVELS = [1, 2, 3, 4]
NOISES = [0.0, 0.1, 0.2, 0.3]
SEEDS = range(5)

try:
    from scipy import stats

    def tcrit(df):
        return float(stats.t.ppf(0.975, df))
except ImportError:
    _T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}

    def tcrit(df):
        return _T[df]

rec = {}
for d in DATASETS:
    for m in METHODS:
        with open(os.path.join(OUT_DIR, f"test_corruption_{d}_{m}.csv")) as f:
            for r in csv.DictReader(f):
                k = (r["dataset"], r["method"], float(r["train_noise"]), int(r["seed"]), r["corruption"], int(r["level"]))
                rec[k] = (float(r["tag_acc"]), int(r["n_collapsed"]), float(r["recency_share"]))
expected = len(DATASETS) * len(METHODS) * len(NOISES) * len(SEEDS) * (1 + len(CORR) * len(LEVELS))
assert len(rec) == expected, f"{len(rec)} records, expected {expected}"


def acc(d, m, p, s, c, l):
    return rec[(d, m, p, s, c, l)][0]


def ret(d, m, p, s, c, l):
    return acc(d, m, p, s, c, l) / acc(d, m, p, s, "none", 0)


def series(fn, *key):
    return {s: fn(key[0], key[1], key[2], s, *key[3:]) for s in SEEDS}


def mean(x):
    return sum(x) / len(x)


def std(x):
    m = mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / len(x))


def paired(a, b):
    seeds = sorted(set(a) & set(b))
    diffs = [a[s] - b[s] for s in seeds]
    n = len(diffs)
    if n < 2:
        return None
    m = mean(diffs)
    sd = math.sqrt(sum((x - m) ** 2 for x in diffs) / (n - 1))
    h = tcrit(n - 1) * sd / math.sqrt(n)
    return n, m, m - h, m + h, sum(x > 0 for x in diffs)


# ---- summary per condition ----
summary = []
for d in DATASETS:
    for m in METHODS:
        for p in NOISES:
            for c, l in [("none", 0)] + [(c, l) for c in CORR for l in LEVELS]:
                a = [acc(d, m, p, s, c, l) for s in SEEDS]
                r = [ret(d, m, p, s, c, l) for s in SEEDS]
                nc = [rec[(d, m, p, s, c, l)][1] for s in SEEDS]
                sh = [rec[(d, m, p, s, c, l)][2] for s in SEEDS]
                summary.append([d, m, p, c, l, round(mean(a), 3), round(std(a), 3), round(mean(r), 4), round(std(r), 4),
                                round(mean(nc), 2), round(mean(sh), 4)])
with open(os.path.join(OUT_DIR, "test_corruption_summary.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "method", "train_noise", "corruption", "level", "tag_acc_mean", "tag_acc_std",
                "retention_mean", "retention_std", "n_collapsed_mean", "recency_share_mean"])
    w.writerows(summary)

# ---- paired comparisons ----
rows = []


def add(family, metric, d, label, p, c, l, res):
    if res is None:
        return
    n, m, lo, hi, w = res
    rows.append([family, metric, d, label, p, c, l, n, round(m, 4), round(lo, 4), round(hi, 4), w, int(lo > 0 or hi < 0)])


for d in DATASETS:
    for c in CORR:
        for l in LEVELS:
            for m in METHODS:
                add("drop_from_uncorrupted", "tag_acc", d, NAMES[m], 0.0, c, l,
                    paired(series(acc, d, m, 0.0, "none", 0), series(acc, d, m, 0.0, c, l)))
                for p in NOISES[1:]:
                    add("train_noise_effect_on_retention", "retention", d, NAMES[m], p, c, l,
                        paired(series(ret, d, m, p, c, l), series(ret, d, m, 0.0, c, l)))
hdr = ["family", "metric", "dataset", "comparison", "train_noise", "corruption", "level", "n", "mean_diff",
       "ci95_lo", "ci95_hi", "wins", "excludes_0"]
with open(os.path.join(OUT_DIR, "test_corruption_paired.csv"), "w", newline="") as f:
    csv.writer(f).writerows([hdr] + rows)

# ---- console summary ----
sm = {(r[0], r[1], r[2], r[3], r[4]): r for r in summary}
print(f"loaded {len(rec)} records; wrote test_corruption_summary.csv and test_corruption_paired.csv\n")

print("== Scenario A (clean-trained checkpoints): mean TAg Acc by level; last column = retention at level 4 ==")
print(f"{'dataset':12s} {'method':6s} {'corruption':11s} {'clean':>6s} {'L1':>6s} {'L2':>6s} {'L3':>6s} {'L4':>6s} {'ret_L4':>7s}")
for d in DATASETS:
    for m in METHODS:
        for c in CORR:
            vals = " ".join(f"{sm[(d, m, 0.0, c, l)][5]:6.2f}" for l in LEVELS)
            print(f"{d:12s} {NAMES[m]:6s} {c:11s} {sm[(d, m, 0.0, 'none', 0)][5]:6.2f} {vals} {sm[(d, m, 0.0, c, 4)][7]:7.3f}")

print("\n== Paired comparisons: intervals that exclude zero ==")
cnt = defaultdict(lambda: [0, 0])
for r in rows:
    cnt[(r[0], r[1])][0] += 1
    cnt[(r[0], r[1])][1] += r[12]
for k in sorted(cnt):
    print(f"{k[0]:34s} {k[1]:9s} {cnt[k][1]:3d} of {cnt[k][0]:3d}")

print("\n== Scenario C: mean retention at level 4 (corrupted / uncorrupted accuracy of the same checkpoint) ==")
print(f"{'dataset':12s} {'method':6s} {'p':>4s} " + " ".join(f"{c:>12s}" for c in CORR))
for d in DATASETS:
    for m in METHODS:
        for p in NOISES:
            print(f"{d:12s} {NAMES[m]:6s} {p:4.1f} " + " ".join(f"{sm[(d, m, p, c, 4)][7]:12.3f}" for c in CORR))

print("\n== Recency under corruption (clean-trained): collapsed classes / recency share, uncorrupted vs level 4 ==")
for d in DATASETS:
    for m in METHODS:
        for c in CORR:
            a, b = sm[(d, m, 0.0, "none", 0)], sm[(d, m, 0.0, c, 4)]
            print(f"{d:12s} {NAMES[m]:6s} {c:11s} {a[9]:5.2f} / {a[10]:.2f}  ->  {b[9]:5.2f} / {b[10]:.2f}")
