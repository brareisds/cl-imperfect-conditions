import csv
import math
import os

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
NAMES = {"ewc": "EWC", "icarl": "iCaRL"}
NOISES = [0.1, 0.2, 0.3]

acc = {}
with open(os.path.join(OUT_DIR, "per_seed.csv")) as f:
    for r in csv.DictReader(f):
        acc.setdefault((r["dataset"], r["method"], float(r["noise"])), {})[int(r["seed"])] = float(r["tag_acc"])


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


def add(kind, d, p, label, res):
    if res is None:
        return
    n, m, lo, hi, w = res
    rows.append([kind, d, p, label, n, round(m, 4), round(lo, 4), round(hi, 4), w, int(lo > 0 or hi < 0)])


for d in DATASETS:
    for p in NOISES:
        for x, y in [("icarl", "ewc")]:
            add("method_diff_TAg_acc", d, p, f"{NAMES[x]} - {NAMES[y]}",
                paired(acc.get((d, x, p), {}), acc.get((d, y, p), {})))
    for m_ in NAMES:
        for p in NOISES:
            add("drop_from_clean_TAg_acc", d, p, NAMES[m_],
                paired(acc.get((d, m_, 0.0), {}), acc.get((d, m_, p), {})))
    ret = {}
    for m_ in NAMES:
        clean = acc.get((d, m_, 0.0), {})
        for p in NOISES:
            ret[(m_, p)] = {s: v / clean[s] for s, v in acc.get((d, m_, p), {}).items() if s in clean and clean[s] > 0}
    for p in NOISES:
        for x, y in [("icarl", "ewc")]:
            add("retention_diff", d, p, f"{NAMES[x]} - {NAMES[y]}", paired(ret[(x, p)], ret[(y, p)]))

for d in DATASETS:
    for x, y in [("icarl", "ewc")]:
        add("method_diff_TAg_acc", d, 0.0, f"{NAMES[x]} - {NAMES[y]}",
            paired(acc.get((d, x, 0.0), {}), acc.get((d, y, 0.0), {})))

hdr = ["analysis", "dataset", "noise", "comparison", "n", "mean_diff", "ci95_lo", "ci95_hi", "wins", "excludes_0"]
with open(os.path.join(OUT_DIR, "paired_stats.csv"), "w", newline="") as f:
    csv.writer(f).writerows([hdr] + rows)

for r in rows:
    print(f"{r[0]:25s} {r[1]:11s} {r[2]:3.1f} {r[3]:16s} n={r[4]} diff={r[5]:8.3f} "
          f"CI[{r[6]:8.3f},{r[7]:8.3f}] wins={r[8]}/{r[4]} {'*' if r[9] else ''}")
