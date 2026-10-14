import argparse
import csv
import math
import os
from collections import defaultdict

try:
    from scipy import stats
    T4 = float(stats.t.ppf(0.975, 4))
except ImportError:
    T4 = 2.776

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT_DIR = os.path.join(ROOT, "analysis", "label_noise")
DATASETS = ["organamnist", "bloodmnist", "pathmnist", "tissuemnist"]


def mean(x):
    return sum(x) / len(x)


def std(x):
    m = mean(x)
    return (sum((v - m) ** 2 for v in x) / len(x)) ** 0.5


def paired(a, b):
    seeds = sorted(set(a) & set(b))
    diff = [a[s] - b[s] for s in seeds]
    n = len(diff)
    m = mean(diff)
    sd = (sum((v - m) ** 2 for v in diff) / (n - 1)) ** 0.5
    h = T4 * sd / math.sqrt(n)
    return n, m, m - h, m + h, sum(1 for v in diff if v > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default=os.path.join(OUT_DIR, "nme_eval.csv"))
    ap.add_argument("--outdir", default=OUT_DIR)
    a = ap.parse_args()
    runs = defaultdict(dict)
    for r in csv.DictReader(open(a.inp)):
        assert r["log_ok"] == "1", f"unvalidated row: {r['phase']} {r['dataset']} {r['method']} {r['noise']} {r['seed']}"
        runs[(r["mode"], r["dataset"], r["method"], float(r["noise"]))][int(r["seed"])] = r

    def col(k, c):
        return {s: float(v[c]) for s, v in runs[k].items()}

    summary = []
    for (mode, d, m, p), v in sorted(runs.items()):
        nme, lg = list(col((mode, d, m, p), "tag_nme").values()), list(col((mode, d, m, p), "tag_logits").values())
        summary.append([mode, d, m, p, len(v), round(mean(nme), 3), round(std(nme), 3), round(mean(lg), 3), round(std(lg), 3)])

    kinds = [
        ("ewc_nme_minus_logits", "EWC NME - EWC logits", "ewc", "tag_nme", "ewc", "tag_logits"),
        ("icarl_nme_minus_logits", "iCaRL NME - iCaRL logits", "icarl", "tag_nme", "icarl", "tag_logits"),
        ("icarl_nme_minus_ewc_nme", "iCaRL NME - EWC NME", "icarl", "tag_nme", "ewc", "tag_nme"),
        ("icarl_logits_minus_ewc_logits", "iCaRL logits - EWC logits", "icarl", "tag_logits", "ewc", "tag_logits"),
    ]
    paired_rows = []
    for kind, label, m1, c1, m2, c2 in kinds:
        for mode in ("intra", "cross", "other"):
            for d in DATASETS:
                for p in sorted({k[3] for k in runs if k[0] == mode}):
                    k1, k2 = (mode, d, m1, p), (mode, d, m2, p)
                    if k1 not in runs or k2 not in runs:
                        continue
                    n, dm, lo, hi, w = paired(col(k1, c1), col(k2, c2))
                    paired_rows.append([kind, mode, d, p, label, n, round(dm, 3), round(lo, 3), round(hi, 3), w, int(lo > 0 or hi < 0)])

    for m, mname in (("ewc", "EWC"), ("icarl", "iCaRL")):
        for c, cname in (("tag_nme", "NME"), ("tag_logits", "logits")):
            for d in DATASETS:
                for p in (0.1, 0.3):
                    k1, k2 = ("cross", d, m, p), ("intra", d, m, p)
                    if k1 not in runs or k2 not in runs:
                        continue
                    n, dm, lo, hi, w = paired(col(k1, c), col(k2, c))
                    paired_rows.append([f"cross_minus_intra_{cname}", "cross", d, p, f"{mname} cross - intra ({cname})",
                                        n, round(dm, 3), round(lo, 3), round(hi, 3), w, int(lo > 0 or hi < 0)])

    with open(os.path.join(a.outdir, "summary_nme.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "dataset", "method", "noise", "n", "tag_nme_mean", "tag_nme_std", "tag_logits_mean", "tag_logits_std"])
        w.writerows(summary)
    with open(os.path.join(a.outdir, "paired_nme.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "mode", "dataset", "noise", "comparison", "n", "mean_diff", "ci95_lo", "ci95_hi", "wins", "excludes_0"])
        w.writerows(paired_rows)

    print(f"t(0.975, df=4) = {T4:.4f}")
    print("\n== paired differences in TAg acc (wins = seeds with positive difference; * = interval excludes zero) ==")
    for r in paired_rows:
        print(f"{r[0]:30s} {r[1]:5s} {r[2]:11s} p={r[3]:.1f} n={r[5]} diff={r[6]:8.3f} CI[{r[7]:8.3f},{r[8]:8.3f}] wins={r[9]}/{r[5]} {'*' if r[10] else ''}")


if __name__ == "__main__":
    main()
