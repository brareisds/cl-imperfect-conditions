import csv
import os
from collections import defaultdict

import numpy as np

try:
    from scipy import stats
    T95 = float(stats.t.ppf(0.975, 4))
except Exception:
    T95 = 2.776

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(OUT_DIR, "test_corruption_nme.csv")
MAXLEVEL = 4


def load():
    acc = {}
    for r in csv.DictReader(open(SRC)):
        k = (r["dataset"], r["method"], float(r["train_noise"]), r["classifier"], r["corruption"], int(r["level"]), int(r["seed"]))
        acc[k] = (float(r["tag_acc"]), float(r["n_collapsed"]), float(r["recency_share"]))
    return acc


def ci(diffs):
    d = np.asarray(diffs, float)
    n = len(d)
    m = d.mean()
    h = T95 * d.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    return n, m, m - h, m + h, int((d > 0).sum())


def main():
    acc = load()
    keys = sorted({k[:6] for k in acc})
    summ = []
    for dset, m, p, c, kind, lv in keys:
        seeds = sorted(k[6] for k in acc if k[:6] == (dset, m, p, c, kind, lv))
        a = [acc[(dset, m, p, c, kind, lv, s)] for s in seeds]
        base = [acc[(dset, m, p, c, "none", 0, s)][0] for s in seeds]
        ret = [x[0] / b for x, b in zip(a, base)]
        summ.append([dset, m, p, c, kind, lv, len(seeds), np.mean([x[0] for x in a]), np.std([x[0] for x in a], ddof=1),
                     np.mean(ret), np.mean([x[1] for x in a]), np.mean([x[2] for x in a])])
    with open(os.path.join(OUT_DIR, "test_corruption_nme_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "method", "train_noise", "classifier", "corruption", "level", "n", "tag_acc_mean",
                    "tag_acc_std", "retention_mean", "n_collapsed_mean", "recency_share_mean"])
        for r in summ:
            w.writerow(r[:7] + [round(x, 4) for x in r[7:]])

    pr = []
    conds = sorted({(k[0], k[2], k[4], k[5]) for k in acc})
    for dset, p, kind, lv in conds:
        def get(m, c):
            return {k[6]: v[0] for k, v in acc.items() if k[:6] == (dset, m, p, c, kind, lv)}

        def diff(x, y):
            s = sorted(set(x) & set(y))
            return [x[i] - y[i] for i in s]
        tests = [("nme_minus_logits", "icarl", get("icarl", "nme"), get("icarl", "logits")),
                 ("nme_minus_logits", "ewc", get("ewc", "nme"), get("ewc", "logits")),
                 ("icarl_minus_ewc_nme", "icarl-ewc", get("icarl", "nme"), get("ewc", "nme")),
                 ("icarl_minus_ewc_logits", "icarl-ewc", get("icarl", "logits"), get("ewc", "logits"))]
        for fam, comp, x, y in tests:
            dd = diff(x, y)
            if len(dd) >= 2:
                n, mm, lo, hi, wins = ci(dd)
                pr.append([fam, dset, comp, p, kind, lv, n, round(mm, 3), round(lo, 3), round(hi, 3), wins, int(lo > 0 or hi < 0)])
        if lv > 0:
            for m in ("icarl", "ewc"):
                for c in ("nme", "logits"):
                    x = get(m, c)
                    y = {k[6]: v[0] for k, v in acc.items() if k[:6] == (dset, m, p, c, "none", 0)}
                    dd = diff(x, y)
                    if len(dd) >= 2:
                        n, mm, lo, hi, wins = ci(dd)
                        pr.append(["change_from_uncorrupted_" + c, dset, m, p, kind, lv, n, round(mm, 3), round(lo, 3), round(hi, 3), wins, int(lo > 0 or hi < 0)])
    with open(os.path.join(OUT_DIR, "test_corruption_nme_paired.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "dataset", "comparison", "train_noise", "corruption", "level", "n", "mean_diff", "ci95_lo", "ci95_hi", "wins", "excludes_0"])
        w.writerows(pr)
    print(f"wrote {len(summ)} summary rows, {len(pr)} paired rows")


if __name__ == "__main__":
    main()
