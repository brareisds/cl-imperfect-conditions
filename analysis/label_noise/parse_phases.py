import csv
import math
import os
import re

from parse_label_noise import DATASETS, NOISES, OUT_DIR, ROOT, load, mean, std

try:
    from scipy import stats

    def tcrit(df):
        return float(stats.t.ppf(0.975, df))
except ImportError:
    print("WARN scipy not found: using a rounded t table")
    _T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}

    def tcrit(df):
        return _T.get(df, 1.96)

PHASES = ["phase1_icarl", "phase2_cross", "phase3_fisher_norm", "ablation_noweight"]
DIRPAT = re.compile(r"^(organamnist|bloodmnist)_(ewc|icarl)_p(\d+(?:\.\d+)?)$")
NAMES = {"ewc": "EWC", "icarl": "iCaRL"}
SANITY_TOL = 0.15


def collect():
    runs = {}
    for ph in PHASES:
        base = os.path.join(ROOT, "results", ph)
        if not os.path.isdir(base):
            print(f"WARN missing {base}")
            continue
        for name in sorted(os.listdir(base)):
            m = DIRPAT.match(name)
            if m and os.path.isdir(os.path.join(base, name)):
                r = load(os.path.join(base, name))
                if r:
                    runs[(ph, m.group(1), m.group(2), float(m.group(3)))] = r
    return runs


def read_baseline():
    base = {}
    with open(os.path.join(OUT_DIR, "per_seed.csv")) as f:
        for r in csv.DictReader(f):
            base.setdefault((r["dataset"], r["method"], float(r["noise"])), {})[int(r["seed"])] = float(r["tag_acc"])
    return base


def read_logits():
    path = os.path.join(OUT_DIR, "phase0_base_runs.csv")
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out.setdefault((r["dataset"], r["method"], float(r["noise"])), {})[int(r["seed"])] = float(r["tag"])
    return out


def paired(a, b):
    seeds = sorted(set(a) & set(b))
    diffs = [a[s] - b[s] for s in seeds]
    n = len(diffs)
    if n < 2:
        return None
    m = mean(diffs)
    sd = math.sqrt(sum((x - m) ** 2 for x in diffs) / (n - 1))
    h = tcrit(n - 1) * sd / math.sqrt(n)
    lo, hi = m - h, m + h
    return n, m, lo, hi, sum(x > 0 for x in diffs), int(lo > 0 or hi < 0), max(abs(x) for x in diffs)


def main():
    runs = collect()
    base = read_baseline()
    logits = read_logits()

    per_seed, summary = [], []
    for (ph, d, a, p), r in sorted(runs.items()):
        for s, v in sorted(r.items()):
            per_seed.append([ph, d, a, p, s, v["tag_acc"], v["taw_acc"], v["tag_forg"], v["taw_forg"]])
        col = lambda k: [v[k] for v in r.values()]
        summary.append([ph, d, a, p, len(r), mean(col("tag_acc")), std(col("tag_acc")),
                        mean(col("tag_forg")), std(col("tag_forg"))])

    rows = []

    def add(kind, d, p, label, x, y):
        res = paired(x, y)
        if res:
            n, m, lo, hi, w, sig, mx = res
            rows.append([kind, d, p, label, n, round(m, 3), round(lo, 4), round(hi, 4), w, sig])

    for (ph, d, a, p), r in sorted(runs.items()):
        cur = {s: v["tag_acc"] for s, v in r.items()}
        ref = base.get((d, a, p), {})
        if ph == "phase1_icarl":
            add("rerun_minus_original_TAg_acc", d, p, f"{NAMES[a]} phase1 - label_noise", cur, ref)
        elif ph == "phase2_cross":
            add("cross_minus_intra_TAg_acc", d, p, f"{NAMES[a]} cross - intra", cur, ref)
        elif ph == "phase3_fisher_norm":
            add("fisher_norm_minus_ewc_TAg_acc", d, p, f"{NAMES[a]} normalized - standard", cur, ref)
        elif ph == "ablation_noweight":
            add("no_weights_minus_weights_TAg_acc", d, p, f"{NAMES[a]} without - with class weights", cur, ref)

    for d in DATASETS:
        for p in NOISES:
            ic, ew = runs.get(("phase2_cross", d, "icarl", p)), runs.get(("phase2_cross", d, "ewc", p))
            if ic and ew:
                add("cross_icarl_minus_ewc_TAg_acc", d, p, "iCaRL - EWC (cross-task noise)",
                    {s: v["tag_acc"] for s, v in ic.items()}, {s: v["tag_acc"] for s, v in ew.items()})

    for d in DATASETS:
        for p in NOISES:
            nme, lg_i, lg_e = base.get((d, "icarl", p), {}), logits.get((d, "icarl", p), {}), logits.get((d, "ewc", p), {})
            add("icarl_nme_minus_logits_TAg_acc", d, p, "iCaRL NME - iCaRL logits", nme, lg_i)
            add("icarl_logits_minus_ewc_TAg_acc", d, p, "iCaRL logits - EWC", lg_i, lg_e)

    print("== sanity: EWC TAg accuracy in the logs vs logits in phase0_base_runs.csv (max abs diff over seeds) ==")
    for d in DATASETS:
        for p in NOISES:
            log, lg = base.get((d, "ewc", p), {}), logits.get((d, "ewc", p), {})
            seeds = sorted(set(log) & set(lg))
            if seeds:
                mx = max(abs(log[s] - lg[s]) for s in seeds)
                print(f"{d:12s} p={p:.1f} n={len(seeds)} max|diff|={mx:.3f} {'OK' if mx <= SANITY_TOL else 'MISMATCH'}")

    with open(os.path.join(OUT_DIR, "per_seed_phases.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase", "dataset", "method", "noise", "seed", "tag_acc", "taw_acc", "tag_forg", "taw_forg"])
        w.writerows(per_seed)
    with open(os.path.join(OUT_DIR, "summary_phases.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase", "dataset", "method", "noise", "n", "tag_acc_mean", "tag_acc_std", "tag_forg_mean", "tag_forg_std"])
        w.writerows([r[:5] + [round(x, 3) for x in r[5:]] for r in summary])
    with open(os.path.join(OUT_DIR, "paired_phases.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["comparison", "dataset", "noise", "label", "n", "mean_diff", "ci_lo", "ci_hi", "wins", "excludes_zero"])
        w.writerows(rows)

    print(f"\n{'phase':20s} {'dataset':12s} {'method':6s} {'noise':>5s} {'n':>2s} {'TAg acc':>15s} {'TAg forg':>15s}")
    for r in summary:
        print(f"{r[0]:20s} {r[1]:12s} {r[2]:6s} {r[3]:5.1f} {r[4]:2d} {r[5]:7.2f} ±{r[6]:5.2f} {r[7]:7.2f} ±{r[8]:5.2f}")
    print("\n== paired differences (wins = seeds with positive difference; * = interval excludes zero) ==")
    for r in rows:
        print(f"{r[0]:34s} {r[1]:11s} p={r[2]:.1f} {r[3]:34s} n={r[4]} diff={r[5]:8.3f} CI[{r[6]:8.3f},{r[7]:8.3f}] wins={r[8]}/{r[4]} {'*' if r[9] else ''}")


if __name__ == "__main__":
    main()
