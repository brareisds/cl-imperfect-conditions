import csv
import glob
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
NOISE_ROOT = os.path.join(ROOT, "results/label_noise")
OFFICIAL_ROOT = os.path.join(ROOT, "results/official_savemodel")
OUT_DIR = os.path.join(ROOT, "analysis", "label_noise")

DATASETS = ["organamnist", "bloodmnist"]
METHODS = ["ewc", "icarl"]
NOISES = [0.0, 0.1, 0.2, 0.3]
NTASKS = 4
STD_DDOF = 0  # 0 = population std (np.std default)

TABLE1 = {  # TAg Acc, lr=0.001, progress report Table 1
    ("organamnist", "ewc"): 43.00, ("organamnist", "icarl"): 62.48,
    ("bloodmnist", "ewc"): 45.92, ("bloodmnist", "icarl"): 66.66,
}

HEADERS = {"TAw Acc": "taw_acc", "TAg Acc": "tag_acc", "TAw Forg": "taw_forg", "TAg Forg": "tag_forg"}
PCT = re.compile(r"-?\d+\.\d+(?=%)")
AVG = re.compile(r"Avg\.:\s*(-?\d+\.\d+)%")


# ---- parsing ----
def parse_tables(path):
    lines = open(path, errors="replace").read().splitlines()
    out, i = {}, 0
    while i < len(lines):
        name = lines[i].strip()
        if name in HEADERS:
            rows, j = [], i + 1
            while j < len(lines) and not lines[j].startswith("*"):
                if "%" in lines[j]:
                    rows.append(lines[j])
                j += 1
            out[HEADERS[name]] = rows
            i = j
        else:
            i += 1
    return out


def last_avg(rows, kind):
    assert len(rows) == NTASKS, f"{len(rows)} rows"
    m = AVG.search(rows[-1])
    assert m, f"no Avg. in {rows[-1]!r}"
    if kind == "acc":
        assert all("Avg." in r for r in rows), "acc table with a row lacking Avg."
        vals = [float(v) for v in PCT.findall(rows[-1].split("Avg.:")[0])]
        assert len(vals) == NTASKS and abs(sum(vals) / NTASKS - float(m.group(1))) <= 0.11, rows[-1]
    else:
        assert "Avg." not in rows[0], "forg table whose first row has Avg."
    return float(m.group(1))


def load(base):
    runs = {}
    for d in sorted(glob.glob(os.path.join(base, "*_fixd_*"))):
        m = re.search(r"_fixd_(\d+)$", d)
        if not (m and os.path.isdir(d)):
            continue
        files = sorted(glob.glob(os.path.join(d, "stdout-*.txt")))
        if not files:
            print(f"WARN no stdout: {d}")
            continue
        if len(files) > 1:
            print(f"WARN {len(files)} stdout files in {d}, using {os.path.basename(files[-1])}")
        t = parse_tables(files[-1])
        if set(t) != set(HEADERS.values()):
            print(f"WARN incomplete tables: {files[-1]}")
            continue
        runs[int(m.group(1))] = {k: last_avg(r, "acc" if k.endswith("acc") else "forg") for k, r in t.items()}
    return runs


# ---- stats ----
def mean(x):
    return sum(x) / len(x)


def std(x):
    n = len(x)
    if n - STD_DDOF <= 0:
        return float("nan")
    m = mean(x)
    return (sum((v - m) ** 2 for v in x) / (n - STD_DDOF)) ** 0.5


# ---- main ----
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    official = {(d, m): load(os.path.join(OFFICIAL_ROOT, f"{d}_{m}")) for d in DATASETS for m in METHODS}
    p0 = {(d, m): load(os.path.join(NOISE_ROOT, f"{d}_{m}_p0")) for d in DATASETS for m in METHODS}
    clean = {k: (p0[k] or official[k]) for k in official}
    for k in clean:
        print(f"clean baseline {k}: {'p0 folder' if p0[k] else 'official folder'}")

    per_seed, summary = [], []
    for d in DATASETS:
        for m in METHODS:
            for p in NOISES:
                runs = clean[(d, m)] if p == 0.0 else load(os.path.join(NOISE_ROOT, f"{d}_{m}_p{p}"))
                if not runs:
                    continue
                for s, r in sorted(runs.items()):
                    per_seed.append([d, m, p, s, r["tag_acc"], r["taw_acc"], r["tag_forg"], r["taw_forg"]])
                ref = clean[(d, m)]
                ret = [runs[s]["tag_acc"] / ref[s]["tag_acc"] for s in runs if s in ref and ref[s]["tag_acc"] > 0]
                col = lambda k: [r[k] for r in runs.values()]
                summary.append([d, m, p, len(runs),
                                mean(col("tag_acc")), std(col("tag_acc")),
                                mean(col("tag_forg")), std(col("tag_forg")),
                                mean(col("taw_acc")),
                                mean(ret) if ret else float("nan"), std(ret) if ret else float("nan")])

    with open(os.path.join(OUT_DIR, "per_seed.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "method", "noise", "seed", "tag_acc", "taw_acc", "tag_forg", "taw_forg"])
        w.writerows(per_seed)
    with open(os.path.join(OUT_DIR, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "method", "noise", "n", "tag_acc_mean", "tag_acc_std",
                    "tag_forg_mean", "tag_forg_std", "taw_acc_mean", "retention_mean", "retention_std"])
        w.writerows([r[:4] + [round(x, 3) for x in r[4:]] for r in summary])

    print(f"\n{'dataset':12s} {'method':6s} {'noise':>5s} {'n':>2s} {'TAg acc':>15s} {'TAg forg':>15s} {'retention':>14s}")
    for r in summary:
        print(f"{r[0]:12s} {r[1]:6s} {r[2]:5.1f} {r[3]:2d} {r[4]:7.2f} ±{r[5]:5.2f} {r[6]:7.2f} ±{r[7]:5.2f} {r[9]:7.3f} ±{r[10]:5.3f}")

    print("\n== Check 1: official runs vs Table 1 (TAg Acc mean) ==")
    for (d, m), exp in TABLE1.items():
        runs = official[(d, m)]
        got = mean([r["tag_acc"] for r in runs.values()]) if runs else float("nan")
        flag = "OK" if abs(got - exp) < 0.01 else "MISMATCH"
        print(f"{d:12s} {m:6s} n={len(runs)} got={got:6.2f} report={exp:6.2f} {flag}")

    print("\n== Check 2: p0 folder minus official folder, TAg Acc per seed ==")
    for d in DATASETS:
        for m in METHODS:
            diffs = {s: round(p0[(d, m)][s]["tag_acc"] - official[(d, m)][s]["tag_acc"], 2)
                     for s in p0[(d, m)] if s in official[(d, m)]}
            print(f"{d:12s} {m:6s} {diffs}")


if __name__ == "__main__":
    main()
