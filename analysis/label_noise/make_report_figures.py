import argparse
import csv
import math
import os
from collections import defaultdict
import statistics
import matplotlib
from matplotlib.lines import Line2D
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DEF_DIR = os.path.join(ROOT, "analysis", "label_noise")

DATASETS = [("organamnist", "OrganAMNIST"), ("bloodmnist", "BloodMNIST")]
METHODS = {"icarl": "iCaRL", "ewc": "EWC"}
COLORS = {"ewc": "#1f4e9c", "icarl": "#2a8a4a"}
MARKERS = {"ewc": "o", "icarl": "^"}
CORR = [("gauss_noise", "Gaussian noise"), ("blur", "Blur"), ("contrast", "Contrast"), ("brightness", "Brightness")]
NOISES = [0.0, 0.1, 0.2, 0.3]

plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.linewidth": 0.6})


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def save(fig, outdir, name, png):
    fig.savefig(os.path.join(outdir, name + ".pdf"))
    if png:
        fig.savefig(os.path.join(outdir, name + ".png"), dpi=300)
    plt.close(fig)
    print("saved", os.path.join(outdir, name + (".pdf (+ .png)" if png else ".pdf")))


def fig_acc_noise(rows, outdir, png):
    val = {(r["dataset"], r["method"], round(float(r["noise"]), 2)): (float(r["tag_acc_mean"]), float(r["tag_acc_std"]))
           for r in rows}
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.5), sharey=True)
    for ax, (ds, title) in zip(axs, DATASETS):
        for m, label in METHODS.items():
            pts = [val[(ds, m, p)] for p in NOISES]
            ax.errorbar(NOISES, [a for a, _ in pts], yerr=[s for _, s in pts], color=COLORS[m], marker=MARKERS[m],
                        ms=3.5, lw=1.2, capsize=2, elinewidth=0.6, label=label)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("label noise $p$ (training)")
        ax.set_xticks(NOISES)
        ax.grid(alpha=0.25, lw=0.4)
    axs[0].set_ylabel("TAg Acc (%)")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, outdir, "fig_acc_noise", png)


def fig_corruption(rows, outdir, png):
    val = {}
    for r in rows:
        if round(float(r["train_noise"]), 2) == 0.0:
            val[(r["dataset"], r["method"], r["corruption"], int(r["level"]))] = float(r["tag_acc_mean"])

    def curve(ds, m, c):
        return [val[(ds, m, "none", 0)]] + [val[(ds, m, c, l)] for l in (1, 2, 3, 4)]

    allv = [v for ds, _ in DATASETS for m in ("ewc",) for c, _ in CORR for v in curve(ds, m, c)]
    lo, hi = 5 * math.floor((min(allv) - 1) / 5), 5 * math.ceil((max(allv) + 1) / 5)
    fig, axs = plt.subplots(2, 4, figsize=(6.4, 3.3), sharex=True)
    for r_i, (ds, title) in enumerate(DATASETS):
        for c_i, (c, cname) in enumerate(CORR):
            ax = axs[r_i][c_i]
            for m in ("ewc",):
                ax.plot(range(5), curve(ds, m, c), color=COLORS[m], marker=MARKERS[m], ms=3, lw=1.1, label=METHODS[m])
            ax.set_ylim(lo, hi)
            ax.grid(alpha=0.25, lw=0.4)
            ax.set_xticks(range(5))
            ax.set_xticklabels(["clean", "1", "2", "3", "4"])
            if r_i == 0:
                ax.set_title(cname, fontsize=7.5)
            if c_i == 0:
                ax.set_ylabel(title + "\nTAg Acc (%)", fontsize=7)
            else:
                ax.set_yticklabels([])
            if r_i == 1:
                ax.set_xlabel("severity level", fontsize=7)
    axs[0][0].legend(frameon=False, fontsize=6.5, loc="lower left")
    fig.tight_layout()
    save(fig, outdir, "fig_corruption", png)


def fig_fisher_change(rows, outdir, png):
    levels = ["0", "0.01", "0.1", "0.2", "0.3"]
    g = defaultdict(list)
    for r in rows:
        g[(r["dataset"], r["method"], r["noise"])].append(float(r["signed_change_pct"]))
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.4), sharey=True)
    for ax, (ds, title) in zip(axs, DATASETS):
        for m in ("ewc",):
            ys = [statistics.median(g[(ds, m, p)]) for p in levels]
            ax.plot(range(5), ys, color=COLORS[m], marker=MARKERS[m], ms=3.5, lw=1.2, label=METHODS[m])
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_xticks(range(5))
        ax.set_xticklabels(levels)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("label noise $p$ (training)")
        ax.grid(alpha=0.25, lw=0.4)
    axs[0].set_ylabel("change of the Fisher magnitude\nvs. clean run (%)")
    axs[0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    save(fig, outdir, "fig_fisher_change", png)


def fig_fisher_mechanism(rows, outdir, png):
    levels, xs = ["0", "0.1", "0.2", "0.3"], [0, 0.1, 0.2, 0.3]
    g = defaultdict(list)
    for r in rows:
        g[(r["dataset"], r["method"], r["noise"])].append(r)

    def med(ds, p, k):
        return statistics.median(float(x[k]) for x in g[(ds, "ewc", p)])

    style = {"organamnist": "-", "bloodmnist": "--"}
    fig, axs = plt.subplots(1, 2, figsize=(4.6, 2.4))
    for ds, _ in DATASETS:
        for ax, k in zip(axs, ("conf", "g_ewc")):
            ax.plot(xs, [med(ds, p, k) for p in levels], color=COLORS["ewc"], ls=style[ds], marker="o", ms=3, lw=1.1)
    for ax, t in zip(axs, ["(a) confidence of the model", "(b) gradient scale w.r.t. the logits"]):
        ax.set_title(t, fontsize=7)
        ax.set_xlabel("label noise $p$ (training)", fontsize=7)
        ax.set_xticks(xs)
        ax.grid(alpha=0.25, lw=0.4)
    axs[1].legend(handles=[Line2D([], [], color="gray", ls="-", lw=1.1, label="OrganAMNIST"),
                           Line2D([], [], color="gray", ls="--", lw=1.1, label="BloodMNIST")],
                  frameon=False, fontsize=6.5, loc="center right")
    fig.tight_layout()
    save(fig, outdir, "fig_fisher_mechanism", png)


def main():
    ap = argparse.ArgumentParser(description="Figures of the progress report, from the analysis CSVs.")
    ap.add_argument("--csv-dir", default=DEF_DIR, help="folder with summary.csv, test_corruption_summary.csv, fisher_by_noise.csv and fisher_confidence_<dataset>.csv")
    ap.add_argument("--outdir", default=os.path.join(DEF_DIR, "report_figs"))
    ap.add_argument("--png", action="store_true", help="also save 300-dpi PNG files")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    fig_acc_noise(read(os.path.join(a.csv_dir, "summary.csv")), a.outdir, a.png)
    fig_corruption(read(os.path.join(a.csv_dir, "test_corruption_summary.csv")), a.outdir, a.png)
    fig_fisher_change(read(os.path.join(a.csv_dir, "fisher_by_noise.csv")), a.outdir, a.png)
    conf_rows = []
    for d, _ in DATASETS:
        conf_rows += read(os.path.join(a.csv_dir, f"fisher_confidence_{d}.csv"))
    fig_fisher_mechanism(conf_rows, a.outdir, a.png)


if __name__ == "__main__":
    main()
