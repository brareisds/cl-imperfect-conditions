import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, INK, GRID = "#2a78d6", "#eb6834", "#222222", "#dddddd"
DATASETS = [("organamnist", "OrganAMNIST"), ("bloodmnist", "BloodMNIST")]
COLOR = {"ewc": BLUE, "icarl": ORANGE}
NAME = {"ewc": "EWC", "icarl": "iCaRL"}
W = 4.8

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": "#888888",
                     "axes.linewidth": 0.6, "pdf.fonttype": 42, "font.family": "sans-serif"})


def rd(path):
    return list(csv.DictReader(open(path)))


def style(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.5)


def fig_tag_vs_noise(summ, out):
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.5), sharey=True)
    for ax, (d, title) in zip(axes, DATASETS):
        for m in ("ewc", "icarl"):
            for clf, ls, mfc, off in (("nme", "-", None, 0.003), ("logits", "--", "white", -0.003)):
                rows = sorted([r for r in summ if r["mode"] == "intra" and r["dataset"] == d and r["method"] == m],
                              key=lambda r: float(r["noise"]))
                x = [float(r["noise"]) + (off if m == "ewc" else -off) for r in rows]
                y = [float(r[f"tag_{clf}_mean"]) for r in rows]
                e = [float(r[f"tag_{clf}_std"]) for r in rows]
                ax.errorbar(x, y, yerr=e, color=COLOR[m], ls=ls, lw=1.2, marker="o" if m == "ewc" else "s", ms=4,
                            mfc=mfc or COLOR[m], mec=COLOR[m], elinewidth=0.6, capsize=1.5, capthick=0.6)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xticks([0, 0.1, 0.2, 0.3])
        ax.set_xlabel("Training label noise $p$ (intra-task)")
        ax.set_ylim(15, 80)
        style(ax)
    axes[0].set_ylabel("TAg accuracy (%)")
    h = [Line2D([], [], color=BLUE, marker="o", ms=4, lw=1.2, label="EWC"),
         Line2D([], [], color=ORANGE, marker="s", ms=4, lw=1.2, label="iCaRL"),
         Line2D([], [], color="#555555", ls="-", lw=1.2, label="NME"),
         Line2D([], [], color="#555555", ls="--", lw=1.2, label="Logits")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)


def forest(ax, rows, y0, color, filled, offset=0.0):
    for i, r in enumerate(rows):
        y = y0[i] + offset
        lo, hi, mean = float(r["ci95_lo"]), float(r["ci95_hi"]), float(r["mean_diff"])
        ax.plot([lo, hi], [y, y], color=color, lw=1.3, solid_capstyle="butt")
        ax.plot([mean], [y], marker="o", ms=4.2, color=color, mfc=color if filled else "white", mew=1.0, zorder=3)
        if r["excludes_0"] == "1":
            ax.text(max(hi, mean) + 0.8, y, "*", va="center", ha="left", fontsize=8, color=INK)


def sel(par, **kw):
    return [r for r in par if all(r[k] == v for k, v in kw.items())]


def fig_classifier_effect(par, out):
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.6), sharex=True)
    for ax, (d, title) in zip(axes, DATASETS):
        noises = ["0.0", "0.1", "0.2", "0.3"]
        base = list(range(len(noises)))[::-1]
        for m, off in (("ewc", 0.17), ("icarl", -0.17)):
            rows = [sel(par, kind=f"{m}_nme_minus_logits", mode="intra", dataset=d, noise=n)[0] for n in noises]
            forest(ax, rows, base, COLOR[m], True, off)
        ax.axvline(0, color="#777777", lw=0.7)
        ax.set_yticks(base)
        ax.set_yticklabels([f"$p$={n}" for n in noises])
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("NME $-$ logits, TAg (points)")
        ax.set_ylim(-0.6, len(noises) - 0.4)
        style(ax, "x")
    axes[1].tick_params(labelleft=True)
    h = [Line2D([], [], color=BLUE, marker="o", ms=4, lw=1.3, label="EWC"),
         Line2D([], [], color=ORANGE, marker="o", ms=4, lw=1.3, label="iCaRL")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)


def fig_method_effect(par, out):
    labels = [("intra", "0.0"), ("intra", "0.1"), ("intra", "0.2"), ("intra", "0.3"), ("cross", "0.1"), ("cross", "0.3")]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.9), sharex=True)
    for ax, (d, title) in zip(axes, DATASETS):
        base = list(range(len(labels)))[::-1]
        for kind, filled, off in (("icarl_nme_minus_ewc_nme", True, 0.17), ("icarl_logits_minus_ewc_logits", False, -0.17)):
            rows = [sel(par, kind=kind, mode=mo, dataset=d, noise=n)[0] for mo, n in labels]
            forest(ax, rows, base, INK, filled, off)
        ax.axvline(0, color="#777777", lw=0.7)
        ax.set_yticks(base)
        ax.set_yticklabels([f"$p$={n}" + (" (cross)" if mo == "cross" else "") for mo, n in labels])
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("iCaRL $-$ EWC, TAg (points)")
        ax.set_ylim(-0.6, len(labels) - 0.4)
        style(ax, "x")
    h = [Line2D([], [], color=INK, marker="o", ms=4, lw=1.3, label="Both with NME"),
         Line2D([], [], color=INK, marker="o", ms=4, mfc="white", lw=1.3, label="Both with logits")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)


def fig_cross_effect(par, out):
    rows_def = [("ewc", "0.1"), ("icarl", "0.1"), ("ewc", "0.3"), ("icarl", "0.3")]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.5), sharex=True)
    for ax, (d, title) in zip(axes, DATASETS):
        base = list(range(len(rows_def)))[::-1]
        for kind, filled, off in (("cross_minus_intra_NME", True, 0.17), ("cross_minus_intra_logits", False, -0.17)):
            rows = []
            for m, n in rows_def:
                cand = [r for r in sel(par, kind=kind, mode="cross", dataset=d, noise=n)
                        if r["comparison"].lower().startswith(NAME[m].lower())]
                rows.append(cand[0])
            forest(ax, rows, base, INK, filled, off)
        ax.axvline(0, color="#777777", lw=0.7)
        ax.set_yticks(base)
        ax.set_yticklabels([f"{NAME[m]}, $p$={n}" for m, n in rows_def])
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("Cross $-$ intra, TAg (points)")
        ax.set_ylim(-0.6, len(rows_def) - 0.4)
        style(ax, "x")
    h = [Line2D([], [], color=INK, marker="o", ms=4, lw=1.3, label="NME"),
         Line2D([], [], color=INK, marker="o", ms=4, mfc="white", lw=1.3, label="Logits")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default=HERE)
    ap.add_argument("--outdir", default=os.path.join(HERE, "paper_figs"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    summ = rd(os.path.join(a.indir, "summary_nme.csv"))
    par = rd(os.path.join(a.indir, "paired_nme.csv"))
    fig_tag_vs_noise(summ, os.path.join(a.outdir, "fig_tag_vs_noise.png"))
    fig_classifier_effect(par, os.path.join(a.outdir, "fig_classifier_effect.png"))
    fig_method_effect(par, os.path.join(a.outdir, "fig_method_effect.png"))
    fig_cross_effect(par, os.path.join(a.outdir, "fig_cross_effect.png"))
    print("figures written to", a.outdir)


if __name__ == "__main__":
    main()
