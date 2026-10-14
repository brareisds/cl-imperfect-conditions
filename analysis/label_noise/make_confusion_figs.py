import argparse, glob, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SPLITS = {"organamnist": [3, 3, 3, 2], "bloodmnist": [2, 2, 2, 2]}
NAMES = {"organamnist": "OrganAMNIST", "bloodmnist": "BloodMNIST"}
MLAB = {"ewc": "EWC (logits)", "icarl": "iCaRL (NME)", "icarl-logits": "iCaRL (logits)", "ewc-nme": "EWC (NME)"}


def load(indir, d, m, p):
    mats = []
    for f in sorted(glob.glob(os.path.join(indir, f"{d}_{m}_p{p}_seed*_confusion.csv"))):
        c = np.loadtxt(f, delimiter=",")
        s = c.sum(1, keepdims=True)
        mats.append(np.divide(c, s, out=np.zeros_like(c), where=s > 0))
    return (np.mean(mats, 0), len(mats)) if mats else (None, 0)


def paper(a):
    p = a.p
    rows = [("Logits", ("ewc", "EWC"), ("icarl-logits", "iCaRL")), ("NME", ("ewc-nme", "EWC"), ("icarl", "iCaRL"))]
    cols = [(d, mk) for d in SPLITS for mk in (0, 1)]
    fig, ax = plt.subplots(2, 4, figsize=(11, 5.6), squeeze=False)
    im = None
    for i, (clf, *ms) in enumerate(rows):
        for j, (d, k) in enumerate(cols):
            A = ax[i][j]
            m, lab = ms[k]
            M, _ = load(a.indir, d, m, p)
            if M is None:
                A.axis("off")
                continue
            im = A.imshow(M, vmin=0, vmax=1, cmap="Blues")
            for c in np.cumsum(SPLITS[d])[:-1] - 0.5:
                A.axvline(c, color="k", lw=0.6, ls="--")
                A.axhline(c, color="k", lw=0.6, ls="--")
            A.set_title(f"{NAMES[d]}, {lab} ({clf}), p={p}", fontsize=8)
            A.set_xticks(range(M.shape[1]))
            A.set_yticks(range(M.shape[0]))
            A.tick_params(labelsize=6)
            if i == 1:
                A.set_xlabel("predicted class", fontsize=8)
            if j == 0:
                A.set_ylabel("true class", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="row-normalised rate")
    out = os.path.join(a.outdir, f"fig_confusion_paper_p{p}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="confusions")
    ap.add_argument("--outdir", default="paper_figs")
    ap.add_argument("--ps", nargs="+", default=["0", "0.3"])
    ap.add_argument("--methods", nargs="+", default=["ewc", "icarl"])
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--p", default="0.3")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    if a.paper:
        paper(a)
        return
    for d, sp in SPLITS.items():
        panels = [(m, p, *load(a.indir, d, m, p)) for m in a.methods for p in a.ps]
        panels = [x for x in panels if x[2] is not None]
        if not panels:
            continue
        ms = [m for m in a.methods if any(x[0] == m for x in panels)]
        ps = [p for p in a.ps if any(x[1] == p for x in panels)]
        fig, ax = plt.subplots(len(ms), len(ps), figsize=(3.2 * len(ps), 3.1 * len(ms)), squeeze=False)
        cuts = np.cumsum(sp)[:-1] - 0.5
        for i, m in enumerate(ms):
            for j, p in enumerate(ps):
                A = ax[i][j]
                hit = [x for x in panels if x[0] == m and x[1] == p]
                if not hit:
                    A.axis("off")
                    continue
                M = hit[0][2]
                im = A.imshow(M, vmin=0, vmax=1, cmap="Blues")
                for c in cuts:
                    A.axvline(c, color="k", lw=0.6, ls="--")
                    A.axhline(c, color="k", lw=0.6, ls="--")
                A.set_title(f"{MLAB.get(m, m)}, p={p}", fontsize=9)
                A.set_xticks(range(M.shape[1]))
                A.set_yticks(range(M.shape[0]))
                A.tick_params(labelsize=7)
                if i == len(ms) - 1:
                    A.set_xlabel("predicted class", fontsize=8)
                if j == 0:
                    A.set_ylabel("true class", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="row-normalised rate")
        fig.suptitle(f"{NAMES[d]}: mean confusion matrix after task 4 (5 seeds)", fontsize=10)
        out = os.path.join(a.outdir, f"fig_confusion_{d}.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
