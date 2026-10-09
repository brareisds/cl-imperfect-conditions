"""
plot_confusion_from_csv.py

Reads a confusion-matrix .csv already produced by
confusion_matrix_check.py and plots it as a row-normalized heatmap,
with axis labels from MedMNIST's own class names, ordered by that
seed's real class_order, and thick lines marking where one task's
classes end and the next one's begin.

Neither class_order nor taskcla is recomputed here -- reproducing
them standalone would require matching the exact sequence of RNG
calls inside get_loaders(), which is fragile and cannot be
self-verified. Both are read directly from the training log's own
printed line ([class-order check] ... class_order=[...]
taskcla=[...]), the same line already used and validated earlier in
this project. Pass that log with --log-file; the script fails loudly
if the line is not found, rather than silently guessing.

Usage (from the repository root, with medmnist installed):
    python analysis/plot_confusion_from_csv.py \
        --csv analysis/organamnist_ewc_seed0_confusion.csv \
        --log-file results/official_savemodel/organamnist_ewc/organamnist_ewc_fixd_0/stdout-*.txt \
        --seed 0 \
        --dataset organamnist \
        --title "OrganAMNIST, EWC, seed 0 (official config)" \
        --output analysis/cm_seed0.png
"""

import argparse
import re
import ast
import numpy as np
import matplotlib.pyplot as plt
from medmnist import INFO


SHORT_LABELS = {
    "immature granulocytes(myelocytes, metamyelocytes and promyelocytes)": "immature gran.",
}


def shorten(name, max_len=18):
    """Abbreviate very long class names so they don't squeeze the plot."""
    if name in SHORT_LABELS:
        return SHORT_LABELS[name]
    return name if len(name) <= max_len else name[:max_len - 1] + "\u2026"


def get_class_order_and_taskcla_from_log(log_path, seed):
    """Extracts the class_order and taskcla printed for a specific
    seed. The log may contain several seeds concatenated
    (Done!-separated blocks); this returns the pair from the block
    whose '[class-order check] seed=N ...' line matches the
    requested seed.
    """
    with open(log_path, "r", errors="ignore") as f:
        text = f.read()

    segments = text.split("Done!")
    order_re = re.compile(r"\[class-order check\] dataset=\S+ class_order=(\[[^\]]*\])")
    seed_re = re.compile(r"\[class-order check\] seed=(\d+) dataset=\S+ taskcla=(\[[^\]]*\])")

    for seg in segments:
        order_match = order_re.search(seg)
        seed_match = seed_re.search(seg)
        if seed_match and order_match and int(seed_match.group(1)) == seed:
            class_order = ast.literal_eval(order_match.group(1))
            taskcla = ast.literal_eval(seed_match.group(2))
            return class_order, taskcla

    raise ValueError(
        f"Could not find class_order/taskcla for seed={seed} in {log_path}."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--log-file", required=True,
                     help="training log containing the "
                          "'[class-order check]' line for this seed")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cm = np.loadtxt(args.csv, delimiter=",")
    n_classes = cm.shape[0]

    class_order, taskcla = get_class_order_and_taskcla_from_log(args.log_file, args.seed)
    assert len(class_order) == n_classes, (
        f"class_order has {len(class_order)} entries but the confusion "
        f"matrix has {n_classes} -- dataset/seed/log mismatch, stopping "
        f"rather than mislabeling the axes."
    )
    assert sum(nc for _, nc in taskcla) == n_classes, (
        f"taskcla classes ({sum(nc for _, nc in taskcla)}) don't sum to "
        f"n_classes ({n_classes}) -- stopping rather than drawing wrong "
        f"task boundaries."
    )

    info = INFO[args.dataset]
    label_names = info["label"]  # dict: "0"->name, "1"->name, ...
    labels = [shorten(label_names[str(c)]) for c in class_order]

    # task boundaries: cumulative class counts, e.g. taskcla=[(0,3),(1,3),
    # (2,3),(3,2)] -> boundaries at positions 3, 6, 9 (not at 0 or 11)
    cum = 0
    boundaries = []
    for _, nc in taskcla[:-1]:
        cum += nc
        boundaries.append(cum)

    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    size = max(4.5, 0.5 * n_classes + 1.5)
    fig, ax = plt.subplots(figsize=(size + 0.7, size))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(args.title or f"{args.dataset}, seed {args.seed}", fontsize=10)
    for i in range(n_classes):
        for j in range(n_classes):
            v = int(cm[i, j])
            if v > 0:
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, str(v), ha="center", va="center",
                         fontsize=6, color=color)

    # task-boundary lines, drawn at the edge between the last class of
    # one task and the first class of the next (position - 0.5 in
    # imshow's pixel coordinates)
    for b in boundaries:
        ax.axhline(b - 0.5, color="red", linewidth=1.3, alpha=0.8)
        ax.axvline(b - 0.5, color="red", linewidth=1.3, alpha=0.8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalized")
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"class_order used for labeling (from log, seed {args.seed}): {class_order}")
    print(f"taskcla (task boundaries drawn at classes {boundaries}): {taskcla}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
