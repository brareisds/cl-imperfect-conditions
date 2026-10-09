"""
audit_dataset_properties.py

Full metadata audit of the MedMNIST datasets used in this project
(organamnist, pathmnist, bloodmnist, tissuemnist), covering:
color/channels, binary vs.
multi-class, number of classes, number of samples, native resolution,
raw intensity range -- plus class imbalance (majority/minority class
ratio).

Reads directly from the already-downloaded .npz files (no network
access needed if they are already present).

Usage:
    python audit_dataset_properties.py --data-root /path/to/data \
        --output-csv dataset_audit.csv
"""

import argparse
import os
import numpy as np
import csv


DATASETS = ["organamnist", "pathmnist", "bloodmnist", "tissuemnist"]

# grayscale vs. RGB, as configured in dataset_config.py (extend_channel
# present = originally grayscale)
NATIVE_COLOR = {
    "organamnist": "grayscale",
    "pathmnist": "RGB",
    "bloodmnist": "RGB",
    "tissuemnist": "grayscale",
}


def load_npz(data_root, dataset_name):
    """Loads the official MedMNIST .npz (28x28 version) for a dataset,
    trying a couple of likely file layouts."""
    candidates = [
        os.path.join(data_root, f"{dataset_name}.npz"),
        os.path.join(data_root, dataset_name, f"{dataset_name}.npz"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return np.load(path)
    raise FileNotFoundError(
        f"Could not find {dataset_name}.npz in any of: {candidates}"
    )


def audit_one(data_root, dataset_name):
    npz = load_npz(data_root, dataset_name)

    train_imgs = npz["train_images"]
    train_labels = npz["train_labels"].flatten()

    n_train = len(train_imgs)
    n_classes = len(np.unique(train_labels))
    is_binary = n_classes == 2

    # raw resolution and channel count as stored (before extend_channel)
    if train_imgs.ndim == 3:
        # (N, H, W) -- grayscale
        h, w = train_imgs.shape[1:]
        channels_raw = 1
    else:
        # (N, H, W, C)
        h, w, channels_raw = train_imgs.shape[1:]

    intensity_min = int(train_imgs.min())
    intensity_max = int(train_imgs.max())

    # class imbalance: count per class, majority/minority ratio
    class_counts = np.bincount(train_labels.astype(int))
    class_counts = class_counts[class_counts > 0]  # drop unused labels
    majority = int(class_counts.max())
    minority = int(class_counts.min())
    imbalance_ratio = round(majority / minority, 2)

    return {
        "dataset": dataset_name,
        "native_color": NATIVE_COLOR[dataset_name],
        "channels_raw": channels_raw,
        "n_classes": n_classes,
        "is_binary": is_binary,
        "n_train_samples": n_train,
        "resolution": f"{h}x{w}",
        "intensity_min": intensity_min,
        "intensity_max": intensity_max,
        "majority_class_n": majority,
        "minority_class_n": minority,
        "imbalance_ratio": imbalance_ratio,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True,
                         help="Directory containing the .npz files "
                              "(e.g. data/)")
    parser.add_argument("--output-csv", default="dataset_audit.csv")
    args = parser.parse_args()

    rows = []
    for dataset_name in DATASETS:
        try:
            row = audit_one(args.data_root, dataset_name)
            rows.append(row)
            print(f"\n=== {dataset_name} ===")
            for k, v in row.items():
                if k != "dataset":
                    print(f"  {k}: {v}")
        except FileNotFoundError as e:
            print(f"\n[skip] {dataset_name}: {e}")

    if not rows:
        print("\nNo datasets found -- check --data-root.")
        return

    fieldnames = list(rows[0].keys())
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved to: {args.output_csv}")

    # quick preview table
    print("\nSummary table:")
    header = f"{'dataset':14s} {'color':10s} {'classes':8s} {'n_train':10s} {'imbalance':10s}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['dataset']:14s} {row['native_color']:10s} "
              f"{row['n_classes']:<8d} {row['n_train_samples']:<10d} "
              f"{row['imbalance_ratio']:<10.2f}")


if __name__ == "__main__":
    main()
