"""
compute_normalize_stats.py

Per-channel (RGB) mean and standard deviation of a MedMNIST training split, in the
format LifeLonger uses (after extend_channel=3, before any normalization). The
result goes into the 'normalize' field of dataset_config.py.

Usage:
    python compute_normalize_stats.py --dataset organamnist
"""

import argparse
import numpy as np
import torch
from torchvision import transforms
import medmnist
from medmnist import INFO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                         help="MedMNIST dataset name, e.g. organamnist")
    parser.add_argument("--download-root", default="../data",
                         help="folder where the dataset is or will be downloaded")
    args = parser.parse_args()

    info = INFO[args.dataset]
    DataClass = getattr(medmnist, info["python_class"])

    # No normalization: tensor in [0, 1], expanded to 3 channels as LifeLonger does before 'normalize'.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    ])

    train_dataset = DataClass(
        split="train", transform=transform,
        download=True, root=args.download_root, as_rgb=True
    )

    print(f"Dataset: {args.dataset}  |  training samples: {len(train_dataset)}")

    # Running sums per channel, so the split is never held in memory at once.
    channel_sum = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)
    n_pixels = 0

    loader = torch.utils.data.DataLoader(train_dataset, batch_size=256,
                                          shuffle=False, num_workers=2)

    for images, _ in loader:
        # images: (batch, 3, H, W)
        b, c, h, w = images.shape
        channel_sum += images.sum(dim=[0, 2, 3])
        channel_sum_sq += (images ** 2).sum(dim=[0, 2, 3])
        n_pixels += b * h * w

    mean = channel_sum / n_pixels
    std = torch.sqrt(channel_sum_sq / n_pixels - mean ** 2)

    print("\nResult -- copy into dataset_config.py:")
    mean_tuple = tuple(round(m.item(), 4) for m in mean)
    std_tuple = tuple(round(s.item(), 4) for s in std)
    print(f"'normalize': ({mean_tuple}, {std_tuple})")


if __name__ == "__main__":
    main()
