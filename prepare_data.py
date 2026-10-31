"""
Step 1: prepare the data BEFORE any training.

    python prepare_data.py

What it does
  1. Downloads BloodMNIST (official train/val/test split, used unchanged).
  2. Saves the split to splits/bloodmnist_split.csv (split, index, label, md5). It is never changed
     afterwards: if the file already exists it is only verified (use --force to rebuild it).
  3. Reports the class distribution of each split (results/data_distribution.csv + figure).
  4. Leakage checks: exact duplicates (MD5) and near-duplicates (thumbnail distance) across splits.
  5. Saves a grid of example images per class.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import config
import data
from utils import get_device, set_seed

set_seed(config.GLOBAL_SEED)

NEAR_DUP_THRESHOLD = 0.01   # mean absolute difference (pixels scaled to 0-1) of 16x16 grey thumbnails


def build_split_dataframe(raw):
    rows = []
    for split, (imgs, labels) in raw.items():
        for i in range(len(imgs)):
            rows.append((split, i, int(labels[i]), data.image_md5(imgs[i])))
    return pd.DataFrame(rows, columns=["split", "index", "label", "md5"])


def thumbnails(imgs, device):
    """uint8 [N,H,W,3] -> float [N,256] grey 16x16 thumbnails in [0,1]."""
    x = torch.from_numpy(imgs).float().div(255.0).permute(0, 3, 1, 2)
    grey = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
    small = F.adaptive_avg_pool2d(grey.unsqueeze(1), 16).flatten(1)
    return small.to(device)


def nearest_train_distance(query, train, chunk=32):
    """For each query thumbnail: mean absolute difference to its nearest TRAIN thumbnail."""
    mins = []
    for i in range(0, len(query), chunk):
        d = torch.cdist(query[i:i + chunk], train, p=1) / train.shape[1]
        mins.append(d.min(dim=1).values.cpu())
    return torch.cat(mins).numpy()


def main(force=False):
    os.makedirs(os.path.dirname(config.SPLIT_CSV), exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.FIG_DIR, exist_ok=True)
    names = data.class_names()

    raw = data.load_raw_splits()
    df = build_split_dataframe(raw)

    # ---- 1) save (or verify) the split file --------------------------------------------------
    if os.path.isfile(config.SPLIT_CSV) and not force:
        old = pd.read_csv(config.SPLIT_CSV)
        if not old.equals(df):
            raise RuntimeError("Existing split CSV differs from the dataset. "
                               "The split must not change after training started (use --force to rebuild).")
        print(f"Split file already exists and is consistent: {config.SPLIT_CSV}")
    else:
        df.to_csv(config.SPLIT_CSV, index=False)
        print(f"Saved split file: {config.SPLIT_CSV}")

    lines = []
    log = lambda s="": (print(s), lines.append(s))

    # ---- 2) basic statistics + class distribution -------------------------------------------
    log(f"Dataset: {config.DATASET_FLAG}, image size {config.IMG_SIZE}x{config.IMG_SIZE}x3, "
        f"{config.NUM_CLASSES} classes")
    total = len(df)
    for s in data.SPLITS:
        n = int((df.split == s).sum())
        log(f"  {s:5s}: {n:6d} images ({100 * n / total:.1f}%)")

    counts = pd.crosstab(df["label"], df["split"])[data.SPLITS]
    counts.index = [names[i] for i in counts.index]
    pct = counts.div(counts.sum(axis=0), axis=1) * 100
    table = counts.join(pct, rsuffix="_%")
    table.to_csv(os.path.join(config.RESULTS_DIR, "data_distribution.csv"))
    log("\nClass distribution (count and % of each split):")
    log(table.round(1).to_string())

    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.27
    for k, s in enumerate(data.SPLITS):
        ax.bar(np.arange(len(names)) + (k - 1) * width, pct[s], width, label=s)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("% of split")
    ax.set_title("BloodMNIST class distribution per split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(config.FIG_DIR, "data_distribution.png"), dpi=200)
    plt.close(fig)

    # ---- 3) leakage checks -------------------------------------------------------------------
    log("\n--- Leakage checks ---")
    groups = df.groupby("md5")["split"].agg(lambda s: sorted(set(s)))
    cross = groups[groups.apply(len) > 1]
    within = int((df.groupby("md5").size() > 1).sum())
    log(f"Exact duplicates (MD5): {len(cross)} hashes appear in more than one split; "
        f"{within} hashes appear more than once overall.")
    if len(cross):
        log(f"  split combinations: {cross.apply(tuple).value_counts().to_dict()}")

    device = get_device()
    thumbs = {s: thumbnails(raw[s][0], device) for s in data.SPLITS}
    for s in ["val", "test"]:
        d = nearest_train_distance(thumbs[s], thumbs["train"])
        n_near = int((d < NEAR_DUP_THRESHOLD).sum())
        log(f"Near-duplicates {s} vs train: {n_near} / {len(d)} images have a train neighbour with mean abs "
            f"difference < {NEAR_DUP_THRESHOLD} (min distance {d.min():.4f}, median {np.median(d):.4f})")
    log("Subject-level check: BloodMNIST provides NO patient/subject IDs, so overlap of subjects between "
        "splits cannot be verified. State this as a limitation in the report.")

    with open(os.path.join(config.RESULTS_DIR, "leakage_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---- 4) example images ---------------------------------------------------------------------
    imgs, labels = raw["train"]
    rng = np.random.RandomState(config.GLOBAL_SEED)
    fig, axes = plt.subplots(config.NUM_CLASSES, 6, figsize=(6, config.NUM_CLASSES))
    for c in range(config.NUM_CLASSES):
        pick = rng.choice(np.where(labels == c)[0], 6, replace=False)
        for j, p in enumerate(pick):
            axes[c, j].imshow(imgs[p])
            axes[c, j].axis("off")
        axes[c, 0].set_title(names[c], fontsize=7, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(config.FIG_DIR, "sample_images.png"), dpi=200)
    plt.close(fig)
    print("\nDone. Figures in figures/, reports in results/.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="rebuild the split CSV even if it exists")
    main(parser.parse_args().force)
