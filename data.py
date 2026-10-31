"""
Data utilities: loading BloodMNIST, reading the saved split CSV, GPU augmentation, normalisation.

BloodMNIST ships with an OFFICIAL train / val / test split, which we use unchanged.
`prepare_data.py` writes that split to a CSV (with an MD5 hash per image) BEFORE any training.
Every training run then reads the split back from the CSV and verifies the hashes, so the
split can never silently change.
"""
import hashlib
import os

import numpy as np
import pandas as pd
import torch

import medmnist
from medmnist import INFO

import config

SPLITS = ["train", "val", "test"]


def class_names():
    """List of the 8 class names, ordered by label index."""
    labels = INFO[config.DATASET_FLAG]["label"]
    # keep only the short name, e.g. "immature granulocytes(myelocytes, ...)" -> "immature granulocytes"
    return [labels[str(i)].split("(")[0].strip() for i in range(len(labels))]


def image_md5(img: np.ndarray) -> str:
    """MD5 hash of the raw pixel bytes of one image (used for the leakage check)."""
    return hashlib.md5(np.ascontiguousarray(img).tobytes()).hexdigest()


def load_raw_splits():
    """
    Load the official BloodMNIST splits with the `medmnist` package.
    Returns {split: (images uint8 [N,H,W,3], labels int64 [N])}.
    The data is downloaded to config.DATA_ROOT the first time.
    """
    os.makedirs(config.DATA_ROOT, exist_ok=True)
    cls = getattr(medmnist, INFO[config.DATASET_FLAG]["python_class"])
    suffix = "" if config.IMG_SIZE == 28 else f"_{config.IMG_SIZE}"
    npz_path = os.path.join(config.DATA_ROOT, f"{config.DATASET_FLAG}{suffix}.npz")
    need_download = not os.path.isfile(npz_path)

    out = {}
    for split in SPLITS:
        ds = cls(split=split, download=need_download, root=config.DATA_ROOT, size=config.IMG_SIZE)
        out[split] = (ds.imgs.copy(), ds.labels.reshape(-1).astype(np.int64))
    return out


def load_split_data(verify=True):
    """
    Return {split: (images, labels)} ordered exactly as in the saved split CSV.
    If verify=True, the MD5 hash and label of every image are checked against the CSV.
    """
    if not os.path.isfile(config.SPLIT_CSV):
        raise FileNotFoundError("Split CSV not found. Run `python prepare_data.py` first.")
    df = pd.read_csv(config.SPLIT_CSV)
    raw = load_raw_splits()
    out = {}
    for split in SPLITS:
        rows = df[df["split"] == split].sort_values("index")
        idx = rows["index"].to_numpy()
        imgs = raw[split][0][idx]
        labels = raw[split][1][idx]
        if verify:
            if not np.array_equal(labels, rows["label"].to_numpy()):
                raise RuntimeError(f"Labels of split '{split}' do not match the split CSV.")
            hashes = [image_md5(im) for im in imgs]
            if hashes != rows["md5"].tolist():
                raise RuntimeError(f"Images of split '{split}' do not match the split CSV.")
        out[split] = (imgs, labels)
    return out


def to_tensor_nchw(imgs: np.ndarray, device) -> torch.Tensor:
    """uint8 [N,H,W,C] numpy -> uint8 [N,C,H,W] tensor on `device` (whole split lives on the GPU)."""
    return torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous().to(device)


def normalize(x: torch.Tensor) -> torch.Tensor:
    """x: float in [0,1], shape [N,3,H,W]. Applies ImageNet mean/std (same for every model)."""
    mean = torch.tensor(config.IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(config.IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def augment(x: torch.Tensor) -> torch.Tensor:
    """
    Batch augmentation on the GPU, input/output float in [0,1], shape [N,3,H,W].
      * random horizontal flip, vertical flip and transpose -> all 8 symmetries of a square
        (blood cells have no preferred orientation),
      * random contrast (x0.8-1.2) and brightness (+-0.1) jitter (staining / illumination changes).
    """
    n = x.shape[0]
    dev = x.device
    m = lambda p: (torch.rand(n, device=dev) < p).view(n, 1, 1, 1)
    x = torch.where(m(0.5), x.flip(3), x)
    x = torch.where(m(0.5), x.flip(2), x)
    x = torch.where(m(0.5), x.transpose(2, 3), x)
    contrast = 0.8 + 0.4 * torch.rand(n, 1, 1, 1, device=dev)
    brightness = (torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 0.2
    return (x * contrast + brightness).clamp(0.0, 1.0)
