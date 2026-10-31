"""Small helpers shared by all scripts."""
import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fix the seeds of Python `random`, NumPy and PyTorch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN kernels (results can still differ slightly across GPUs / driver versions).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mean_std(values, scale=1.0, decimals=2) -> str:
    """Format a list of numbers as 'mean ± std' (sample std, ddof=1)."""
    arr = np.asarray(values, dtype=float) * scale
    std = arr.std(ddof=1) if len(arr) > 1 else 0.0
    return f"{arr.mean():.{decimals}f} ± {std:.{decimals}f}"
