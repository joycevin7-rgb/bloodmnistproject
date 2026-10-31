"""
Central configuration for the BloodMNIST deep learning project.

Every hyperparameter of every model lives in this file, so that
(1) all models are trained under exactly the same budget, and
(2) the report can quote the values directly.
"""
import os

# ----------------------------------------------------------------------------
# Reproducibility: seeds
# ----------------------------------------------------------------------------
GLOBAL_SEED = 42              # used by data preparation / analysis scripts
SEEDS = [42, 43, 44]          # one training run per seed for every model (>= 3 required)

# ----------------------------------------------------------------------------
# Paths (all relative to this folder, so the project runs from anywhere)
# ----------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_DIR, "data")                       # dataset download (NOT submitted)
SPLIT_CSV = os.path.join(PROJECT_DIR, "splits", "bloodmnist_split.csv")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
CKPT_DIR = os.path.join(PROJECT_DIR, "checkpoints")                 # best checkpoints (NOT submitted)
FIG_DIR = os.path.join(PROJECT_DIR, "figures")

# "Quick" mode (smoke test) writes to separate folders so real results are never mixed up.
QUICK_RESULTS_DIR = os.path.join(PROJECT_DIR, "results_quick")
QUICK_CKPT_DIR = os.path.join(PROJECT_DIR, "checkpoints_quick")
QUICK_FIG_DIR = os.path.join(PROJECT_DIR, "figures_quick")

# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
DATASET_FLAG = "bloodmnist"
IMG_SIZE = 64                 # MedMNIST+ resolutions: 28, 64, 128, 224. 64 fits a 4 GB GPU comfortably.
NUM_CLASSES = 8
IMAGENET_MEAN = (0.485, 0.456, 0.406)   # the SAME normalisation is used for every model
IMAGENET_STD = (0.229, 0.224, 0.225)

# ----------------------------------------------------------------------------
# Shared training budget (identical for ALL models -> fair comparison)
# ----------------------------------------------------------------------------
EPOCHS = 20                   # fixed number of epochs for every model
BATCH_SIZE = 128
USE_AMP = True                # mixed precision on GPU (ignored on CPU)
# Model selection rule (identical for all models): keep the epoch with the best
# validation macro-F1 (ties broken by lower validation loss). Test set is NOT used.

# ----------------------------------------------------------------------------
# Models
# A "method" = the complete training setup (architecture, loss, optimiser, lr, augmentation ...)
# ----------------------------------------------------------------------------
MODELS = {
    # Baseline: tiny CNN written by hand, trained from scratch, plain setup.
    "baseline": dict(
        description="Small 3-block CNN from scratch, plain cross-entropy, no augmentation",
        arch="small_cnn",
        pretrained=False,
        remove_maxpool=False,      # not used by small_cnn
        dropout=0.0,
        augment=False,
        class_weighted_loss=False,
        optimizer="adam",
        lr=1e-3,                   # lr of the whole network
        backbone_lr=None,          # None -> same lr everywhere
        weight_decay=0.0,
    ),
    # Method A: ResNet-18 from scratch + augmentation.
    # Design choices: deeper residual architecture adapted to small images (stem max-pool removed),
    # dihedral flips + brightness/contrast augmentation, AdamW with weight decay.
    "method_a": dict(
        description="ResNet-18 from scratch (stem max-pool removed) + augmentation, AdamW",
        arch="resnet18",
        pretrained=False,
        remove_maxpool=True,
        dropout=0.0,
        augment=True,
        class_weighted_loss=False,
        optimizer="adamw",
        lr=1e-3,
        backbone_lr=None,
        weight_decay=1e-4,
    ),
    # Method B: same network as A, but ImageNet-pretrained, with our own design decisions:
    #   (1) class-weighted cross-entropy (inverse class frequency from the TRAIN split only),
    #   (2) new head = Dropout + Linear,
    #   (3) discriminative learning rates (small lr for pretrained backbone, larger lr for the new head),
    #   (4) same augmentation as Method A.
    "method_b": dict(
        description="ImageNet-pretrained ResNet-18 fine-tuned, class-weighted loss, dropout head, "
                    "discriminative learning rates, augmentation",
        arch="resnet18",
        pretrained=True,
        remove_maxpool=True,
        dropout=0.2,
        augment=True,
        class_weighted_loss=True,
        optimizer="adamw",
        lr=1e-3,                   # lr of the new head
        backbone_lr=1e-4,          # lr of the pretrained backbone
        weight_decay=1e-4,
    ),
}


def _variant(base, **changes):
    """Copy a model config and change some entries (used for the optional ablation study)."""
    cfg = dict(MODELS[base])
    cfg.update(changes)
    return cfg


# Optional ablation study: remove ONE component of Method B at a time.
ABLATIONS = {
    "abl_b_no_pretrained": _variant(
        "method_b", description="Method B without ImageNet weights (random init)",
        pretrained=False, backbone_lr=None),
    "abl_b_no_classweights": _variant(
        "method_b", description="Method B with plain cross-entropy", class_weighted_loss=False),
    "abl_b_no_augment": _variant(
        "method_b", description="Method B without data augmentation", augment=False),
}

ALL_MODELS = {**MODELS, **ABLATIONS}
MAIN_MODELS = list(MODELS.keys())

DISPLAY_NAMES = {
    "baseline": "Baseline (SmallCNN)",
    "method_a": "Method A (ResNet-18 scratch)",
    "method_b": "Method B (ResNet-18 pretrained)",
    "abl_b_no_pretrained": "B w/o pretrained",
    "abl_b_no_classweights": "B w/o class weights",
    "abl_b_no_augment": "B w/o augmentation",
}


def get_dirs(quick=False):
    """Return (results_dir, ckpt_dir, fig_dir) for normal or quick mode."""
    if quick:
        return QUICK_RESULTS_DIR, QUICK_CKPT_DIR, QUICK_FIG_DIR
    return RESULTS_DIR, CKPT_DIR, FIG_DIR
