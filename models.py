"""Model definitions: hand-written baseline CNN and ResNet-18 variants (torchvision)."""
import torch
import torch.nn as nn
import torchvision


class SmallCNN(nn.Module):
    """
    Baseline: three conv blocks + two fully connected layers, written by hand, trained from scratch.
    For a 64x64 RGB input the layer output shapes are
      Conv(3->16)  [16,64,64] -> MaxPool [16,32,32]
      Conv(16->32) [32,32,32] -> MaxPool [32,16,16]
      Conv(32->64) [64,16,16] -> MaxPool [64, 8, 8]
      Flatten [4096] -> Linear [128] -> Linear [num_classes]
    """

    def __init__(self, num_classes=8, img_size=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        feat = 64 * (img_size // 8) * (img_size // 8)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(feat, 128), nn.ReLU(), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(cfg: dict, num_classes=8, img_size=64) -> nn.Module:
    """Create the network described by a config entry of config.MODELS."""
    if cfg["arch"] == "small_cnn":
        return SmallCNN(num_classes, img_size)

    if cfg["arch"] == "resnet18":
        # ImageNet weights come from torchvision (public source, must be cited in the report).
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if cfg["pretrained"] else None
        net = torchvision.models.resnet18(weights=weights)
        if cfg["remove_maxpool"]:
            # Images are only 64x64: dropping the stem max-pool keeps larger feature maps.
            # (It has no weights, so pretrained parameters stay valid.)
            net.maxpool = nn.Identity()
        net.fc = nn.Sequential(nn.Dropout(cfg["dropout"]), nn.Linear(net.fc.in_features, num_classes))
        return net

    raise ValueError(f"Unknown architecture: {cfg['arch']}")


def count_params(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_optimizer(model: nn.Module, cfg: dict):
    """Adam / AdamW, optionally with a smaller lr for the pretrained backbone (discriminative lrs)."""
    if cfg["backbone_lr"] is not None and hasattr(model, "fc"):
        head_params = list(model.fc.parameters())
        head_ids = {id(p) for p in head_params}
        backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
        groups = [{"params": backbone_params, "lr": cfg["backbone_lr"]},
                  {"params": head_params, "lr": cfg["lr"]}]
    else:
        groups = [{"params": model.parameters(), "lr": cfg["lr"]}]

    if cfg["optimizer"] == "adam":
        return torch.optim.Adam(groups, weight_decay=cfg["weight_decay"])
    if cfg["optimizer"] == "adamw":
        return torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    raise ValueError(f"Unknown optimizer: {cfg['optimizer']}")


@torch.no_grad()
def layer_shapes(model: nn.Module, input_shape=(1, 3, 64, 64)):
    """
    Output shape of every leaf layer, from input to output (required in the report for the baseline).
    Returns a list of (layer name, output shape).
    """
    rows, hooks = [], []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(
                lambda m, i, o, n=name: rows.append((f"{n} ({m.__class__.__name__})", tuple(o.shape[1:])))))
    was_training = model.training
    model.eval()
    model(torch.zeros(input_shape))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return [("input", tuple(input_shape[1:]))] + rows
