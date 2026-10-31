"""
Train and evaluate ONE model with ONE seed.

    python train.py --model baseline --seed 42
    python train.py --model method_b --seed 43
    python train.py --model method_a --seed 42 --quick      # fast smoke test (tiny subset, 2 epochs)

For every run this script saves
    results/<model>/seed<seed>.json              params, train time, learning curves, val/test metrics
    results/<model>/seed<seed>_test_preds.npz    test labels, predictions, probabilities
    checkpoints/<model>_seed<seed>.pt            best checkpoint (selected on validation macro-F1)
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

import config
import data
import models
from utils import get_device, set_seed

# Seeds are fixed at the very top (train_one() sets the seed of each run again).
set_seed(config.GLOBAL_SEED)


@torch.no_grad()
def predict_logits(model, X_uint8, device, use_amp, batch_size=512):
    """Forward pass over a whole split (no augmentation), returns logits [N, num_classes]."""
    model.eval()
    outs = []
    for i in range(0, len(X_uint8), batch_size):
        xb = data.normalize(X_uint8[i:i + batch_size].float() / 255.0)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outs.append(model(xb).float())
    return torch.cat(outs)


def evaluate(logits, y, criterion):
    """Loss + the two reported metrics: accuracy and macro-F1."""
    y_np = y.cpu().numpy()
    pred = logits.argmax(1).cpu().numpy()
    return {
        "loss": float(criterion(logits, y).item()),
        "accuracy": float(accuracy_score(y_np, pred)),
        "macro_f1": float(f1_score(y_np, pred, average="macro", labels=list(range(config.NUM_CLASSES)))),
    }


def train_one(model_name, seed, quick=False, verbose=True):
    cfg = config.ALL_MODELS[model_name]
    results_dir, ckpt_dir, _ = config.get_dirs(quick)
    epochs = 2 if quick else config.EPOCHS
    device = get_device()
    use_amp = bool(config.USE_AMP and device.type == "cuda")

    set_seed(seed)

    # ---- data (read from the saved split CSV, hashes verified) --------------------------------
    splits = data.load_split_data(verify=True)
    X, Y = {}, {}
    for s in data.SPLITS:
        imgs, labels = splits[s]
        if quick and s == "train":                       # smoke test: tiny subset
            imgs, labels = imgs[:512], labels[:512]
        elif quick:
            imgs, labels = imgs[:256], labels[:256]
        X[s] = data.to_tensor_nchw(imgs, device)
        Y[s] = torch.from_numpy(labels).to(device)
    n_train = len(X["train"])

    # ---- model, loss, optimiser ------------------------------------------------------------------
    model = models.build_model(cfg, config.NUM_CLASSES, config.IMG_SIZE).to(device)
    n_params = models.count_params(model)

    if cfg["class_weighted_loss"]:
        # Inverse-frequency class weights computed from the TRAIN split only.
        counts = torch.bincount(Y["train"], minlength=config.NUM_CLASSES).float().clamp(min=1)
        weights = (counts.sum() / (config.NUM_CLASSES * counts)).to(device)
    else:
        weights = None
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = models.build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    if verbose:
        print(f"\n=== {model_name} | seed {seed} | {n_params:,} params | device={device} | "
              f"epochs={epochs} | train={n_train} ===")

    # ---- training loop ------------------------------------------------------------------------------
    gen = torch.Generator()
    gen.manual_seed(seed)                                # controls the shuffling order
    history, best_key, best_state, best_epoch = [], None, None, -1

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, generator=gen).to(device)
        loss_sum, seen = 0.0, 0
        for i in range(0, n_train, config.BATCH_SIZE):
            idx = perm[i:i + config.BATCH_SIZE]
            if len(idx) < 2:                             # BatchNorm needs >= 2 samples
                continue
            xb = X["train"][idx].float() / 255.0
            yb = Y["train"][idx]
            if cfg["augment"]:
                xb = data.augment(xb)
            xb = data.normalize(xb)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(xb)
            loss = criterion(out.float(), yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(idx)
            seen += len(idx)
        scheduler.step()

        val = evaluate(predict_logits(model, X["val"], device, use_amp), Y["val"], criterion)
        rec = {"epoch": epoch, "train_loss": loss_sum / max(seen, 1), "val_loss": val["loss"],
               "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"]}
        history.append(rec)

        # model selection on VALIDATION only: best macro-F1, ties -> lower val loss
        key = (val["macro_f1"], -val["loss"])
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch
            best_state = copy.deepcopy(model.state_dict())
        if verbose:
            print(f"epoch {epoch:2d}/{epochs} | train loss {rec['train_loss']:.4f} | "
                  f"val loss {val['loss']:.4f} | val acc {val['accuracy']:.4f} | val F1 {val['macro_f1']:.4f}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    train_time = time.time() - t0

    # ---- final evaluation with the selected checkpoint ---------------------------------------------
    # (The test set is only evaluated here, after all decisions - checkpoint choice - are made.)
    model.load_state_dict(best_state)
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(best_state, os.path.join(ckpt_dir, f"{model_name}_seed{seed}.pt"))

    val_logits = predict_logits(model, X["val"], device, use_amp)
    test_logits = predict_logits(model, X["test"], device, use_amp)
    val_res = evaluate(val_logits, Y["val"], criterion)
    test_res = evaluate(test_logits, Y["test"], criterion)

    y_true = Y["test"].cpu().numpy()
    y_pred = test_logits.argmax(1).cpu().numpy()
    probs = torch.softmax(test_logits, dim=1).cpu().numpy().astype(np.float32)
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=list(range(config.NUM_CLASSES)),
                            zero_division=0)
    test_res["per_class_f1"] = [float(v) for v in per_class_f1]

    out_dir = os.path.join(results_dir, model_name)
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"seed{seed}_test_preds.npz"),
                        y_true=y_true, y_pred=y_pred, probs=probs)
    record = {
        "model": model_name, "seed": seed, "config": cfg, "epochs": epochs,
        "batch_size": config.BATCH_SIZE, "img_size": config.IMG_SIZE, "n_params": n_params,
        "train_time_sec": train_time, "best_epoch": best_epoch,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "history": history, "val": val_res, "test": test_res, "quick": quick,
    }
    with open(os.path.join(out_dir, f"seed{seed}.json"), "w") as f:
        json.dump(record, f, indent=2)

    if verbose:
        print(f"-> best epoch {best_epoch} | VAL acc {val_res['accuracy']:.4f} F1 {val_res['macro_f1']:.4f} | "
              f"TEST acc {test_res['accuracy']:.4f} F1 {test_res['macro_f1']:.4f} | time {train_time:.0f}s")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(config.ALL_MODELS.keys()))
    parser.add_argument("--seed", type=int, default=config.SEEDS[0])
    parser.add_argument("--quick", action="store_true", help="smoke test: tiny subset, 2 epochs")
    args = parser.parse_args()
    train_one(args.model, args.seed, quick=args.quick)
