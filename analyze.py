"""
Step 3: aggregate the results and create all tables / figures for the report.

    python analyze.py

Outputs (in results/ and figures/)
  results/results_table.md / .csv     main table: params, val/test accuracy and macro-F1 (mean ± std over seeds)
  results/per_class_f1.csv            test F1 per class (mean over seeds)
  results/failure_cases.csv           misclassified test samples of the best model (fill in your hypotheses!)
  figures/learning_curves.png         train/val loss per epoch (first seed of every model)
  figures/results_bar.png             test accuracy and macro-F1 with error bars
  figures/confusion_matrix.png        confusion matrix of the best model on the test set
  figures/failure_cases.png           >= 3 failure cases with prediction and ground truth

The "best model" is chosen by mean VALIDATION macro-F1 (never by test results).
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import data
import models
from utils import mean_std, set_seed

set_seed(config.GLOBAL_SEED)


def load_runs(results_dir):
    """{model name: [run dicts sorted by seed]} for every model that has results."""
    runs = {}
    for name in config.ALL_MODELS:
        files = sorted(glob.glob(os.path.join(results_dir, name, "seed*.json")))
        if files:
            runs[name] = sorted((json.load(open(f)) for f in files), key=lambda r: r["seed"])
    return runs


def build_table(runs):
    rows = []
    for name, rs in runs.items():
        g = lambda split, key: [r[split][key] for r in rs]
        rows.append({
            "Model": config.DISPLAY_NAMES.get(name, name),
            "Params": f"{rs[0]['n_params']:,}",
            "Seeds": len(rs),
            "Val Acc (%)": mean_std(g("val", "accuracy"), 100),
            "Val Macro-F1 (%)": mean_std(g("val", "macro_f1"), 100),
            "Test Acc (%)": mean_std(g("test", "accuracy"), 100),
            "Test Macro-F1 (%)": mean_std(g("test", "macro_f1"), 100),
            "Train time (s)": mean_std([r["train_time_sec"] for r in rs], 1, 0),
        })
    return pd.DataFrame(rows)


def plot_learning_curves(runs, fig_dir):
    names = [n for n in config.MAIN_MODELS if n in runs]
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.2), squeeze=False)
    for ax, name in zip(axes[0], names):
        run = runs[name][0]                                    # one representative seed per model
        ep = [h["epoch"] for h in run["history"]]
        ax.plot(ep, [h["train_loss"] for h in run["history"]], label="train")
        ax.plot(ep, [h["val_loss"] for h in run["history"]], label="validation")
        ax.set_title(f"{config.DISPLAY_NAMES[name]}\n(seed {run['seed']})", fontsize=9)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "learning_curves.png"), dpi=200)
    plt.close(fig)


def plot_results_bar(runs, fig_dir):
    names = [n for n in config.MAIN_MODELS if n in runs]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for k, (key, label) in enumerate([("accuracy", "Accuracy"), ("macro_f1", "Macro-F1")]):
        vals = [[r["test"][key] * 100 for r in runs[n]] for n in names]
        ax.bar(x + (k - 0.5) * 0.35, [np.mean(v) for v in vals], 0.35,
               yerr=[np.std(v, ddof=1) if len(v) > 1 else 0 for v in vals], capsize=4, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([config.DISPLAY_NAMES[n].replace(" (", "\n(") for n in names], fontsize=8)
    ax.set_ylabel("test score (%)")
    lo = min(np.mean([r["test"]["macro_f1"] * 100 for r in runs[n]]) for n in names)
    ax.set_ylim(max(0, lo - 10), 100)
    ax.set_title("Test results (mean ± std over seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "results_bar.png"), dpi=200)
    plt.close(fig)


def plot_confusion(y_true, y_pred, title, fig_dir):
    names = data.class_names()
    k = len(names)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, fraction=0.046, label="row-normalised")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "confusion_matrix.png"), dpi=200)
    plt.close(fig)
    return cm


def failure_cases(y_true, y_pred, probs, n_show=6):
    """The n most confident mistakes (highest probability given to the wrong class)."""
    names = data.class_names()
    wrong = np.where(y_true != y_pred)[0]
    order = wrong[np.argsort(-probs[wrong, y_pred[wrong]])][:n_show]
    test_imgs = data.load_split_data(verify=False)["test"][0]
    fig, axes = plt.subplots(2, 3, figsize=(7, 5.2))
    rows = []
    for ax, i in zip(axes.ravel(), order):
        ax.imshow(test_imgs[i], interpolation="nearest")
        ax.set_title(f"test #{i}\ntrue: {names[y_true[i]]}\npred: {names[y_pred[i]]} "
                     f"(p={probs[i, y_pred[i]]:.2f})", fontsize=8)
        ax.axis("off")
        rows.append({"test_index": int(i), "true": names[y_true[i]], "predicted": names[y_pred[i]],
                     "confidence": float(probs[i, y_pred[i]]), "your_hypothesis": ""})
    for ax in axes.ravel()[len(order):]:
        ax.axis("off")
    fig.tight_layout()
    return fig, pd.DataFrame(rows)


def main(quick=False):
    results_dir, _, fig_dir = config.get_dirs(quick)
    os.makedirs(fig_dir, exist_ok=True)
    runs = load_runs(results_dir)
    if not runs:
        raise SystemExit(f"No results found in {results_dir}. Run run_all.py first.")

    # ---- main results table -------------------------------------------------------------------
    table = build_table(runs)
    table.to_csv(os.path.join(results_dir, "results_table.csv"), index=False)
    try:
        md = table.to_markdown(index=False)          # needs the optional `tabulate` package
    except ImportError:
        md = table.to_string(index=False)
    with open(os.path.join(results_dir, "results_table.md"), "w") as f:
        f.write(md + "\n\nValues are mean ± sample std (ddof=1) over seeds.\n")
    print(md)

    # ---- per-class F1 (mean over seeds) ---------------------------------------------------------
    names = data.class_names()
    pc = pd.DataFrame({config.DISPLAY_NAMES.get(n, n): np.mean([r["test"]["per_class_f1"] for r in rs], axis=0)
                       for n, rs in runs.items()}, index=names)
    pc.round(4).to_csv(os.path.join(results_dir, "per_class_f1.csv"))
    print("\nTest F1 per class (mean over seeds):\n", pc.round(3).to_string())

    # ---- output shape of every baseline layer (required in the Method section of the report) ----
    shapes = models.layer_shapes(models.build_model(config.MODELS["baseline"], config.NUM_CLASSES,
                                                    config.IMG_SIZE),
                                 (1, 3, config.IMG_SIZE, config.IMG_SIZE))
    with open(os.path.join(results_dir, "baseline_layer_shapes.txt"), "w") as f:
        for lname, shp in shapes:
            f.write(f"{lname:35s} -> {list(shp)}\n")

    # ---- figures -------------------------------------------------------------------------------
    plot_learning_curves(runs, fig_dir)
    plot_results_bar(runs, fig_dir)

    # best MAIN model by mean VALIDATION macro-F1, representative run = its best-val seed
    main_names = [n for n in config.MAIN_MODELS if n in runs]
    best_name = max(main_names, key=lambda n: np.mean([r["val"]["macro_f1"] for r in runs[n]]))
    best_run = max(runs[best_name], key=lambda r: r["val"]["macro_f1"])
    print(f"\nBest model by validation macro-F1: {best_name} (representative seed {best_run['seed']})")

    preds = np.load(os.path.join(results_dir, best_name, f"seed{best_run['seed']}_test_preds.npz"))
    y_true, y_pred, probs = preds["y_true"], preds["y_pred"], preds["probs"]
    cm = plot_confusion(y_true, y_pred,
                        f"{config.DISPLAY_NAMES[best_name]} - test set (seed {best_run['seed']})", fig_dir)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    top = np.dstack(np.unravel_index(np.argsort(-off.ravel())[:3], off.shape))[0]
    print("Most frequent confusions (true -> predicted):")
    for t, p in top:
        if off[t, p] > 0:
            print(f"  {names[t]} -> {names[p]}: {off[t, p]} samples")

    fig, fail_df = failure_cases(y_true, y_pred, probs)
    fig.savefig(os.path.join(fig_dir, "failure_cases.png"), dpi=200)
    plt.close(fig)
    fail_df.to_csv(os.path.join(results_dir, "failure_cases.csv"), index=False)
    print(f"\nSaved tables to {results_dir} and figures to {fig_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="analyse the smoke-test results")
    main(parser.parse_args().quick)
