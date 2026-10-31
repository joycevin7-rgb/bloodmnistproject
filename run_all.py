"""
Run every model with every seed (baseline, method_a, method_b  x  3 seeds = 9 runs).
Finished runs are skipped, so the script can be stopped and restarted at any time.

    python run_all.py                    # the 9 main runs
    python run_all.py --ablation         # also the optional ablation study (9 extra runs)
    python run_all.py --quick            # smoke test of the whole pipeline (a few minutes at most)
"""
import argparse
import os

import config
from train import train_one
from utils import set_seed

set_seed(config.GLOBAL_SEED)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", action="store_true", help="also run the ablation variants of Method B")
    parser.add_argument("--quick", action="store_true", help="smoke test with tiny data and 2 epochs")
    args = parser.parse_args()

    names = list(config.MODELS.keys()) + (list(config.ABLATIONS.keys()) if args.ablation else [])
    results_dir, _, _ = config.get_dirs(args.quick)

    for name in names:
        for seed in config.SEEDS:
            if os.path.isfile(os.path.join(results_dir, name, f"seed{seed}.json")):
                print(f"skip {name} seed {seed} (already done)")
                continue
            train_one(name, seed, quick=args.quick)
    print("\nAll runs finished. Next: python analyze.py" + (" --quick" if args.quick else ""))
