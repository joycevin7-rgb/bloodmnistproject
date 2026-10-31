# Hypotheses (written BEFORE running the experiments)

Date written: ____ (fill in before you run `run_all.py`)

**H1 (Baseline vs. Method A).** Method A (ResNet-18 from scratch + augmentation) will achieve a higher
test macro-F1 than the Baseline (small CNN), because it is deeper, has residual connections and sees
augmented images, which should generalise better than a 3-layer CNN trained on raw images.

**H2 (Method A vs. Method B).** Method B (ImageNet-pretrained ResNet-18, class-weighted loss, dropout head,
discriminative learning rates) will achieve a higher test macro-F1 than Method A, because pretrained features
give a better starting point on ~12k images, and the class-weighted loss gives more weight to the rarer cell
types, which is what macro-F1 rewards.

**Why macro-F1 and accuracy?** BloodMNIST classes are not perfectly balanced, so accuracy alone can hide poor
performance on rare classes; macro-F1 averages the F1 of all 8 classes equally.

> Edit this text to your own reasoning if you wish, but do it BEFORE looking at any results.
