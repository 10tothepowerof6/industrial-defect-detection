# Experiment protocol

## Required models and categories

Evaluate PatchCore, EfficientAD, and U-Net with an ImageNet-pretrained ResNet-18
encoder on `tile` and `metal_nut`. Fit separate models per category. U-Net predicts
one binary defect mask. Dinomaly is the first optional model.

## Official-split anomaly benchmark

Preserve the official MVTec AD test set. Reserve normal training images for
validation, and record the exact split. Do not select checkpoints, thresholds,
normalization statistics, or hyperparameters using official test images or masks.
Audit Anomalib's datamodule and postprocessor settings to enforce this rule.

Use a normal-only validation false-positive target for image decisions and
document how pixel thresholds are calibrated. Report the resulting training
subset and calibration procedure alongside metrics. Differences from a paper's
validation or threshold protocol must be disclosed when presenting comparisons.

## Custom supervised comparison

Create one versioned split definition shared by all models. Separate original
images before applying augmentation, crops, or patches. Check duplicate content
across splits. Report counts per category, split, and defect type.

Use the same normal fitting images for all three models. Only supervised training
receives known defective images and masks. Reserve disjoint validation and test
images, including good images. Custom results must be labeled separately from
the official-split benchmark.

## Held-out defect experiment

For each category, repeat the experiment with each defect type held out in turn.
No images or masks of the held-out type may be used for fitting, validation,
checkpoint selection, calibration, or hyperparameter selection.

Split the other defect types into supervised training, validation, and test
subsets. All held-out-type images are test-only. Reuse fixed normal splits and
fixed known-type image assignments across folds where applicable. Select model
settings before examining held-out results. Model-specific thresholds may be
selected using the same permitted validation pool and objective. Explicitly
distinguish normal-only fitting from access to labeled calibration data.

Evaluate every model on the same test images within a fold. Report detection of
held-out defects independently of defect-name recognition. Normal-only anomaly
models may reuse their fitted artifact across folds when training images and
settings are identical. Recalibrate only using the permitted fold validation set.

## Evaluation and artifacts

- Image-level AUROC and average precision.
- Image precision, recall, F1, and normal-image false-positive rate at a threshold
  selected on validation and frozen before testing.
- Pixel-level AUROC, average precision, and Dice/IoU at a frozen pixel threshold.
  State aggregation and empty-mask handling rules before reporting results.
- Separate known-defect and held-out-defect recall, with category/type counts.
- Inference latency, model size, and RAM/VRAM on named hardware. Record input
  resolution, batch size, warm-up, and whether preprocessing is included.

Save package versions, configuration, random seeds, manifest identity, thresholds,
and raw predictions for every run. Where feasible, repeat stochastic training
with three seeds and report variation. Do not present repeat runs on one test set
as independent datasets.

For the primary-track data-size study, fix validation/test and evaluate nested
subsets of the available good training images, for example 25%, 50%, and 100%.
Report false positives, missed defects, and selected failure-case overlays.
