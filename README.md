# Industrial Defect Detection

Visual inspection of **tile** and **metal nut** products from MVTec AD, with a
primary focus on learning normal appearance and detecting anomalous regions.

Dataset auditing, image/mask reading, and reproducible split manifests are
implemented. A good-only PatchCore fit/load/raw-prediction pipeline is available;
the Streamlit application remains a planned deliverable. See the
[PatchCore guide](docs/patchcore-guide.md) for commands, scope, and H02/H03 integration.

Shared raw/calibrated prediction objects and a synthetic integration example are
available. See the [day 19 handoff](docs/day19-handoff.md) for audit results,
manifest counts, commands, and the review items for Hung.

## Scope

| Track | Required models | Optional extension | Owner |
| --- | --- | --- | --- |
| Anomaly detection (primary, about 70% of effort) | PatchCore, EfficientAD | Dinomaly | Triệu |
| Supervised binary segmentation | U-Net with ImageNet-pretrained ResNet-18 encoder | SegFormer-B0 | Hùng |

Train a separate model for each product category initially. Both tracks produce
image-level decisions and spatial defect predictions. Compare known and held-out
defect types without assuming which approach will win.

## Repository layout

```text
industrial-defect-detection/
|-- app/                         # Streamlit interface
|-- configs/                     # Versioned experiment configurations
|-- datasets/                    # Local dataset files (ignored)
|-- splits/                      # Versioned image-level split manifests
|-- notebooks/
|   |-- anomaly/                 # Colab/Kaggle anomaly experiments
|   `-- supervised/              # Colab/Kaggle supervised experiments
|-- src/industrial_defect_detection/
|   |-- data/                    # Dataset readers and transforms
|   |-- models/
|   |   |-- anomaly/             # Anomalib integration
|   |   `-- supervised/          # U-Net training integration
|   |-- evaluation/              # Shared metrics and threshold selection
|   |-- inference/               # Artifact loading and prediction
|   `-- visualization/           # Heatmaps, masks, and overlays
|-- scripts/                     # Future command-line entry points
|-- artifacts/                   # Local model artifacts (ignored)
|-- outputs/                     # Local logs and generated results (ignored)
|-- docs/
|   |-- development-contract.md  # Shared interfaces and experiment rules
|   `-- assignments/             # Work plans for Triệu and Hùng
|-- requirement.txt
`-- pyproject.toml
```

Reusable code belongs in `src/`. Notebooks and scripts will call that code.
Keep split manifests and selected final result tables/figures in Git. Store
generated runs in `outputs/` and copy reviewed publication assets into `docs/`.

## Environment setup

The local baseline is **CPython 3.11.9 (64-bit)**, installed at
`C:\Users\TN\AppData\Local\Programs\Python\Python311\python.exe`.
Create an isolated environment using this executable.

Run these commands from the repository root using the chosen Python executable.

```powershell
& 'C:\Users\TN\AppData\Local\Programs\Python\Python311\python.exe' -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

The shared GPU baseline is **PyTorch 2.10.0 + CUDA 12.8** with
**torchvision 0.25.0 + CUDA 12.8**, for the GTX 1650 and RTX 4050 used by the team.
`requirement.txt` pins the `+cu128` builds and adds the official PyTorch package
index. Both machines need a compatible NVIDIA driver. The CUDA runtime is
provided by the PyTorch installation; a separate CUDA Toolkit is only needed for
tasks such as compiling custom CUDA extensions.

Install project dependencies and the local package.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirement.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; x = torch.ones((32, 32), device='cuda'); y = x @ x; torch.cuda.synchronize(); assert torch.all(y == 32).item(); print('GPU computation PASS:', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), y.device)"
.\.venv\Scripts\python.exe -c "from anomalib.models import Patchcore, EfficientAd; import segmentation_models_pytorch; import industrial_defect_detection; print('Imports OK')"
.\.venv\Scripts\python.exe -m ruff check src
```

The GPU check must print `2.10.0+cu128`, CUDA `12.8`, the expected NVIDIA device,
and `cuda:0`. Perform this check on each machine before training.

On Linux/Colab/Kaggle, use the active environment's `python -m pip` commands
instead of the Windows executable path. Check Python, NVIDIA driver, and CUDA
wheel compatibility before installing. Restart the notebook kernel after changing
PyTorch.

The requirement file pins the core model packages and bounds supporting
dependencies. It is not a complete environment lock. An installation and model
smoke test are needed before claiming runtime compatibility.

Dependency references are [Anomalib 2.7.0](https://pypi.org/project/anomalib/2.7.0/),
[SMP 0.5.0](https://pypi.org/project/segmentation-models-pytorch/0.5.0/), and the
[official PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/).

## Data

The current local dataset root is `datasets/`, containing `tile/` and
`metal_nut/`. Preserve the original `train/`, `test/`, and `ground_truth/`
subdirectories. The scripts accept a different root through `--dataset-root`.
Dataset files, downloaded archives,
credentials, checkpoints, and generated runs are excluded from Git.

Use the [shared technical contract](docs/development-contract.md) before creating train,
validation, or test splits. Official MVTec AD training images are normal only.
Supervised experiments therefore require a documented custom split.

Audit the dataset and create manifests using the confirmed held-out defect types.
The commands refuse to overwrite an existing output directory.

```powershell
.\.venv\Scripts\python.exe scripts/check_environment.py
.\.venv\Scripts\python.exe scripts/audit_dataset.py --dataset-root datasets
.\.venv\Scripts\python.exe scripts/create_splits.py --dataset-root datasets --seed 42 --tile-held-out crack --metal-nut-held-out scratch
```

Audit output and previews are in `outputs/day19/audit/`. The four manifest CSVs
and their metadata are in `splits/seed42/`. They include official AD splits and
one held-out comparison per category. Metadata records the normal validation
split, rounding rule, dataset image fingerprint, manifest hash, and pending peer
review. Review the previews and splits with Hung before training.

`load_sample` returns RGB uint8 images and binary uint8 masks. An optional size
uses `(height, width)` and resizes masks with nearest-neighbor. No random
augmentation or normalization is applied by this shared reader.

Run the data checks from PowerShell without installing more packages.

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m ruff check src scripts tests
```

MVTec AD is provided by MVTec Software GmbH under CC BY-NC-SA 4.0. A Kaggle mirror
does not replace the original dataset attribution or terms. See the
[official dataset page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad).

## Implementation milestones

1. Audit images and masks, then freeze split manifests.
2. Complete PatchCore and U-Net baselines for both categories.
3. Add EfficientAD and a shared evaluation pipeline.
4. Run held-out defect experiments and the normal-training-data-size study.
5. Integrate image inference into Streamlit and publish reproducible results.
6. Evaluate Dinomaly first if additional time is available.

Triệu owns anomaly experiments, shared data, and inference integration. Hùng owns
supervised training, shared evaluation, and the demo. Both review splits, failure
cases, and final results. See the detailed work plans for
[Triệu](docs/assignments/trieu.md) and [Hùng](docs/assignments/hung.md), including
branch names, commit titles, PR titles, and handoff checklists.

See the [workflow diagram](docs/workflow.md) for task dependencies, parallel work,
and integration checkpoints.
