# Industrial Defect Detection

Visual inspection of **tile** and **metal nut** products from MVTec AD, with a
primary focus on learning normal appearance and detecting anomalous regions.

The project is currently at the repository setup stage. Training pipelines,
evaluation results, and the Streamlit application are planned deliverables.

## Scope

| Track | Required models | Optional extension | Owner |
| --- | --- | --- | --- |
| Anomaly detection (primary, about 70% of effort) | PatchCore, EfficientAD | Dinomaly | Member A |
| Supervised binary segmentation | U-Net with ImageNet-pretrained ResNet-18 encoder | SegFormer-B0 | Member B |

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
|   `-- experiment-protocol.md
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

Choose one PyTorch backend before installing the remaining dependencies.

```powershell
# NVIDIA GPU, including GTX 1650, with a compatible NVIDIA driver
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126

# CPU alternative
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
```

Then install project dependencies and the local package.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirement.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
.\.venv\Scripts\python.exe -c "from anomalib.models import Patchcore, EfficientAd; import segmentation_models_pytorch; import industrial_defect_detection; print('Imports OK')"
.\.venv\Scripts\python.exe -m ruff check src
```

On Linux/Colab/Kaggle, use the active environment's `python -m pip` commands
instead of the Windows executable path. Check its Python version and select a
compatible PyTorch backend. Restart the notebook kernel after changing PyTorch.

The requirement file pins the core model packages and bounds supporting
dependencies. It is not a complete environment lock. An installation and model
smoke test are needed before claiming runtime compatibility.

Dependency references are [Anomalib 2.7.0](https://pypi.org/project/anomalib/2.7.0/),
[SMP 0.5.0](https://pypi.org/project/segmentation-models-pytorch/0.5.0/), and the
[official PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/).

## Data

Place the extracted categories under `datasets/mvtec_ad/tile/` and
`datasets/mvtec_ad/metal_nut/`, preserving their original `train/`, `test/`, and
`ground_truth/` subdirectories. Dataset files, downloaded archives,
credentials, checkpoints, and generated runs are excluded from Git.

Use the [experiment protocol](docs/experiment-protocol.md) before creating train,
validation, or test splits. Official MVTec AD training images are normal only.
Supervised experiments therefore require a documented custom split.

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

Member A owns anomaly experiments and model selection. Member B owns supervised
training, shared evaluation integration, and the demo. Both members review splits,
failure cases, and final results.
