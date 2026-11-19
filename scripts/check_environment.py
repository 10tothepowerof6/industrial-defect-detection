"""Verify installed libraries and run small CUDA kernels without downloading weights."""

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/day19/environment.json"))
    args = parser.parse_args()
    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "torchvision",
            "anomalib",
            "segmentation-models-pytorch",
            "numpy",
            "pillow",
        )
    }
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment")
    tensor = torch.ones((64, 64), device="cuda")
    result = tensor @ tensor
    torch.cuda.synchronize()
    if not torch.all(result == 64).item():
        raise RuntimeError("CUDA matrix multiplication produced an incorrect result")
    conv = torch.nn.Conv2d(3, 4, 3).to("cuda")
    with torch.inference_mode():
        output = conv(torch.ones((1, 3, 16, 16), device="cuda"))
    torch.cuda.synchronize()
    if not torch.isfinite(output).all().item():
        raise RuntimeError("CUDA convolution produced nonfinite values")
    from anomalib.models import EfficientAd, Patchcore
    from segmentation_models_pytorch import Unet

    report = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "compiled_architectures": torch.cuda.get_arch_list(),
        "cuda_matmul_passed": True,
        "cuda_convolution_passed": True,
        "imports": [cls.__name__ for cls in (Patchcore, EfficientAd, Unet)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
