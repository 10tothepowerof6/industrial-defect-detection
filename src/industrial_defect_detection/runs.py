"""Run IDs, configs, and provenance shared by training, calibration, and evaluation."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGES = (
    "torch",
    "torchvision",
    "segmentation-models-pytorch",
    "anomalib",
    "numpy",
    "pillow",
    "scikit-learn",
)


def make_run_id(seed: int, fold: str, now: datetime | None = None) -> str:
    """<YYYYMMDD-HHMMSS>-seed<seed>-<fold>, using Asia/Bangkok time as the contract says."""
    moment = now or datetime.now(ZoneInfo("Asia/Bangkok"))
    return f"{moment:%Y%m%d-%H%M%S}-seed{seed}-{fold}"


def run_dirs(config: dict, run_id: str) -> tuple[Path, Path]:
    """artifacts/<protocol>/<model_id>/<category>/<run_id> and the matching outputs dir."""
    exp = config["experiment"]
    parts = (exp["protocol"], exp["model_id"], exp["category"], run_id)
    artifact_dir = resolve(config["paths"]["artifact_root"]).joinpath(*parts)
    output_dir = resolve(config["paths"]["output_root"]).joinpath(*parts)
    return artifact_dir, output_dir


def resolve(path: str | Path) -> Path:
    """Relative paths in configs are relative to the repository root."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    for section in ("experiment", "data", "model", "training", "calibration", "paths"):
        if section not in config:
            raise ValueError(f"Config {path} is missing the '{section}' section")
    return config


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_state() -> dict:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        )
        return result.stdout.strip()

    return {"commit": run("rev-parse", "HEAD") or None, "dirty": bool(run("status", "--porcelain"))}


def environment() -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": versions,
    }
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_runtime"] = torch.version.cuda
        info["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except ImportError:
        pass
    return info


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
