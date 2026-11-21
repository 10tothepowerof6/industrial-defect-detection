"""Fit good-only PatchCore without Lightning callbacks or test-set access."""

import csv
import importlib.metadata
import json
import math
import platform
import random
import re
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
import yaml

from industrial_defect_detection.data.manifests import load_manifest_sample, read_manifest
from industrial_defect_detection.runs import git_state as provenance
from industrial_defect_detection.runs import resolve, write_json

from .patchcore import (
    MEAN,
    STD,
    WEIGHTS_URL,
    load_pretrained,
    make_model,
    preprocess,
    select_device,
    sha256,
)


def validate_experiment(config: dict, rows: list[dict]) -> list[dict]:
    exp, model, data = config["experiment"], config["model"], config["data"]
    expected = (exp["category"], exp["protocol"])
    if exp["model_id"] != "patchcore":
        raise ValueError("Expected patchcore model_id")
    if any((r["category"], r["protocol"]) != expected for r in rows):
        raise ValueError("Config category/protocol does not match manifest")
    held = rows[0]["held_out_defect"]
    expected_fold = f"holdout-{held}" if held else "baseline"
    if exp["fold"] != expected_fold:
        raise ValueError(f"Config fold must be {expected_fold}, got {exp['fold']}")
    if min(data["input_height"], data["input_width"]) < 32:
        raise ValueError("Input dimensions must be at least 32")
    if not 0 < model["coreset_sampling_ratio"] <= 1:
        raise ValueError("coreset_sampling_ratio must be in (0,1]")
    if model["num_neighbors"] < 1 or config["training"]["batch_size"] < 1:
        raise ValueError("num_neighbors and batch_size must be positive")
    normal = [r for r in rows if r["split"] == "train" and not r["is_anomalous"]]
    if not normal:
        raise ValueError("No good training images")
    return normal


def fit(config_path: Path, run_id: str | None = None, smoke_test: bool = False) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    exp, data, settings, training = (
        config["experiment"],
        config["data"],
        config["model"],
        config["training"],
    )
    root, manifest = resolve(data["dataset_root"]), resolve(data["manifest"])
    rows = read_manifest(manifest, root)
    normal = validate_experiment(config, rows)
    sidecar = json.loads(manifest.with_suffix(".json").read_text(encoding="utf-8"))
    if sha256(manifest) != sidecar["manifest_sha256"]:
        raise ValueError("Manifest checksum mismatch")
    if sidecar["seed"] != exp["seed"]:
        raise ValueError("Config seed does not match split metadata")
    if sidecar["review_status"] != "approved_by_hung":
        raise ValueError("Manifest must be reviewed before fitting")
    if smoke_test:
        normal = normal[:4]
    device = select_device(training["device"])
    seed = exp["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(training.get("cpu_threads", 4))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cudnn.benchmark = False
    run_id = run_id or (
        f"{datetime.now(ZoneInfo('Asia/Bangkok')):%Y%m%d-%H%M%S}-seed{seed}-"
        f"{'smoke-' if smoke_test else ''}{exp['fold']}"
    )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id):
        raise ValueError("run_id must be a simple directory name")
    parts = (exp["protocol"], exp["model_id"], exp["category"], run_id)
    artifact = resolve(config["paths"]["artifact_root"]).joinpath(*parts)
    output = resolve(config["paths"]["output_root"]).joinpath(*parts)
    if artifact.exists() or output.exists():
        raise FileExistsError("Run already exists; choose a new run_id")
    model = make_model(settings)
    weights_hash = load_pretrained(model, resolve(settings["weights_path"]))
    artifact.mkdir(parents=True)
    output.mkdir(parents=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    preprocessing = {
        "color": "RGB",
        "resize": {
            "height": data["input_height"],
            "width": data["input_width"],
            "interpolation": "bilinear",
        },
        "normalize_mean": MEAN,
        "normalize_std": STD,
    }
    model.to(device).train()
    # Frozen features must use ImageNet BatchNorm running statistics while collecting embeddings.
    model.feature_extractor.eval()
    started = time.perf_counter()
    batch_size = training["batch_size"]
    with torch.inference_mode():
        for offset in range(0, len(normal), batch_size):
            batch = normal[offset : offset + batch_size]
            images = [load_manifest_sample(root, row)[0] for row in batch]
            tensors = torch.stack([preprocess(image, preprocessing) for image in images]).to(device)
            model(tensors)
            print(
                f"Features {min(offset + batch_size, len(normal))}/{len(normal)} good images",
                flush=True,
            )
        points = sum(t.shape[0] for t in model.embedding_store)
        bank_count = math.floor(points * settings["coreset_sampling_ratio"])
        if bank_count < settings["num_neighbors"]:
            raise ValueError("Coreset too small for num_neighbors; increase training data or ratio")
        print(f"Coreset: {points} patches -> {bank_count} representatives", flush=True)
        model.subsample_embedding(settings["coreset_sampling_ratio"])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    checkpoint = artifact / "model.ckpt"
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, checkpoint)
    packages = {
        name: importlib.metadata.version(name)
        for name in ("torch", "torchvision", "anomalib", "timm", "numpy", "pillow")
    }
    metadata = {
        "schema_version": 1,
        "status": "uncalibrated",
        "model_id": "patchcore",
        "category": exp["category"],
        "protocol": exp["protocol"],
        "fold": exp["fold"],
        "seed": seed,
        "run_id": run_id,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256(checkpoint),
        "model": settings,
        "preprocessing": preprocessing,
        "score": {
            "image_score": "native Anomalib PatchCore score with neighbor reweighting",
            "map": "native anomaly distances, bilinearly resized to original image size",
        },
        "training_manifest": {
            "path": data["manifest"],
            "sha256": sha256(manifest),
            "train_images": len(normal),
            "val_images": sum(r["split"] == "val" and not r["is_anomalous"] for r in rows),
            "train_sample_ids": [r["sample_id"] for r in normal],
            "val_sample_ids": [r["sample_id"] for r in rows if r["split"] == "val"],
        },
        "config": config,
        "git": provenance(),
        "smoke_test": smoke_test,
        "pretrained": {"url": WEIGHTS_URL, "sha256": weights_hash},
        "training_seconds": seconds,
        "memory_bank_shape": list(model.memory_bank.shape),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": packages,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
    }
    write_json(artifact / "metadata.json", metadata)
    (artifact / "manifest.csv").write_bytes(manifest.read_bytes())
    write_json(output / "fit-summary.json", metadata)
    (output / "environment.txt").write_text(
        json.dumps(metadata["environment"], indent=2) + "\n", encoding="utf-8"
    )
    with (output / "fit-samples.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "split", "is_anomalous"])
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in normal)
    print(f"Artifact: {artifact}\nOutputs: {output}\nFit seconds: {seconds:.2f}", flush=True)
    return artifact
