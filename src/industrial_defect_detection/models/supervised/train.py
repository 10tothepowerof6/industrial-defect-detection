"""Train U-Net from a manifest and save a self-contained artifact."""

import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from industrial_defect_detection.data.manifests import read_manifest
from industrial_defect_detection.evaluation.metrics import dice_iou
from industrial_defect_detection.runs import (
    environment,
    file_sha256,
    git_state,
    make_run_id,
    resolve,
    run_dirs,
    write_json,
)

from .data import IMAGENET_MEAN, IMAGENET_STD, ManifestSegmentationDataset
from .model import BCEDiceLoss, build_unet

MODEL_ID = "unet_resnet18"
CHECKPOINT = "model.ckpt"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_rows(rows: list[dict], config: dict) -> dict[str, list[dict]]:
    """Train on every train row; validate on val rows. Test rows are never loaded here."""
    exp = config["experiment"]
    for row in rows:
        if row["category"] != exp["category"] or row["protocol"] != exp["protocol"]:
            raise ValueError(
                f"Manifest row {row['sample_id']} is {row['category']}/{row['protocol']}, "
                f"config expects {exp['category']}/{exp['protocol']}"
            )
        held_out = row["held_out_defect"]
        if held_out and row["defect_type"] == held_out and row["split"] != "test":
            raise ValueError(f"Held-out defect appears outside test: {row['sample_id']}")
    splits = {name: [r for r in rows if r["split"] == name] for name in ("train", "val")}
    if not any(r["is_anomalous"] for r in splits["val"]):
        raise ValueError("Validation has no known-defect images to choose a checkpoint")
    return splits


@torch.no_grad()
def validate(model, loader, loss_fn, device) -> dict:
    model.eval()
    losses, defect_dice, good_alarms, good_count = [], [], 0, 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        losses.append(loss_fn(logits, masks).item() * len(images))
        predicted = (torch.sigmoid(logits) > 0.5).cpu().numpy()[:, 0]
        targets = masks.cpu().numpy()[:, 0].astype(bool)
        for pred, target, anomalous in zip(
            predicted, targets, batch["is_anomalous"].tolist(), strict=True
        ):
            if anomalous:
                defect_dice.append(dice_iou(pred, target)[0])
            else:
                good_count += 1
                good_alarms += int(pred.any())
    return {
        "val_loss": sum(losses) / len(loader.dataset),
        "val_dice_defect": float(np.mean(defect_dice)),
        # Share of good val images with any pixel above 0.5: an early false-alarm signal.
        "val_good_any_pixel_rate": good_alarms / good_count if good_count else None,
    }


def train(config: dict, run_id: str | None = None) -> Path:
    """Train, keep the best validation checkpoint, and return its artifact directory."""
    exp, data_cfg, model_cfg, train_cfg = (
        config["experiment"],
        config["data"],
        config["model"],
        config["training"],
    )
    if exp["model_id"] != MODEL_ID:
        raise ValueError(f"This trainer only supports {MODEL_ID}, config has {exp['model_id']}")
    seed_everything(exp["seed"])
    dataset_root = resolve(data_cfg["dataset_root"])
    manifest_path = resolve(data_cfg["manifest"])
    run_id = run_id or make_run_id(exp["seed"], exp["fold"])
    artifact_dir, output_dir = run_dirs(config, run_id)
    for directory in (artifact_dir, output_dir):
        if directory.exists():
            raise FileExistsError(f"Run directory already exists, use a new run ID: {directory}")

    splits = split_rows(read_manifest(manifest_path, dataset_root), config)
    size = (data_cfg["input_height"], data_cfg["input_width"])
    train_set = ManifestSegmentationDataset(
        splits["train"], dataset_root, size, train_cfg.get("augmentation"), exp["seed"]
    )
    val_set = ManifestSegmentationDataset(splits["val"], dataset_root, size)
    generator = torch.Generator().manual_seed(exp["seed"])
    train_loader = DataLoader(
        train_set,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 0),
        generator=generator,
        drop_last=len(train_set) > train_cfg["batch_size"],
    )
    val_loader = DataLoader(val_set, batch_size=train_cfg["batch_size"], shuffle=False)

    device = torch.device(train_cfg["device"] if torch.cuda.is_available() else "cpu")
    if train_cfg["device"].startswith("cuda") and device.type != "cuda":
        print("WARNING: CUDA unavailable, training on CPU")
    model = build_unet(model_cfg["encoder"], model_cfg["encoder_weights"]).to(device)
    loss_fn = BCEDiceLoss(train_cfg["bce_weight"], train_cfg["dice_weight"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])
    use_amp = bool(train_cfg.get("mixed_precision")) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    max_batches = train_cfg.get("max_batches_per_epoch")

    output_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)
    env = environment()
    (output_dir / "environment.txt").write_text(
        yaml.safe_dump(env, sort_keys=False), encoding="utf-8"
    )

    history, best, best_epoch = [], -1.0, None
    started = time.perf_counter()
    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        total, seen = 0.0, 0
        for step, batch in enumerate(train_loader):
            if max_batches and step >= max_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=use_amp):
                logits = model(images)
            loss = loss_fn(logits, masks)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(images)
            seen += len(images)
        scheduler.step()
        metrics = {"epoch": epoch, "train_loss": total / seen, **validate(model, val_loader, loss_fn, device)}
        metrics["lr"] = scheduler.get_last_lr()[0]
        history.append(metrics)
        improved = metrics[train_cfg["checkpoint_metric"]] > best
        if improved:
            best, best_epoch = metrics[train_cfg["checkpoint_metric"]], epoch
            torch.save({"state_dict": model.state_dict(), "epoch": epoch}, artifact_dir / CHECKPOINT)
        print(
            f"epoch {epoch:3d} train_loss {metrics['train_loss']:.4f} "
            f"val_loss {metrics['val_loss']:.4f} val_dice_defect {metrics['val_dice_defect']:.4f} "
            f"good_any_pixel {metrics['val_good_any_pixel_rate']:.3f}" + (" *" if improved else ""),
            flush=True,
        )

    with (output_dir / "history.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    metadata = {
        "schema_version": 1,
        "status": "uncalibrated",
        "model_id": MODEL_ID,
        "category": exp["category"],
        "protocol": exp["protocol"],
        "fold": exp["fold"],
        "seed": exp["seed"],
        "run_id": run_id,
        "checkpoint": CHECKPOINT,
        "best_epoch": best_epoch,
        "best_checkpoint_metric": {train_cfg["checkpoint_metric"]: best},
        "model": {"encoder": model_cfg["encoder"], "pretrained": model_cfg["encoder_weights"]},
        "preprocessing": {
            "color": "RGB",
            "resize": {"height": size[0], "width": size[1], "interpolation": "bilinear"},
            "normalize_mean": IMAGENET_MEAN.tolist(),
            "normalize_std": IMAGENET_STD.tolist(),
        },
        "score": {
            "map": "sigmoid(logits), bilinearly resized to the original image size",
            "image_score": "max of the probability map at original image size",
        },
        "training_manifest": {
            "path": data_cfg["manifest"],
            "sha256": file_sha256(manifest_path),
            "train_images": len(train_set),
            "val_images": len(val_set),
        },
        "config": config,
        "environment": env,
        "git": git_state(),
        "training_seconds": round(time.perf_counter() - started, 1),
        "smoke_test": bool(max_batches),
    }
    write_json(artifact_dir / "metadata.json", metadata)
    print(f"Artifact: {artifact_dir}\nOutputs:  {output_dir}")
    return artifact_dir
