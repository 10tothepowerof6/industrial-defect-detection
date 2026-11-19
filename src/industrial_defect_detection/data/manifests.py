"""Reproducible manifests for official AD and selected held-out comparisons."""

import csv
import hashlib
import io
import json
import math
import random
from collections import Counter
from pathlib import Path

from .mvtec import MVTecSample, dataset_path, load_sample

FIELDS = (
    "sample_id",
    "image_path",
    "mask_path",
    "category",
    "defect_type",
    "is_anomalous",
    "split",
    "protocol",
    "held_out_defect",
)


def partition(samples: list[MVTecSample], seed: int, key: str, good: bool) -> dict[str, str]:
    ordered = sorted(samples, key=lambda s: s.sample_id)
    random.Random(f"{seed}:{key}").shuffle(ordered)
    train_count = math.floor(len(ordered) * (0.8 if good else 0.6))
    val_count = len(ordered) - train_count if good else math.floor(len(ordered) * 0.2)
    if not train_count or not val_count or (not good and train_count + val_count >= len(ordered)):
        raise ValueError(f"Too few images to split group: {key}")
    assignments = {}
    for i, sample in enumerate(ordered):
        if i < train_count:
            split = "train"
        elif i < train_count + val_count:
            split = "val"
        else:
            split = "test"
        assignments[sample.sample_id] = split
    return assignments


def build_manifest(
    samples: list[MVTecSample],
    category: str,
    protocol: str,
    seed: int,
    held_out_defect: str = "",
) -> list[dict]:
    if protocol not in {"official_ad", "custom_comparison"}:
        raise ValueError(f"Unsupported protocol: {protocol}")
    selected = [s for s in samples if s.category == category]
    if not selected:
        raise ValueError(f"No samples for category: {category}")
    defects = {s.defect_type for s in selected if s.is_anomalous}
    if protocol == "custom_comparison" and held_out_defect not in defects:
        raise ValueError(f"Choose an existing held-out defect for {category}: {sorted(defects)}")
    if protocol == "official_ad" and held_out_defect:
        raise ValueError("official_ad does not take a held-out defect")
    train_good = [s for s in selected if s.source_split == "train"]
    assignments = partition(train_good, seed, f"{category}/good", good=True)
    if protocol == "custom_comparison":
        for defect in sorted(defects - {held_out_defect}):
            assignments.update(
                partition(
                    [s for s in selected if s.defect_type == defect],
                    seed,
                    f"{category}/{defect}",
                    good=False,
                )
            )
    rows = []
    for sample in sorted(selected, key=lambda s: s.sample_id):
        rows.append(
            {
                "sample_id": sample.sample_id,
                "image_path": sample.image_path,
                "mask_path": sample.mask_path,
                "category": category,
                "defect_type": sample.defect_type,
                "is_anomalous": int(sample.is_anomalous),
                "split": assignments.get(sample.sample_id, "test"),
                "protocol": protocol,
                "held_out_defect": held_out_defect,
            }
        )
    validate_manifest(rows)
    return rows


def validate_manifest(rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Manifest is empty")
    if any(set(row) != set(FIELDS) for row in rows):
        raise ValueError("Manifest columns do not match the shared schema")
    identity = {(r["category"], r["protocol"], r["held_out_defect"]) for r in rows}
    if len(identity) != 1:
        raise ValueError("A manifest must describe exactly one category/protocol/fold")
    held_out = rows[0]["held_out_defect"]
    if rows[0]["protocol"] == "custom_comparison" and (
        not held_out or not any(row["defect_type"] == held_out for row in rows)
    ):
        raise ValueError("Custom comparison must include its named held-out defect")
    seen = set()
    for row in rows:
        if row["sample_id"] in seen or row["sample_id"] != row["image_path"]:
            raise ValueError(f"Duplicate or inconsistent sample ID: {row['sample_id']}")
        seen.add(row["sample_id"])
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {row['split']}")
        if row["protocol"] not in {"official_ad", "custom_comparison"}:
            raise ValueError(f"Invalid protocol: {row['protocol']}")
        if int(row["is_anomalous"]) != int(row["defect_type"] != "good"):
            raise ValueError(f"Image label disagrees with defect type: {row['sample_id']}")
        if bool(row["mask_path"]) != bool(int(row["is_anomalous"])):
            raise ValueError(f"Mask path disagrees with image label: {row['sample_id']}")
        if row["held_out_defect"] == row["defect_type"] and row["split"] != "test":
            raise ValueError(f"Held-out defect leaked into development: {row['sample_id']}")
        parts = Path(row["image_path"]).parts
        if len(parts) != 4 or parts[0] != row["category"] or parts[2] != row["defect_type"]:
            raise ValueError(f"Unexpected original image path: {row['image_path']}")
        if parts[1] == "train" and (int(row["is_anomalous"]) or row["split"] == "test"):
            raise ValueError(f"Invalid original normal training assignment: {row['sample_id']}")
        if parts[1] not in {"train", "test"}:
            raise ValueError(f"Invalid original source split: {parts[1]}")
        if row["protocol"] == "official_ad" and (
            row["held_out_defect"] or (parts[1] == "test" and row["split"] != "test")
        ):
            raise ValueError("official_ad must preserve all original test images")
        if parts[1] == "test" and row["defect_type"] == "good" and row["split"] != "test":
            raise ValueError("Original test/good images must remain test-only")


def manifest_files(rows: list[dict], seed: int, inventory: list[dict]) -> tuple[bytes, bytes]:
    validate_manifest(rows)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    content = buffer.getvalue().encode("utf-8")
    hashes = {r["sample_id"]: r["image_sha256"] for r in inventory}
    dataset_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    metadata = {
        "schema_version": 1,
        "seed": seed,
        "category": rows[0]["category"],
        "protocol": rows[0]["protocol"],
        "held_out_defect": rows[0]["held_out_defect"],
        "manifest_sha256": hashlib.sha256(content).hexdigest(),
        "dataset_image_inventory_sha256": dataset_hash,
        "source": "MVTec AD via Kaggle ipythonx/mvtec-ad",
        "rounding": "floor train, floor defect val, remainder test; good val is remainder",
        "normal_ratios": {"train": 0.8, "val": 0.2},
        "known_defect_ratios": {"train": 0.6, "val": 0.2, "test": 0.2},
        "counts": dict(sorted(Counter(f"{r['split']}/{r['defect_type']}" for r in rows).items())),
        "review_status": "pending_hung_review",
    }
    return content, (json.dumps(metadata, indent=2, ensure_ascii=True) + "\n").encode()


def read_manifest(path: Path, root: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    validate_manifest(rows)
    for row in rows:
        if not dataset_path(root, row["image_path"]).is_file():
            raise FileNotFoundError(row["image_path"])
        if row["mask_path"] and not dataset_path(root, row["mask_path"]).is_file():
            raise FileNotFoundError(row["mask_path"])
        row["is_anomalous"] = int(row["is_anomalous"])
    return rows


def load_manifest_sample(root: Path, row: dict, image_size: tuple[int, int] | None = None):
    sample = MVTecSample(
        row["sample_id"],
        row["image_path"],
        row["mask_path"],
        row["category"],
        row["defect_type"],
        Path(row["image_path"]).parts[1],
    )
    return load_sample(root, sample, image_size)
