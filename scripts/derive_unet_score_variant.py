"""Create a v2 U-Net artifact that reuses a v1 checkpoint with a border-aware image score.

The v1 artifact is never modified. The new artifact gets its own run ID
(<v1 run_id>-border<b>), the same checkpoint, and no thresholds: calibrate it again.

  python scripts/derive_unet_score_variant.py --artifact-dir <unet v1 artifact> --border 8
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from industrial_defect_detection.models.supervised.scoring import (  # noqa: E402
    image_score_border,
)
from industrial_defect_detection.runs import (  # noqa: E402
    file_sha256,
    git_state,
    read_json,
    write_json,
)


def derive(artifact_dir: Path, border: int, selection: Path | None = None) -> Path:
    metadata = read_json(artifact_dir / "metadata.json")
    if metadata.get("model_id") != "unet_resnet18":
        raise ValueError(f"{artifact_dir} is not a U-Net artifact")
    if image_score_border(metadata) != 0 or metadata.get("derived_from"):
        raise ValueError("Derive from the original v1 artifact, not from a variant")
    if border <= 0:
        raise ValueError("border must be positive; 0 is the v1 rule")
    run_id = f"{metadata['run_id']}-border{border}"
    target = artifact_dir.parent / run_id
    if target.exists():
        raise FileExistsError(f"Artifact already exists: {target}")
    checkpoint = artifact_dir / metadata["checkpoint"]
    checkpoint_sha = file_sha256(checkpoint)
    target.mkdir()
    shutil.copy2(checkpoint, target / metadata["checkpoint"])
    derived = {k: v for k, v in metadata.items() if k != "calibration"}
    derived.update(
        status="uncalibrated",
        run_id=run_id,
        score_version="v2",
        score={
            **metadata.get("score", {}),
            "image_score": (
                f"max of the probability map at original image size, ignoring a {border}-pixel "
                "border band (v2)"
            ),
            "image_score_border_px": border,
        },
        derived_from={
            "run_id": metadata["run_id"],
            "checkpoint_sha256": checkpoint_sha,
            "change": "image score only; same checkpoint, map and preprocessing",
            "selection": selection.as_posix() if selection else None,
        },
        derivation_git=git_state(),
    )
    write_json(target / "metadata.json", derived)
    if file_sha256(target / metadata["checkpoint"]) != checkpoint_sha:
        raise RuntimeError("Copied checkpoint does not match the v1 checkpoint")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--border", type=int, required=True)
    parser.add_argument("--selection", type=Path, help="JSON written by select_score_border.py")
    args = parser.parse_args()
    try:
        target = derive(args.artifact_dir, args.border, args.selection)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(f"Created {target}; run scripts/calibrate.py on it next")


if __name__ == "__main__":
    main()
