"""Train U-Net (ResNet-18 encoder) from a YAML config.

Examples, from the repository root:
  python scripts/train_supervised.py --config configs/supervised/unet_resnet18_tile.yaml
  python scripts/train_supervised.py --config configs/supervised/unet_resnet18_tile.yaml \
      --smoke-test   # 2 epochs x 2 batches, checks the pipeline only
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from industrial_defect_detection.models.supervised.train import train  # noqa: E402
from industrial_defect_detection.runs import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", help="Override data.dataset_root (e.g. on Colab)")
    parser.add_argument("--epochs", type=int, help="Override training.epochs")
    parser.add_argument("--batch-size", type=int, help="Override training.batch_size")
    parser.add_argument("--run-id", help="Use this run ID instead of a timestamp")
    parser.add_argument("--smoke-test", action="store_true", help="2 epochs x 2 batches")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.dataset_root:
        config["data"]["dataset_root"] = args.dataset_root
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.smoke_test:
        config["training"]["epochs"] = 2
        config["training"]["max_batches_per_epoch"] = 2
        config["experiment"]["fold"] = f"smoke-{config['experiment']['fold']}"
    train(config, args.run_id)


if __name__ == "__main__":
    main()
