"""Fit PatchCore from an approved manifest; no automatic weight downloads."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from industrial_defect_detection.models.anomaly.train import fit  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    fit(args.config, args.run_id, args.smoke_test)


if __name__ == "__main__":
    main()
