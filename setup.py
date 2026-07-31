"""Set up Lay Language AI data artifacts.

AI assistance: OpenAI Codex helped generate this project code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.preprocess import preprocess
from scripts.train import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up Lay Language AI.")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Also fine-tune the model after preparing the training data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = preprocess()
    print(f"Prepared {count} training examples")
    if args.train:
        output_dir = train_model()
        print(f"Wrote fine-tuned model to {output_dir}")
    else:
        print("Run `python scripts/train.py` to fine-tune the model.")


if __name__ == "__main__":
    main()
