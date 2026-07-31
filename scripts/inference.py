"""Command-line inference for Lay Language AI.

AI assistance: OpenAI Codex helped generate this project code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lay_language_ai.inference import Rewriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite medical text in plain language.")
    parser.add_argument("medical_text", help="Medical text to rewrite")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = Rewriter.from_default_paths().compare(args.medical_text)
    print("Base Model:")
    print(comparison.base_output)
    print()
    print("Fine-Tuned Model:")
    print(comparison.adapted_output)


if __name__ == "__main__":
    main()
