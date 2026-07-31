"""Prepare Lay Language AI training data.

AI assistance: OpenAI Codex helped generate this project code.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lay_language_ai.data import load_rewrite_examples, write_training_jsonl


RAW_PATH = PROJECT_ROOT / "data" / "raw" / "medical_rewrites.jsonl"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "train.jsonl"


def preprocess(raw_path: Path = RAW_PATH, processed_path: Path = PROCESSED_PATH) -> int:
    examples = load_rewrite_examples(raw_path)
    write_training_jsonl(examples, processed_path)
    return len(examples)


def main() -> None:
    count = preprocess()
    print(f"Wrote {count} examples to {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
