"""Tests for Lay Language AI data helpers.

AI assistance: OpenAI Codex helped generate this project code.
"""

from pathlib import Path

from lay_language_ai.data import load_rewrite_examples, write_training_jsonl


def test_load_rewrite_examples() -> None:
    examples = load_rewrite_examples(Path("data/raw/medical_rewrites.jsonl"))

    assert examples
    assert examples[0].medical_text
    assert examples[0].plain_language


def test_write_training_jsonl(tmp_path: Path) -> None:
    examples = load_rewrite_examples(Path("data/raw/medical_rewrites.jsonl"))
    output_path = tmp_path / "train.jsonl"

    write_training_jsonl(examples[:1], output_path)

    content = output_path.read_text(encoding="utf-8")
    assert "Medical text:" in content
    assert "Plain language:" in content
