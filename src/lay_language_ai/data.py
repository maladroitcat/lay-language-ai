"""Data helpers for Lay Language AI.

AI assistance: OpenAI Codex helped generate this project code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class RewriteExample:
    medical_text: str
    plain_language: str


def read_jsonl(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}") from exc
    return rows


def load_rewrite_examples(path: Path) -> list[RewriteExample]:
    examples: list[RewriteExample] = []
    for row in read_jsonl(path):
        medical_text = row.get("medical_text", "").strip()
        plain_language = row.get("plain_language", "").strip()
        if not medical_text or not plain_language:
            raise ValueError(f"Each row in {path} must include medical_text and plain_language")
        examples.append(RewriteExample(medical_text=medical_text, plain_language=plain_language))
    return examples


def format_instruction(example: RewriteExample) -> str:
    return (
        "Rewrite this medical text in plain language for a patient.\n\n"
        f"Medical text: {example.medical_text}\n\n"
        f"Plain language: {example.plain_language}"
    )


def write_training_jsonl(examples: Iterable[RewriteExample], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for example in examples:
            payload = {
                "medical_text": example.medical_text,
                "plain_language": example.plain_language,
                "text": format_instruction(example),
            }
            file.write(json.dumps(payload, ensure_ascii=True) + "\n")

