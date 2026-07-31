"""Fine-tune a small text model for Lay Language AI.

AI assistance: OpenAI Codex helped generate this project code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lay_language_ai.data import RewriteExample, load_rewrite_examples
from lay_language_ai.inference import adapted_prompt, select_device


RAW_PATH = PROJECT_ROOT / "data" / "raw" / "medical_rewrites.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "models" / "lay_language_ai_model"
DEFAULT_MODEL_NAME = "google/flan-t5-small"


class RewriteDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        examples: list[RewriteExample],
        tokenizer: AutoTokenizer,
        max_input_length: int = 384,
        max_target_length: int = 160,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        inputs = self.tokenizer(
            adapted_prompt(example.medical_text),
            max_length=self.max_input_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        labels = self.tokenizer(
            example.plain_language,
            max_length=self.max_target_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Lay Language AI.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--raw-path", type=Path, default=RAW_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    return parser.parse_args()


def train_model(
    model_name: str = DEFAULT_MODEL_NAME,
    raw_path: Path = RAW_PATH,
    output_dir: Path = OUTPUT_DIR,
    epochs: int = 4,
    batch_size: int = 1,
    learning_rate: float = 5e-5,
) -> Path:
    examples = load_rewrite_examples(raw_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = select_device()
    model.to(device)
    model.train()

    dataset = RewriteDataset(examples, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()
            total_loss += float(outputs.loss.detach().cpu())
        mean_loss = total_loss / max(len(dataloader), 1)
        print(f"epoch={epoch + 1} loss={mean_loss:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def main() -> None:
    args = parse_args()
    output_dir = train_model(
        model_name=args.model_name,
        raw_path=args.raw_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    print(f"Wrote fine-tuned model to {output_dir}")


if __name__ == "__main__":
    main()

