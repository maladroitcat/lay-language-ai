"""Model-backed inference helpers for Lay Language AI.

AI assistance: OpenAI Codex helped generate this project code.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_MODEL = "google/flan-t5-small"
DEFAULT_ADAPTED_MODEL_PATH = PROJECT_ROOT / "models" / "lay_language_ai_model"


@dataclass(frozen=True)
class RewriteComparison:
    base_output: str
    adapted_output: str


class Rewriter:
    """Compare a base instruction model with the fine-tuned medical rewrite model."""

    def __init__(
        self,
        base_model_name: str = DEFAULT_BASE_MODEL,
        fine_tuned_model_path: Path = DEFAULT_ADAPTED_MODEL_PATH,
    ) -> None:
        self.device = select_device()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name).to(self.device)
        self.base_model.eval()

        if not fine_tuned_model_path.exists():
            raise FileNotFoundError(
                f"Fine-tuned model not found at {fine_tuned_model_path}. "
                "Run `python scripts/train.py` before starting the app."
            )
        self.fine_tuned_tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)
        self.fine_tuned_model = AutoModelForSeq2SeqLM.from_pretrained(fine_tuned_model_path).to(self.device)
        self.fine_tuned_model.eval()

    @classmethod
    def from_default_paths(cls) -> "Rewriter":
        return cls()

    def compare(self, medical_text: str) -> RewriteComparison:
        return RewriteComparison(
            base_output=escape(self.base_rewrite(medical_text)),
            adapted_output=escape(self.adapted_rewrite(medical_text)),
        )

    def base_rewrite(self, medical_text: str) -> str:
        return self.generate(
            model=self.base_model,
            tokenizer=self.tokenizer,
            prompt=rewrite_prompt(medical_text),
        )

    def adapted_rewrite(self, medical_text: str) -> str:
        return self.generate(
            model=self.fine_tuned_model,
            tokenizer=self.fine_tuned_tokenizer,
            prompt=rewrite_prompt(medical_text),
        )

    def generate(self, model: AutoModelForSeq2SeqLM, tokenizer: AutoTokenizer, prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to(self.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=140,
                num_beams=4,
                do_sample=False,
                no_repeat_ngram_size=3,
            )
        output = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        return output or "I could not generate a rewrite for this note."


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def rewrite_prompt(medical_text: str) -> str:
    return (
        "Rewrite this medical note in patient-friendly plain English. "
        "Explain medical jargon briefly and preserve the original meaning.\n\n"
        f"Medical note: {medical_text}\n\n"
        "Patient-friendly rewrite:"
    )

