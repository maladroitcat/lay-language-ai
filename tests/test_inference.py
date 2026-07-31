"""Tests for Lay Language AI inference prompts.

AI assistance: OpenAI Codex helped generate this project code.
"""

from lay_language_ai.inference import rewrite_prompt


def test_rewrite_prompt_contains_medical_text() -> None:
    prompt = rewrite_prompt("MRI demonstrates stenosis.")

    assert "MRI demonstrates stenosis." in prompt
    assert "patient-friendly" in prompt
    assert "Patient-friendly rewrite:" in prompt
