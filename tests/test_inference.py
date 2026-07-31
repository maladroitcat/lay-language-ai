"""Tests for Lay Language AI inference prompts.

AI assistance: OpenAI Codex helped generate this project code.
"""

from lay_language_ai.inference import adapted_prompt, base_prompt


def test_base_prompt_contains_medical_text() -> None:
    prompt = base_prompt("MRI demonstrates stenosis.")

    assert "MRI demonstrates stenosis." in prompt
    assert "Plain English:" in prompt


def test_adapted_prompts_are_patient_friendly() -> None:
    prompt = adapted_prompt("The biopsy reveals benign changes.")

    assert "patient-friendly" in prompt
    assert "Patient-friendly rewrite:" in prompt
