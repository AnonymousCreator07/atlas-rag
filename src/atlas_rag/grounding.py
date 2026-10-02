"""Post-generation grounding and citation checks."""

import re
from typing import Any

from atlas_rag.config import Settings


CITATION_RE = re.compile(r"\[S(\d+)\]")


def citation_numbers(answer: str) -> list[int]:
    return [int(match) for match in CITATION_RE.findall(answer)]


def validate_citations(answer: str, source_count: int) -> tuple[bool, str]:
    numbers = citation_numbers(answer)
    if not numbers:
        return False, "The generated answer did not contain a source citation."
    invalid = [number for number in numbers if number < 1 or number > source_count]
    if invalid:
        return False, f"Invalid source markers: {invalid}."
    return True, ""


def verify_grounding(
    answer: str,
    sources: list[dict[str, Any]],
    settings: Settings,
) -> tuple[bool, str]:
    """Use deterministic citation checks by default and an optional LLM verifier."""
    valid, reason = validate_citations(answer, len(sources))
    if not valid:
        return False, reason
    if not settings.grounding_check_enabled or not settings.openai_api_key:
        return True, ""

    from openai import OpenAI

    evidence = "\n\n".join(
        f"[S{i}] {source['document']}" for i, source in enumerate(sources, start=1)
    )
    prompt = f"""Evaluate whether the answer is fully supported by the evidence.
Return only JSON: {{"supported": true|false, "reason": "brief reason"}}.
Do not use outside knowledge.

Answer:
{answer}

Evidence:
{evidence}
"""
    response = OpenAI(api_key=settings.openai_api_key).chat.completions.create(
        model=settings.openai_chat_model,
        messages=[{"role": "system", "content": "You are a strict factual grounding verifier."},
                  {"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    import json

    result = json.loads(response.choices[0].message.content or "{}")
    return bool(result.get("supported")), str(result.get("reason", ""))


def fallback_answer(reason: str) -> str:
    return (
        "I can't provide a grounded answer from the indexed documents. "
        f"Grounding check: {reason}"
    )
