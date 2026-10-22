"""Quality assessment via open-source (Ollama) or OpenAI LLM, with heuristic fallback."""

from __future__ import annotations

import json
import re
from typing import Any

from src.features import ClassFeatures
from src.llm_config import LlmSettings, resolve_llm_settings
from src.prompts import SYSTEM_PROMPT, USER_TEMPLATE


def _parse_json_content(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def _build_user_message(features: ClassFeatures) -> str:
    return USER_TEMPLATE.format(
        metrics_json=json.dumps(features.to_dict(), indent=2),
        opening=features.opening_excerpt or "(no opening transcript)",
        closing=features.closing_excerpt or "(no closing transcript)",
        quotes="\n".join(features.sample_quotes) or "(no quotes)",
    )


def _chat_assessment(settings: LlmSettings, user_msg: str) -> dict[str, Any]:
    from openai import OpenAI

    kwargs: dict[str, Any] = {"api_key": settings.api_key}
    if settings.base_url:
        kwargs["base_url"] = settings.base_url
    client = OpenAI(**kwargs)

    system = SYSTEM_PROMPT
    if settings.provider == "ollama":
        system += "\nYou MUST respond with a single valid JSON object only, no markdown."

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]

    attempts: list[dict[str, Any]] = [
        {"temperature": 0.2, "response_format": {"type": "json_object"}},
        {"temperature": 0.2},
    ]
    last_error: Exception | None = None
    for opts in attempts:
        try:
            resp = client.chat.completions.create(
                model=settings.model,
                messages=messages,
                **opts,
            )
            content = resp.choices[0].message.content or "{}"
            result = _parse_json_content(content)
            result["engine"] = settings.engine
            result["model"] = settings.model
            return result
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(
        f"LLM assessment failed ({settings.provider}/{settings.model}). "
        f"Is Ollama running? Try: ollama pull {settings.model}"
    ) from last_error


def assess_with_llm(
    features: ClassFeatures,
    *,
    provider: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    settings = resolve_llm_settings(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    if not settings:
        return assess_heuristic(features)

    user_msg = _build_user_message(features)
    try:
        return _chat_assessment(settings, user_msg)
    except Exception as e:
        fallback = assess_heuristic(features)
        fallback["summary"] = (
            f"LLM call failed ({e}). Showing heuristic fallback — "
            "check Ollama is running (`ollama serve`) and the model is pulled."
        )
        fallback["llm_error"] = str(e)
        return fallback


def assess_heuristic(features: ClassFeatures) -> dict[str, Any]:
    """Fallback when LLM is unavailable — transparent rule-based score."""
    score = 0
    notes: list[str] = []

    if features.homework_mentions >= 1:
        score += 2
        notes.append("Homework or assignment mentioned in transcript.")
    else:
        notes.append("No clear homework mention detected.")

    if features.question_count >= 15:
        score += 2
    elif features.question_count >= 8:
        score += 1
    notes.append(f"Question/interaction signals: {features.question_count}.")

    if features.intro_signal_count >= 3:
        score += 1
    if features.encouragement_count >= 3:
        score += 1
    if features.camera_background_mentions >= 1:
        score += 1

    if features.longest_gap_sec > 120:
        score -= 2
        notes.append(f"Long silence gap (~{features.longest_gap_sec}s) detected between cues.")
    if features.teacher_talk_ratio > 0.92:
        score -= 1
        notes.append("Teacher-dominated talk ratio; limited student voice in transcript.")

    if score >= 6:
        quality = "Excellent"
    elif score >= 4:
        quality = "Good"
    elif score >= 2:
        quality = "Average"
    else:
        quality = "Poor"

    observations = []
    for q in features.sample_quotes[:5]:
        m = re.match(r"\[([^\]]+)\]\s*([^:]+):\s*(.+)", q)
        if m:
            observations.append({"time": m.group(1), "quote": m.group(3)[:200], "note": "Sample transcript evidence"})
        else:
            observations.append({"time": "—", "quote": q[:200], "note": "Sample transcript evidence"})

    return {
        "quality": quality,
        "confidence": "low",
        "summary": (
            "Heuristic assessment (LLM not configured or unreachable). "
            "Use Ollama locally (LLM_PROVIDER=ollama) or set OPENAI_API_KEY."
        ),
        "strengths": [n for n in notes if "mentioned" in n or "signals" in n][:3],
        "improvements": [
            "Run `./scripts/setup_ollama.sh` or `ollama pull qwen2.5:7b-instruct`.",
            "Optionally add video frame checks for background and face visibility.",
        ],
        "observations": observations,
        "metrics_used": [
            f"question_count={features.question_count}",
            f"homework_mentions={features.homework_mentions}",
            f"longest_gap_sec={features.longest_gap_sec}",
            f"teacher_talk_ratio={features.teacher_talk_ratio}",
        ],
        "engine": "heuristic",
        "model": "rules",
    }
