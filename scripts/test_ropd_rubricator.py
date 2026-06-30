"""
End-to-end test of the ROPD Rubricator → Verifier pipeline.

Sends one hardcoded math question through the full judge loop and prints the
rubric JSON and per-answer scores.  Does NOT require a dataset, a teacher
index, or a running training job.

Usage:
    uv run python scripts/test_ropd_rubricator.py

Set in .env (or environment):
    OPENAI_API_KEY, OPENAI_BASE_URL
    ROPD_RUBRICATOR_MODEL  (defaults to gpt-4o-mini)
    ROPD_VERIFIER_MODEL    (defaults to gpt-4o-mini)
    ROPD_RUBRICATOR_REASONING_EFFORT  (leave empty for DeepSeek / non-o-series)
    ROPD_VERIFIER_REASONING_EFFORT    (leave empty for DeepSeek / non-o-series)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Repo root on sys.path so algo.* imports work
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

_ENV_FILE = _REPO_ROOT / ".env"
if _ENV_FILE.exists():
    from dotenv import load_dotenv
    load_dotenv(_ENV_FILE, override=False)

# ---------------------------------------------------------------------------
# Hardcoded test fixtures
# ---------------------------------------------------------------------------
_QUESTION = "A store sells apples for $0.75 each and oranges for $1.25 each. If Maya buys 4 apples and 3 oranges, how much does she spend in total?"
_TEACHER_ANSWER = "Maya spends 4 × $0.75 = $3.00 on apples and 3 × $1.25 = $3.75 on oranges. Total = $3.00 + $3.75 = $6.75."
_STUDENT_ANSWERS = [
    "4 apples × $0.75 = $3.00. 3 oranges × $1.25 = $3.75. Total = $6.75.",
    "Apples: 4 × 0.75 = 3. Oranges: 3 × 1.25 = 3.75. Total = 3 + 3.75 = $6.25.",  # wrong answer
]

_RAW_PROMPT = [
    {"role": "system", "content": "You are a helpful math tutor. Solve problems step by step."},
    {"role": "user", "content": _QUESTION},
]


def _make_client(model: str, reasoning_effort: str | None) -> Any:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    return OpenAI(api_key=api_key, base_url=base_url), model, reasoning_effort


def _chat(client_tuple: tuple, messages: list[dict], max_tokens: int) -> str:
    client, model, reasoning_effort = client_tuple
    kwargs: dict[str, Any] = dict(model=model, messages=messages, max_tokens=max_tokens)
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def main() -> None:
    # Windows consoles default to cp1252; reconfigure to UTF-8 so JSON with
    # non-ASCII characters prints correctly.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    rubricator_model = os.environ.get("ROPD_RUBRICATOR_MODEL", "gpt-4o-mini").strip()
    verifier_model = os.environ.get("ROPD_VERIFIER_MODEL", "gpt-4o-mini").strip()
    rubricator_effort = os.environ.get("ROPD_RUBRICATOR_REASONING_EFFORT", "high").strip() or None
    verifier_effort = os.environ.get("ROPD_VERIFIER_REASONING_EFFORT", "high").strip() or None

    print(f"Rubricator model : {rubricator_model}  (reasoning_effort={rubricator_effort!r})")
    print(f"Verifier model   : {verifier_model}  (reasoning_effort={verifier_effort!r})")

    from algo.ropd.prompts import build_ropd_rubricator_prompt, build_ropd_verifier_prompt

    # ------------------------------------------------------------------
    # Step 1: Rubricator
    # ------------------------------------------------------------------
    print("\n--- Step 1: Building rubricator prompt ---")
    rubricator_prompt = build_ropd_rubricator_prompt(
        _RAW_PROMPT,
        teacher_answer=_TEACHER_ANSWER,
        student_answers=_STUDENT_ANSWERS,
        extra_rubric_instructions="",
        model=rubricator_model,
    )
    print(f"Prompt length: {len(rubricator_prompt)} chars")
    print("Calling rubricator...")

    rubricator_client = _make_client(rubricator_model, rubricator_effort)
    rubric_raw = _chat(
        rubricator_client,
        [{"role": "user", "content": rubricator_prompt}],
        max_tokens=4096,
    )

    # Extract JSON from response
    rubric_json: Any = None
    start = rubric_raw.find("[")
    end = rubric_raw.rfind("]")
    if start != -1 and end != -1:
        try:
            rubric_json = json.loads(rubric_raw[start : end + 1])
        except json.JSONDecodeError:
            pass

    if rubric_json is None:
        start = rubric_raw.find("{")
        end = rubric_raw.rfind("}")
        if start != -1 and end != -1:
            try:
                rubric_json = json.loads(rubric_raw[start : end + 1])
            except json.JSONDecodeError:
                pass

    if rubric_json is None:
        print("WARNING: Could not parse rubric JSON from response.")
        print("Raw response repr:", repr(rubric_raw[:500]))
        rubric_json = []
    else:
        print(f"Rubric ({len(rubric_json) if isinstance(rubric_json, list) else 1} criteria):")
        print(json.dumps(rubric_json, indent=2, ensure_ascii=False))

    # ------------------------------------------------------------------
    # Step 2: Verifier
    # ------------------------------------------------------------------
    print("\n--- Step 2: Calling verifier ---")
    verifier_prompt = build_ropd_verifier_prompt(
        _RAW_PROMPT,
        rubrics=rubric_json,
        answers=_STUDENT_ANSWERS,
        extra_scoring_instructions="",
        model=verifier_model,
    )
    print(f"Prompt length: {len(verifier_prompt)} chars")
    print("Calling verifier...")

    verifier_client = _make_client(verifier_model, verifier_effort)
    verifier_raw = _chat(
        verifier_client,
        [{"role": "user", "content": verifier_prompt}],
        max_tokens=2048,
    )

    print("Verifier raw response:")
    print(verifier_raw[:1000])

    print("\nRubricator → Verifier pipeline completed successfully.")


if __name__ == "__main__":
    main()
