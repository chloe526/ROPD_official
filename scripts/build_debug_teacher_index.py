"""
Build a teacher answer index for the debug/tiny-math dataset.

Reads each row from datasets/unified/debug/tiny-math/train.parquet, calls the
teacher LLM (ROPD_TEACHER_MODEL at OPENAI_BASE_URL) for each prompt, and
writes a JSONL teacher index that training will consume via the offline_index
provider.

The stored teacher_fingerprint matches exactly what verl's training loop
computes from the resolved teacher role config, so the fingerprint validation
in OfflineTeacherIndex.load() passes without errors.

Usage:
    uv run python scripts/build_debug_teacher_index.py

Required env vars (set in .env):
    OPENAI_API_KEY
    OPENAI_BASE_URL
    ROPD_TEACHER_MODEL
    ROPD_TEACHER_INDEX_PATH   (output path; defaults to the debug location)
    ROPD_TEACHER_ANSWER_COUNT (number of answers per prompt; default 1)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

_ENV_FILE = _REPO_ROOT / ".env"
if _ENV_FILE.exists():
    from dotenv import load_dotenv
    load_dotenv(_ENV_FILE, override=False)

_DEFAULT_INDEX_PATH = (
    _REPO_ROOT
    / "datasets/unified/debug/tiny-math/artifacts/teacher_index/debug-teacher-index.jsonl"
)
_DEFAULT_TRAIN_PARQUET = (
    _REPO_ROOT / "datasets/unified/debug/tiny-math/train.parquet"
)


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        print(f"ERROR: {name} is not set. Check your .env file.", file=sys.stderr)
        sys.exit(1)
    return value


def _call_teacher(client: Any, model: str, messages: list[dict], max_tokens: int) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip()


def _build_messages_from_raw_prompt(raw_prompt: Any) -> list[dict]:
    if isinstance(raw_prompt, str):
        return [{"role": "user", "content": raw_prompt}]
    if isinstance(raw_prompt, list):
        return raw_prompt
    return list(raw_prompt)


def main() -> None:
    api_key = _require_env("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    teacher_model = os.environ.get("ROPD_TEACHER_MODEL", "").strip()
    if not teacher_model:
        print("ERROR: ROPD_TEACHER_MODEL not set.", file=sys.stderr)
        sys.exit(1)

    answer_count = int(os.environ.get("ROPD_TEACHER_ANSWER_COUNT", "1"))
    index_path = Path(os.environ.get("ROPD_TEACHER_INDEX_PATH", "").strip() or _DEFAULT_INDEX_PATH)
    train_parquet = _DEFAULT_TRAIN_PARQUET

    if not train_parquet.exists():
        print(
            f"ERROR: Training parquet not found at {train_parquet}\n"
            "Run scripts/create_debug_dataset.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"teacher model : {teacher_model}")
    print(f"base_url      : {base_url or '(OpenAI default)'}")
    print(f"answer_count  : {answer_count}")
    print(f"output        : {index_path}")

    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas not installed. Run: uv sync", file=sys.stderr)
        sys.exit(1)

    from openai import OpenAI
    from algo.ropd_teacher_index import (
        OFFLINE_TEACHER_MULTI_ANSWER_SCHEMA_VERSION,
        build_teacher_fingerprint_payload,
        hash_canonical_raw_prompt,
        hash_teacher_fingerprint,
    )
    from algo.ropd_prompts import PROMPT_TEMPLATE_VERSION

    client = OpenAI(api_key=api_key, base_url=base_url)

    # Build the fingerprint that training will expect.
    # Fields must match the resolved teacher role config:
    #   - provider comes from the entrypoints.build_teacher_index override → openai_compatible
    #   - timeout_seconds comes from ropd.yaml overrides.teacher.timeout_seconds → 480.0
    #   - reasoning_effort: teacher role has no reasoning_effort set → None
    teacher_fingerprint = build_teacher_fingerprint_payload(
        provider="openai_compatible",
        model=teacher_model,
        base_url=base_url,
        reasoning_effort=None,
        max_output_tokens=8192,
        temperature=None,
        top_p=None,
        timeout_seconds=480.0,
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
    )
    fingerprint_digest = hash_teacher_fingerprint(teacher_fingerprint)
    print(f"fingerprint   : {fingerprint_digest[:16]}...")

    df = pd.read_parquet(train_parquet)
    print(f"Dataset rows  : {len(df)}")

    index_path.parent.mkdir(parents=True, exist_ok=True)
    records_written = 0

    with index_path.open("w", encoding="utf-8") as out:
        for _, row in df.iterrows():
            uid = str(row["extra_info"]["index"])
            raw_prompt = row["prompt"]
            messages = _build_messages_from_raw_prompt(raw_prompt)
            prompt_hash = hash_canonical_raw_prompt(raw_prompt)

            teacher_answers: list[str] = []
            for attempt in range(answer_count):
                print(f"  uid={uid}  answer {attempt + 1}/{answer_count} ...", end=" ", flush=True)
                try:
                    answer = _call_teacher(client, teacher_model, messages, max_tokens=1024)
                    teacher_answers.append(answer)
                    print("OK")
                except Exception as exc:
                    print(f"ERROR: {exc}", file=sys.stderr)
                    raise
                if attempt < answer_count - 1:
                    time.sleep(0.3)

            record = {
                "schema_version": OFFLINE_TEACHER_MULTI_ANSWER_SCHEMA_VERSION,
                "uid": uid,
                "raw_prompt_hash": prompt_hash,
                "teacher_answers": teacher_answers,
                "teacher_fingerprint": teacher_fingerprint,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            records_written += 1

    print(f"\nWrote {records_written} records: {index_path}")
    print("Teacher index ready.")
    print(f"\nSet in .env:")
    print(f'  ROPD_TEACHER_INDEX_PATH="{index_path}"')


if __name__ == "__main__":
    main()
