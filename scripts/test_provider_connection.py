"""
Quick provider connectivity check.

Reads OPENAI_API_KEY, OPENAI_BASE_URL, and ROPD_RUBRICATOR_MODEL from the
repo .env (or the current environment) and sends a single chat completion
request.  No ROPD-internal imports — just the openai SDK.

Usage:
    uv run python scripts/test_provider_connection.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Load .env from repo root if present
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_ENV_FILE = _REPO_ROOT / ".env"
if _ENV_FILE.exists():
    from dotenv import load_dotenv
    load_dotenv(_ENV_FILE, override=False)


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        print(f"ERROR: {name} is not set. Check your .env file.", file=sys.stderr)
        sys.exit(1)
    return value


def main() -> None:
    api_key = _require_env("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    model = os.environ.get("ROPD_RUBRICATOR_MODEL", "").strip() or "gpt-4o-mini"

    print(f"base_url : {base_url or '(OpenAI default)'}")
    print(f"model    : {model}")
    print("Sending test request...")

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with exactly: ROPD connection OK"}],
        max_tokens=32,
    )

    choice = response.choices[0]
    content = choice.message.content or ""
    print(f"\nResponse  : {content.strip()}")
    print(f"Finish    : {choice.finish_reason}")
    print(f"Tokens    : prompt={response.usage.prompt_tokens} "
          f"completion={response.usage.completion_tokens}")
    print("\nProvider connection OK.")


if __name__ == "__main__":
    main()
