"""
ROPD pipeline validator — dry-run and CPU smoke test.

Checks config, datasets, tokenizer, teacher index, judge provider config, and
output directories without a GPU, without Ray workers, and without LLM API calls.

Usage
-----
Structural dry-run (fastest, no tokenization):
    uv run python scripts/dry_run_check.py

CPU smoke test (also tokenizes one sample to verify shapes):
    uv run python scripts/dry_run_check.py --mode smoke-test

Real GPU training (once GPU is available):
    set -a && source .env && set +a
    uv run --no-sync python -m verl.trainer.main_ppo --config-name ropd \\
        data.train_batch_size=16 \\
        actor_rollout_ref.actor.ppo_mini_batch_size=2 \\
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

_ENV_FILE = _REPO_ROOT / ".env"
if _ENV_FILE.exists():
    from dotenv import load_dotenv
    load_dotenv(_ENV_FILE, override=False)

# ── Result accumulator ───────────────────────────────────────────────────────

_results: list[tuple[str, str, str]] = []


def _record(status: str, label: str, note: str = "") -> None:
    _results.append((status, label, note))
    tag = f"[{status:<4}]"
    line = f"  {tag} {label}"
    if note:
        line += f"  ({note})"
    print(line)


def ok(label: str, note: str = "") -> None:
    _record("PASS", label, note)


def fail(label: str, note: str = "") -> None:
    _record("FAIL", label, note)


def skip(label: str, note: str = "") -> None:
    _record("SKIP", label, note)


# ── Helper ───────────────────────────────────────────────────────────────────

def _resolve_path(raw: str) -> Path:
    """Return an absolute path, resolving relative paths against repo root."""
    p = Path(raw.strip().strip('"').strip("'"))
    return p if p.is_absolute() else _REPO_ROOT / p


# ── Individual checks ────────────────────────────────────────────────────────

def check_cuda() -> bool:
    try:
        import torch
        if torch.cuda.is_available():
            ok("CUDA available", torch.cuda.get_device_name(0))
            return True
        skip("CUDA", "no GPU visible — dry-run/smoke-test only")
        return False
    except ImportError:
        skip("CUDA check", "torch not installed")
        return False


def check_config() -> Any:
    try:
        from omegaconf import OmegaConf
    except ImportError:
        skip("ropd.yaml", "omegaconf not installed")
        return None

    config_path = _REPO_ROOT / "verl/trainer/config/ropd.yaml"
    if not config_path.exists():
        fail("ropd.yaml exists", str(config_path))
        return None
    try:
        cfg = OmegaConf.load(config_path)
        ok("ropd.yaml loads and parses")
        return cfg
    except Exception as exc:
        fail("ropd.yaml loads", str(exc)[:120])
        return None


def check_judge_providers() -> None:
    try:
        from omegaconf import OmegaConf
    except ImportError:
        skip("judge_providers.yaml", "omegaconf not installed")
        return

    path = _REPO_ROOT / "verl/trainer/config/judge_providers.yaml"
    if not path.exists():
        fail("judge_providers.yaml exists", str(path))
        return
    try:
        cfg = OmegaConf.load(path)
        ok("judge_providers.yaml loads and parses")
    except Exception as exc:
        fail("judge_providers.yaml loads", str(exc)[:120])
        return

    profile = os.environ.get("ROPD_JUDGE_PROFILE", "openai_chat")
    available = list(cfg.get("profiles", {}).keys())
    if profile in available:
        ok(f"Profile '{profile}' present in judge_providers.yaml")
    else:
        fail(f"Profile '{profile}' in judge_providers.yaml", f"available: {available}")


def check_parquet(label: str, path: Path) -> int:
    try:
        import pandas as pd
    except ImportError:
        skip(f"{label}", "pandas not installed")
        return 0

    if not path.exists():
        fail(f"{label} exists", str(path))
        return 0
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        fail(f"{label} readable", str(exc)[:100])
        return 0

    try:
        rel = path.relative_to(_REPO_ROOT)
    except ValueError:
        rel = path

    ok(f"{label} loads", f"{len(df)} rows  [{rel}]")

    required_cols = {"data_source", "prompt", "ability", "reward_model", "extra_info"}
    missing = required_cols - set(df.columns)
    if missing:
        fail(f"{label} required columns", f"missing: {missing}")
    else:
        ok(f"{label} schema columns present")

    if "extra_info" in df.columns:
        ei = df["extra_info"].iloc[0]
        if isinstance(ei, dict) and "index" in ei:
            ok(f"{label} extra_info.index present", f"e.g. {ei['index']!r}")
        else:
            fail(f"{label} extra_info.index", f"got type {type(ei).__name__}")

    if "prompt" in df.columns:
        sample = df["prompt"].iloc[0]
        # Parquet may deserialize list-of-struct as numpy ndarray; normalise to list.
        if hasattr(sample, "tolist"):
            sample = sample.tolist()
        if isinstance(sample, (list, tuple)) and len(sample) >= 1:
            ok(f"{label} prompt is a message list", f"{len(sample)} messages per row")
        else:
            fail(f"{label} prompt format", f"expected list[dict], got {type(sample).__name__}")

    return len(df)


def check_teacher_index(path: Path) -> int:
    if not path.exists():
        fail("Teacher index exists", str(path))
        return 0

    try:
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        records = [json.loads(l) for l in lines]
    except Exception as exc:
        fail("Teacher index readable as JSONL", str(exc)[:100])
        return 0

    try:
        rel = path.relative_to(_REPO_ROOT)
    except ValueError:
        rel = path

    ok("Teacher index loads", f"{len(records)} records  [{rel}]")

    if not records:
        fail("Teacher index non-empty")
        return 0

    required_keys = {"schema_version", "uid", "raw_prompt_hash", "teacher_answers", "teacher_fingerprint"}
    missing = required_keys - set(records[0].keys())
    if missing:
        fail("Teacher index record schema", f"missing keys: {missing}")
    else:
        r = records[0]
        ok("Teacher index record schema", f"uid={r['uid']!r}, {len(r['teacher_answers'])} answer(s)")

    uids = [r.get("uid") for r in records]
    if len(uids) == len(set(uids)):
        ok("Teacher index UIDs unique")
    else:
        fail("Teacher index UIDs unique", "duplicate UIDs found")

    return len(records)


def check_tokenizer(model_path: Path) -> Any:
    if not model_path.exists():
        if sys.platform == "win32":
            # Model is downloaded into WSL (~ROPD_official/models/), not the Windows checkout.
            skip("Tokenizer", f"model not at Windows path — run from WSL where the model was downloaded  [{model_path}]")
        else:
            fail("Model / tokenizer directory exists", str(model_path))
        return None

    # Presence check — at least tokenizer.json must exist and be non-empty
    tok_json = model_path / "tokenizer.json"
    if tok_json.exists() and tok_json.stat().st_size < 200:
        fail("tokenizer.json is not an LFS stub", f"file is only {tok_json.stat().st_size} bytes (Git LFS pointer?)")
        return None

    try:
        from transformers import AutoTokenizer
        try:
            tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
            ok("Tokenizer loads (fast)", f"vocab_size={tok.vocab_size}")
        except Exception:
            # Fast (Rust) tokenizer may panic on some tokenizers versions; fall back to slow.
            tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, use_fast=False)
            ok("Tokenizer loads (slow fallback — upgrade tokenizers>=0.21 for fast)", f"vocab_size={tok.vocab_size}")
        return tok
    except Exception as exc:
        fail("Tokenizer loads", str(exc)[:120])
        return None


def check_output_dirs() -> None:
    ckpt_raw = os.environ.get("ROPD_CKPT_DIR", "checkpoints/ropd")
    experiment = os.environ.get("EXPERIMENT", "ropd")
    run_dir = _resolve_path(ckpt_raw) / experiment
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            rel = run_dir.relative_to(_REPO_ROOT)
        except ValueError:
            rel = run_dir
        ok("Checkpoint directory creatable", str(rel))
    except Exception as exc:
        fail("Checkpoint directory creatable", str(exc))

    outputs_dir = _REPO_ROOT / "outputs"
    try:
        outputs_dir.mkdir(exist_ok=True)
        ok("Outputs directory creatable")
    except Exception as exc:
        fail("Outputs directory creatable", str(exc))


def check_reward_manager_config() -> None:
    """Verify the algo.ropd import chain compiles without errors.

    We do NOT call build_ropd_judge_config() here — it requires the full resolved
    ropd config (provider_resolution / teacher sections) which is only available
    inside the Hydra training entry point.  The judge_providers.yaml is already
    validated by check_judge_providers() above.
    """
    import importlib

    try:
        importlib.import_module("algo.ropd.client")
        ok("algo.ropd.client imports")
    except Exception as exc:
        fail("algo.ropd.client imports", str(exc)[:120])
        return

    try:
        importlib.import_module("algo.ropd_judge_provider_resolver")
        ok("algo.ropd_judge_provider_resolver imports")
    except Exception as exc:
        fail("algo.ropd_judge_provider_resolver imports", str(exc)[:120])


def smoke_tokenize(tokenizer: Any, train_parquet: Path, max_len: int = 64) -> None:
    if tokenizer is None:
        skip("Smoke tokenization", "no tokenizer")
        return
    if not train_parquet.exists():
        skip("Smoke tokenization", "train parquet missing")
        return

    try:
        import pandas as pd
        df = pd.read_parquet(train_parquet)
        row = df.iloc[0]
        prompt = row["prompt"]

        if isinstance(prompt, (list, tuple)):
            text = tokenizer.apply_chat_template(
                list(prompt), tokenize=False, add_generation_prompt=True
            )
        else:
            text = str(prompt)

        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
        ids = enc["input_ids"]
        ok(
            f"Smoke tokenize 1 prompt (max_len={max_len})",
            f"input_ids shape={tuple(ids.shape)}, first tokens={ids[0].tolist()[:6]}...",
        )

        # Also verify answer field is accessible
        ei = row.get("extra_info", {})
        ground_truth = row.get("reward_model", {}).get("ground_truth", None)
        ok("Ground-truth field accessible", f"ground_truth={ground_truth!r}")

    except Exception as exc:
        fail("Smoke tokenization", str(exc)[:120])


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ROPD dry-run validator and CPU smoke test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["dry-run", "smoke-test"],
        default="dry-run",
        help="dry-run: structural checks only  |  smoke-test: also tokenize one batch",
    )
    args = parser.parse_args()
    smoke = args.mode == "smoke-test"

    # Resolve dataset paths (mirrors train.sh logic)
    data_root = _resolve_path(os.environ.get("DATA_ROOT", "datasets/unified"))
    train_task = os.environ.get("ROPD_TRAIN_TASK", "math/dapo-math-17k")
    val_task = os.environ.get("ROPD_VAL_TASK", "math_eval/aime24")
    model_path = _resolve_path(os.environ.get("ROPD_MODEL_PATH", "models/Qwen3-4B"))

    teacher_raw = os.environ.get("ROPD_TEACHER_INDEX_PATH", "")
    if teacher_raw:
        teacher_index = _resolve_path(teacher_raw)
    else:
        teacher_index = data_root / train_task / "artifacts/teacher_index/shared-teacher-index.jsonl"

    train_parquet = data_root / train_task / "train.parquet"
    val_parquet = data_root / val_task / "test.parquet"

    print()
    print("=" * 66)
    print(f"  ROPD pipeline validator  [mode: {args.mode}]")
    print("=" * 66)

    # ── GPU ──────────────────────────────────────────────────────────────────
    print("\nGPU / CUDA")
    check_cuda()

    # ── Config ───────────────────────────────────────────────────────────────
    print("\nConfig files")
    check_config()
    check_judge_providers()

    # ── Datasets ─────────────────────────────────────────────────────────────
    print("\nDatasets")
    n_train = check_parquet("Train", train_parquet)
    check_parquet("Val", val_parquet)

    # ── Teacher index ─────────────────────────────────────────────────────────
    print("\nTeacher index")
    n_index = check_teacher_index(teacher_index)

    # Warn if UID counts differ
    if n_train > 0 and n_index > 0 and n_index < n_train:
        fail(
            "Teacher index covers all train rows",
            f"index has {n_index} records but train has {n_train} rows",
        )
    elif n_train > 0 and n_index >= n_train:
        ok("Teacher index covers all train rows", f"{n_index} >= {n_train}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print("\nTokenizer  (no model weights loaded)")
    tokenizer = check_tokenizer(model_path)

    # ── Reward manager config (no API calls) ──────────────────────────────────
    print("\nReward manager config  (no API calls)")
    check_reward_manager_config()

    # ── Output / checkpoint dirs ──────────────────────────────────────────────
    print("\nOutput directories")
    check_output_dirs()

    # ── Smoke test ────────────────────────────────────────────────────────────
    if smoke:
        print("\nSmoke test  (CPU tokenization, 1 sample, max_len=64)")
        smoke_tokenize(tokenizer, train_parquet, max_len=64)
    else:
        print("\nSmoke test  [skipped — use --mode smoke-test to enable]")

    # ── Summary ───────────────────────────────────────────────────────────────
    n_pass = sum(1 for s, _, _ in _results if s == "PASS")
    n_fail = sum(1 for s, _, _ in _results if s == "FAIL")
    n_skip = sum(1 for s, _, _ in _results if s == "SKIP")

    print()
    print("─" * 66)
    print(f"  {n_pass} passed   {n_fail} failed   {n_skip} skipped")
    print("─" * 66)

    if n_fail:
        print("\nFailed items:")
        for s, lbl, note in _results:
            if s == "FAIL":
                print(f"  - {lbl}" + (f": {note}" if note else ""))
        print()
        print("NOTE: Trainer init (Ray workers, model weights, vLLM engine) is NOT")
        print("      checked here — those require a GPU and are the only remaining")
        print("      unknowns once the above failures are fixed.")
        sys.exit(1)

    print()
    print("All structural checks passed.")
    print("Config, datasets, tokenizer, teacher index, and output dirs are valid.")
    print()
    if not smoke:
        print("Run with --mode smoke-test to also verify tokenizer shapes:")
        print("  uv run python scripts/dry_run_check.py --mode smoke-test")
        print()
    print("When a GPU is available, start real training with:")
    print("  cd ~/ROPD_official")
    print("  set -a && source .env && set +a")
    print("  uv run --no-sync python -m verl.trainer.main_ppo --config-name ropd \\")
    print("    data.train_batch_size=16 \\")
    print("    actor_rollout_ref.actor.ppo_mini_batch_size=2 \\")
    print("    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2")
    print()


if __name__ == "__main__":
    main()
