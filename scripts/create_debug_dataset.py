"""
Create a tiny debug dataset for ROPD end-to-end testing.

Writes two parquet files:
  datasets/unified/debug/tiny-math/train.parquet  (20 rows)
  datasets/unified/debug/tiny-val/test.parquet    (5 rows)

Each row has the columns expected by verl's RLHFDataset:
  data_source, prompt, ability, reward_model, extra_info

The extra_info["index"] field is the stable uid used for teacher index lookup.

Usage:
    uv run python scripts/create_debug_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

_SYSTEM_PROMPT = (
    "You are a helpful math assistant. "
    "Solve the problem step by step and give the final numeric answer."
)

# (question, answer) pairs — simple integer arithmetic for fast student rollouts
_PROBLEMS = [
    ("What is 14 + 29?", "43"),
    ("What is 87 - 34?", "53"),
    ("What is 6 × 8?", "48"),
    ("What is 72 ÷ 9?", "8"),
    ("What is 25 + 37?", "62"),
    ("A rectangle has length 12 and width 5. What is its area?", "60"),
    ("What is 15% of 200?", "30"),
    ("If a train travels at 60 km/h for 2.5 hours, how far does it go?", "150"),
    ("What is 9 squared?", "81"),
    ("What is the sum of the first 5 positive integers?", "15"),
    ("A bag has 3 red and 7 blue marbles. How many marbles total?", "10"),
    ("What is 144 ÷ 12?", "12"),
    ("What is 2 to the power of 8?", "256"),
    ("What is 17 × 6?", "102"),
    ("A square has side length 9. What is its perimeter?", "36"),
    ("What is 1000 - 437?", "563"),
    ("What is 3/4 of 80?", "60"),
    ("What is 45 + 55?", "100"),
    ("If you have $50 and spend $18.50, how much is left?", "31.5"),
    ("What is the product of 11 and 13?", "143"),
]

_VAL_PROBLEMS = [
    ("What is 23 + 41?", "64"),
    ("What is 100 - 37?", "63"),
    ("What is 7 × 9?", "63"),
    ("A circle has diameter 10. What is its circumference? Use π ≈ 3.14.", "31.4"),
    ("What is 50% of 90?", "45"),
]


def _make_row(uid: str, question: str, answer: str) -> dict:
    return {
        "data_source": "debug",
        "prompt": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "ability": "math",
        "reward_model": {
            "ground_truth": answer,
            "style": "rule",
        },
        "extra_info": {
            "index": uid,
            "question": question,
            "solution": answer,
        },
    }


def main() -> None:
    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas not installed. Run: uv sync", file=sys.stderr)
        sys.exit(1)

    train_rows = [
        _make_row(f"debug_{i:04d}", q, a)
        for i, (q, a) in enumerate(_PROBLEMS)
    ]
    val_rows = [
        _make_row(f"debug_val_{i:04d}", q, a)
        for i, (q, a) in enumerate(_VAL_PROBLEMS)
    ]

    train_path = _REPO_ROOT / "datasets/unified/debug/tiny-math/train.parquet"
    val_path = _REPO_ROOT / "datasets/unified/debug/tiny-val/test.parquet"

    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(train_rows).to_parquet(train_path, index=False)
    pd.DataFrame(val_rows).to_parquet(val_path, index=False)

    print(f"Wrote {len(train_rows)} training rows: {train_path}")
    print(f"Wrote {len(val_rows)} validation rows: {val_path}")
    print("\nDataset created. Set in .env:")
    print('  ROPD_TRAIN_TASK="debug/tiny-math"')
    print('  ROPD_VAL_TASK="debug/tiny-val"')


if __name__ == "__main__":
    main()
