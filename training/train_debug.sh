#!/usr/bin/env bash
# Tiny debug training run — 10 PPO steps, batch=4, no wandb.
#
# Uses the debug dataset and teacher index created by:
#   python scripts/create_debug_dataset.py
#   python scripts/build_debug_teacher_index.py
#
# Usage:
#   bash training/train_debug.sh [extra hydra overrides...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# WSL2 cgroups don't expose memory stats; disable Ray's OOM killer to prevent
# false-positive Raylet kills from the "negative cgroup memory" misread.
export RAY_memory_monitor_refresh_ms=0

exec bash "$SCRIPT_DIR/train.sh" \
    trainer.total_training_steps=10 \
    trainer.test_freq=5 \
    trainer.save_freq=10 \
    "trainer.logger=[console]" \
    data.train_batch_size=4 \
    data.val_batch_size=2 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    'reward_model.reward_kwargs.ropd.teacher_answer_count=1' \
    'reward_model.reward_kwargs.ropd.max_concurrency=2' \
    actor_rollout_ref.model.path=/home/degirum/ROPD_official_chloe/models/Qwen3-0.6B \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.40 \
    "$@"
