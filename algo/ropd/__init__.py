from .client import (
    ROPD_BATCH_SCHEMA_VERSION,
    RopdAnswerScore,
    RopdJudgeConfig,
    RopdVerifierScores,
    build_ropd_clients,
    build_ropd_judge_config,
    parse_ropd_scores,
)
from .prompts import (
    build_ropd_rubricator_prompt,
    build_ropd_verifier_prompt,
)

# RopdRewardManager is loaded lazily to avoid pulling in verl/Ray when only
# the client or prompt utilities are imported (e.g. provider tests, scripts).
# `from algo.ropd import RopdRewardManager` still works via __getattr__.
# Training loads it directly from algo.ropd.reward_manager via importlib.
def __getattr__(name: str):
    if name == "RopdRewardManager":
        from .reward_manager import RopdRewardManager
        return RopdRewardManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ROPD_BATCH_SCHEMA_VERSION",
    "RopdAnswerScore",
    "RopdJudgeConfig",
    "RopdRewardManager",
    "RopdVerifierScores",
    "build_ropd_clients",
    "build_ropd_judge_config",
    "build_ropd_rubricator_prompt",
    "build_ropd_verifier_prompt",
    "parse_ropd_scores",
]
