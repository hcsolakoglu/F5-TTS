"""
Reward plugin system for RL training in F5-TTS.

This module provides a flexible, extensible reward system for reinforcement learning
fine-tuning of TTS models. It supports multiple reward providers that can be combined
with configurable weights.

Example usage:
    from f5_tts.rewards import RewardRegistry, RewardCombiner, RewardInput

    # Create reward providers from config
    providers = [
        RewardRegistry.create_from_config({
            "name": "asr_wer",
            "provider": "f5_tts.rewards.providers.funasr_sensevoice:FunASRWerReward",
            "weight": 1.0,
            "cfg": {"model_id": "iic/SenseVoiceSmall"}
        })
    ]

    # Combine rewards
    combiner = RewardCombiner(providers, weights=[1.0], mode="sum")

    # Compute rewards
    inputs = [RewardInput(audio=audio_tensor, text="hello", sample_rate=24000)]
    outputs = combiner.compute(inputs)
"""

from f5_tts.rewards.base import (
    RewardInput,
    RewardOutput,
    RewardProvider,
)
from f5_tts.rewards.registry import RewardRegistry
from f5_tts.rewards.combiner import RewardCombiner

__all__ = [
    "RewardInput",
    "RewardOutput",
    "RewardProvider",
    "RewardRegistry",
    "RewardCombiner",
]
