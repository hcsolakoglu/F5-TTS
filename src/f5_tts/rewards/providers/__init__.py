"""
Reward providers for RL training.

This module contains implementations of various reward providers:
- FunASR-based ASR WER reward
- WeSpeaker-based speaker similarity reward

Each provider uses lazy imports to avoid loading dependencies
when the provider is not used.
"""

from f5_tts.rewards.providers.dummy import DummyRewardProvider

__all__ = [
    "DummyRewardProvider",
]

# Conditionally export FunASR and WeSpeaker providers
# They will be imported lazily when needed
