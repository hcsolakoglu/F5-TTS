"""
Dummy reward provider for testing.
"""

from __future__ import annotations

from typing import Any

import torch

from f5_tts.rewards.base import RewardInput, RewardOutput, RewardProvider


class DummyRewardProvider(RewardProvider):
    """Dummy reward provider that returns a fixed reward.

    Useful for testing the RL training pipeline without
    loading actual reward models.

    Attributes:
        fixed_reward: The fixed reward value to return.
    """

    name = "dummy"
    required_extras: list[str] = []

    def __init__(self, fixed_reward: float = 1.0):
        """Initialize the dummy provider.

        Args:
            fixed_reward: The fixed reward value to return.
        """
        super().__init__()
        self.fixed_reward = fixed_reward

    def setup(self, cfg: dict[str, Any] | None = None) -> None:
        """Setup the provider with configuration.

        Args:
            cfg: Configuration dictionary. Supports:
                - fixed_reward: Override the fixed reward value.
        """
        super().setup(cfg)
        if cfg and "fixed_reward" in cfg:
            self.fixed_reward = cfg["fixed_reward"]

    def compute(self, batch: list[RewardInput]) -> list[RewardOutput]:
        """Return fixed rewards for the batch.

        Args:
            batch: List of RewardInput objects.

        Returns:
            List of RewardOutput objects with fixed rewards.
        """
        outputs = []
        for inp in batch:
            # Determine device from input audio
            device = inp.audio.device if inp.audio is not None else "cpu"

            reward = torch.tensor(self.fixed_reward, device=device)
            outputs.append(
                RewardOutput(
                    total_reward=reward,
                    components={"fixed": reward},
                    logs={"text_length": len(inp.text) if inp.text else 0},
                )
            )
        return outputs
