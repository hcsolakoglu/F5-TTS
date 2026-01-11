"""
Reward combiner for aggregating multiple reward providers.
"""

from __future__ import annotations

from typing import Literal

import torch

from f5_tts.rewards.base import RewardInput, RewardOutput, RewardProvider


class RewardCombiner:
    """Combines multiple reward providers with configurable weights and modes.

    Supports different combination strategies:
    - 'sum': Weighted sum of rewards (default)
    - 'normalized_sum': Weighted sum normalized by total weight
    - 'rank': Rank-based reward shaping

    Example:
        providers = [asr_reward, spk_reward]
        weights = [1.0, 0.5]
        combiner = RewardCombiner(providers, weights, mode='sum')

        inputs = [RewardInput(audio=audio, text="hello")]
        outputs = combiner.compute(inputs)
    """

    def __init__(
        self,
        providers: list[RewardProvider],
        weights: list[float] | None = None,
        mode: Literal["sum", "normalized_sum", "rank"] = "sum",
        normalize: bool = False,
    ):
        """Initialize the reward combiner.

        Args:
            providers: List of reward providers to combine.
            weights: Optional list of weights for each provider.
                     Defaults to equal weights of 1.0.
            mode: Combination mode ('sum', 'normalized_sum', 'rank').
            normalize: Whether to normalize the final reward.
        """
        self.providers = providers
        self.weights = weights or [1.0] * len(providers)
        self.mode = mode
        self.normalize = normalize

        if len(self.weights) != len(self.providers):
            raise ValueError(
                f"Number of weights ({len(self.weights)}) must match "
                f"number of providers ({len(self.providers)})"
            )

    def compute(self, batch: list[RewardInput]) -> list[RewardOutput]:
        """Compute combined rewards for a batch of inputs.

        Args:
            batch: List of RewardInput objects.

        Returns:
            List of combined RewardOutput objects.
        """
        if not batch:
            return []

        # Collect rewards from all providers
        all_provider_outputs: list[list[RewardOutput]] = []
        for provider in self.providers:
            provider_outputs = provider.compute(batch)
            all_provider_outputs.append(provider_outputs)

        # Combine rewards for each input
        combined_outputs = []
        for i in range(len(batch)):
            combined = self._combine_rewards(
                [outputs[i] for outputs in all_provider_outputs]
            )
            combined_outputs.append(combined)

        return combined_outputs

    def _combine_rewards(
        self, provider_outputs: list[RewardOutput]
    ) -> RewardOutput:
        """Combine rewards from multiple providers for a single input.

        Args:
            provider_outputs: List of RewardOutput from each provider.

        Returns:
            Combined RewardOutput.
        """
        # Collect all component rewards
        all_components: dict[str, torch.Tensor] = {}
        all_logs: dict[str, any] = {}

        weighted_rewards = []
        for output, weight, provider in zip(
            provider_outputs, self.weights, self.providers
        ):
            # Add weighted reward
            weighted_rewards.append(weight * output.total_reward)

            # Merge components with provider prefix
            for name, value in output.components.items():
                key = f"{provider.name}/{name}"
                all_components[key] = value

            # Merge logs with provider prefix
            for name, value in output.logs.items():
                key = f"{provider.name}/{name}"
                all_logs[key] = value

            # Also store the provider's total reward as a component
            all_components[f"{provider.name}/total"] = output.total_reward

        # Combine based on mode
        if self.mode == "sum":
            total_reward = sum(weighted_rewards)
        elif self.mode == "normalized_sum":
            total_weight = sum(self.weights)
            total_reward = sum(weighted_rewards) / max(total_weight, 1e-8)
        elif self.mode == "rank":
            total_reward = self._rank_based_combine(weighted_rewards)
        else:
            raise ValueError(f"Unknown combination mode: {self.mode}")

        # Optionally normalize
        if self.normalize and self.mode != "normalized_sum":
            total_weight = sum(self.weights)
            total_reward = total_reward / max(total_weight, 1e-8)

        return RewardOutput(
            total_reward=total_reward,
            components=all_components,
            logs=all_logs,
        )

    def _rank_based_combine(
        self, weighted_rewards: list[torch.Tensor]
    ) -> torch.Tensor:
        """Combine rewards using rank-based shaping.

        This method ranks the rewards and uses the rank as a weight,
        which can help with reward scale differences.

        Args:
            weighted_rewards: List of weighted reward tensors.

        Returns:
            Combined reward tensor.
        """
        # Stack rewards
        stacked = torch.stack(weighted_rewards)

        # Compute ranks (higher reward = higher rank)
        ranks = torch.argsort(torch.argsort(stacked)) + 1

        # Weight by rank
        rank_weights = ranks.float() / ranks.float().sum()

        # Compute rank-weighted sum
        return (stacked * rank_weights).sum()

    @classmethod
    def from_config(
        cls,
        reward_configs: list[dict],
        combine_config: dict | None = None,
    ) -> "RewardCombiner":
        """Create a RewardCombiner from configuration.

        Args:
            reward_configs: List of reward provider configurations.
            combine_config: Optional combiner configuration with keys:
                - mode: Combination mode
                - normalize: Whether to normalize

        Returns:
            Configured RewardCombiner instance.

        Example:
            reward_configs = [
                {
                    "name": "asr_wer",
                    "provider": "f5_tts.rewards.providers.funasr_sensevoice:FunASRWerReward",
                    "weight": 1.0,
                    "cfg": {"model_id": "iic/SenseVoiceSmall"}
                },
                {
                    "name": "spk_sim",
                    "provider": "f5_tts.rewards.providers.wespeaker:WeSpeakerSimReward",
                    "weight": 1.0,
                    "cfg": {}
                }
            ]
            combine_config = {"mode": "sum", "normalize": True}
            combiner = RewardCombiner.from_config(reward_configs, combine_config)
        """
        from f5_tts.rewards.registry import RewardRegistry

        providers = []
        weights = []

        for cfg in reward_configs:
            provider = RewardRegistry.create_from_config(cfg)
            providers.append(provider)
            weights.append(cfg.get("weight", 1.0))

        combine_config = combine_config or {}
        return cls(
            providers=providers,
            weights=weights,
            mode=combine_config.get("mode", "sum"),
            normalize=combine_config.get("normalize", False),
        )
