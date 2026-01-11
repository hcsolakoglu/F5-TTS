"""
Base classes and dataclasses for the reward plugin system.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class RewardInput:
    """Input data for reward computation.

    Attributes:
        audio: Audio waveform tensor of shape (samples,) or (channels, samples).
        text: Target text that the audio should represent.
        speaker_ref: Optional reference audio for speaker similarity computation.
        sample_rate: Sample rate of the audio in Hz.
        meta: Optional metadata dictionary for additional information.
    """

    audio: torch.Tensor
    text: str
    speaker_ref: torch.Tensor | None = None
    sample_rate: int = 24000
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class RewardOutput:
    """Output from reward computation.

    Attributes:
        total_reward: Combined reward value as a tensor.
        components: Dictionary mapping component names to their reward values.
        logs: Optional dictionary for logging additional information.
    """

    total_reward: torch.Tensor
    components: dict[str, torch.Tensor] = field(default_factory=dict)
    logs: dict[str, Any] = field(default_factory=dict)


class RewardProvider(ABC):
    """Base class for reward providers.

    All reward providers must inherit from this class and implement the
    required abstract methods.

    Attributes:
        name: Unique identifier for this reward provider.
        required_extras: List of extra package groups required for this provider.
    """

    name: str = "base"
    required_extras: list[str] = []

    def __init__(self):
        """Initialize the reward provider."""
        self._is_setup = False

    def setup(self, cfg: dict[str, Any] | None = None) -> None:
        """Optional setup method called before first use.

        Override this method to perform lazy initialization of models,
        download weights, or configure the provider.

        Args:
            cfg: Configuration dictionary for the provider.
        """
        self._is_setup = True

    @abstractmethod
    def compute(self, batch: list[RewardInput]) -> list[RewardOutput]:
        """Compute rewards for a batch of inputs.

        Args:
            batch: List of RewardInput objects to compute rewards for.

        Returns:
            List of RewardOutput objects, one per input.

        Raises:
            RuntimeError: If required dependencies are not installed.
        """
        pass

    def supports_language(self, lang: str) -> bool:
        """Check if this provider supports a specific language.

        Args:
            lang: Language code (e.g., 'en', 'zh').

        Returns:
            True if the language is supported, False otherwise.
        """
        return True

    def _check_dependencies(self) -> None:
        """Check if required dependencies are installed.

        Raises:
            RuntimeError: If required dependencies are missing.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
