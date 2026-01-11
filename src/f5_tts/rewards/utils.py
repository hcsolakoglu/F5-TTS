"""
Utility functions for the reward system.
"""

from __future__ import annotations

import hashlib
from typing import Any

import torch


def compute_audio_hash(audio: torch.Tensor, sample_rate: int = 24000) -> str:
    """Compute a stable hash for an audio tensor.

    This can be used for caching reward computations on the same audio.

    Args:
        audio: Audio waveform tensor.
        sample_rate: Sample rate (included in hash for uniqueness).

    Returns:
        Hex string hash of the audio content.
    """
    # Convert to numpy bytes for hashing
    audio_bytes = audio.detach().cpu().numpy().tobytes()
    sr_bytes = str(sample_rate).encode()

    hasher = hashlib.sha256()
    hasher.update(audio_bytes)
    hasher.update(sr_bytes)

    return hasher.hexdigest()


def resample_audio(
    audio: torch.Tensor,
    orig_sr: int,
    target_sr: int,
) -> torch.Tensor:
    """Resample audio to a target sample rate.

    Uses lazy import of torchaudio for resampling.

    Args:
        audio: Audio waveform tensor.
        orig_sr: Original sample rate.
        target_sr: Target sample rate.

    Returns:
        Resampled audio tensor.
    """
    if orig_sr == target_sr:
        return audio

    import torchaudio

    resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
    resampler = resampler.to(audio.device)

    return resampler(audio)


def normalize_audio(
    audio: torch.Tensor,
    target_db: float = -20.0,
) -> torch.Tensor:
    """Normalize audio to a target dB level.

    Args:
        audio: Audio waveform tensor.
        target_db: Target dB level (default: -20 dB).

    Returns:
        Normalized audio tensor.
    """
    # Compute current RMS
    rms = torch.sqrt(torch.mean(audio ** 2) + 1e-8)

    # Convert target dB to linear
    target_rms = 10 ** (target_db / 20)

    # Scale audio
    return audio * (target_rms / rms)


def check_package_installed(package_name: str) -> bool:
    """Check if a package is installed.

    Args:
        package_name: Name of the package to check.

    Returns:
        True if installed, False otherwise.
    """
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False


def get_device(device: str | torch.device = "auto") -> torch.device:
    """Get the appropriate device.

    Args:
        device: Device specification. Can be 'auto', 'cpu', 'cuda',
                'cuda:0', etc.

    Returns:
        torch.device object.
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class RewardCache:
    """Simple in-memory cache for reward computations.

    Uses audio hashes as keys to avoid recomputing rewards
    for the same audio.

    Example:
        cache = RewardCache(max_size=1000)

        # Try to get cached result
        result = cache.get(audio_hash, provider_name)
        if result is None:
            result = compute_reward(audio)
            cache.set(audio_hash, provider_name, result)
    """

    def __init__(self, max_size: int = 1000):
        """Initialize the cache.

        Args:
            max_size: Maximum number of entries to cache.
        """
        self.max_size = max_size
        self._cache: dict[str, Any] = {}
        self._access_order: list[str] = []

    def _make_key(self, audio_hash: str, provider_name: str) -> str:
        """Create a cache key from audio hash and provider name."""
        return f"{provider_name}:{audio_hash}"

    def get(self, audio_hash: str, provider_name: str) -> Any | None:
        """Get a cached result.

        Args:
            audio_hash: Hash of the audio.
            provider_name: Name of the provider.

        Returns:
            Cached result or None if not found.
        """
        key = self._make_key(audio_hash, provider_name)
        return self._cache.get(key)

    def set(self, audio_hash: str, provider_name: str, value: Any) -> None:
        """Set a cached result.

        Args:
            audio_hash: Hash of the audio.
            provider_name: Name of the provider.
            value: Value to cache.
        """
        key = self._make_key(audio_hash, provider_name)

        # Evict oldest entries if at capacity
        while len(self._cache) >= self.max_size and self._access_order:
            oldest_key = self._access_order.pop(0)
            self._cache.pop(oldest_key, None)

        self._cache[key] = value
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._access_order.clear()

    def __len__(self) -> int:
        """Return the number of cached entries."""
        return len(self._cache)
