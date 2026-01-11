"""
WeSpeaker-based speaker similarity reward provider.

This provider computes speaker embedding similarity between
generated audio and a reference speaker audio.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from f5_tts.rewards.base import RewardInput, RewardOutput, RewardProvider
from f5_tts.rewards.utils import get_device, resample_audio


class WeSpeakerSimReward(RewardProvider):
    """Speaker similarity reward using WeSpeaker embeddings.

    Computes cosine similarity between speaker embeddings of
    generated audio and reference audio.

    Attributes:
        model_dir: Path to WeSpeaker model directory.
        device: Device for inference.
    """

    name = "spk_sim"
    required_extras = ["reward_wespeaker"]

    def __init__(self):
        """Initialize the WeSpeaker similarity reward provider."""
        super().__init__()
        self.model = None
        self.model_dir = None
        self.device = "auto"
        self._target_sample_rate = 16000

    def setup(self, cfg: dict[str, Any] | None = None) -> None:
        """Setup the speaker embedding model.

        Args:
            cfg: Configuration dictionary. Supports:
                - model_dir: Path to WeSpeaker model directory
                - device: Device for inference ('auto', 'cpu', 'cuda')
        """
        super().setup(cfg)
        cfg = cfg or {}

        self.model_dir = cfg.get("model_dir", self.model_dir)
        self.device = cfg.get("device", self.device)

        # Lazy model loading
        self._load_model()

    def _check_dependencies(self) -> None:
        """Check if required dependencies are installed."""
        # Check for the ECAPA-TDNN model used in evaluation
        try:
            from f5_tts.eval.ecapa_tdnn import ECAPA_TDNN_SMALL  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "ECAPA-TDNN model is required for WeSpeakerSimReward. "
                "This should be available in the f5_tts.eval module."
            ) from e

    def _load_model(self) -> None:
        """Load the speaker embedding model."""
        self._check_dependencies()

        from f5_tts.eval.ecapa_tdnn import ECAPA_TDNN_SMALL

        device = get_device(self.device)

        self.model = ECAPA_TDNN_SMALL(
            feat_dim=1024,
            feat_type="wavlm_large",
            config_path=None,
        )

        # Load pretrained weights if model_dir is specified
        if self.model_dir:
            import os

            ckpt_path = self.model_dir
            if os.path.isdir(ckpt_path):
                # Try to find checkpoint file in directory
                for name in ["model.pt", "model.pth", "checkpoint.pt"]:
                    candidate = os.path.join(ckpt_path, name)
                    if os.path.exists(candidate):
                        ckpt_path = candidate
                        break

            if os.path.isfile(ckpt_path):
                state_dict = torch.load(
                    ckpt_path, weights_only=True, map_location="cpu"
                )
                if "model" in state_dict:
                    state_dict = state_dict["model"]
                self.model.load_state_dict(state_dict, strict=False)

        self.model = self.model.to(device)
        self.model.eval()
        self._device = device

    def compute(self, batch: list[RewardInput]) -> list[RewardOutput]:
        """Compute speaker similarity rewards for the batch.

        Args:
            batch: List of RewardInput objects. Each should have
                   a speaker_ref for comparison.

        Returns:
            List of RewardOutput objects with similarity rewards.
        """
        if self.model is None:
            self._load_model()

        outputs = []
        for inp in batch:
            try:
                if inp.speaker_ref is None:
                    # No reference, return neutral reward
                    reward = torch.tensor(0.5, device=inp.audio.device)
                    outputs.append(
                        RewardOutput(
                            total_reward=reward,
                            components={"similarity": reward},
                            logs={"warning": "No speaker reference provided"},
                        )
                    )
                    continue

                sim = self._compute_similarity(inp)
                # Convert similarity from [-1, 1] to [0, 1] reward
                reward = (sim + 1.0) / 2.0

                outputs.append(
                    RewardOutput(
                        total_reward=reward,
                        components={
                            "similarity": sim,
                            "reward": reward,
                        },
                        logs={},
                    )
                )
            except Exception as e:
                # On error, return neutral reward
                outputs.append(
                    RewardOutput(
                        total_reward=torch.tensor(0.0, device=inp.audio.device),
                        components={
                            "similarity": torch.tensor(0.0, device=inp.audio.device),
                        },
                        logs={"error": str(e)},
                    )
                )

        return outputs

    def _compute_similarity(self, inp: RewardInput) -> torch.Tensor:
        """Compute speaker similarity for a single input.

        Args:
            inp: RewardInput with audio and speaker_ref.

        Returns:
            Cosine similarity tensor.
        """
        # Resample both audios to model's expected sample rate
        gen_audio = resample_audio(
            inp.audio, inp.sample_rate, self._target_sample_rate
        )
        ref_audio = resample_audio(
            inp.speaker_ref, inp.sample_rate, self._target_sample_rate
        )

        # Ensure correct shape [batch, samples]
        if gen_audio.ndim == 1:
            gen_audio = gen_audio.unsqueeze(0)
        if ref_audio.ndim == 1:
            ref_audio = ref_audio.unsqueeze(0)

        # Move to device
        gen_audio = gen_audio.to(self._device)
        ref_audio = ref_audio.to(self._device)

        # Get embeddings
        with torch.no_grad():
            gen_emb = self.model(gen_audio)
            ref_emb = self.model(ref_audio)

        # Compute cosine similarity
        sim = F.cosine_similarity(gen_emb, ref_emb, dim=-1)

        return sim.squeeze()

    def supports_language(self, lang: str) -> bool:
        """Check if a language is supported.

        Speaker embeddings are language-agnostic.

        Args:
            lang: Language code.

        Returns:
            Always True.
        """
        return True
