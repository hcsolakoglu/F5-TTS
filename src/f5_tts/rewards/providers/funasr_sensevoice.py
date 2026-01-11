"""
FunASR-based ASR reward provider for computing WER (Word Error Rate).

This provider uses FunASR's SenseVoice model to transcribe generated audio
and compute WER against the target text, converting it to a reward signal.
"""

from __future__ import annotations

import string
from typing import Any

import torch

from f5_tts.rewards.base import RewardInput, RewardOutput, RewardProvider
from f5_tts.rewards.utils import get_device, resample_audio


class FunASRWerReward(RewardProvider):
    """ASR-based WER reward using FunASR SenseVoice model.

    Computes Word Error Rate between generated audio transcription
    and target text, and converts it to a reward signal:
        reward = 1.0 - WER

    Attributes:
        model_id: FunASR model ID or path.
        device: Device for inference.
        lang: Language code ('zh', 'en', 'auto').
    """

    name = "asr_wer"
    required_extras = ["reward_funasr"]

    def __init__(self):
        """Initialize the FunASR WER reward provider."""
        super().__init__()
        self.model = None
        self.model_id = "iic/SenseVoiceSmall"
        self.cache_dir = None
        self.device = "auto"
        self.lang = "auto"
        self._target_sample_rate = 16000

    def setup(self, cfg: dict[str, Any] | None = None) -> None:
        """Setup the ASR model.

        Args:
            cfg: Configuration dictionary. Supports:
                - model_id: FunASR model ID (default: "iic/SenseVoiceSmall")
                - cache_dir: Directory for caching model weights
                - device: Device for inference ('auto', 'cpu', 'cuda')
                - lang: Language code ('zh', 'en', 'auto')
        """
        super().setup(cfg)
        cfg = cfg or {}

        self.model_id = cfg.get("model_id", self.model_id)
        self.cache_dir = cfg.get("cache_dir", self.cache_dir)
        self.device = cfg.get("device", self.device)
        self.lang = cfg.get("lang", self.lang)

        # Lazy import and model loading
        self._load_model()

    def _check_dependencies(self) -> None:
        """Check if FunASR is installed."""
        try:
            import funasr  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "FunASR is required for FunASRWerReward. "
                "Install it with: pip install 'f5-tts[reward_funasr]' "
                "or: pip install funasr"
            ) from e

    def _load_model(self) -> None:
        """Load the FunASR model."""
        self._check_dependencies()

        from funasr import AutoModel

        device = get_device(self.device)

        model_kwargs = {
            "model": self.model_id,
            "disable_update": True,
        }
        if self.cache_dir:
            model_kwargs["model"] = self.cache_dir

        self.model = AutoModel(**model_kwargs)
        self._device = device

    def compute(self, batch: list[RewardInput]) -> list[RewardOutput]:
        """Compute WER-based rewards for the batch.

        Args:
            batch: List of RewardInput objects.

        Returns:
            List of RewardOutput objects with WER-based rewards.
        """
        if self.model is None:
            self._load_model()

        outputs = []
        for inp in batch:
            try:
                wer, hypo = self._compute_wer(inp)
                reward = torch.tensor(1.0 - wer, device=inp.audio.device)

                outputs.append(
                    RewardOutput(
                        total_reward=reward,
                        components={
                            "wer": torch.tensor(wer, device=inp.audio.device),
                            "accuracy": reward,
                        },
                        logs={
                            "hypothesis": hypo,
                            "reference": inp.text,
                        },
                    )
                )
            except Exception as e:
                # On error, return zero reward
                outputs.append(
                    RewardOutput(
                        total_reward=torch.tensor(0.0, device=inp.audio.device),
                        components={
                            "wer": torch.tensor(1.0, device=inp.audio.device),
                            "accuracy": torch.tensor(0.0, device=inp.audio.device),
                        },
                        logs={"error": str(e)},
                    )
                )

        return outputs

    def _compute_wer(self, inp: RewardInput) -> tuple[float, str]:
        """Compute WER for a single input.

        Args:
            inp: RewardInput object.

        Returns:
            Tuple of (WER value, hypothesis text).
        """
        # Resample audio to model's expected sample rate
        audio = resample_audio(
            inp.audio, inp.sample_rate, self._target_sample_rate
        )

        # Convert to numpy for FunASR
        audio_np = audio.detach().cpu().numpy()
        if audio_np.ndim == 2:
            audio_np = audio_np.squeeze(0)

        # Run ASR
        result = self.model.generate(input=audio_np, batch_size_s=300, disable_pbar=True)
        hypo = result[0]["text"] if result else ""

        # Detect language and preprocess
        lang = self._detect_language(inp.text) if self.lang == "auto" else self.lang

        # Normalize text for WER computation
        truth = self._normalize_text(inp.text, lang)
        hypo = self._normalize_text(hypo, lang)

        # Compute WER
        wer = self._calculate_wer(truth, hypo, lang)

        return wer, hypo

    def _detect_language(self, text: str) -> str:
        """Detect language from text.

        Args:
            text: Text to analyze.

        Returns:
            Language code ('zh' or 'en').
        """
        # Simple heuristic: if contains CJK characters, assume Chinese
        for char in text:
            if "\u4e00" <= char <= "\u9fff":
                return "zh"
        return "en"

    def _normalize_text(self, text: str, lang: str) -> str:
        """Normalize text for WER computation.

        Args:
            text: Text to normalize.
            lang: Language code.

        Returns:
            Normalized text.
        """
        # Remove punctuation
        try:
            from zhon.hanzi import punctuation as zh_punctuation

            all_punctuation = zh_punctuation + string.punctuation
        except ImportError:
            all_punctuation = string.punctuation

        for p in all_punctuation:
            text = text.replace(p, "")

        # Normalize whitespace
        text = " ".join(text.split())

        if lang == "zh":
            # Convert to simplified Chinese if zhconv available
            try:
                import zhconv

                text = zhconv.convert(text, "zh-cn")
            except ImportError:
                pass
            # Character-level tokenization for Chinese
            text = " ".join(list(text.replace(" ", "")))
        else:
            # Lowercase for English
            text = text.lower()

        return text

    def _calculate_wer(self, reference: str, hypothesis: str, lang: str) -> float:
        """Calculate Word Error Rate.

        Args:
            reference: Reference text.
            hypothesis: Hypothesis text.
            lang: Language code.

        Returns:
            WER value between 0 and 1.
        """
        try:
            from jiwer import compute_measures

            measures = compute_measures(reference, hypothesis)
            return min(measures["wer"], 1.0)  # Cap at 1.0
        except ImportError:
            # Fallback to simple character error rate
            return self._simple_cer(reference, hypothesis)

    def _simple_cer(self, reference: str, hypothesis: str) -> float:
        """Simple character error rate as fallback.

        Args:
            reference: Reference text.
            hypothesis: Hypothesis text.

        Returns:
            Character error rate.
        """
        ref_chars = list(reference.replace(" ", ""))
        hyp_chars = list(hypothesis.replace(" ", ""))

        if not ref_chars:
            return 1.0 if hyp_chars else 0.0

        # Levenshtein distance
        m, n = len(ref_chars), len(hyp_chars)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_chars[i - 1] == hyp_chars[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        return min(dp[m][n] / m, 1.0)

    def supports_language(self, lang: str) -> bool:
        """Check if a language is supported.

        SenseVoice supports Chinese and English.

        Args:
            lang: Language code.

        Returns:
            True if supported.
        """
        return lang.lower() in ["zh", "en", "auto"]
