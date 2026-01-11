from f5_tts.model.backbones.dit import DiT
from f5_tts.model.backbones.mmdit import MMDiT
from f5_tts.model.backbones.unett import UNetT
from f5_tts.model.cfm import CFM
from f5_tts.model.trainer import Trainer


__all__ = ["CFM", "UNetT", "DiT", "MMDiT", "Trainer"]


# Lazy import for GRPOTrainer to avoid loading RL dependencies
def __getattr__(name):
    if name == "GRPOTrainer":
        from f5_tts.train.trainer_grpo import GRPOTrainer

        return GRPOTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
