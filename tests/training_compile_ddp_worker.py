"""Two-process CPU/gloo probe for strict compile-setup failure under DDP.

Launched by ``test_real_two_rank_invalid_compile_backend_setup_raises_without_hanging``.
Kept outside pytest collection so each torchrun rank executes exactly one probe.
"""

from __future__ import annotations

import sys

import torch
import torch.distributed as dist
from accelerate import Accelerator

from f5_tts.model.trainer import Trainer


class _Compilable(torch.nn.Module):
    def forward(self, value):
        return value + 1

    def compile_training_core(self, **compile_kwargs):
        compile_kwargs.pop("target")
        compile_kwargs.pop("runtime_fallback")
        return torch.compile(self.forward, **compile_kwargs)

    def clear_training_compile(self):
        return None


def invalid_compile_backend_main():
    """Prove strict invalid compile setup raises coherently after the rank collective."""
    accelerator = Accelerator(cpu=True)
    assert accelerator.num_processes == 2

    trainer = Trainer.__new__(Trainer)
    trainer.accelerator = accelerator
    trainer.compile_enabled = True
    trainer.compile_backend = "definitely-not-a-real-backend"
    trainer.compile_target = "cfm_loss_core"
    trainer.compile_mode = None
    trainer.compile_fullgraph = False
    trainer.compile_dynamic = True
    trainer.compile_fallback_to_eager = False
    trainer.compile_active = False
    trainer.compile_fallback_active = False
    trainer._unwrapped_model = _Compilable()

    caught = False
    try:
        trainer._configure_compile()
    except Exception as exc:
        caught = "Invalid backend" in str(exc)

    caught_by_rank = accelerator.gather(torch.tensor([int(caught)], device=accelerator.device))
    if accelerator.is_main_process:
        assert caught_by_rank.tolist() == [1, 1]
        print("REAL_DDP_INVALID_BACKEND_OK", flush=True)

    accelerator.wait_for_everyone()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    if "--invalid-compile-backend" not in sys.argv:
        raise SystemExit("this worker only supports --invalid-compile-backend")
    invalid_compile_backend_main()
