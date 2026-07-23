"""Two-process CPU/gloo probe for Trainer global masked-mean gradient scaling.

Launched by ``test_real_two_rank_ddp_global_masked_mean_matches_reference``.
Kept outside pytest collection so each torchrun rank executes exactly one probe.
"""

from __future__ import annotations

import copy
import sys

import torch
import torch.distributed as dist
from accelerate import Accelerator

from f5_tts.model.trainer import Trainer


def _batch(rank: int, microbatch: int):
    generator = torch.Generator().manual_seed(10_000 + rank * 100 + microbatch)
    frames = ((9, 5), (7, 4))[rank][microbatch]
    masked = ((8, 3), (5, 2))[rank][microbatch]
    x = torch.randn(1, frames, 4, generator=generator)
    target = torch.randn(1, frames, 4, generator=generator)
    mask = torch.zeros(1, frames, dtype=torch.bool)
    mask[:, :masked] = True
    return x, target, mask


def _loss_components(model, x, target, mask):
    prediction = model(x)
    squared_error = torch.nn.functional.mse_loss(prediction, target, reduction="none")
    loss_mask = mask[..., None].to(squared_error.dtype)
    loss_sum = (squared_error * loss_mask).sum()
    denominator = (loss_mask.sum() * squared_error.shape[-1]).clamp(min=1.0)
    return loss_sum, denominator


def main():
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2)
    assert accelerator.num_processes == 2

    torch.manual_seed(7)
    model = torch.nn.Linear(4, 4, bias=False)
    initial_state = copy.deepcopy(model.state_dict())
    model = accelerator.prepare(model)

    trainer = Trainer.__new__(Trainer)
    trainer.model = model
    trainer.accelerator = accelerator
    trainer.grad_accumulation_steps = 2

    local_denominator = torch.zeros((), device=accelerator.device)
    for microbatch in range(2):
        x, target, mask = (tensor.to(accelerator.device) for tensor in _batch(accelerator.process_index, microbatch))
        with accelerator.accumulate(model):
            loss_sum, denominator = _loss_components(model, x, target, mask)
            local_denominator = local_denominator + denominator.detach()
            accelerator.backward(loss_sum)
            if accelerator.sync_gradients:
                global_denominator = trainer._global_loss_denom(local_denominator)
                trainer._scale_gradients_by_loss_denom(global_denominator)

    actual = accelerator.unwrap_model(model).weight.grad
    assert actual is not None

    reference = torch.nn.Linear(4, 4, bias=False).to(accelerator.device)
    reference.load_state_dict(initial_state)
    total_loss_sum = torch.zeros((), device=accelerator.device)
    total_denominator = torch.zeros((), device=accelerator.device)
    for rank in range(2):
        for microbatch in range(2):
            x, target, mask = (tensor.to(accelerator.device) for tensor in _batch(rank, microbatch))
            loss_sum, denominator = _loss_components(reference, x, target, mask)
            total_loss_sum = total_loss_sum + loss_sum
            total_denominator = total_denominator + denominator.detach()
    (total_loss_sum / total_denominator).backward()

    assert reference.weight.grad is not None
    torch.testing.assert_close(actual, reference.weight.grad, atol=1e-6, rtol=1e-6)

    gathered = accelerator.gather(actual.detach())
    if accelerator.is_main_process:
        rank_gradients = gathered.reshape(2, *actual.shape)
        torch.testing.assert_close(rank_gradients[0], rank_gradients[1], atol=0.0, rtol=0.0)
        print("REAL_DDP_GLOBAL_MASKED_MEAN_OK", flush=True)

    accelerator.wait_for_everyone()
    if dist.is_initialized():
        dist.destroy_process_group()


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
    if "--invalid-compile-backend" in sys.argv:
        invalid_compile_backend_main()
    else:
        main()
