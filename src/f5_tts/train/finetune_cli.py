import argparse
import os
import shutil
from importlib.resources import files

from cached_path import cached_path

from f5_tts.model import CFM, DiT, Trainer, UNetT
from f5_tts.model.dataset import load_dataset
from f5_tts.model.utils import get_tokenizer


# -------------------------- Dataset Settings --------------------------- #
target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"  # 'vocos' or 'bigvgan'


# -------------------------- Argument Parsing --------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="Train CFM Model")

    parser.add_argument(
        "--exp_name",
        type=str,
        default="F5TTS_v1_Base",
        choices=["F5TTS_v1_Base", "F5TTS_Base", "E2TTS_Base"],
        help="Experiment name",
    )
    parser.add_argument("--dataset_name", type=str, default="Emilia_ZH_EN", help="Name of the dataset to use")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--batch_size_per_gpu", type=int, default=3200, help="Batch size per GPU")
    parser.add_argument(
        "--batch_size_type", type=str, default="frame", choices=["frame", "sample"], help="Batch size type"
    )
    parser.add_argument("--max_samples", type=int, default=64, help="Max sequences per batch")
    parser.add_argument("--grad_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--num_warmup_updates", type=int, default=20000, help="Warmup updates")
    parser.add_argument("--save_per_updates", type=int, default=50000, help="Save checkpoint every N updates")
    parser.add_argument(
        "--keep_last_n_checkpoints",
        type=int,
        default=-1,
        help="-1 to keep all, 0 to not save intermediate, > 0 to keep last N checkpoints",
    )
    parser.add_argument("--last_per_updates", type=int, default=5000, help="Save last checkpoint every N updates")
    parser.add_argument("--finetune", action="store_true", help="Use Finetune")
    parser.add_argument("--pretrain", type=str, default=None, help="the path to the checkpoint")
    parser.add_argument(
        "--tokenizer", type=str, default="pinyin", choices=["pinyin", "char", "custom"], help="Tokenizer type"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to custom tokenizer vocab file (only used if tokenizer = 'custom')",
    )
    parser.add_argument(
        "--log_samples",
        action="store_true",
        help="Log inferenced samples per ckpt save updates",
    )
    parser.add_argument("--logger", type=str, default=None, choices=[None, "wandb", "tensorboard"], help="logger")
    parser.add_argument(
        "--bnb_optimizer",
        action="store_true",
        help="Use 8-bit Adam optimizer from bitsandbytes",
    )
    # torch.compile options (optional, default-off)
    parser.add_argument("--compile_enabled", action="store_true", help="Enable optional torch.compile training path")
    parser.add_argument("--compile_backend", type=str, default="inductor", help="torch.compile backend")
    parser.add_argument(
        "--compile_target",
        type=str,
        default="auto",
        choices=["auto", "cfm_loss_core", "dit_blocks"],
        help="torch.compile training target (auto chooses a model-compatible target)",
    )
    parser.add_argument(
        "--compile_mode", type=str, default=None, help="torch.compile mode (e.g. default, reduce-overhead)"
    )
    parser.add_argument("--compile_fullgraph", action="store_true", help="Require fullgraph torch.compile")
    parser.add_argument(
        "--compile_dynamic",
        type=str,
        default="default",
        choices=["default", "true", "false"],
        help="torch.compile dynamic shape setting (default leaves it unset)",
    )
    parser.add_argument(
        "--compile_no_fallback",
        action="store_true",
        help="Raise torch.compile errors instead of falling back to eager",
    )
    parser.add_argument(
        "--global_masked_mean",
        action="store_true",
        help="Opt-in: weight every masked frame equally across gradient accumulation and DDP "
        "(default off preserves the historical per-microbatch mean loss)",
    )

    return parser.parse_args()


# -------------------------- Training Settings -------------------------- #


def main():
    args = parse_args()
    compile_dynamic = None if args.compile_dynamic == "default" else args.compile_dynamic == "true"

    checkpoint_path = str(files("f5_tts").joinpath(f"../../ckpts/{args.dataset_name}"))

    # Model parameters based on experiment name

    if args.exp_name == "F5TTS_v1_Base":
        wandb_resume_id = None
        model_cls = DiT
        model_cfg = dict(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_dim=512,
            conv_layers=4,
        )
        if args.finetune:
            if args.pretrain is None:
                ckpt_path = str(cached_path("hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"))
            else:
                ckpt_path = args.pretrain

    elif args.exp_name == "F5TTS_Base":
        wandb_resume_id = None
        model_cls = DiT
        model_cfg = dict(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_dim=512,
            text_mask_padding=False,
            conv_layers=4,
            pe_attn_head=1,
        )
        if args.finetune:
            if args.pretrain is None:
                ckpt_path = str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.pt"))
            else:
                ckpt_path = args.pretrain

    elif args.exp_name == "E2TTS_Base":
        wandb_resume_id = None
        model_cls = UNetT
        model_cfg = dict(
            dim=1024,
            depth=24,
            heads=16,
            ff_mult=4,
            text_mask_padding=False,
            pe_attn_head=1,
        )
        if args.finetune:
            if args.pretrain is None:
                ckpt_path = str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.pt"))
            else:
                ckpt_path = args.pretrain

    if args.finetune:
        if not os.path.isdir(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)

        file_checkpoint = os.path.basename(ckpt_path)
        if not file_checkpoint.startswith("pretrained_"):  # Change: Add 'pretrained_' prefix to copied model
            file_checkpoint = "pretrained_" + file_checkpoint
        file_checkpoint = os.path.join(checkpoint_path, file_checkpoint)
        if not os.path.isfile(file_checkpoint):
            shutil.copy2(ckpt_path, file_checkpoint)
            print("copy checkpoint for finetune")

    # Use the tokenizer and tokenizer_path provided in the command line arguments

    tokenizer = args.tokenizer
    if tokenizer == "custom":
        if not args.tokenizer_path:
            raise ValueError("Custom tokenizer selected, but no tokenizer_path provided.")
        tokenizer_path = args.tokenizer_path
    else:
        tokenizer_path = args.dataset_name

    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)

    print("\nvocab : ", vocab_size)
    print("\nvocoder : ", mel_spec_type)

    mel_spec_kwargs = dict(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        target_sample_rate=target_sample_rate,
        mel_spec_type=mel_spec_type,
    )

    model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )
    compile_target = args.compile_target
    if compile_target == "auto":
        compile_target = "dit_blocks" if hasattr(model.transformer, "compile_training_target") else "cfm_loss_core"

    trainer = Trainer(
        model,
        args.epochs,
        args.learning_rate,
        num_warmup_updates=args.num_warmup_updates,
        save_per_updates=args.save_per_updates,
        keep_last_n_checkpoints=args.keep_last_n_checkpoints,
        checkpoint_path=checkpoint_path,
        batch_size_per_gpu=args.batch_size_per_gpu,
        batch_size_type=args.batch_size_type,
        max_samples=args.max_samples,
        grad_accumulation_steps=args.grad_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        logger=args.logger,
        wandb_project=args.dataset_name,
        wandb_run_name=args.exp_name,
        wandb_resume_id=wandb_resume_id,
        log_samples=args.log_samples,
        last_per_updates=args.last_per_updates,
        bnb_optimizer=args.bnb_optimizer,
        compile_enabled=args.compile_enabled,
        compile_backend=args.compile_backend,
        compile_target=compile_target,
        compile_mode=args.compile_mode,
        compile_fullgraph=args.compile_fullgraph,
        compile_dynamic=compile_dynamic,
        compile_fallback_to_eager=not args.compile_no_fallback,
        global_masked_mean=args.global_masked_mean,
    )

    train_dataset = load_dataset(args.dataset_name, tokenizer, mel_spec_kwargs=mel_spec_kwargs)

    trainer.train(
        train_dataset,
        resumable_with_seed=666,  # seed for shuffling dataset
    )


if __name__ == "__main__":
    main()
