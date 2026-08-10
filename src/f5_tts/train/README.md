# Training

Check your FFmpeg installation:
```bash
ffmpeg -version
```
If not found, install it first (or skip assuming you know of other backends available).

## Prepare Dataset

Example data processing scripts, and you may tailor your own one along with a Dataset class in `src/f5_tts/model/dataset.py`.

### 1. Some specific Datasets preparing scripts
Download corresponding dataset first, and fill in the path in scripts.

```bash
# Prepare the Emilia dataset
python src/f5_tts/train/datasets/prepare_emilia.py

# Prepare the Wenetspeech4TTS dataset
python src/f5_tts/train/datasets/prepare_wenetspeech4tts.py

# Prepare the LibriTTS dataset
python src/f5_tts/train/datasets/prepare_libritts.py

# Prepare the LJSpeech dataset
python src/f5_tts/train/datasets/prepare_ljspeech.py
```

### 2. Create custom dataset with CSV
Prepare a CSV with two columns using a required header: `audio_file|text`. Audio paths must be absolute.
Use guidance see [#57 here](https://github.com/SWivid/F5-TTS/discussions/57#discussioncomment-10959029).

```bash
python src/f5_tts/train/datasets/prepare_csv_wavs.py /path/to/metadata.csv /path/to/output
```

## Training & Finetuning

Once your datasets are prepared, you can start the training process.

### 1. Training script used for pretrained model

```bash
# setup accelerate config, e.g. use multi-gpu ddp, fp16
# will be to: ~/.cache/huggingface/accelerate/default_config.yaml     
accelerate config

# .yaml files are under src/f5_tts/configs directory
accelerate launch src/f5_tts/train/train.py --config-name F5TTS_v1_Base.yaml

# possible to overwrite accelerate and hydra config
accelerate launch --mixed_precision=fp16 src/f5_tts/train/train.py --config-name F5TTS_v1_Base.yaml ++datasets.batch_size_per_gpu=19200
```

### 2. Finetuning practice
Discussion board for Finetuning [#57](https://github.com/SWivid/F5-TTS/discussions/57).

Gradio UI training/finetuning with `src/f5_tts/train/finetune_gradio.py` see [#143](https://github.com/SWivid/F5-TTS/discussions/143).

If want to finetune with a variant version e.g. *F5TTS_v1_Base_no_zero_init*, manually download pretrained checkpoint from model weight repository and fill in the path correspondingly on web interface.

If use tensorboard as logger, install it first with `pip install tensorboard`.

<ins>The `use_ema = True` might be harmful for early-stage finetuned checkpoints</ins> (which goes just few updates, thus ema weights still dominated by pretrained ones), try turn it off with finetune gradio option or `load_model(..., use_ema=False)`, see if offer better results.

### 3. W&B Logging

The `wandb/` dir will be created under path you run training/finetuning scripts.

By default, the training script does NOT use logging (assuming you didn't manually log in using `wandb login`).

To turn on wandb logging, you can either:

1. Manually login with `wandb login`: Learn more [here](https://docs.wandb.ai/ref/cli/wandb-login)
2. Automatically login programmatically by setting an environment variable: Get an API KEY at https://wandb.ai/authorize and set the environment variable as follows:

On Mac & Linux:

```
export WANDB_API_KEY=<YOUR WANDB API KEY>
```

On Windows:

```
set WANDB_API_KEY=<YOUR WANDB API KEY>
```
Moreover, if you couldn't access W&B and want to log metrics offline, you can set the environment variable as follows:

```
export WANDB_MODE=offline
```

## Global masked-mean training

`global_masked_mean=True` computes one masked-frame mean across each distributed
accumulation window. Accelerate duplicate-padding is disabled for this mode, and
`split_batches=True` is rejected because equal dataloader lengths do not guarantee
an equal number of incomplete-tail iterations. A multi-process dataloader whose
batch count is not divisible by process count is also rejected before preparation
instead of replaying real samples or silently dropping a tail group. Choose a batch
size or explicit dataset policy that produces equal per-rank batch counts.

For this mode, dataloader construction and preparation, batch fetching, batch
validation, batch device transfer, training-mask preparation, duration prediction,
and the model forward phase coordinate rank-local exceptions before any rank enters
backward. A worker failure or malformed batch therefore fails all ranks with a
launcher-visible error instead of leaving healthy ranks waiting in a collective.
Failures inside backward or a dead process remain launcher-fatal by design; they
cannot be made safely recoverable with a later in-loop status collective.

Scheduler horizons are computed after Accelerate prepares the dataloader and read
`gradient_state.num_steps`, `sync_with_dataloader`, and `adjust_scheduler` rather
than trusting only Trainer's requested GAS. The raw scheduler horizon accounts for
Accelerate's process step multiplier, including `split_batches` and
`step_scheduler_with_optimizer`. With `sync_with_dataloader=False`, best-effort
resume uses a global raw-batch cursor and floor horizon; `global_masked_mean` rejects
that mode because its host accumulation window cannot cross an epoch boundary, and
exact resume rejects it because pending gradient state is not checkpointed.
When Accelerate reports `adjust_scheduler=True`, the wrapped scheduler is called on
every accumulation microbatch. Non-boundary calls update Accelerate's wrapper
bookkeeping but do not advance the underlying scheduler, whose horizon remains the
number of successful optimizer boundaries. Exact resume stores this horizon
signature and rejects a changed prepared loader or schedule configuration.

Checkpoint resume defaults to `resume_mode="best_effort"`: it restores the update
cursor and compatible process-local state when the persisted resume signature
matches. Legacy checkpoints without the new signatures may restore their legacy
RNG payload in best-effort mode after the existing topology-count check, but they
cannot claim exact resume. If the dataset, loader contract, execution topology,
precision, or AMP scaler state is incompatible, it skips incompatible scaler/RNG
restoration and emits a warning. `resume_mode="exact"` additionally requires the
checkpoint to contain matching model, optimizer, scheduler, RNG, AMP scaler, and
resume signatures. The resume signature includes an immutable HF dataset fingerprint or
an explicit `exact_resume_signature`, dataset preprocessing/mel configuration,
sampler/batch configuration, effective accumulation plugin settings, optimizer and
compile configuration, prepared loader flags, loss mode, and execution topology.

The supported exact subset is deliberately narrow: an unsharded single process,
pure CPU/Gloo DDP, or CUDA DDP resumed with the same rank-to-device topology; a
map-style dataset that explicitly sets
`supports_exact_resume=True` and provides an immutable HF fingerprint or
`exact_resume_signature`, `num_workers=0`, `batch_size_type="sample"`, and a
`resumable_with_seed`, with accumulation synchronized at dataloader boundaries.
FSDP, DeepSpeed, Megatron-LM, multi-worker loading, frame batching, and non-CPU/CUDA
RNG backends are rejected at `train()` entry before dataloader or checkpoint work.
Built-in wrappers expose the dataset marker only for stateless map-style HF
datasets; arbitrary wrappers and custom mel modules must opt in explicitly.

HF datasets with raw external audio also require an explicit content signature. The
production loader accepts it through `datasets.exact_resume_content_signature`, for
example `++datasets.exact_resume_content_signature=raw-audio-manifest-v1`; the value
must identify an immutable manifest or equivalent content hash. Custom mel modules
require an explicit preprocessing signature.

Custom models and duration predictors must set `supports_exact_resume=True` and
provide an `exact_resume_signature`. This opt-in certifies that every mutable value
which can affect training is registered in `state_dict()`; non-parameter state uses
PyTorch's paired `get_extra_state()` / `set_extra_state()` protocol. Trainer does not
guess by serializing arbitrary Python attributes. The production CFM path uses its
resolved model config as the behavior signature, and EMA signatures include resolved
library defaults plus user overrides. A duration predictor is supported only in a
single process because its per-process state is not globally checkpointed.

A repeatable two-rank Gloo coordination probe is available at
`tests/probes/coordinated_train_failure_probe.py` and requires exactly two ranks. It
asserts phase-specific peer propagation for DataLoader construction, preparation,
device transfer, and forward failures.
