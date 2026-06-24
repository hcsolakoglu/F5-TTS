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

Optional runtime metrics are disabled by default. Enable them when you need comparable eager/optimized training timings:

```bash
accelerate launch src/f5_tts/train/train.py --config-name F5TTS_v1_Base.yaml ++metrics.enabled=True ++metrics.log_every=50
```

The metrics include optimizer-update wall time, summed compute-region time, data-wait, forward, backward, optimizer, scheduler, zero-grad, checkpoint, sampling, samples/frames/audio-seconds throughput, padding, mean/max/p95 sequence length, grad norm, compile first-forward time, and CUDA memory values where available. Gradient-accumulation microbatches are aggregated into one optimizer-update record. With an Accelerate-prepared dataloader, data-wait includes device dispatch; host-to-device time is not isolated by the production Trainer. CUDA synchronization is enabled for accurate timings when metrics are on; disable it only when you accept lower timing precision:

```bash
accelerate launch src/f5_tts/train/train.py --config-name F5TTS_v1_Base.yaml ++metrics.enabled=True ++metrics.sync_cuda=False
```

`torch.compile` support is optional and disabled by default. For F5-TTS models, the current compile path targets the deterministic CFM training loss core after eager text/mask/randomness preparation. Dataset loading, collation, optimizer orchestration, logging, checkpointing, and sampling stay eager:

```bash
accelerate launch src/f5_tts/train/train.py --config-name F5TTS_v1_Base.yaml ++compile.enabled=True ++metrics.enabled=True
```

Compile options are under the `compile` config section:

```yaml
compile:
  enabled: False
  target: cfm_loss_core
  backend: inductor
  mode: null
  fullgraph: False
  dynamic: null
  fallback_to_eager: True
```

Leave `fallback_to_eager` enabled for exploratory runs on new hardware. A lazy compile failure disables the compiled core and retries only the deterministic loss core with the same prepared mask, noise, time, CFG decisions, and restored PyTorch RNG state. Disable fallback when you want compile failures to stop training immediately and to remove the small RNG-state snapshot overhead.

For variable-length real-audio batches, treat `compile.dynamic` as an experiment knob rather than a default recommendation. `True` can reduce shape-specialized recompiles, but it can also reduce specialization and steady-state speed. `False` or `null` may be faster when sequence lengths are stable or bucketed. Compare `null`, `True`, and `False` with the same dataset, batch policy, precision, and measured warmup; compile is lazy, so graph capture/codegen must be reported separately from steady-state throughput.

For DiT and UNetT training, integer text tokens are eagerly cropped/padded to the audio tensor's existing padded length before the compiled boundary. This is equivalent to the backbones' existing internal crop/pad behavior and prevents raw text width from becoming an additional compile guard. Audio length variation still remains; this does not make `dynamic=False` a general recommendation for unbucketed datasets.

### 2. Finetuning practice
Discussion board for Finetuning [#57](https://github.com/SWivid/F5-TTS/discussions/57).

Gradio UI training/finetuning with `src/f5_tts/train/finetune_gradio.py` see [#143](https://github.com/SWivid/F5-TTS/discussions/143).

If want to finetune with a variant version e.g. *F5TTS_v1_Base_no_zero_init*, manually download pretrained checkpoint from model weight repository and fill in the path correspondingly on web interface.

If use tensorboard as logger, install it first with `pip install tensorboard`.

The CLI finetune path has matching opt-in flags:

```bash
python src/f5_tts/train/finetune_cli.py --metrics_enabled --metrics_log_every 50
python src/f5_tts/train/finetune_cli.py --compile_enabled --metrics_enabled
```

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
