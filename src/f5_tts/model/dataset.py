import json
import math
import warnings
from importlib.resources import files

import torch
import torch.nn.functional as F
import torchaudio
from datasets import Dataset as Dataset_
from datasets import load_from_disk
from torch import nn
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import default


class HFDataset(Dataset):
    def __init__(
        self,
        hf_dataset: Dataset,
        target_sample_rate=24_000,
        n_mel_channels=100,
        hop_length=256,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
    ):
        self.data = hf_dataset
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length

        self.mel_spectrogram = MelSpec(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        )
        self._resamplers = {}

    def get_frame_len(self, index):
        row = self.data[index]
        audio = row["audio"]["array"]
        sample_rate = row["audio"]["sampling_rate"]
        return audio.shape[-1] / sample_rate * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        audio = row["audio"]["array"]

        sample_rate = row["audio"]["sampling_rate"]
        duration = audio.shape[-1] / sample_rate

        if duration > 30 or duration < 0.3:
            return self.__getitem__((index + 1) % len(self.data))

        audio_tensor = torch.from_numpy(audio).float()

        if sample_rate != self.target_sample_rate:
            if sample_rate not in self._resamplers:
                self._resamplers[sample_rate] = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            audio_tensor = self._resamplers[sample_rate](audio_tensor)

        audio_tensor = audio_tensor.unsqueeze(0)  # 't -> 1 t')

        mel_spec = self.mel_spectrogram(audio_tensor)

        mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'

        text = row["text"]

        return dict(
            mel_spec=mel_spec,
            text=text,
        )


class CustomDataset(Dataset):
    def __init__(
        self,
        custom_dataset: Dataset,
        durations=None,
        target_sample_rate=24_000,
        hop_length=256,
        n_mel_channels=100,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
        preprocessed_mel=False,
        mel_spec_module: nn.Module | None = None,
    ):
        self.data = custom_dataset
        self.durations = durations
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.mel_spec_type = mel_spec_type
        self.preprocessed_mel = preprocessed_mel

        if not preprocessed_mel:
            self.mel_spectrogram = default(
                mel_spec_module,
                MelSpec(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    n_mel_channels=n_mel_channels,
                    target_sample_rate=target_sample_rate,
                    mel_spec_type=mel_spec_type,
                ),
            )
        self._resamplers = {}

    def get_frame_len(self, index):
        if (
            self.durations is not None
        ):  # Please make sure the separately provided durations are correct, otherwise 99.99% OOM
            return self.durations[index] * self.target_sample_rate / self.hop_length
        return self.data[index]["duration"] * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        while True:
            row = self.data[index]
            audio_path = row["audio_path"]
            text = row["text"]
            duration = row["duration"]

            # filter by given length
            if 0.3 <= duration <= 30:
                break  # valid

            index = (index + 1) % len(self.data)

        if self.preprocessed_mel:
            mel_spec = torch.tensor(row["mel_spec"])
        else:
            audio, source_sample_rate = torchaudio.load(audio_path)

            # make sure mono input
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            # resample if necessary
            if source_sample_rate != self.target_sample_rate:
                if source_sample_rate not in self._resamplers:
                    self._resamplers[source_sample_rate] = torchaudio.transforms.Resample(
                        source_sample_rate, self.target_sample_rate
                    )
                audio = self._resamplers[source_sample_rate](audio)

            # to mel spectrogram
            mel_spec = self.mel_spectrogram(audio)
            mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'

        return {
            "mel_spec": mel_spec,
            "text": text,
        }


def padded_mel_frames(frame_len: float, mel_spec_type: str) -> int:
    """Exact padded mel-frame count for one clip, per mel frontend.

    ``frame_len`` is the float returned by ``get_frame_len`` (resampled samples /
    hop_length). ``collate_fn`` pads every clip in a batch to the batch max of this
    count, so the padded rectangle the GPU allocates is
    ``len(batch) * max(padded_mel_frames(frame_len_i, mel_spec_type))``.

    vocos uses ``torchaudio.MelSpectrogram(center=True)``, whose output width is
    ``1 + floor(n_samples / hop_length)`` == ``floor(frame_len) + 1``. The previous
    ``ceil(frame_len)`` was correct for non-integral frame_len (where ceil == floor+1)
    but one frame short at exact hop multiples (where ceil == floor), so the cap was
    violated by up to ``len(batch)`` frames.

    bigvgan uses ``center=False`` with symmetric ``(n_fft - hop_length)//2`` reflect
    padding, whose output width is ``floor(n_samples / hop_length)`` ==
    ``floor(frame_len)``; ``ceil`` was already a safe (loose) upper bound there.
    """
    if mel_spec_type == "vocos":
        return math.floor(frame_len) + 1
    if mel_spec_type == "bigvgan":
        return math.floor(frame_len)
    raise ValueError(f"unsupported mel_spec_type for padded-frame cap: {mel_spec_type!r}")


# Dynamic Batch Sampler
class DynamicBatchSampler(Sampler[list[int]]):
    """Extension of Sampler that will do the following:
    1.  Change the batch size (essentially number of sequences)
        in a batch to ensure that the total number of frames are less
        than a certain threshold.
    2.  Make sure the padding efficiency in the batch is high.
    3.  Shuffle batches each epoch while maintaining reproducibility.
    """

    def __init__(
        self,
        sampler: Sampler[int],
        frames_threshold: int,
        max_samples=0,
        random_seed=None,
        drop_residual: bool = False,
        max_padded_frames: int = 0,
        mel_spec_type: str = "vocos",
    ):
        """``max_padded_frames`` optionally bounds the *padded* batch rectangle.

        ``frames_threshold`` budgets the sum of raw frame lengths, but the tensor the GPU
        actually allocates is ``len(batch) * max(frame_len)`` -- padding included. Because
        batches are built from length-sorted indices those two are usually within a rounding
        error of each other, so this is not a general problem. They diverge when a batch of
        short utterances gets closed out by a much longer one, which happens on bimodal
        corpora (short prompts mixed with long-form audio).

        Measured at N=100k, frames_threshold=38400, max_samples=64 -- worst-case padded
        rectangle as a multiple of the requested budget:

            LibriSpeech-shaped  1.00x      Emilia-shaped  1.00x
            uniform 1-15s       1.00x      bimodal        1.82x   <-- the case this exists for

        Choosing a value (cost measured on the same workload):

            0                      off; batch composition byte-identical to upstream. DEFAULT.
            == frames_threshold    RECOMMENDED for bigvgan. Bounds the rectangle to exactly
                                   the budget already requested. For vocos (center=True) the
                                   longest clip's padded width is ``floor(frame_len)+1``, so a
                                   clip exactly at frames_threshold is one frame over and is
                                   dropped; use ``frames_threshold + 1`` for vocos to keep it.
            == frames_threshold+1  RECOMMENDED for vocos. Same bound as above while accounting
                                   for the center=True +1 frame, so no admitted sample is dropped.
            >  frames_threshold    deliberate slack. Only has any effect on bimodal-style data,
                                   where it permits proportional overshoot (1.5x cap -> 1.47x
                                   observed). Use if the recommended value costs you throughput
                                   on a corpus not represented above.
            <  frames_threshold    NOT RECOMMENDED. A padded rectangle is never smaller than the
                                   frame sum, so this makes frames_threshold dead and simply
                                   shrinks batches: at 0.5x, batch count roughly doubles and mean
                                   batch size halves for no memory benefit you could not get by
                                   lowering frames_threshold itself. It can also discard every
                                   sample longer than the cap; that emits a RuntimeWarning.

        This is a memory-safety guard and is unrelated to torch.compile: compile behaviour is
        driven by dynamic-shape promotion, not by shape count, and compiled runs measured ~12%
        *lower* peak VRAM than eager. Enable it based on the shape of your corpus, not on
        whether compile is on.
        """
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.max_padded_frames = max_padded_frames
        self.mel_spec_type = mel_spec_type
        self.epoch = 0

        indices, batches = [], []
        data_source = self.sampler.data_source

        for idx in tqdm(
            self.sampler, desc="Sorting with sampler... if slow, check whether dataset is provided with duration"
        ):
            indices.append((idx, data_source.get_frame_len(idx)))
        indices.sort(key=lambda elem: elem[1])

        batch = []
        batch_frames = 0
        dropped_by_cap = 0
        for idx, frame_len in tqdm(
            indices, desc=f"Creating dynamic batches with {frames_threshold} audio frames per gpu"
        ):
            # `indices` is sorted ascending by frame_len, so the incoming element is always
            # the batch maximum and the padded rectangle is exactly (len(batch)+1)*frame_len.
            # get_frame_len returns a float (duration * sample_rate / hop_length) but collate_fn
            # pads to a whole number of mel frames, so the cap must be checked against the
            # frontend-specific mel-frame count (vocos center=True adds one frame at exact hop
            # multiples; see ``padded_mel_frames``), not the raw float.
            padded_fits = (
                self.max_padded_frames == 0
                or (len(batch) + 1) * padded_mel_frames(frame_len, self.mel_spec_type) <= self.max_padded_frames
            )
            if (
                batch_frames + frame_len <= self.frames_threshold
                and (max_samples == 0 or len(batch) < max_samples)
                and padded_fits
            ):
                batch.append(idx)
                batch_frames += frame_len
            else:
                if len(batch) > 0:
                    batches.append(batch)
                # A sample that cannot fit alone is dropped, matching the existing
                # frames_threshold behaviour rather than emitting an over-budget batch.
                fits_threshold = frame_len <= self.frames_threshold
                fits_cap = (
                    self.max_padded_frames == 0
                    or padded_mel_frames(frame_len, self.mel_spec_type) <= self.max_padded_frames
                )
                if fits_threshold and fits_cap:
                    batch = [idx]
                    batch_frames = frame_len
                else:
                    # Only the cap rejecting a sample that frames_threshold would have kept
                    # is new data loss introduced by this option, so count that case alone.
                    if fits_threshold and not fits_cap:
                        dropped_by_cap += 1
                    batch = []
                    batch_frames = 0

        if not drop_residual and len(batch) > 0:
            batches.append(batch)

        if dropped_by_cap:
            # Never silent: max_padded_frames < frames_threshold discards every sample
            # between the two, which is almost always a misconfiguration rather than intent.
            warnings.warn(
                f"max_padded_frames={self.max_padded_frames} dropped {dropped_by_cap} sample(s) that "
                f"frames_threshold={self.frames_threshold} would have kept. Set max_padded_frames >= "
                f"frames_threshold (a padded rectangle is never smaller than the frame sum) to bound "
                f"batch memory without discarding data.",
                RuntimeWarning,
                stacklevel=2,
            )

        del indices
        self.batches = batches

        # Ensure even batches with accelerate BatchSamplerShard cls under frame_per_batch setting
        self.drop_last = True

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler."""
        self.epoch = epoch

    def __iter__(self):
        # Use both random_seed and epoch for deterministic but different shuffling per epoch
        if self.random_seed is not None:
            g = torch.Generator()
            g.manual_seed(self.random_seed + self.epoch)
            # Use PyTorch's random permutation for better reproducibility across PyTorch versions
            indices = torch.randperm(len(self.batches), generator=g).tolist()
            batches = [self.batches[i] for i in indices]
        else:
            batches = self.batches
        return iter(batches)

    def __len__(self):
        return len(self.batches)


# Load dataset


def load_dataset(
    dataset_name: str,
    tokenizer: str = "pinyin",
    dataset_type: str = "CustomDataset",
    audio_type: str = "raw",
    mel_spec_module: nn.Module | None = None,
    mel_spec_kwargs: dict = dict(),
) -> CustomDataset | HFDataset:
    """
    dataset_type    - "CustomDataset" if you want to use tokenizer name and default data path to load for train_dataset
                    - "CustomDatasetPath" if you just want to pass the full path to a preprocessed dataset without relying on tokenizer
    """

    print("Loading dataset ...")

    if dataset_type == "CustomDataset":
        rel_data_path = str(files("f5_tts").joinpath(f"../../data/{dataset_name}_{tokenizer}"))
        if audio_type == "raw":
            try:
                train_dataset = load_from_disk(f"{rel_data_path}/raw")
            except:  # noqa: E722
                train_dataset = Dataset_.from_file(f"{rel_data_path}/raw.arrow")
            preprocessed_mel = False
        elif audio_type == "mel":
            train_dataset = Dataset_.from_file(f"{rel_data_path}/mel.arrow")
            preprocessed_mel = True
        with open(f"{rel_data_path}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        train_dataset = CustomDataset(
            train_dataset,
            durations=durations,
            preprocessed_mel=preprocessed_mel,
            mel_spec_module=mel_spec_module,
            **mel_spec_kwargs,
        )

    elif dataset_type == "CustomDatasetPath":
        try:
            train_dataset = load_from_disk(f"{dataset_name}/raw")
        except:  # noqa: E722
            train_dataset = Dataset_.from_file(f"{dataset_name}/raw.arrow")

        with open(f"{dataset_name}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        train_dataset = CustomDataset(
            train_dataset, durations=durations, preprocessed_mel=preprocessed_mel, **mel_spec_kwargs
        )

    elif dataset_type == "HFDataset":
        print(
            "Should manually modify the path of huggingface dataset to your need.\n"
            + "May also the corresponding script cuz different dataset may have different format."
        )
        pre, post = dataset_name.split("_")
        train_dataset = HFDataset(
            load_dataset(f"{pre}/{pre}", split=f"train.{post}", cache_dir=str(files("f5_tts").joinpath("../../data"))),
        )

    return train_dataset


# collation


def collate_fn(batch):
    mel_specs = [item["mel_spec"].squeeze(0) for item in batch]
    mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
    max_mel_length = mel_lengths.amax()

    padded_mel_specs = []
    for spec in mel_specs:
        padding = (0, max_mel_length - spec.size(-1))
        padded_spec = F.pad(spec, padding, value=0)
        padded_mel_specs.append(padded_spec)

    mel_specs = torch.stack(padded_mel_specs)

    text = [item["text"] for item in batch]
    text_lengths = torch.LongTensor([len(item) for item in text])

    return dict(
        mel=mel_specs,
        mel_lengths=mel_lengths,  # records for padding mask
        text=text,
        text_lengths=text_lengths,
    )
