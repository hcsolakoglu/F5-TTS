import os
from pathlib import Path

import numpy
import torch
import torch.nn.functional as F
import torchaudio
import wespeaker
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from wespeaker.cli.speaker import Speaker


class SpeakerEmb(Speaker):
    def __init__(self, model_dir: str):
        super().__init__(model_dir)

    def extract_embedding_from_pcm(self, pcm: torch.Tensor, sample_rate: int):
        pcm = pcm.to(torch.float)
        if sample_rate != self.resample_rate:
            pcm = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.resample_rate)(pcm)
        feats = self.compute_fbank(pcm, sample_rate=self.resample_rate, cmn=True)
        feats = feats.unsqueeze(0)
        feats = feats.to(self.device)

        with torch.no_grad():
            outputs = self.model(feats)
            outputs = outputs[-1] if isinstance(outputs, tuple) else outputs
        return outputs


def _resolve_resource_dir(relative: str) -> str:
    base = Path(__file__).parent
    return str((base / relative).resolve())


default_spk_id = "english"  # wespeaker hub id; set to "chinese" for Chinese model
model_spk_dir = os.getenv("F5TTS_RL_SPK_MODEL", default_spk_id)
_spk_backend = None
if os.path.isdir(model_spk_dir):
    try:
        _spk_backend = SpeakerEmb(model_spk_dir)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to load speaker model for RL. Set F5TTS_RL_SPK_MODEL to a valid path or allow download."
        ) from exc
else:
    # Try auto-download via wespeaker hub name or HF/modelscope id
    try:
        _spk_backend = wespeaker.load_model(model_spk_dir)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Could not download/load speaker model. Set F5TTS_RL_SPK_MODEL to a local path or a wespeaker model id (e.g., 'chinese')."
        ) from exc


def get_emb(wav: torch.Tensor, sr: int):
    result = []
    for i in range(wav.size(0)):
        item = wav[i].unsqueeze(0)
        if hasattr(_spk_backend, "extract_embedding_from_pcm"):
            item = _spk_backend.extract_embedding_from_pcm(item, sr).squeeze(0)
        else:  # wespeaker high-level API
            item = _spk_backend.extract_embedding(item, sample_rate=sr)
        result.append(item)
    return torch.stack(result, dim=0)


def cal_sim(emb1: torch.Tensor, emb2: torch.Tensor):
    return F.cosine_similarity(emb1, emb2)


model_asr_dir = os.getenv("F5TTS_RL_ASR_MODEL", "FunAudioLLM/SenseVoiceSmall")
try:
    model_asr = AutoModel(model=model_asr_dir, device="cpu", disable_update=True)
except Exception as exc:  # noqa: BLE001
    raise RuntimeError(
        "Failed to load ASR model for RL. Set F5TTS_RL_ASR_MODEL to a local path or a valid repo id (e.g., FunAudioLLM/SenseVoiceSmall)."
    ) from exc


def get_asr(audios: torch.Tensor, sr: int):
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        list_audios = [resampler(audios[i, :].unsqueeze(0))[0] for i in range(audios.size(0))]
    else:
        list_audios = [audios[i, :] for i in range(audios.size(0))]

    results = model_asr.inference(
        input=list_audios,
        cache={},
        language="auto",
        use_itn=True,
        disable_pbar=True,
        batch_size=len(list_audios),
    )
    text = [rich_transcription_postprocess(res["text"]) for res in results]
    return text


def editDistance(r, h):
    d = numpy.zeros((len(r) + 1) * (len(h) + 1), dtype=numpy.uint8).reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitute = d[i - 1][j - 1] + 1
                insert = d[i][j - 1] + 1
                delete = d[i - 1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    return d


def cal_wer(r, h):
    d = editDistance(r, h)
    result = float(d[len(r)][len(h)]) / max(1, len(r))
    return result
