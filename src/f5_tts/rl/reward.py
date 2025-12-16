import torch
import torch.nn.functional as F

from f5_tts.infer.utils_infer import load_vocoder
from f5_tts.rl import utils


vocos = load_vocoder(vocoder_name="vocos", device="cpu")


def get_reward(gen_mel: torch.Tensor, trg_mel: torch.Tensor):
    gen_mel = gen_mel.permute(0, 2, 1)
    trg_mel = trg_mel.permute(0, 2, 1)
    gen_wav = vocos.decode(gen_mel.cpu())
    trg_wav = vocos.decode(trg_mel.cpu())
    gen_emb = utils.get_emb(gen_wav, 24000)
    trg_emb = utils.get_emb(trg_wav, 24000)
    sim = utils.cal_sim(gen_emb, trg_emb).to(gen_mel.device)
    gen_txt = utils.get_asr(gen_wav, 24000)
    trg_txt = utils.get_asr(trg_wav, 24000)
    acc = []
    for r, h in zip(trg_txt, gen_txt):
        acc.append(1 - utils.cal_wer(r, h))
    acc = torch.tensor(acc, device=gen_mel.device)
    return sim, acc


def cal_kl(gen, ref):
    gen_mu, gen_sig = gen
    ref_mu, ref_sig = ref
    kl = ref_sig - gen_sig
    kl += ((torch.exp(gen_sig) ** 2) + F.mse_loss(gen_mu, ref_mu, reduction="none")) / (2 * (torch.exp(ref_sig) ** 2))
    return kl


def get_kl(gen_pros, ref_pros):
    loss = 0
    for gen, ref in zip(gen_pros, ref_pros):
        loss += cal_kl(gen[1:3], ref[1:3])
    return loss
