#!/usr/bin/env python3
"""Colab real-data Trainer benchmark for F5-TTS torch.compile candidates.

This script is intentionally self-contained for `colab run --gpu G4`:
- clones the requested F5-TTS branch on the Colab VM,
- installs the package,
- builds a real LibriSpeech subset with precomputed mel spectrograms,
- runs each candidate in a fresh Python process with the real Trainer/Accelerate stack,
- records timing, memory, fallback, loss, and shape-cardinality evidence.

It does not modify the source-of-truth checkout. Candidate-only behavior that is not
exposed by the current PR CLI, such as torch.compile options and cudagraph mark-step,
is injected by the benchmark driver around Trainer construction so it can be validated
before deciding whether to add upstream API fields.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import statistics
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

REPO_URL = "https://github.com/hcsolakoglu/F5-TTS.git"
BRANCH = "torch-compile-upstream-integration"
COMMIT = "e04c641e6cc38f280d1987e58e853313c3cc44db"
DATASET_NAME = "openslr/librispeech_asr"
DATASET_CONFIG = "clean"
DATASET_SPLIT = "train.100"

CANDIDATES: dict[str, dict[str, Any]] = {
    "eager": {"compile": False, "mark_step": False},
    "compiled_default": {"compile": True, "mode": None, "options": None, "mark_step": False},
    "compiled_bucket64_r20_align": {
        "compile": True,
        "mode": None,
        "options": None,
        "mark_step": False,
        "bucket_mel_size": 64,
        "bucket_max_padding_ratio": 0.20,
        "align_text_len_to_mel": True,
        "compile_mark_dynamic": False,
    },
    "compiled_bucket128_r20_align": {
        "compile": True,
        "mode": None,
        "options": None,
        "mark_step": False,
        "bucket_mel_size": 128,
        "bucket_max_padding_ratio": 0.20,
        "align_text_len_to_mel": True,
        "compile_mark_dynamic": False,
    },
    "compiled_bucket256_r20_align": {
        "compile": True,
        "mode": None,
        "options": None,
        "mark_step": False,
        "bucket_mel_size": 256,
        "bucket_max_padding_ratio": 0.20,
        "align_text_len_to_mel": True,
        "compile_mark_dynamic": False,
    },
}

BUCKET_AGENT_PATCH_B64 = """ZGlmZiAtLWdpdCBhL3NyYy9mNV90dHMvbW9kZWwvY2ZtLnB5IGIvc3JjL2Y1X3R0cy9tb2RlbC9jZm0ucHkKaW5kZXggNTBhMzUxOC4uZTdhODZlOSAxMDA2NDQKLS0tIGEvc3JjL2Y1X3R0cy9tb2RlbC9jZm0ucHkKKysrIGIvc3JjL2Y1X3R0cy9tb2RlbC9jZm0ucHkKQEAgLTU2LDEwICs1NiwyMiBAQCBjbGFzcyBDRk0obm4uTW9kdWxlKToKICAgICAgICAgbWVsX3NwZWNfa3dhcmdzOiBkaWN0ID0gZGljdCgpLAogICAgICAgICBmcmFjX2xlbmd0aHNfbWFzazogdHVwbGVbZmxvYXQsIGZsb2F0XSA9ICgwLjcsIDEuMCksCiAgICAgICAgIHZvY2FiX2NoYXJfbWFwOiBkaWN0W3N0cjppbnRdIHwgTm9uZSA9IE5vbmUsCisgICAgICAgICMgT3B0LWluIGNvbXBpbGUgc2hhcGUtc3RhYmlsaXNhdGlvbiAoZGVmYXVsdCBvZmY7IHNlZSB0cmFpbmVyIGJ1Y2tldF8qIGFyZ3MpLgorICAgICAgICAjIGFsaWduX3RleHRfbGVuX3RvX21lbCBwYWRzL2N1cnRhaWxzIHRva2VuaXplZCB0ZXh0IHRvIHRoZSBtZWwgc2VxdWVuY2UgbGVuZ3RoCisgICAgICAgICMgc28gdGhlIGNvbXBpbGVkIGxvc3MgY29yZSBzZWVzIGEgc2luZ2xlIGR5bmFtaWMgZGltIChuID09IG50KSBpbnN0ZWFkIG9mIHR3bworICAgICAgICAjIGluZGVwZW5kZW50IG9uZXMuIFNlbWFudGljYWxseSBhIG5vLW9wOiB0aGUgRGlUIHRleHQgZW1iZWRkaW5nIGFscmVhZHkKKyAgICAgICAgIyBjdXJ0YWlscy9wYWRzIHRleHQgdG8gbWVsIHNlcV9sZW4gaW50ZXJuYWxseTsgdGhpcyBvbmx5IGZpeGVzIHRoZSBpbnB1dCBzaGFwZS4KKyAgICAgICAgYWxpZ25fdGV4dF9sZW5fdG9fbWVsOiBib29sID0gRmFsc2UsCisgICAgICAgICMgRXhwZXJpbWVudGFsOiBtYXJrIHRoZSBtZWwgYW5kIHRleHQgc2VxdWVuY2UgZGltcyBhcyBkeW5hbWljIHRvIER5bmFtbyBiZWZvcmUKKyAgICAgICAgIyB0aGUgY29tcGlsZWQgZm9yd2FyZC4gQ29tcGFuaW9uIHRvIGJvdW5kZWQgYnVja2V0aW5nOyBkbyBub3QgcmVseSBvbiBnbG9iYWwKKyAgICAgICAgIyBkeW5hbWljPVRydWUgYXMgYSBkZWZhdWx0LiBObyBlZmZlY3Qgd2hlbiBjb21waWxlIGlzIGluYWN0aXZlLgorICAgICAgICBjb21waWxlX21hcmtfZHluYW1pYzogYm9vbCA9IEZhbHNlLAogICAgICk6CiAgICAgICAgIHN1cGVyKCkuX19pbml0X18oKQogCiAgICAgICAgIHNlbGYuZnJhY19sZW5ndGhzX21hc2sgPSBmcmFjX2xlbmd0aHNfbWFzaworICAgICAgICBzZWxmLmFsaWduX3RleHRfbGVuX3RvX21lbCA9IGFsaWduX3RleHRfbGVuX3RvX21lbAorICAgICAgICBzZWxmLmNvbXBpbGVfbWFya19keW5hbWljID0gY29tcGlsZV9tYXJrX2R5bmFtaWMKIAogICAgICAgICAjIG1lbCBzcGVjCiAgICAgICAgIHNlbGYubWVsX3NwZWMgPSBkZWZhdWx0KG1lbF9zcGVjX21vZHVsZSwgTWVsU3BlYygqKm1lbF9zcGVjX2t3YXJncykpCkBAIC0yMDYsNiArMjE4LDE2IEBAIGNsYXNzIENGTShubi5Nb2R1bGUpOgogICAgICAgICAgICAgICAgIHRleHQgPSBsaXN0X3N0cl90b190ZW5zb3IodGV4dCkudG8oZGV2aWNlKQogICAgICAgICAgICAgYXNzZXJ0IHRleHQuc2hhcGVbMF0gPT0gYmF0Y2gKIAorICAgICAgICAjIE9wdC1pbjogYWxpZ24gdG9rZW5pemVkIHRleHQgbGVuZ3RoIHRvIHRoZSBtZWwgc2VxdWVuY2UgbGVuZ3RoIHNvIHRoZQorICAgICAgICAjIGNvbXBpbGVkIGNvcmUgc2VlcyBuID09IG50IChvbmUgZHluYW1pYyBkaW0pLiBUaGUgRGlUIHRleHQgZW1iZWRkaW5nCisgICAgICAgICMgYWxyZWFkeSBjdXJ0YWlscy9wYWRzIHRleHQgdG8gbWVsIHNlcV9sZW4sIHNvIHRoaXMgaXMgc2hhcGUtb25seS4KKyAgICAgICAgaWYgc2VsZi5hbGlnbl90ZXh0X2xlbl90b19tZWw6CisgICAgICAgICAgICBjdXJfdGV4dF9sZW4gPSB0ZXh0LnNoYXBlWzFdCisgICAgICAgICAgICBpZiBjdXJfdGV4dF9sZW4gPCBzZXFfbGVuOgorICAgICAgICAgICAgICAgIHRleHQgPSBGLnBhZCh0ZXh0LCAoMCwgc2VxX2xlbiAtIGN1cl90ZXh0X2xlbiksIHZhbHVlPS0xKQorICAgICAgICAgICAgZWxpZiBjdXJfdGV4dF9sZW4gPiBzZXFfbGVuOgorICAgICAgICAgICAgICAgIHRleHQgPSB0ZXh0WzosIDpzZXFfbGVuXQorCiAgICAgICAgICMgbGVucyBhbmQgbWFzazogbG9uZyBkdHlwZSBmb3IgbWFzayBpbmRleCBhcml0aG1ldGljCiAgICAgICAgIGlmIG5vdCBleGlzdHMobGVucyk6ICAjIGlmIGxlbnMgbm90IGFjcXVpcmVkIGJ5IHRyYWluZXIgZnJvbSBjb2xsYXRlX2ZuCiAgICAgICAgICAgICBsZW5zID0gdG9yY2guZnVsbCgoYmF0Y2gsKSwgc2VxX2xlbiwgZGV2aWNlPWRldmljZSwgZHR5cGU9dG9yY2gubG9uZykKQEAgLTIzNCw2ICsyNTYsMTUgQEAgY2xhc3MgQ0ZNKG5uLk1vZHVsZSk6CiAgICAgICAgIGVsc2U6CiAgICAgICAgICAgICBkcm9wX3RleHQgPSBGYWxzZQogCisgICAgICAgICMgRXhwZXJpbWVudGFsIG9wdC1pbjogaGludCBEeW5hbW8gdGhhdCB0aGUgbWVsIChkaW0gMSkgYW5kIHRleHQgKGRpbSAxKQorICAgICAgICAjIHNlcXVlbmNlIGRpbXMgYXJlIGR5bmFtaWMgYmVmb3JlIGVudGVyaW5nIHRoZSBjb21waWxlZCBsb3NzIGNvcmUuIE9ubHkKKyAgICAgICAgIyBlbWl0dGVkIHdoZW4gY29tcGlsZSBpcyBhY3RpdmU7IGhhcm1sZXNzIGluIGVhZ2VyIG1vZGUuCisgICAgICAgIGlmIHNlbGYuY29tcGlsZV9tYXJrX2R5bmFtaWMgYW5kIHNlbGYuX2NvbXBpbGVkX2xvc3NfY29yZSBpcyBub3QgTm9uZToKKyAgICAgICAgICAgIGltcG9ydCB0b3JjaC5fZHluYW1vIGFzIF9keW5hbW8KKworICAgICAgICAgICAgX2R5bmFtby5tYXJrX2R5bmFtaWMoeDEsIDEpCisgICAgICAgICAgICBfZHluYW1vLm1hcmtfZHluYW1pYyh0ZXh0LCAxKQorCiAgICAgICAgIHJldHVybiB4MSwgdGV4dCwgbWFzaywgcmFuZF9zcGFuX21hc2ssIHgwLCB0aW1lLCBkcm9wX2F1ZGlvX2NvbmQsIGRyb3BfdGV4dAogCiAgICAgZGVmIF9mb3J3YXJkX2xvc3NfY29yZV9jb21wb25lbnRzKApkaWZmIC0tZ2l0IGEvc3JjL2Y1X3R0cy9tb2RlbC9kYXRhc2V0LnB5IGIvc3JjL2Y1X3R0cy9tb2RlbC9kYXRhc2V0LnB5CmluZGV4IDFmNzI5ZjkuLjUyNzU4ZjEgMTAwNjQ0Ci0tLSBhL3NyYy9mNV90dHMvbW9kZWwvZGF0YXNldC5weQorKysgYi9zcmMvZjVfdHRzL21vZGVsL2RhdGFzZXQucHkKQEAgLTMxMCwxNSArMzEwLDExMyBAQCBkZWYgbG9hZF9kYXRhc2V0KAogIyBjb2xsYXRpb24KIAogCi1kZWYgY29sbGF0ZV9mbihiYXRjaCk6CitjbGFzcyBCdWNrZXRTdGF0czoKKyAgICAiIiJBY2N1bXVsYXRlcyBwYWRkaW5nL3NoYXBlIHN0YXRpc3RpY3MgZm9yIHRoZSBvcHQtaW4gY29tcGlsZS1idWNrZXRpbmcgcG9saWN5LgorCisgICAgTm90IHRocmVhZC9wcm9jZXNzLXNhZmU7IGludGVuZGVkIGZvciBhIHNpbmdsZSB0cmFpbmluZyBwcm9jZXNzLiBVcGRhdGVkIGJ5CisgICAgOmZ1bmM6YGJ1Y2tldGVkX2NvbGxhdGVfZm5gIHdoZW4gYSBgYHN0YXRzYGAgaGFuZGxlIGlzIHN1cHBsaWVkLiBVc2UKKyAgICA6bWV0aDpgc3VtbWFyeWAgdG8gcmVwb3J0IHVuaXF1ZS1zaGFwZSBjYXJkaW5hbGl0eSBhbmQgbWVhc3VyZWQgcGFkZGluZworICAgIG92ZXJoZWFkIHNvIHRoZSBlZmZlY3Qgb2YgYnVja2V0aW5nIG9uIHRvcmNoLmNvbXBpbGUgcmVjb21waWxlcyBpcyBvYnNlcnZlZAorICAgIHJhdGhlciB0aGFuIGd1ZXNzZWQuCisgICAgIiIiCisKKyAgICBfX3Nsb3RzX18gPSAoCisgICAgICAgICJyYXdfbWVsX2xlbmd0aHMiLAorICAgICAgICAicmF3X2JhdGNoX21heF9tZWwiLAorICAgICAgICAiYnVja2V0ZWRfbWVsX2xlbmd0aHMiLAorICAgICAgICAiYmF0Y2hlcyIsCisgICAgICAgICJ0b3RhbF9hY3R1YWxfZnJhbWVzIiwKKyAgICAgICAgInRvdGFsX3BhZGRlZF9mcmFtZXMiLAorICAgICAgICAic2tpcHBlZF9iYXRjaGVzIiwKKyAgICApCisKKyAgICBkZWYgX19pbml0X18oc2VsZik6CisgICAgICAgIHNlbGYucmF3X21lbF9sZW5ndGhzID0gc2V0KCkgICMgcGVyLXNhbXBsZSBhY3R1YWwgbWVsIGZyYW1lcworICAgICAgICBzZWxmLnJhd19iYXRjaF9tYXhfbWVsID0gc2V0KCkgICMgcGVyLWJhdGNoIG1heCBtZWwgbGVuZ3RoIGJlZm9yZSBidWNrZXRpbmcKKyAgICAgICAgc2VsZi5idWNrZXRlZF9tZWxfbGVuZ3RocyA9IHNldCgpICAjIHBlci1iYXRjaCBwYWRkZWQgdGFyZ2V0IG1lbCBsZW5ndGgKKyAgICAgICAgc2VsZi5iYXRjaGVzID0gMAorICAgICAgICBzZWxmLnRvdGFsX2FjdHVhbF9mcmFtZXMgPSAwICAjIHN1bSBvZiBhY3R1YWwgbWVsIGZyYW1lcyBhY3Jvc3Mgc2FtcGxlcworICAgICAgICBzZWxmLnRvdGFsX3BhZGRlZF9mcmFtZXMgPSAwICAjIHRhcmdldCAqIGJhdGNoX3NpemUgc3VtbWVkIG92ZXIgYmF0Y2hlcworICAgICAgICBzZWxmLnNraXBwZWRfYmF0Y2hlcyA9IDAgICMgYmF0Y2hlcyB3aGVyZSB0aGUgcmF0aW8gY2FwIGtlcHQgdGhlIHBlci1iYXRjaCBtYXgKKworICAgIGRlZiByZWNvcmQoc2VsZiwgbWVsX2xlbmd0aHM6IHRvcmNoLlRlbnNvciwgdGFyZ2V0OiBpbnQpIC0+IE5vbmU6CisgICAgICAgICIiIlJlY29yZCBvbmUgYmF0Y2g6IGFjdHVhbCBwZXItc2FtcGxlIGxlbmd0aHMgYW5kIHRoZSBjaG9zZW4gcGFkIHRhcmdldC4iIiIKKyAgICAgICAgc2VsZi5iYXRjaGVzICs9IDEKKyAgICAgICAgc2VsZi5yYXdfYmF0Y2hfbWF4X21lbC5hZGQoaW50KG1lbF9sZW5ndGhzLmFtYXgoKSkpCisgICAgICAgIHNlbGYuYnVja2V0ZWRfbWVsX2xlbmd0aHMuYWRkKGludCh0YXJnZXQpKQorICAgICAgICBmb3IgbGVuZ3RoIGluIG1lbF9sZW5ndGhzLnRvbGlzdCgpOgorICAgICAgICAgICAgc2VsZi5yYXdfbWVsX2xlbmd0aHMuYWRkKGludChsZW5ndGgpKQorICAgICAgICBzZWxmLnRvdGFsX2FjdHVhbF9mcmFtZXMgKz0gaW50KG1lbF9sZW5ndGhzLnN1bSgpKQorICAgICAgICBzZWxmLnRvdGFsX3BhZGRlZF9mcmFtZXMgKz0gaW50KHRhcmdldCAqIGxlbihtZWxfbGVuZ3RocykpCisKKyAgICBkZWYgcmVjb3JkX3NraXAoc2VsZikgLT4gTm9uZToKKyAgICAgICAgc2VsZi5za2lwcGVkX2JhdGNoZXMgKz0gMQorCisgICAgZGVmIHN1bW1hcnkoc2VsZikgLT4gZGljdDoKKyAgICAgICAgYWN0dWFsID0gc2VsZi50b3RhbF9hY3R1YWxfZnJhbWVzCisgICAgICAgIG92ZXJoZWFkID0gKHNlbGYudG90YWxfcGFkZGVkX2ZyYW1lcyAtIGFjdHVhbCkgLyBtYXgoYWN0dWFsLCAxKQorICAgICAgICByZXR1cm4geworICAgICAgICAgICAgImJhdGNoZXMiOiBzZWxmLmJhdGNoZXMsCisgICAgICAgICAgICAidW5pcXVlX3Jhd19tZWxfbGVuZ3RocyI6IGxlbihzZWxmLnJhd19tZWxfbGVuZ3RocyksCisgICAgICAgICAgICAidW5pcXVlX3Jhd19iYXRjaF9tYXhfbWVsIjogbGVuKHNlbGYucmF3X2JhdGNoX21heF9tZWwpLAorICAgICAgICAgICAgInVuaXF1ZV9idWNrZXRlZF9tZWxfbGVuZ3RocyI6IGxlbihzZWxmLmJ1Y2tldGVkX21lbF9sZW5ndGhzKSwKKyAgICAgICAgICAgICJ0b3RhbF9hY3R1YWxfZnJhbWVzIjogYWN0dWFsLAorICAgICAgICAgICAgInRvdGFsX3BhZGRlZF9mcmFtZXMiOiBzZWxmLnRvdGFsX3BhZGRlZF9mcmFtZXMsCisgICAgICAgICAgICAicGFkZGluZ19vdmVyaGVhZF9yYXRpbyI6IG92ZXJoZWFkLAorICAgICAgICAgICAgInNraXBwZWRfYmF0Y2hlc19yYXRpb19jYXAiOiBzZWxmLnNraXBwZWRfYmF0Y2hlcywKKyAgICAgICAgfQorCisKK2RlZiBfYnVja2V0X2NlaWwodmFsdWU6IGludCwgYnVja2V0X3NpemU6IGludCkgLT4gaW50OgorICAgICMgU21hbGxlc3QgbXVsdGlwbGUgb2YgYnVja2V0X3NpemUgPj0gdmFsdWU7IGJ1Y2tldF9zaXplIGFzc3VtZWQgPiAwLgorICAgIHJldHVybiAoKHZhbHVlICsgYnVja2V0X3NpemUgLSAxKSAvLyBidWNrZXRfc2l6ZSkgKiBidWNrZXRfc2l6ZQorCisKK2RlZiBidWNrZXRlZF9jb2xsYXRlX2ZuKAorICAgIGJhdGNoLAorICAgICosCisgICAgbWVsX2J1Y2tldF9zaXplOiBpbnQgfCBOb25lID0gTm9uZSwKKyAgICBtYXhfcGFkZGluZ19yYXRpbzogZmxvYXQgPSAwLjA1LAorICAgIHN0YXRzOiBCdWNrZXRTdGF0cyB8IE5vbmUgPSBOb25lLAorKToKKyAgICAiIiJDb2xsYXRlIHdpdGggb3B0aW9uYWwgYm91bmRlZCBtZWwtbGVuZ3RoIGJ1Y2tldGluZyBmb3IgdG9yY2guY29tcGlsZS4KKworICAgIFdoZW4gYGBtZWxfYnVja2V0X3NpemVgYCBpcyBOb25lIChkZWZhdWx0KSB0aGlzIGlzIGlkZW50aWNhbCB0bworICAgIDpmdW5jOmBjb2xsYXRlX2ZuYDogbWVsIGZyYW1lcyBhcmUgcGFkZGVkIHRvIHRoZSBwZXItYmF0Y2ggbWF4aW11bS4KKworICAgIFdoZW4gYGBtZWxfYnVja2V0X3NpemVgYCBpcyBhIHBvc2l0aXZlIGludCwgbWVsIGZyYW1lcyBhcmUgcGFkZGVkIHRvIHRoZQorICAgIG5leHQgbXVsdGlwbGUgb2YgYGBtZWxfYnVja2V0X3NpemVgYCBhYm92ZSB0aGUgcGVyLWJhdGNoIG1heCwgYnV0IG9ubHkgaWYKKyAgICB0aGUgKmFkZGVkKiBwYWRkaW5nIChjZWlsaW5nIC0gcGVyLWJhdGNoIG1heCkgc3RheXMgd2l0aGluCisgICAgYGBtYXhfcGFkZGluZ19yYXRpb2BgIG9mIHRoZSBwZXItYmF0Y2ggbWF4OyBvdGhlcndpc2UgdGhlIHBlci1iYXRjaCBtYXggaXMKKyAgICB1c2VkIGFuZCB0aGUgYmF0Y2ggaXMgY291bnRlZCBhcyBza2lwcGVkLiBUaGlzIGJvdW5kcyBzaGFwZSBjYXJkaW5hbGl0eQorICAgIHdpdGhvdXQgZ2xvYmFsIHBhZGRpbmcgYW5kIGNhcHMgdGhlIHdvcnN0LWNhc2UgcGFkZGluZyBvdmVyaGVhZCBwZXIgYmF0Y2guCisKKyAgICBBY3R1YWwgbWVsIGxlbmd0aHMgYXJlIGFsd2F5cyByZXR1cm5lZCB1bmNoYW5nZWQgaW4gYGBtZWxfbGVuZ3Roc2BgLCBzbyB0aGUKKyAgICBkb3duc3RyZWFtIG1hc2sgYW5kIG1hc2tlZCBmbG93LW1hdGNoaW5nIGxvc3MgYXJlIHVuYWZmZWN0ZWQgYnkgdGhlIHBhZGRpbmcKKyAgICB0YXJnZXQgKHBhZGRlZCBmcmFtZXMgYXJlIG1hc2tlZCBvdXQgb2YgdGhlIGxvc3MpLgorICAgICIiIgogICAgIG1lbF9zcGVjcyA9IFtpdGVtWyJtZWxfc3BlYyJdLnNxdWVlemUoMCkgZm9yIGl0ZW0gaW4gYmF0Y2hdCiAgICAgbWVsX2xlbmd0aHMgPSB0b3JjaC5Mb25nVGVuc29yKFtzcGVjLnNoYXBlWy0xXSBmb3Igc3BlYyBpbiBtZWxfc3BlY3NdKQotICAgIG1heF9tZWxfbGVuZ3RoID0gbWVsX2xlbmd0aHMuYW1heCgpCisgICAgbWF4X21lbF9sZW5ndGggPSBpbnQobWVsX2xlbmd0aHMuYW1heCgpKQorCisgICAgaWYgbWVsX2J1Y2tldF9zaXplIGlzIG5vdCBOb25lIGFuZCBtZWxfYnVja2V0X3NpemUgPiAwOgorICAgICAgICBjZWlsaW5nID0gX2J1Y2tldF9jZWlsKG1heF9tZWxfbGVuZ3RoLCBtZWxfYnVja2V0X3NpemUpCisgICAgICAgIGFkZGVkID0gY2VpbGluZyAtIG1heF9tZWxfbGVuZ3RoCisgICAgICAgIGlmIG1heF9tZWxfbGVuZ3RoID4gMCBhbmQgYWRkZWQgLyBtYXhfbWVsX2xlbmd0aCA8PSBtYXhfcGFkZGluZ19yYXRpbzoKKyAgICAgICAgICAgIHRhcmdldCA9IGNlaWxpbmcKKyAgICAgICAgZWxzZToKKyAgICAgICAgICAgIHRhcmdldCA9IG1heF9tZWxfbGVuZ3RoCisgICAgICAgICAgICBpZiBzdGF0cyBpcyBub3QgTm9uZToKKyAgICAgICAgICAgICAgICBzdGF0cy5yZWNvcmRfc2tpcCgpCisgICAgZWxzZToKKyAgICAgICAgdGFyZ2V0ID0gbWF4X21lbF9sZW5ndGgKKworICAgIGlmIHN0YXRzIGlzIG5vdCBOb25lOgorICAgICAgICBzdGF0cy5yZWNvcmQobWVsX2xlbmd0aHMsIHRhcmdldCkKIAogICAgIHBhZGRlZF9tZWxfc3BlY3MgPSBbXQogICAgIGZvciBzcGVjIGluIG1lbF9zcGVjczoKLSAgICAgICAgcGFkZGluZyA9ICgwLCBtYXhfbWVsX2xlbmd0aCAtIHNwZWMuc2l6ZSgtMSkpCi0gICAgICAgIHBhZGRlZF9zcGVjID0gRi5wYWQoc3BlYywgcGFkZGluZywgdmFsdWU9MCkKKyAgICAgICAgcGFkZGVkX3NwZWMgPSBGLnBhZChzcGVjLCAoMCwgdGFyZ2V0IC0gc3BlYy5zaXplKC0xKSksIHZhbHVlPTApCiAgICAgICAgIHBhZGRlZF9tZWxfc3BlY3MuYXBwZW5kKHBhZGRlZF9zcGVjKQogCiAgICAgbWVsX3NwZWNzID0gdG9yY2guc3RhY2socGFkZGVkX21lbF9zcGVjcykKQEAgLTMzMiwzICs0MzAsOSBAQCBkZWYgY29sbGF0ZV9mbihiYXRjaCk6CiAgICAgICAgIHRleHQ9dGV4dCwKICAgICAgICAgdGV4dF9sZW5ndGhzPXRleHRfbGVuZ3RocywKICAgICApCisKKworZGVmIGNvbGxhdGVfZm4oYmF0Y2gpOgorICAgICMgRGVmYXVsdCBjb2xsYXRlIHByZXNlcnZlcyB0aGUgaGlzdG9yaWNhbCBiZWhhdmlvdXI6IHBhZCB0byB0aGUgcGVyLWJhdGNoCisgICAgIyBtYXggbWVsIGxlbmd0aC4gRGVsZWdhdGVzIHRvIGJ1Y2tldGVkX2NvbGxhdGVfZm4gd2l0aCBidWNrZXRpbmcgZGlzYWJsZWQuCisgICAgcmV0dXJuIGJ1Y2tldGVkX2NvbGxhdGVfZm4oYmF0Y2gpCmRpZmYgLS1naXQgYS9zcmMvZjVfdHRzL21vZGVsL3RyYWluZXIucHkgYi9zcmMvZjVfdHRzL21vZGVsL3RyYWluZXIucHkKaW5kZXggMTEzYzVjMC4uZjBlNmI0ZSAxMDA2NDQKLS0tIGEvc3JjL2Y1X3R0cy9tb2RlbC90cmFpbmVyLnB5CisrKyBiL3NyYy9mNV90dHMvbW9kZWwvdHJhaW5lci5weQpAQCAtMTcsNyArMTcsOSBAQCBmcm9tIHRvcmNoLnV0aWxzLmRhdGEgaW1wb3J0IERhdGFMb2FkZXIsIERhdGFzZXQsIFNlcXVlbnRpYWxTYW1wbGVyCiBmcm9tIHRxZG0gaW1wb3J0IHRxZG0KIAogZnJvbSBmNV90dHMubW9kZWwgaW1wb3J0IENGTQotZnJvbSBmNV90dHMubW9kZWwuZGF0YXNldCBpbXBvcnQgRHluYW1pY0JhdGNoU2FtcGxlciwgY29sbGF0ZV9mbgoraW1wb3J0IGZ1bmN0b29scworCitmcm9tIGY1X3R0cy5tb2RlbC5kYXRhc2V0IGltcG9ydCBCdWNrZXRTdGF0cywgRHluYW1pY0JhdGNoU2FtcGxlciwgYnVja2V0ZWRfY29sbGF0ZV9mbiwgY29sbGF0ZV9mbgogZnJvbSBmNV90dHMubW9kZWwudXRpbHMgaW1wb3J0IGRlZmF1bHQsIGV4aXN0cwogCiAKQEAgLTYwLDYgKzYyLDEyIEBAIGNsYXNzIFRyYWluZXI6CiAgICAgICAgIGNvbXBpbGVfZnVsbGdyYXBoOiBib29sID0gRmFsc2UsCiAgICAgICAgIGNvbXBpbGVfZHluYW1pYzogYm9vbCB8IE5vbmUgPSBOb25lLAogICAgICAgICBjb21waWxlX2ZhbGxiYWNrX3RvX2VhZ2VyOiBib29sID0gVHJ1ZSwKKyAgICAgICAgIyBPcHQtaW4gYm91bmRlZCBjb21waWxlLWJ1Y2tldGluZyAoZGVmYXVsdCBvZmY7IGRlZmF1bHQgY29sbGF0ZS9zYW1wbGVyCisgICAgICAgICMgYmVoYXZpb3VyIGlzIHVuY2hhbmdlZCB1bmxlc3MgbWVsX2J1Y2tldF9zaXplIGlzIGEgcG9zaXRpdmUgaW50KS4KKyAgICAgICAgYnVja2V0X21lbF9zaXplOiBpbnQgfCBOb25lID0gTm9uZSwKKyAgICAgICAgYnVja2V0X21heF9wYWRkaW5nX3JhdGlvOiBmbG9hdCA9IDAuMDUsCisgICAgICAgIGFsaWduX3RleHRfbGVuX3RvX21lbDogYm9vbCA9IEZhbHNlLAorICAgICAgICBjb21waWxlX21hcmtfZHluYW1pYzogYm9vbCA9IEZhbHNlLAogICAgICAgICBnbG9iYWxfbWFza2VkX21lYW46IGJvb2wgPSBGYWxzZSwKICAgICApOgogICAgICAgICBkZHBfa3dhcmdzID0gRGlzdHJpYnV0ZWREYXRhUGFyYWxsZWxLd2FyZ3MoZmluZF91bnVzZWRfcGFyYW1ldGVycz1UcnVlKQpAQCAtMTUwLDYgKzE1OCwxNCBAQCBjbGFzcyBUcmFpbmVyOgogICAgICAgICBzZWxmLmNvbXBpbGVfZnVsbGdyYXBoID0gY29tcGlsZV9mdWxsZ3JhcGgKICAgICAgICAgc2VsZi5jb21waWxlX2R5bmFtaWMgPSBjb21waWxlX2R5bmFtaWMKICAgICAgICAgc2VsZi5jb21waWxlX2ZhbGxiYWNrX3RvX2VhZ2VyID0gY29tcGlsZV9mYWxsYmFja190b19lYWdlcgorICAgICAgICAjIE9wdC1pbiBib3VuZGVkIGNvbXBpbGUtYnVja2V0aW5nIChkZWZhdWx0IG9mZikuIFdoZW4gYnVja2V0X21lbF9zaXplIGlzIHNldCwKKyAgICAgICAgIyB0aGUgY29sbGF0ZSBwYWRzIG1lbCBsZW5ndGhzIHRvIGEgYm91bmRlZCBidWNrZXQgY2VpbGluZyBhbmQgYSBCdWNrZXRTdGF0cworICAgICAgICAjIGhhbmRsZSByZWNvcmRzIG1lYXN1cmVkIHBhZGRpbmcgb3ZlcmhlYWQgYW5kIHVuaXF1ZS1zaGFwZSBjYXJkaW5hbGl0eS4KKyAgICAgICAgc2VsZi5idWNrZXRfbWVsX3NpemUgPSBidWNrZXRfbWVsX3NpemUKKyAgICAgICAgc2VsZi5idWNrZXRfbWF4X3BhZGRpbmdfcmF0aW8gPSBidWNrZXRfbWF4X3BhZGRpbmdfcmF0aW8KKyAgICAgICAgc2VsZi5hbGlnbl90ZXh0X2xlbl90b19tZWwgPSBhbGlnbl90ZXh0X2xlbl90b19tZWwKKyAgICAgICAgc2VsZi5jb21waWxlX21hcmtfZHluYW1pYyA9IGNvbXBpbGVfbWFya19keW5hbWljCisgICAgICAgIHNlbGYuYnVja2V0X3N0YXRzID0gQnVja2V0U3RhdHMoKSBpZiBidWNrZXRfbWVsX3NpemUgaXMgbm90IE5vbmUgZWxzZSBOb25lCiAgICAgICAgICMgT3B0LWluIGdsb2JhbCBtYXNrZWQtbWVhbiBsb3NzIG5vcm1hbGl6YXRpb24gKGRlZmF1bHQgb2ZmKTogd2hlbiBlbmFibGVkIHRoZQogICAgICAgICAjIHRyYWluZXIgYmFja3Byb3BzIHBlci1taWNyb2JhdGNoIGxvc3Nfc3VtIGFuZCByZXNjYWxlcyBzeW5jZWQgZ3JhZGllbnRzIGJ5IHRoZQogICAgICAgICAjIGdsb2JhbCBtYXNrZWQtZnJhbWUgZGVub21pbmF0b3IgYWNyb3NzIGFjY3VtdWxhdGlvbiB3aW5kb3dzIGFuZCBERFAgcmFua3MsIHNvCkBAIC0xNzEsNiArMTg3LDEyIEBAIGNsYXNzIFRyYWluZXI6CiAgICAgICAgICAgICBzZWxmLm9wdGltaXplciA9IEFkYW1XKG1vZGVsLnBhcmFtZXRlcnMoKSwgbHI9bGVhcm5pbmdfcmF0ZSwgZnVzZWQ9dXNlX2Z1c2VkKQogICAgICAgICBzZWxmLm1vZGVsLCBzZWxmLm9wdGltaXplciA9IHNlbGYuYWNjZWxlcmF0b3IucHJlcGFyZShzZWxmLm1vZGVsLCBzZWxmLm9wdGltaXplcikKICAgICAgICAgc2VsZi5fdW53cmFwcGVkX21vZGVsID0gc2VsZi5hY2NlbGVyYXRvci51bndyYXBfbW9kZWwoc2VsZi5tb2RlbCkKKyAgICAgICAgIyBQcm9wYWdhdGUgb3B0LWluIHNoYXBlLXN0YWJpbGlzYXRpb24gZmxhZ3MgdG8gdGhlIENGTSBiZWZvcmUgYW55IGNvbXBpbGUgY2FsbC4KKyAgICAgICAgIyBUaGVzZSBkZWZhdWx0IHRvIEZhbHNlIG9uIENGTTsgb25seSBvdmVycmlkZSB3aGVuIHRoZSB0cmFpbmVyIG9wdHMgaW4uCisgICAgICAgIGlmIGhhc2F0dHIoc2VsZi5fdW53cmFwcGVkX21vZGVsLCAiYWxpZ25fdGV4dF9sZW5fdG9fbWVsIik6CisgICAgICAgICAgICBzZWxmLl91bndyYXBwZWRfbW9kZWwuYWxpZ25fdGV4dF9sZW5fdG9fbWVsID0gc2VsZi5hbGlnbl90ZXh0X2xlbl90b19tZWwKKyAgICAgICAgaWYgaGFzYXR0cihzZWxmLl91bndyYXBwZWRfbW9kZWwsICJjb21waWxlX21hcmtfZHluYW1pYyIpOgorICAgICAgICAgICAgc2VsZi5fdW53cmFwcGVkX21vZGVsLmNvbXBpbGVfbWFya19keW5hbWljID0gc2VsZi5jb21waWxlX21hcmtfZHluYW1pYwogICAgICAgICBzZWxmLl9jb25maWd1cmVfY29tcGlsZSgpCiAKICAgICBAcHJvcGVydHkKQEAgLTQyNiwxMCArNDQ4LDI1IEBAIGNsYXNzIFRyYWluZXI6CiAKICAgICAgICAgcGVyc2lzdGVudF93b3JrZXJzID0gbnVtX3dvcmtlcnMgPiAwCiAKKyAgICAgICAgIyBPcHQtaW4gYm91bmRlZCBjb21waWxlLWJ1Y2tldGluZzogd2hlbiBidWNrZXRfbWVsX3NpemUgaXMgc2V0LCBwYWQgbWVsCisgICAgICAgICMgbGVuZ3RocyB0byBhIGJvdW5kZWQgYnVja2V0IGNlaWxpbmcgYW5kIGFjY3VtdWxhdGUgcGFkZGluZy9zaGFwZSBzdGF0cy4KKyAgICAgICAgIyBEZWZhdWx0IChidWNrZXRfbWVsX3NpemUgaXMgTm9uZSkgdXNlcyB0aGUgdW5jaGFuZ2VkIGNvbGxhdGVfZm4uCisgICAgICAgICMgZ2V0YXR0ciBrZWVwcyB0cmFpbigpIHJvYnVzdCB0byB0ZXN0IGhhcm5lc3NlcyB0aGF0IGJ5cGFzcyBfX2luaXRfXy4KKyAgICAgICAgYnVja2V0X21lbF9zaXplID0gZ2V0YXR0cihzZWxmLCAiYnVja2V0X21lbF9zaXplIiwgTm9uZSkKKyAgICAgICAgaWYgYnVja2V0X21lbF9zaXplIGlzIG5vdCBOb25lOgorICAgICAgICAgICAgY29sbGF0ZSA9IGZ1bmN0b29scy5wYXJ0aWFsKAorICAgICAgICAgICAgICAgIGJ1Y2tldGVkX2NvbGxhdGVfZm4sCisgICAgICAgICAgICAgICAgbWVsX2J1Y2tldF9zaXplPWJ1Y2tldF9tZWxfc2l6ZSwKKyAgICAgICAgICAgICAgICBtYXhfcGFkZGluZ19yYXRpbz1nZXRhdHRyKHNlbGYsICJidWNrZXRfbWF4X3BhZGRpbmdfcmF0aW8iLCAwLjA1KSwKKyAgICAgICAgICAgICAgICBzdGF0cz1nZXRhdHRyKHNlbGYsICJidWNrZXRfc3RhdHMiLCBOb25lKSwKKyAgICAgICAgICAgICkKKyAgICAgICAgZWxzZToKKyAgICAgICAgICAgIGNvbGxhdGUgPSBjb2xsYXRlX2ZuCisKICAgICAgICAgaWYgc2VsZi5iYXRjaF9zaXplX3R5cGUgPT0gInNhbXBsZSI6CiAgICAgICAgICAgICB0cmFpbl9kYXRhbG9hZGVyID0gRGF0YUxvYWRlcigKICAgICAgICAgICAgICAgICB0cmFpbl9kYXRhc2V0LAotICAgICAgICAgICAgICAgIGNvbGxhdGVfZm49Y29sbGF0ZV9mbiwKKyAgICAgICAgICAgICAgICBjb2xsYXRlX2ZuPWNvbGxhdGUsCiAgICAgICAgICAgICAgICAgbnVtX3dvcmtlcnM9bnVtX3dvcmtlcnMsCiAgICAgICAgICAgICAgICAgcGluX21lbW9yeT1UcnVlLAogICAgICAgICAgICAgICAgIHBlcnNpc3RlbnRfd29ya2Vycz1wZXJzaXN0ZW50X3dvcmtlcnMsCkBAIC00NDksNyArNDg2LDcgQEAgY2xhc3MgVHJhaW5lcjoKICAgICAgICAgICAgICkKICAgICAgICAgICAgIHRyYWluX2RhdGFsb2FkZXIgPSBEYXRhTG9hZGVyKAogICAgICAgICAgICAgICAgIHRyYWluX2RhdGFzZXQsCi0gICAgICAgICAgICAgICAgY29sbGF0ZV9mbj1jb2xsYXRlX2ZuLAorICAgICAgICAgICAgICAgIGNvbGxhdGVfZm49Y29sbGF0ZSwKICAgICAgICAgICAgICAgICBudW1fd29ya2Vycz1udW1fd29ya2VycywKICAgICAgICAgICAgICAgICBwaW5fbWVtb3J5PVRydWUsCiAgICAgICAgICAgICAgICAgcGVyc2lzdGVudF93b3JrZXJzPXBlcnNpc3RlbnRfd29ya2VycywKQEAgLTYyOCw0ICs2NjUsMTcgQEAgY2xhc3MgVHJhaW5lcjoKIAogICAgICAgICBzZWxmLnNhdmVfY2hlY2twb2ludChnbG9iYWxfdXBkYXRlLCBsYXN0PVRydWUpCiAKKyAgICAgICAgIyBSZXBvcnQgb3B0LWluIGJ1Y2tldGluZyBzdGF0cyAobWVhc3VyZWQgcGFkZGluZyBvdmVyaGVhZCArIHVuaXF1ZS1zaGFwZQorICAgICAgICAjIGNhcmRpbmFsaXR5KSBzbyB0aGUgY29tcGlsZS1yZWNvbXBpbGUgdHJhZGUtb2ZmIGlzIG9ic2VydmFibGUuIE1haW4gcmFuayBvbmx5LgorICAgICAgICBidWNrZXRfc3RhdHMgPSBnZXRhdHRyKHNlbGYsICJidWNrZXRfc3RhdHMiLCBOb25lKQorICAgICAgICBpZiBidWNrZXRfc3RhdHMgaXMgbm90IE5vbmUgYW5kIHNlbGYuaXNfbWFpbjoKKyAgICAgICAgICAgIHN1bW1hcnkgPSBidWNrZXRfc3RhdHMuc3VtbWFyeSgpCisgICAgICAgICAgICBwcmludCgKKyAgICAgICAgICAgICAgICBmIltidWNrZXRpbmddIG1lbF9idWNrZXRfc2l6ZT17Z2V0YXR0cihzZWxmLCAnYnVja2V0X21lbF9zaXplJywgTm9uZSl9ICIKKyAgICAgICAgICAgICAgICBmIm1heF9wYWRkaW5nX3JhdGlvPXtnZXRhdHRyKHNlbGYsICdidWNrZXRfbWF4X3BhZGRpbmdfcmF0aW8nLCAwLjA1KX0gIgorICAgICAgICAgICAgICAgIGYiYWxpZ25fdGV4dF9sZW5fdG9fbWVsPXtnZXRhdHRyKHNlbGYsICdhbGlnbl90ZXh0X2xlbl90b19tZWwnLCBGYWxzZSl9ICIKKyAgICAgICAgICAgICAgICBmImNvbXBpbGVfbWFya19keW5hbWljPXtnZXRhdHRyKHNlbGYsICdjb21waWxlX21hcmtfZHluYW1pYycsIEZhbHNlKX0gfCB7c3VtbWFyeX0iCisgICAgICAgICAgICApCisgICAgICAgICAgICBzZWxmLmFjY2VsZXJhdG9yLmxvZyh7ImJ1Y2tldGluZy8iICsgazogdiBmb3IgaywgdiBpbiBzdW1tYXJ5Lml0ZW1zKCl9LCBzdGVwPWludChnbG9iYWxfdXBkYXRlKSkKKwogICAgICAgICBzZWxmLmFjY2VsZXJhdG9yLmVuZF90cmFpbmluZygpCg=="""


def run(cmd: list[str] | str, cwd: str | Path | None = None, env: dict[str, str] | None = None) -> None:
    printable = cmd if isinstance(cmd, str) else " ".join(cmd)
    print(f"\n$ {printable}", flush=True)
    completed = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        shell=isinstance(cmd, str),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n", flush=True)
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, cmd)


def capture(cmd: list[str] | str, cwd: str | Path | None = None, env: dict[str, str] | None = None) -> str:
    printable = cmd if isinstance(cmd, str) else " ".join(cmd)
    print(f"\n$ {printable}", flush=True)
    return subprocess.check_output(cmd, cwd=cwd, env=env, shell=isinstance(cmd, str), text=True, stderr=subprocess.STDOUT)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def apply_bucket_agent_patch(repo: Path) -> None:
    patch_text = base64.b64decode(BUCKET_AGENT_PATCH_B64.encode("ascii")).decode("utf-8")
    patch_path = repo.parent / "bucket-targeted-dynamic.patch"
    patch_path.write_text(patch_text, encoding="utf-8")
    check = subprocess.run(["git", "apply", "--check", str(patch_path)], cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if check.returncode == 0:
        run(["git", "apply", str(patch_path)], cwd=repo)
        print("Applied bucket-targeted-dynamic patch", flush=True)
        return
    reverse = subprocess.run(["git", "apply", "--reverse", "--check", str(patch_path)], cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if reverse.returncode == 0:
        print("Bucket-targeted-dynamic patch already applied", flush=True)
        return
    print(check.stdout, flush=True)
    raise RuntimeError("bucket-targeted-dynamic patch did not apply cleanly")


def setup_repo(work_dir: Path, repo_url: str, branch: str, commit: str | None) -> Path:
    repo = work_dir / "F5-TTS"
    if repo.exists():
        run(["git", "-C", str(repo), "fetch", "origin", branch, "--depth", "1"])
    else:
        run(["git", "clone", "--branch", branch, "--single-branch", "--depth", "1", repo_url, str(repo)])
    if commit:
        run(["git", "-C", str(repo), "fetch", "origin", commit, "--depth", "1"])
        run(["git", "-C", str(repo), "checkout", commit])
    head = capture(["git", "rev-parse", "HEAD"], cwd=repo).strip()
    print(f"Repo HEAD: {head}", flush=True)
    return repo


def install_repo(repo: Path) -> None:
    # Colab normally ships CUDA torch/torchaudio. Keep them, install project deps.
    run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip", "setuptools", "wheel"])
    run([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo)])


def prepare_librispeech_subset(
    repo: Path,
    out_file: Path,
    samples: int,
    fetch_rows: int,
    max_duration: float,
    min_duration: float,
) -> dict[str, Any]:
    sys.path.insert(0, str(repo / "src"))
    import torch
    import torchaudio
    from datasets import Audio, load_dataset
    from f5_tts.model.modules import MelSpec

    if out_file.exists():
        print(f"Using existing preprocessed dataset: {out_file}", flush=True)
        obj = torch.load(out_file, map_location="cpu")
        return obj["metadata"]

    print(
        f"Streaming real dataset {DATASET_NAME}/{DATASET_CONFIG} split={DATASET_SPLIT} scan_rows<={fetch_rows}",
        flush=True,
    )
    # Streaming avoids materialising the full LibriSpeech config. Non-streaming split slices
    # still resolve/download many parquet files on Colab, which is slow and wasteful.
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT, streaming=True)
    ds = ds.shuffle(buffer_size=min(max(fetch_rows, samples), 5000), seed=666)
    # Decode at source sample rate first. Repo dataset wrapper resamples to 24 kHz.
    ds = ds.cast_column("audio", Audio(decode=True))

    mel_spec = MelSpec(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=24_000,
        mel_spec_type="vocos",
    )
    resamplers: dict[int, Any] = {}
    mels: list[torch.Tensor] = []
    texts: list[str] = []
    durations: list[float] = []
    speakers: set[str] = set()
    frame_lengths: list[int] = []

    start = time.perf_counter()
    with torch.no_grad():
        scanned = 0
        for row in ds:
            scanned += 1
            if scanned > fetch_rows:
                break
            audio = row["audio"]
            array = audio["array"]
            sr = int(audio["sampling_rate"])
            duration = float(len(array) / sr)
            if duration < min_duration or duration > max_duration:
                continue
            wav = torch.as_tensor(array, dtype=torch.float32).unsqueeze(0)
            if sr != 24_000:
                if sr not in resamplers:
                    resamplers[sr] = torchaudio.transforms.Resample(sr, 24_000)
                wav = resamplers[sr](wav)
            mel = mel_spec(wav).squeeze(0).contiguous().cpu()
            text = str(row.get("text") or row.get("sentence") or "").strip()
            if not text:
                continue
            mels.append(mel)
            texts.append(text)
            durations.append(duration)
            frame_lengths.append(int(mel.shape[-1]))
            if "speaker_id" in row:
                speakers.add(str(row["speaker_id"]))
            elif "speaker" in row:
                speakers.add(str(row["speaker"]))
            if len(mels) >= samples:
                break

    if len(mels) < samples:
        raise RuntimeError(f"Only prepared {len(mels)} valid samples, requested {samples}")

    metadata = {
        "dataset": DATASET_NAME,
        "config": DATASET_CONFIG,
        "split": DATASET_SPLIT,
        "samples": len(mels),
        "fetch_rows": fetch_rows,
        "streaming": True,
        "min_duration_s": min(durations),
        "median_duration_s": statistics.median(durations),
        "max_duration_s": max(durations),
        "total_hours": sum(durations) / 3600,
        "unique_speakers": len(speakers) if speakers else None,
        "min_frames": min(frame_lengths),
        "median_frames": statistics.median(frame_lengths),
        "max_frames": max(frame_lengths),
        "unique_frame_lengths": len(set(frame_lengths)),
        "preprocess_wall_s": time.perf_counter() - start,
        "dtype": "float32",
        "sample_rate": 24_000,
        "n_mel_channels": 100,
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mels": mels, "texts": texts, "durations": durations, "metadata": metadata}, out_file)
    write_json(out_file.with_suffix(".metadata.json"), metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    return metadata


def candidate_worker(args: argparse.Namespace) -> None:
    repo = Path(args.repo).resolve()
    data_file = Path(args.data_file).resolve()
    out_dir = Path(args.out_dir).resolve()
    candidate = args.candidate
    cfg = CANDIDATES[candidate]

    env_info: dict[str, Any] = {}
    sys.path.insert(0, str(repo / "src"))

    import torch
    from torch.utils.data import Dataset, SequentialSampler
    from f5_tts.model import CFM, DiT, Trainer
    import f5_tts.model.trainer as trainer_mod
    from f5_tts.model.dataset import DynamicBatchSampler
    from f5_tts.model.utils import get_tokenizer

    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        try:
            torch.set_float32_matmul_precision("highest")
        except Exception:
            pass

    try:
        import torch._dynamo as dynamo

        dynamo.reset()
        dynamo.utils.counters.clear()
        if cfg.get("automatic_dynamic_shapes") is not None:
            dynamo.config.automatic_dynamic_shapes = bool(cfg["automatic_dynamic_shapes"])
    except Exception:
        dynamo = None

    class PrecomputedMelDataset(Dataset):
        def __init__(self, path: Path):
            obj = torch.load(path, map_location="cpu")
            self.mels = obj["mels"]
            self.texts = obj["texts"]
            self.durations = obj["durations"]
            self.metadata = obj["metadata"]

        def __len__(self) -> int:
            return len(self.mels)

        def get_frame_len(self, index: int) -> int:
            return int(self.mels[index].shape[-1])

        def __getitem__(self, index: int) -> dict[str, Any]:
            return {"mel_spec": self.mels[index], "text": self.texts[index]}

    dataset = PrecomputedMelDataset(data_file)
    sampler = SequentialSampler(dataset)
    batch_sampler = DynamicBatchSampler(
        sampler,
        args.batch_size_per_gpu,
        max_samples=args.max_samples,
        random_seed=args.seed,
        drop_residual=False,
    )
    batches_per_epoch = len(batch_sampler)
    updates_expected = batches_per_epoch * args.epochs
    batch_shapes = []
    for batch in batch_sampler.batches:
        lengths = [dataset.get_frame_len(i) for i in batch]
        texts = [len(dataset.texts[i]) for i in batch]
        batch_shapes.append((len(batch), max(lengths), max(texts)))

    vocab_char_map, vocab_size = get_tokenizer("", "byte")
    model_cfg = dict(
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        ff_mult=2,
        text_dim=args.text_dim,
        text_mask_padding=True,
        qk_norm=None,
        conv_layers=4,
        pe_attn_head=None,
        attn_backend="torch",
        attn_mask_enabled=False,
        checkpoint_activations=False,
    )
    mel_spec_kwargs = dict(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=24_000,
        mel_spec_type="vocos",
    )
    model = CFM(
        transformer=DiT(**model_cfg, text_num_embeds=vocab_size, mel_dim=100),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )

    compile_enabled_for_trainer = bool(cfg["compile"] and cfg.get("options") is None)
    trainer = Trainer(
        model,
        epochs=args.epochs,
        learning_rate=7.5e-5,
        num_warmup_updates=1,
        save_per_updates=10**9,
        keep_last_n_checkpoints=0,
        checkpoint_path=str(out_dir / "ckpts" / candidate),
        batch_size_per_gpu=args.batch_size_per_gpu,
        batch_size_type="frame",
        max_samples=args.max_samples,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        logger=None,
        wandb_project="f5tts-real-trainer-bench",
        wandb_run_name=candidate,
        log_samples=False,
        last_per_updates=10**9,
        bnb_optimizer=False,
        mel_spec_type="vocos",
        compile_enabled=compile_enabled_for_trainer,
        compile_backend="inductor",
        compile_mode=cfg.get("mode"),
        compile_fullgraph=False,
        compile_dynamic=cfg.get("dynamic"),
        compile_fallback_to_eager=False,
        bucket_mel_size=cfg.get("bucket_mel_size"),
        bucket_max_padding_ratio=cfg.get("bucket_max_padding_ratio", 0.05),
        align_text_len_to_mel=bool(cfg.get("align_text_len_to_mel", False)),
        compile_mark_dynamic=bool(cfg.get("compile_mark_dynamic", False)),
        global_masked_mean=False,
    )
    # Avoid checkpoint I/O dominating short benchmark timing while preserving Trainer/Accelerate training behavior.
    trainer.save_checkpoint = lambda *a, **k: None  # type: ignore[method-assign]

    if cfg["compile"] and cfg.get("options") is not None:
        trainer._unwrapped_model.compile_training_core(
            backend="inductor",
            fullgraph=False,
            dynamic=cfg.get("dynamic"),
            options=cfg["options"],
            runtime_fallback=False,
        )
        trainer.compile_active = True

    if cfg.get("mark_dynamic"):
        if dynamo is None or not hasattr(dynamo, "mark_dynamic"):
            raise RuntimeError("torch._dynamo.mark_dynamic is unavailable")
        module = trainer._unwrapped_model
        original_run_loss_core = module._run_loss_core

        def run_loss_core_with_mark_dynamic(*loss_args: Any):
            # _run_loss_core args: x1, text, mask, rand_span_mask, x0, time, drop_audio_cond, drop_text.
            # The real LibriSpeech dynamic sampler produced 65 unique (mel_seq, text_seq) shapes;
            # marking sequence dims lets Dynamo generalize those dimensions instead of specialising every batch.
            for index, dim in ((0, 1), (1, 1), (2, 1), (3, 1), (4, 1)):
                value = loss_args[index]
                if torch.is_tensor(value) and value.dim() > dim:
                    dynamo.mark_dynamic(value, dim)
            return original_run_loss_core(*loss_args)

        module._run_loss_core = run_loss_core_with_mark_dynamic  # type: ignore[method-assign]

    if cfg.get("mark_step"):
        mark_step = getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None)
        if mark_step is None:
            raise RuntimeError("torch.compiler.cudagraph_mark_step_begin is unavailable")
        module = trainer._unwrapped_model
        original_forward = module.forward

        def forward_with_mark_step(*forward_args: Any, **forward_kwargs: Any):
            if trainer.compile_active:
                mark_step()
            return original_forward(*forward_args, **forward_kwargs)

        module.forward = forward_with_mark_step  # type: ignore[method-assign]

    update_records: list[dict[str, Any]] = []
    original_tqdm = trainer_mod.tqdm
    candidate_progress_start = time.perf_counter()

    class BenchProgress:
        def __init__(self, iterable, desc=None, unit=None, disable=False, initial=0, **kwargs):
            self.iterable = iterable
            self.desc = desc or ""
            self.disable = disable
            self.count = int(initial or 0)
            self.last = time.perf_counter()

        def update(self, n=1):
            now = time.perf_counter()
            self.count += int(n)
            rec = {"desc": self.desc, "update_in_epoch": self.count, "dt_s": now - self.last}
            if torch.cuda.is_available():
                rec["max_alloc_mb"] = torch.cuda.max_memory_allocated() / 1024**2
                rec["max_reserved_mb"] = torch.cuda.max_memory_reserved() / 1024**2
            update_records.append(rec)
            progress = {
                "candidate": candidate,
                "update": self.count,
                "dt_s": rec["dt_s"],
                "elapsed_s": now - candidate_progress_start,
                "max_alloc_mb": rec.get("max_alloc_mb"),
                "max_reserved_mb": rec.get("max_reserved_mb"),
            }
            print("UPDATE_JSON=" + json.dumps(progress, sort_keys=True), flush=True)
            self.last = now

        def set_postfix(self, **kwargs):
            if update_records:
                update_records[-1]["postfix"] = kwargs

    trainer_mod.tqdm = BenchProgress
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    error = None
    try:
        trainer.train(dataset, num_workers=args.num_workers, resumable_with_seed=args.seed)
    except Exception as exc:  # record failures as evidence, then re-raise after JSON write
        error = repr(exc)
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall = time.perf_counter() - start
        trainer_mod.tqdm = original_tqdm

    losses = []
    for rec in update_records:
        postfix = rec.get("postfix") or {}
        try:
            losses.append(float(postfix.get("loss")))
        except Exception:
            pass

    counters = {}
    if dynamo is not None:
        try:
            counters = {k: dict(v) for k, v in dynamo.utils.counters.items()}
        except Exception:
            counters = {"error": "could not serialize dynamo counters"}

    state = getattr(trainer._unwrapped_model, "training_compile_state", None)
    bucket_stats = getattr(trainer, "bucket_stats", None)
    bucket_summary = bucket_stats.summary() if bucket_stats is not None else None
    result = {
        "candidate": candidate,
        "error": error,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "precision": {
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else None,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32 if torch.cuda.is_available() else None,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "amp_or_quantization": "none",
        },
        "model_cfg": model_cfg,
        "params": sum(p.numel() for p in trainer._unwrapped_model.parameters()),
        "dataset_metadata": dataset.metadata,
        "batch_size_per_gpu_frames": args.batch_size_per_gpu,
        "max_samples_per_batch": args.max_samples,
        "epochs": args.epochs,
        "batches_per_epoch": batches_per_epoch,
        "updates_expected": updates_expected,
        "updates_recorded": len(update_records),
        "unique_batch_shapes": len(set(batch_shapes)),
        "first_12_batch_shapes": batch_shapes[:12],
        "wall_s": wall,
        "mean_update_s": wall / max(1, len(update_records)),
        "median_recorded_update_s": statistics.median([r["dt_s"] for r in update_records]) if update_records else None,
        "p90_recorded_update_s": statistics.quantiles([r["dt_s"] for r in update_records], n=10)[8]
        if len(update_records) >= 10
        else None,
        "first_recorded_update_s": update_records[0]["dt_s"] if update_records else None,
        "last_recorded_update_s": update_records[-1]["dt_s"] if update_records else None,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_finite": all(map(lambda x: x == x and abs(x) < float("inf"), losses)) if losses else None,
        "compile_active_end": bool(getattr(trainer, "compile_active", False)),
        "compile_fallback_active_end": bool(getattr(trainer, "compile_fallback_active", False)),
        "model_training_compile_state": state,
        "dynamo_counters": counters,
        "candidate_config": cfg,
        "bucket_summary": bucket_summary,
    }
    if torch.cuda.is_available():
        result.update(
            {
                "max_memory_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
                "max_memory_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
            }
        )
    write_json(out_dir / f"{candidate}.json", result)
    with (out_dir / "results.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(result, sort_keys=True) + "\n")
    print("RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)
    if error is not None:
        raise RuntimeError(error)


def run_correctness_probe(repo: Path, data_file: Path, out_dir: Path, candidate_names: list[str], args: argparse.Namespace) -> None:
    # Run correctness probes as separate workers with tiny one-batch script code to avoid contaminating training processes.
    probe = out_dir / "correctness_probe.py"
    code = textwrap.dedent(
        f"""
        import copy
        import json
        import sys
        from pathlib import Path

        import torch

        repo = Path({str(repo)!r})
        data_file = Path({str(data_file)!r})
        out_dir = Path({str(out_dir)!r})
        sys.path.insert(0, str(repo / "src"))

        from f5_tts.model import CFM, DiT
        from f5_tts.model.dataset import collate_fn
        from f5_tts.model.utils import get_tokenizer

        torch.manual_seed({args.seed})
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all({args.seed})
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")

        obj = torch.load(data_file, map_location="cpu")
        batch = [{{"mel_spec": obj["mels"][i], "text": obj["texts"][i]}} for i in range(min(4, len(obj["mels"])))]
        b = collate_fn(batch)
        mel = b["mel"].permute(0, 2, 1).cuda()
        lens = b["mel_lengths"].cuda()
        text = b["text"]
        vocab_char_map, vocab_size = get_tokenizer("", "byte")
        model_cfg = dict(
            dim={args.dim},
            depth={args.depth},
            heads={args.heads},
            ff_mult=2,
            text_dim={args.text_dim},
            text_mask_padding=True,
            qk_norm=None,
            conv_layers=4,
            pe_attn_head=None,
            attn_backend="torch",
            attn_mask_enabled=False,
            checkpoint_activations=False,
        )
        mel_spec_kwargs = dict(
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mel_channels=100,
            target_sample_rate=24000,
            mel_spec_type="vocos",
        )

        def build():
            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)
            return CFM(
                transformer=DiT(**model_cfg, text_num_embeds=vocab_size, mel_dim=100),
                mel_spec_kwargs=mel_spec_kwargs,
                vocab_char_map=vocab_char_map,
            ).cuda().train()

        base = build()
        cand_template = copy.deepcopy(base)
        torch.manual_seed(999)
        torch.cuda.manual_seed_all(999)
        prepared = base._prepare_training_inputs(mel.clone(), text, lens.clone())
        loss, cond, pred = base._run_loss_core(*prepared)
        loss.backward()
        base_grads = [p.grad.detach().clone() if p.grad is not None else None for p in base.parameters()]
        candidates = json.loads({json.dumps(json.dumps(CANDIDATES))})
        selected = set({candidate_names!r})
        results = []
        for name, cfg in candidates.items():
            if name not in selected or not cfg.get("compile"):
                continue
            m = copy.deepcopy(cand_template).cuda().train()
            m.align_text_len_to_mel = bool(cfg.get("align_text_len_to_mel", False))
            m.compile_mark_dynamic = bool(cfg.get("compile_mark_dynamic", False))
            kwargs = {{
                "backend": "inductor",
                "fullgraph": False,
                "dynamic": cfg.get("dynamic"),
                "runtime_fallback": False,
            }}
            if cfg.get("options") is not None:
                kwargs["options"] = cfg["options"]
            elif cfg.get("mode") is not None:
                kwargs["mode"] = cfg["mode"]
            m.compile_training_core(**kwargs)
            if cfg.get("mark_step"):
                torch.compiler.cudagraph_mark_step_begin()
            args2 = tuple(x.detach().clone() if torch.is_tensor(x) else x for x in prepared)
            closs, ccond, cpred = m._run_loss_core(*args2)
            closs.backward()
            max_grad = 0.0
            for bg, p in zip(base_grads, m.parameters()):
                if bg is not None and p.grad is not None:
                    max_grad = max(max_grad, float((bg - p.grad).abs().max().detach().cpu()))
            results.append(
                {{
                    "candidate": name,
                    "loss_abs_diff": float(abs(loss.detach().cpu() - closs.detach().cpu())),
                    "pred_max_abs_diff": float((pred.detach().cpu() - cpred.detach().cpu()).abs().max()),
                    "grad_max_abs_diff": max_grad,
                    "state": m.training_compile_state,
                }}
            )

        (out_dir / "correctness.json").write_text(json.dumps(results, indent=2, sort_keys=True) + chr(10))
        print("CORRECTNESS_JSON=" + json.dumps(results, sort_keys=True), flush=True)
        """
    ).lstrip()
    probe.write_text(code, encoding="utf-8")
    run([sys.executable, str(probe)])


def orchestrate(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir).resolve()
    out_dir = work_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=== host ===", flush=True)
    print(capture("uname -a || true"), flush=True)
    print(capture("nvidia-smi || true"), flush=True)

    repo = setup_repo(work_dir, args.repo_url, args.branch, args.commit)
    apply_bucket_agent_patch(repo)
    print(capture(["git", "diff", "--stat"], cwd=repo), flush=True)
    install_repo(repo)
    metadata = prepare_librispeech_subset(
        repo=repo,
        out_file=work_dir / "data" / f"librispeech_{args.samples}_mel.pt",
        samples=args.samples,
        fetch_rows=args.fetch_rows,
        max_duration=args.max_duration,
        min_duration=args.min_duration,
    )

    selected = [name.strip() for name in args.candidates.split(",") if name.strip()]
    for name in selected:
        if name not in CANDIDATES:
            raise ValueError(f"Unknown candidate: {name}")

    if args.skip_correctness:
        print("Skipping correctness probe; relying on prior same-script probe run.", flush=True)
    else:
        run_correctness_probe(repo, work_dir / "data" / f"librispeech_{args.samples}_mel.pt", out_dir, selected, args)

    common = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--repo",
        str(repo),
        "--data-file",
        str(work_dir / "data" / f"librispeech_{args.samples}_mel.pt"),
        "--out-dir",
        str(out_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size-per-gpu",
        str(args.batch_size_per_gpu),
        "--max-samples",
        str(args.max_samples),
        "--num-workers",
        str(args.num_workers),
        "--dim",
        str(args.dim),
        "--depth",
        str(args.depth),
        "--heads",
        str(args.heads),
        "--text-dim",
        str(args.text_dim),
        "--seed",
        str(args.seed),
    ]
    for name in selected:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo / "src")
        env["TORCHINDUCTOR_CACHE_DIR"] = str(out_dir / "inductor_cache" / name)
        if not env.get("TORCH_LOGS"):
            env.pop("TORCH_LOGS", None)
        env["WANDB_DISABLED"] = "true"
        print(f"\n=== candidate {name} ===", flush=True)
        start = time.perf_counter()
        try:
            cmd = common + ["--candidate", name]
            proc = subprocess.Popen(
                cmd,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="" if line.endswith("\n") else "\n", flush=True)
            return_code = proc.wait()
            if return_code:
                raise subprocess.CalledProcessError(return_code, cmd)
        finally:
            print(f"candidate {name} elapsed_s={time.perf_counter()-start:.3f}", flush=True)

    rows = []
    results_path = out_dir / "results.jsonl"
    if results_path.exists():
        rows = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
    by_name = {r["candidate"]: r for r in rows}
    base = by_name.get("eager")
    compiled = by_name.get("compiled_default")
    summary = {
        "repo": {"url": args.repo_url, "branch": args.branch, "commit": capture(["git", "rev-parse", "HEAD"], cwd=repo).strip()},
        "dataset_metadata": metadata,
        "args": vars(args),
        "correctness": json.loads((out_dir / "correctness.json").read_text()) if (out_dir / "correctness.json").exists() else None,
        "results": rows,
    }
    for r in rows:
        if base and r.get("wall_s") and base.get("wall_s"):
            r["speedup_vs_eager_wall"] = base["wall_s"] / r["wall_s"]
        if compiled and r.get("wall_s") and compiled.get("wall_s"):
            r["speedup_vs_compiled_default_wall"] = compiled["wall_s"] / r["wall_s"]
    write_json(out_dir / "summary.json", summary)
    print("\n=== FINAL SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True)[:60000], flush=True)
    print(f"ARTIFACT_DIR={out_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--repo-url", default=REPO_URL)
    parser.add_argument("--branch", default=BRANCH)
    parser.add_argument("--commit", default=COMMIT)
    parser.add_argument("--work-dir", default="/content/f5tts_real_trainer_bench")
    parser.add_argument("--repo")
    parser.add_argument("--data-file")
    parser.add_argument("--out-dir")
    parser.add_argument("--candidate", choices=sorted(CANDIDATES))
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--fetch-rows", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size-per-gpu", type=int, default=12000)
    parser.add_argument("--max-samples", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--min-duration", type=float, default=0.5)
    parser.add_argument("--max-duration", type=float, default=12.0)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--depth", type=int, default=18)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--text-dim", type=int, default=512)
    parser.add_argument(
        "--candidates",
        default="compiled_default,compiled_bucket64_r20_align,compiled_bucket128_r20_align,compiled_bucket256_r20_align",
    )
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--skip-correctness", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        candidate_worker(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
