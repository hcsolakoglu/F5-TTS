"""Keep validation tests independent of audio frontend binary packages.

The tests exercise tensor-only CFM/Trainer helpers. Some developer machines do
not have a torchaudio wheel matching their CUDA build, and librosa can be
present with an incompatible numba/numpy pair. Minimal import stubs prevent
those unrelated optional frontends from blocking collection; no stubbed
function is called by this suite.
"""

from __future__ import annotations

import sys
import types


try:
    import torchaudio  # noqa: F401
except (ImportError, OSError):
    torchaudio_stub = types.ModuleType("torchaudio")
    torchaudio_stub.transforms = types.SimpleNamespace()
    torchaudio_stub.load = lambda *args, **kwargs: None
    torchaudio_stub.save = lambda *args, **kwargs: None
    sys.modules["torchaudio"] = torchaudio_stub


try:
    from librosa.filters import mel as _librosa_mel  # noqa: F401
except (ImportError, AttributeError):
    sys.modules.pop("librosa", None)
    sys.modules.pop("librosa.filters", None)
    librosa_stub = types.ModuleType("librosa")
    librosa_stub.__path__ = []
    librosa_filters_stub = types.ModuleType("librosa.filters")
    librosa_filters_stub.mel = lambda *args, **kwargs: None
    librosa_stub.filters = librosa_filters_stub
    sys.modules["librosa"] = librosa_stub
    sys.modules["librosa.filters"] = librosa_filters_stub
