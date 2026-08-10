from types import SimpleNamespace

import pytest

import f5_tts.model.trainer as trainer_module
from f5_tts.model.trainer import Trainer


class _Dataset:
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return index

    def get_frame_len(self, index):
        return 1


class _DataLoaderCreated(Exception):
    pass


@pytest.mark.parametrize("batch_size_type", ["sample", "frame"])
@pytest.mark.parametrize(("num_workers", "expected"), [(0, False), (2, True)])
def test_persistent_workers_requires_worker_processes(monkeypatch, batch_size_type, num_workers, expected):
    captured = {}

    def capture_dataloader(*args, **kwargs):
        captured.update(kwargs)
        raise _DataLoaderCreated

    monkeypatch.setattr(trainer_module, "DataLoader", capture_dataloader)

    trainer = Trainer.__new__(Trainer)
    trainer.log_samples = False
    trainer.batch_size_type = batch_size_type
    trainer.batch_size_per_gpu = 1
    trainer.max_samples = 1
    trainer.accelerator = SimpleNamespace(even_batches=True)

    with pytest.raises(_DataLoaderCreated):
        trainer.train(_Dataset(), num_workers=num_workers)

    assert captured["persistent_workers"] is expected
