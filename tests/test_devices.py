import torch

from alqueries.devices import get_default_device, get_device_rng_state, seed_device


def test_default_device_prefers_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert get_default_device() == torch.device("cuda")


def test_default_device_uses_mps_when_cuda_is_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert get_default_device() == torch.device("mps")


def test_default_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert get_default_device() == torch.device("cpu")


def test_seed_device_seeds_mps_when_available(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.mps, "manual_seed", lambda seed: calls.append(seed), raising=False)

    seed_device(17, "mps")

    assert calls == [17]


def test_device_rng_state_keeps_checkpoint_keys_for_cpu():
    state = get_device_rng_state("cpu")

    assert state == {"cuda_rng": [], "mps_rng": None}
