from __future__ import annotations

import torch


def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_device(seed: int, device: str | torch.device) -> None:
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif torch_device.type == "mps" and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)


def get_device_rng_state(device: str | torch.device) -> dict[str, object]:
    torch_device = torch.device(device)
    state: dict[str, object] = {"cuda_rng": [], "mps_rng": None}
    if torch_device.type == "cuda":
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    if torch_device.type == "mps" and hasattr(torch.mps, "get_rng_state"):
        state["mps_rng"] = torch.mps.get_rng_state()
    return state


def set_device_rng_state(device: str | torch.device, state: dict) -> None:
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and state.get("cuda_rng"):
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    elif torch_device.type == "mps" and state.get("mps_rng") is not None:
        torch.mps.set_rng_state(state["mps_rng"])
