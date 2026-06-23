from __future__ import annotations

import numpy as np
import torch


def pool_rows(values: torch.Tensor, unlabeled_indices: np.ndarray) -> torch.Tensor:
    return values[unlabeled_indices]


def resolve_mc_probs(
    mc_probs: torch.Tensor | None,
    probs: torch.Tensor | None = None,
) -> torch.Tensor:
    if mc_probs is None:
        if probs is None:
            raise ValueError("mc_probs or probs must be provided.")
        mc_probs = probs
    return mc_probs


def mean_dropout_probs(mc_probs: torch.Tensor, unlabeled_indices: np.ndarray) -> torch.Tensor:
    return pool_rows(mc_probs.mean(dim=0), unlabeled_indices)


def negative_entropy(probs: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log(probs.clamp_min(1e-12))
    return (probs * log_probs).sum(dim=-1)


def max_confidence(probs: torch.Tensor) -> torch.Tensor:
    return probs.max(dim=1).values


def top_two_margin(probs: torch.Tensor) -> torch.Tensor:
    top2_probs, _ = torch.topk(probs, k=2, dim=1)
    return top2_probs[:, 0] - top2_probs[:, 1]


def select_by_score(
    unlabeled_indices: np.ndarray,
    scores: torch.Tensor,
    n_samples: int,
    *,
    descending: bool = False,
) -> np.ndarray:
    idx = scores.argsort(descending=descending)[:n_samples]
    if torch.is_tensor(idx):
        idx = idx.detach().cpu().numpy()
    return np.asarray(unlabeled_indices[idx])
