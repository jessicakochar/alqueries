# alqueries/extractors/base.py
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.utils.data import DataLoader


class FeatureExtractor(ABC):
    """
    Runs inference over a pool dataloader and returns arrays aligned to the
    loader's iteration order. Subclass per task (classification, detection,
    segmentation, ...).
    """

    def __init__(self, model: torch.nn.Module, device: torch.device | str = "cpu"):
        self._model = model
        self._device = torch.device(device) if isinstance(device, str) else device

    @abstractmethod
    def extract(self, loader: DataLoader) -> dict[str, np.ndarray | torch.Tensor]:
        """Run inference and return a dict of feature arrays."""