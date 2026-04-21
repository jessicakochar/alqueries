# alqueries/base.py
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class QueryStrategy(ABC):
    """
    Pure strategy: takes precomputed arrays, returns absolute pool indices.
    Subclasses declare what they need via explicit kwargs on `query`.
    """

    @abstractmethod
    def query(self, pool_indices: np.ndarray, n_samples: int, **kwargs) -> np.ndarray:
        ...