# alqueries/registry.py
from __future__ import annotations

from typing import Callable, Type

from alqueries.base import QueryStrategy

_REGISTRY: dict[str, Type[QueryStrategy]] = {}


def register_strategy(name: str) -> Callable[[Type[QueryStrategy]], Type[QueryStrategy]]:
    def decorator(cls):
        if name in _REGISTRY:
            raise ValueError(f"Strategy '{name}' already registered.")
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_strategy(name: str, **kwargs) -> QueryStrategy:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown strategy '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def list_strategies() -> list[str]:
    return sorted(_REGISTRY)