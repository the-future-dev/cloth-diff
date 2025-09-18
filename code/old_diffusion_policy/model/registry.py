"""
Simple registries for pluggable components (encoders, fusion, policies, models).
This enables easy extension without modifying factory code.
"""
from typing import Callable, Dict, Any

# Global registries
ENCODERS: Dict[str, Callable[..., Any]] = {}
FUSIONS: Dict[str, Callable[..., Any]] = {}
SHARED_ENCODERS: Dict[str, Callable[..., Any]] = {}
POLICIES: Dict[str, Callable[..., Any]] = {}
MODELS: Dict[str, Callable[..., Any]] = {}


def register_encoder(name: str):
    def deco(fn: Callable[..., Any]):
        key = name.lower()
        if key in ENCODERS:
            raise ValueError(f"Encoder '{name}' already registered")
        ENCODERS[key] = fn
        return fn
    return deco


def get_encoder(name: str) -> Callable[..., Any]:
    fn = ENCODERS.get(name.lower())
    if fn is None:
        raise KeyError(f"Unknown encoder '{name}'. Registered: {list(ENCODERS)}")
    return fn


def register_fusion(name: str):
    def deco(fn: Callable[..., Any]):
        key = name.lower()
        if key in FUSIONS:
            raise ValueError(f"Fusion '{name}' already registered")
        FUSIONS[key] = fn
        return fn
    return deco


def get_fusion(name: str) -> Callable[..., Any]:
    fn = FUSIONS.get(name.lower())
    if fn is None:
        raise KeyError(f"Unknown fusion '{name}'. Registered: {list(FUSIONS)}")
    return fn


def register_shared_encoder(name: str):
    def deco(fn: Callable[..., Any]):
        key = name.lower()
        if key in SHARED_ENCODERS:
            raise ValueError(f"Shared encoder '{name}' already registered")
        SHARED_ENCODERS[key] = fn
        return fn
    return deco


def get_shared_encoder(name: str) -> Callable[..., Any]:
    fn = SHARED_ENCODERS.get(name.lower())
    if fn is None:
        raise KeyError(f"Unknown shared encoder '{name}'. Registered: {list(SHARED_ENCODERS)}")
    return fn


def register_policy(name: str):
    def deco(fn: Callable[..., Any]):
        key = name.lower()
        if key in POLICIES:
            raise ValueError(f"Policy '{name}' already registered")
        POLICIES[key] = fn
        return fn
    return deco


def get_policy(name: str) -> Callable[..., Any]:
    fn = POLICIES.get(name.lower())
    if fn is None:
        raise KeyError(f"Unknown policy '{name}'. Registered: {list(POLICIES)}")
    return fn


def register_model(name: str):
    def deco(fn: Callable[..., Any]):
        key = name.lower()
        if key in MODELS:
            raise ValueError(f"Model '{name}' already registered")
        MODELS[key] = fn
        return fn
    return deco


def get_model(name: str) -> Callable[..., Any]:
    fn = MODELS.get(name.lower())
    if fn is None:
        raise KeyError(f"Unknown model '{name}'. Registered: {list(MODELS)}")
    return fn
