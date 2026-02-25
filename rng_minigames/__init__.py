"""Standalone RNG mini-games with a documented discovery API."""

from .api import MinigameContext
from .registry import (
    MinigameDescriptor,
    get_descriptor,
    load_launcher,
    load_registry,
    refresh_registry,
)

__all__ = [
    "MinigameContext",
    "MinigameDescriptor",
    "get_descriptor",
    "load_registry",
    "refresh_registry",
    "load_launcher",
]
