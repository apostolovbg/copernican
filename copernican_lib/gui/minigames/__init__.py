"""Seed mini-games exposed by the GUI Run Builder."""

from __future__ import annotations

from .alien_invasion import launch_alien_invasion
from .constellation import launch_constellation
from .emoji_meteors import launch_emoji_meteors

__all__ = [
    "launch_alien_invasion",
    "launch_constellation",
    "launch_emoji_meteors",
]
