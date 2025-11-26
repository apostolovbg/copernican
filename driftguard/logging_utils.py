# Last Updated: 2025-11-26
"""Centralised logging helpers for DriftGuard.

The module wires a consistent logger that defaults to verbose output so CI logs
capture the policy engine's inner workings. Callers can tune verbosity via the
``DRIFTGUARD_LOGLEVEL`` environment variable; absent that override, the helper
leans toward DEBUG-level chatter to aid troubleshooting on fresh runners.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

_LOGGER_NAME = "driftguard"


def get_logger() -> logging.Logger:
    """Return the shared DriftGuard logger instance."""

    return logging.getLogger(_LOGGER_NAME)


def _determine_level(level: Optional[int]) -> int:
    """Resolve the logger level from caller input or environment."""

    if level is not None:
        return level
    env_level = os.environ.get("DRIFTGUARD_LOGLEVEL")
    if env_level:
        try:
            return logging._nameToLevel.get(env_level.upper(), logging.DEBUG)
        except Exception:
            # Fallback to DEBUG if the environment variable is malformed.
            return logging.DEBUG
    return logging.DEBUG


def ensure_logger(level: Optional[int] = None) -> logging.Logger:
    """Configure the shared logger with a stream handler when needed."""

    resolved_level = _determine_level(level)
    logger = get_logger()
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(resolved_level)
        formatter = logging.Formatter("DriftGuard [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(resolved_level)
    return logger
