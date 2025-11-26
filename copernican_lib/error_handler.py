# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

# Rationale: Error routing is centralised because parsers and engines need a
# lightweight path to surface issues without reconfiguring the shared logger.
"""Logging and warning helpers for Copernican components.

Parsers use :func:`report_error` to emit messages without depending on the
global logging configuration.  The new :func:`configure_warnings` routine
forwards all :mod:`warnings` to the central logger and optionally upgrades
them to errors for deterministic CI runs.  Centralising this behaviour keeps
log handling consistent across the project.
"""

from __future__ import annotations

import logging
import warnings
from typing import Type


def report_error(message: str) -> None:
    """Log ``message`` to the shared application logger."""
    # Parsers call this helper instead of accessing the root logger directly so
    # that logging configuration stays centralised. Any error messages end up
    # in the same log file as the main application output.
    logging.getLogger().error(message)


def configure_warnings(strict: bool = False) -> None:
    """Forward warnings to the logger and optionally treat them as errors.

    Parameters
    ----------
    strict:
        When ``True`` all warnings raise exceptions via
        ``warnings.filterwarnings("error")``.  This flag is designed for
        reproducible continuous integration runs where unexpected warnings
        should fail the build.
    """

    def _log_warning(
        message: str,
        category: Type[Warning],
        filename: str,
        lineno: int,
        file: object | None = None,
        line: str | None = None,
    ) -> None:
        """Log formatted warning details to the shared logger."""

        logging.getLogger().warning(
            "%s:%s: %s: %s",
            filename,
            lineno,
            category.__name__,
            message,
        )

    warnings.showwarning = _log_warning
    if strict:
        warnings.filterwarnings("error")


__all__ = ["report_error", "configure_warnings"]
