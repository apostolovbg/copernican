# See LICENSE.md in the repository root for details.

"""Sampler package exposing available inference backends."""

# Both bundled samplers remain discoverable through this package so callers can
# select an inference strategy without importing implementation paths directly.

from . import sampler_mcmc, sampler_nested

__all__ = [
    "sampler_mcmc",
    "sampler_nested",
]
