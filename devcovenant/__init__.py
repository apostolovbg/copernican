"""
DevCovenant - Self-enforcing policy system for Copernican Suite.

This system parses policy definitions from AGENTS.md, maintains Python
policy scripts, and enforces policies automatically during development.
"""

__version__ = "1.0.0"

from .engine import DevCovenantEngine
from .parser import PolicyParser
from .registry import PolicyRegistry

__all__ = ["DevCovenantEngine", "PolicyParser", "PolicyRegistry"]
