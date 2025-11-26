# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Core helper modules for the Copernican Suite.

This package bundles general-purpose utilities used across the
`copernican` command-line tool and the computational engines. The
modules live together so shared logic stays consistent and so engines
remain lightweight. Centralising helpers keeps interfaces aligned and
reduces the chance that engines quietly drift apart in behaviour.
The package exists as a single entry point because Copernican relies on
shared helpers for logging, plotting and orchestration, and keeping them
co-located prevents regressions when adding new engines or GUIs.
"""

# Nothing else is defined here. Importing this package simply exposes the
# submodules like ``logger`` and ``plotter`` used throughout the code.
# The flat surface keeps imports cheap because callers avoid pulling heavy
# dependencies until they intentionally reach into a specific helper.
