# Rationale: We keep the CLI helpers together because the launcher imports
# them lazily to avoid pulling heavy plotting dependencies during startup.
"""CLI helpers for the Copernican Suite.

These helpers group interactive and startup routines used by the primary
``copernican.py`` entrypoint. Modules are split by responsibility so
dependency handling remains isolated from menu rendering and long-running
workflows.  The separation matters because startup dependency checks can run
without importing heavy plotting stacks, keeping the initial CLI responsive.
"""
