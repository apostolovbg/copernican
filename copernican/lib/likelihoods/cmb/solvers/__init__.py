"""Selectable CMB solver backends and registry infrastructure.

Solver adapters implement one stable protocol so sampler code can select a
reference CPU implementation or a future Taichi device implementation
without importing backend-specific numerical kernels.
"""
