"""Selectable CMB solver backends and registry infrastructure.

Solver adapters implement one stable protocol so sampler code can select a
reference CPU implementation or an optional Taichi device boundary without
importing backend-specific numerical kernels.
"""
