# Spec

Implement a scheduler-facing backend layer that can route distributed CPU
workloads through optional Dask or Ray adapters while preserving the stable
analysis contract and local CPU fallback behavior.
