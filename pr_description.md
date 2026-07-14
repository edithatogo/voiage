# Performance optimization summary

This file records prior optimization work and is retained as a human-readable
summary. The repository's authoritative implementation and benchmark evidence
remain in the corresponding source files and tests.

The tracked optimization work covers:

- pre-allocated NumPy state trajectories for Markov cohort simulation;
- vectorized Pareto strategy comparisons; and
- vectorized adaptive-trial net-benefit updates.

Benchmark claims should be regenerated before reuse in release or publication
materials.
