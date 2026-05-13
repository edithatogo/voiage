# Working Notes: ASIC Implementation

ASIC/custom-circuit capability is a separate implementation lane. It remains
the most conservative and should only proceed where the workload is regular
enough to preserve the same contract.

Current blocker:
- No ASIC/custom-circuit runtime exists in the repository yet, so the
  implementation lane is still a planning and evidence task rather than a
  completed backend.

Later action:
- Resume this track only when an ASIC/custom-circuit execution environment is
  available and reproducible CPU/ASIC comparison evidence can be collected.
