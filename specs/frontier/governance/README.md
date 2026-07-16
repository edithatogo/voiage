# Frontier Governance Contracts

This directory holds the machine-readable governance artifacts for the frontier
VOI architecture and dependency policy.

## Contents

- **Maturity taxonomy**: the ordered promotion ladder (`planned` →
  `experimental` → `fixture-backed` → `stable`) and per-level promotion
  criteria. The authoritative source is `voiage.governance.MATURITY_LEVELS`.
- **Backend boundary**: ownership rules for each architectural layer (schema,
  methods, backends, CLI, Rust core). The authoritative source is
  `voiage.governance.BACKEND_OWNERSHIP`.
- **Dependency policy**: the split between lightweight base dependencies and
  optional bleeding-edge backends. The authoritative source is
  `voiage.governance.DEPENDENCY_POLICY`.
- **Stable-promotion matrix**: `promotion-matrix.json` defines the evidence
  required for each promotion state and `promotion-checklist.json` applies it
  to every family in the frontier fixture registry.

The family checklist records the owner, current evidence state, next gate,
blocker state, and committed artifact paths. It is repository-owned evidence;
it does not imply external registry approval, hardware access, cloud quota, or
maintainer sign-off.

## Validation

The governance rules are validated by `tests/test_frontier_governance.py` and
can be checked programmatically via:

```python
from voiage.governance import (
    validate_dependency_policy,
    validate_maturity_label,
    validate_promotion_evidence,
)

validate_maturity_label("fixture-backed")   # passes
validate_dependency_policy()                 # passes
```

The developer-facing prose lives in
`docs/developer_guide/frontier_governance.rst`.
