# Signed/social information implementation review

## Scope and independence

A fresh reviewer inspected issue #598 / C18-M29 at signed clean head
`9ff84a126e58cea56fc9342080ef062c757d2e7d`. The reviewer did not author the
original implementation. Review covered the complete finite joint-world law,
policy selection, selective-sharing comparators, private/role/social values,
transfers, costs, consent and purpose receipts, Blackwell applicability,
schemas, result validation, fixtures, API/CLI, documentation and governance.

This is an engineering implementation review, not independent scientific
approval, polyglot parity, stable-promotion evidence, release evidence or
programme closure.

## Findings and remediation

1. **High — portable derived results were not semantically bound.** The result
   validator accepted arbitrary role values, centralized selected policies and
   ties, optimum designs and values, winners/losers, Blackwell checked values
   and evaluated-world counts. Eight independent mutations all passed before
   remediation. This contradicted M29-S2/S3's strict signed role/social, tie,
   diagnostic and Blackwell result contract.
2. **Medium — Blackwell applicability admitted excluded cases.** A design made
   infeasible by denied consent could still report an applicable nonnegativity
   check. Transfer- or cost-bearing designs and comparators were also admitted,
   despite the governed reference excluding rights constraints and
   transfer/cost accounting from that theorem boundary.

Both findings were reported before their respective edits. Signed commit
`da119389` adds a portable agent-role map and exact evaluated world, policy and
design identifiers; recomputes policy and design ties, optima, signed role
values, feasibility, switches, harm, avoidance, winners/losers, externalities,
Blackwell values and assurance counts; and returns explicit inapplicability for
infeasible or transfer/cost-bearing comparisons. Adversarial mutations cover
each repaired boundary. Since the reviewer authored the remediation, another
fresh independent implementation reviewer must inspect the resulting head.

## Independent recomputation

The Li-Pozzi-inspired fixture contains two worlds of probability 0.5. The
signal policy selects `expand` in the beneficial/positive world and `restrict`
in the adverse/negative world. Direct expectation gives agent values
`alice=-2`, `board=0`, `public=4`; the declared weighted social value is 2 and
the baseline is zero. A declared information cost of 10 assigned to Alice
produces signed social value -8. Both negative private and negative social
values remain present with `clipped_at_zero=false`.

The fixture is a governed finite construction informed by the paper's negative
information phenomenon. It is not a reproduction or empirical calibration of
Li and Pozzi's sequential model.

## Local assurance

- 71 focused tests passed with 100% statements and branches across both new
  modules (424 statements, 200 branches).
- 87 combined signed/social, package-export and supported-frontier tests passed,
  including the public API and signed/social CLI path.
- Ruff check and format verification passed. BasedPyright reported zero errors
  for the contracts, runtime and deterministic exporter.
- Consecutive contract exports were byte-stable. Frontier-contract and GitHub
  cross-reference validation passed; full Conductor validation covered 148
  tracks with zero errors and warnings.

The shared source environment does not contain the Rust extension required by
the stable EVPI, EVPPI, ENBS and CEAF CLI paths. Seven unrelated tests in the
broad CLI file therefore failed with the expected native-runtime requirement;
they do not exercise the experimental #598 command. Hosted installed-wheel
assurance remains mandatory.

The Python 3.14 coverage loader required NumPy and JAX to be imported before
pytest to avoid their known repeated-load/circular-import defects. The same
source and focused tests then produced the complete coverage result above.

## Fresh independent re-review

PASS at signed head `d7d569b28664e9c529c5c039817f4c11f9805754`.
A fresh reviewer who authored neither the original implementation nor the
remediation independently repeated the contract, runtime, fixture, API/CLI,
governance and assurance review. No Critical, High or Medium findings remain.

The reviewer directly mutated each repaired semantic boundary. Altered role
values, policy ties, selected policy, optimum design, winners, Blackwell
checked values and assurance counts were all rejected. The normative result
was accepted and independently reproduced role values `controller=0`,
`decision_maker=-2`, `recipient=-2` and `stakeholder=4`. The signed negative
social value remains unclipped.

The re-review also passed 88 combined feature, API, CLI, export and frontier
tests, Ruff check and formatting, BasedPyright, deterministic frontier and
cross-reference validation, append-only evidence validation, and full
Conductor validation of 148 tracks with zero errors and warnings.

This re-review closes the local engineering-review gate only. Hosted exact-head
and installed-wheel checks and merge remain pending. Independent scientific
review, Rust/R/Julia parity, stable promotion, release, parent #598 closure and
umbrella #318 closure also remain pending.
