# Specification — Uncertainty-Modelling Value

## Objective

Deliver the smallest coherent issue #594 contract that distinguishes the value
of representing uncertainty from the value of acquiring information. Compare a
declared deterministic point-estimate solution with exact finite
nonanticipative stochastic policies and statewise wait-and-see solutions.

## Contract

The request declares minimization or maximization, common value units,
population/horizon/discount bases, the point-estimate functional and value,
finite scenario probabilities, ordered stages, shared history partitions,
candidate policies, statewise recourse feasibility, deterministic candidates,
expected-value risk, tie tolerances and exact-enumeration solver evidence.

Every policy has one first-stage decision and exactly one recourse decision per
shared history. Histories partition the states at every recourse stage, making
nonanticipativity auditable rather than inferred. Later history partitions
must refine earlier partitions and available information must be cumulative,
forming a coherent filtration. State outcomes are either
finite and feasible or explicitly infeasible with null objective value.

For minimization, `VSS = EEV - RP` and `EVPI = RP - WS`; maximization reverses
both contrasts. EVIU is the v1 VSS presentation because its comparator is the
declared point-estimate EV solution. Infeasible induced recourse returns null
EEV/VSS/EVIU; no relatively-complete candidate policy fails closed.

## Acceptance criteria

- **AC-01:** Match independent exact two-stage nonlinear and three-stage
  references under both objective directions, preserving all ties.
- **AC-02:** Validate probabilities, stages, history partitions,
  nonanticipativity, outcomes, recourse, units, point estimate and solver
  assurance fail closed.
- **AC-03:** Return EV problem/solution, EEV status, RP/SP policy/value, WS
  state policies/value, VSS/EVIU/EVPI, policy audit and exact diagnostics.
- **AC-04:** Supply strict v1 schemas, deterministic serialization, pathology
  fixtures, experimental Python API/CLI, discovery and user documentation.
- **AC-05:** Keep information acquisition separate and record DVSS/VMS,
  approximate/external solvers, risk criteria and language parity honestly.

## Boundaries

This contract does not treat ordinary PSA, EVPI alone, a solver adapter or a
deterministic-versus-stochastic chart as EVIU evidence. It does not model
sampling or signals. DVSS and VMS are reviewed adjacent multistage diagnostics
but deferred pending separate scientific contracts. Python is experimental;
Rust, R and Julia are not implemented and Mojo remains external.
