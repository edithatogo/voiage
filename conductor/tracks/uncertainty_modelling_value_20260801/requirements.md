# MoSCoW requirements — planned v1.3.0

## Must

- **UMV-M1:** Declare direction, common value units, population/horizon/
  discount bases, expected-value risk and the exact point-estimate functional.
- **UMV-M2:** Declare finite states, probabilities, ordered stages, shared
  histories, nonanticipative candidate policies, recourse and feasibility.
- **UMV-M3:** Return EV problem/solution, EEV, RP/SP, WS, direction-aware
  VSS/EVIU and EVPI, full ties, policy audit and solver assurance.
- **UMV-M4:** Provide strict schemas, exact two-/multistage fixtures, nonlinear
  point-estimate and infeasible-recourse cases, deterministic API/CLI and docs.

## Should

- Preserve every candidate policy evaluation so the optimum and feasibility
  can be independently audited.
- Keep the EVIU comparator and its equality to v1 VSS explicit.

## Could

- Add separately reviewed DVSS/VMS and supported solver certificates in later
  contract versions.

## Won't

- Model information acquisition, silently replace infeasible recourse, claim
  generic optimization, or advertise stable/polyglot execution in v1.
