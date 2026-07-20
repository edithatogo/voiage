# C15 requirements: Operational Assurance Excellence

## MoSCoW

### Must

- **C15-M01:** Current, N-1, and incompatible migrations shall be independently tested.
- **C15-M02:** Aggregate, critical-module, and changed-line branch coverage shall be enforced.
- **C15-M03:** Mutation evidence shall bind source/configuration cohort identity and retain debt metrics.
- **C15-M04:** Linux and Windows normalized artifacts shall be compared by digest.
- **C15-M05:** Performance evidence shall contain runner identity, repeated samples, and confidence intervals.
- **C15-M06:** Exported collector payloads shall preserve correlation and contain no secrets.
- **C15-M07:** Boundary, near-tie, tail, and higher-dimensional scientific oracles shall carry units and tolerances.
- **C15-M08:** Exact-head local/hosted evidence and independent review shall pass.

### Should

- **C15-S01:** Differential coverage should annotate pull requests read-only.
- **C15-S02:** Performance comparisons should use bootstrap confidence intervals.
- **C15-S03:** Standalone bundle descriptors should be OCI-compatible when available.

### Could

- **C15-C01:** Approved releases may attach transparency-log receipts.
- **C15-C02:** Trustworthy hardware runners may define separate performance cohorts.

### Won't

- **C15-W01:** C15 will not autonomously merge, trust, sign, publish, or close issues.
- **C15-W02:** Experimental dependencies will not replace the frozen stable lane.
