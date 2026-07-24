# Round 4 independent numerical verification

Date: 24 July 2026  
Score: 932/1000  
Recommendation: major revision

This is an internal AI-assisted verification report, not a formal JOSS review.

## Verdict

Every reported estimate, interval, EVSI value, ENBS sign change, sensitivity
result, equation, unit, and binding-parity claim was independently reproduced.
No reported numerical result was found to be incorrect.

## Independent results

| Quantity | Result |
| --- | ---: |
| Programme positive incremental net benefit | 49.24% |
| EVPI | 644.153547 |
| Health-gain EVPPI | 589.666167 |
| Programme-cost EVPPI | 249.594994 |
| EVSI at total sample size 200 | 124.179366 |
| Immediate/full-realisation break-even | between 100 and 200 |
| Delayed/60%-realisation break-even | between 800 and 1,200 |

The analytical population references for EVPI and the two EVPPI quantities lie
inside the paired-bootstrap intervals. The normal--normal preposterior variance
and the Rust implementation agree.

## Remaining deductions

- The supplied PDF and public release do not represent the current source.
- Generated artefacts needed regeneration after replacing “uptake” with “value
  realisation”.
- The manuscript omitted the exact PSA and bootstrap seeds.
- The developing generic estimator lacks a structured MCSE and stronger
  predeclared convergence tests.
- Exact v2 release and archive provenance do not yet exist.
