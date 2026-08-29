# Prior-artifact reconciliation

Audited 2026-08-01 against repository revision `9495fc3f` plus the current
umbrella branch. This reconciliation records compatibility inputs only. It does
not classify an estimand, promote maturity, or establish parity.

| Issue | Candidate family | Existing repository evidence | Reconciliation boundary |
| --- | --- | --- | --- |
| #593 | implementation/information decomposition | `specs/frontier/implementation/v1/` and implementation-loss/uptake helpers | An uptake multiplier or implementation-loss statistic is not the requested joint information/implementation matrix or interaction decomposition. |
| #594 | uncertainty-modelling value | PSA, EVPI and solver/portfolio helpers | Propagating uncertainty or exposing a solver does not establish EVIU, EEV, VSS, wait-and-see, recourse or nonanticipativity. |
| #595 | expected-utility information pricing | merged PR #712 and `risk_adjusted_information_pricing_20260731` | Experimental EUI/CEI/BPI/SPI and VoC presentation are independently governed; scientific review and stable promotion remain open. |
| #596 | event-localized information value | threshold/event-like plotting helpers | A threshold plot, tail probability or conditional table is not an event-information estimand or a density with an integral-recovery assurance contract. |
| #597 | belief-state sequential information value | `voiage.methods.sequential`, dynamic real options, monitoring and adaptive-bandit helpers | A sequential EVPI trace, acquisition score or generic policy solver does not establish belief-state observation/control value. |
| #598 | signed/social information value | strategic, privacy, equity and federated helpers | Expected-minus-baseline values that assume aligned Bayes-optimal agents or clip at zero cannot establish signed recipient, agent and social value. |
| #599 | heterogeneity-value decomposition | stable `value_of_heterogeneity`, preference and distributional/equity surfaces | Subgroup or preference heterogeneity alone does not distinguish static policy value from dynamic research value. |
| #600 | outcome-conditional sample-information value | EVSI and study-design/COSS diagnostics | An EVSI standard error, interval or generic risk statistic is not the outcome-weighted distribution of sample-information value. |
| #619 | estimation-focused variance VOI | merged PR #676 and `estimation_focused_variance_voi_20260727` | Scalar experimental execution is independently governed; vector covariance functionals, scientific review and stable promotion remain open. |

The portable Decision Problem extension in #566 remains a separate native child
of #314. Its incomplete interchange contract does not erase the independently
frozen #595 and #619 contracts, but it still blocks completion of the census
track and any claim that the portable registry is complete.
