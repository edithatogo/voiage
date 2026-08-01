# Belief-state information remediation review

## Scope

Independent implementation review of issue #597 / C18-M28 at signed branch
head `395aedde`, followed by bounded remediation. This record is an engineering
review, not independent scientific approval or stable-promotion evidence.
A fresh re-review at signed head `5693b3ee` found one residual runtime-bound
defect. The fresh reviewer then became the second remediator, so a third
independent implementation review remains required.

## Findings and disposition

1. **High — false dual-control diagnostic.** State-independent observation
   laws with action-dependent marginal frequencies triggered
   `action_dependent_learning` and `dual_control_diagnostic`, despite exact zero
   information value. Remediated by requiring a state-informative posterior-law
   difference, action dependence and a usable downstream control response.
   Transition dependence remains a separate diagnostic and cannot trigger dual
   control alone.
2. **High — unsafe exact-enumeration surface.** The advertised horizon bound
   permitted exponential work without an operation budget; the normative
   cardinalities rose from about 0.06 seconds at horizon three to 0.62 seconds
   at horizon four and 8.99 seconds at horizon five in the pre-remediation
   implementation. Remediated with a conservative preflight expansion estimate,
   a 50,000-expansion fail-closed budget and exact repeated-belief memoization.
   Small-branching problems may still use horizon twelve.
3. **High — incomplete strict and semantic validation.** Runtime accepted extra
   latent-state, control-action and observation fields; result schema subtrees
   were open; and generated results were not checked for numerical identities,
   policy-tree integrity or assurance drift. Remediated with strict recursive
   schemas, runtime nested validation, result identities, recursive policy and
   posterior-martingale checks, conditional null-comparator reconciliation and
   adversarial mutation tests.
4. **Medium — inconsistent probability tolerance.** Initial, likelihood,
   observation-branch and martingale checks used different relative or absolute
   tolerances. Remediated by bounding the declared tolerance at `1e-6`, applying
   it consistently with zero relative tolerance and preserving declared inputs
   without silent normalization.
5. **High — incomplete recursive-work preflight.** The first remediation
   bounded adaptive observation branches but omitted the fully observed regret
   comparator's latent-state recursion. A valid 20-state, four-stage model
   reported only 18 estimated expansions while exceeding 50,000 fully observed
   calls. Remediated by conservatively summing every recursive evaluator run:
   full and horizon-curve adaptive policies, the myopic policy, conditional
   sensing, matched no-information comparators and the fully observed regret
   comparator. The adversarial model now rejects before recursion at an exact
   conservative estimate of 168,453 calls; the accepted one-action,
   one-sensor, two-state horizon-twelve boundary reports 24,649 calls.

## Independent recomputation

The normative two-stage fixture was recomputed from its declared 0.5/0.5 prior.
`probe` has immediate reward -1, the diagnostic costs 0.5, each observation has
probability 0.5, the posterior is 0.9/0.1 or 0.1/0.9, and the optimal second
control has conditional expected reward 8. Therefore closed-loop gross value is
7, closed-loop net value is 6.5, the matched no-information value is 0, myopic
value is 0, nonmyopic net information value is 6.5, fully observed value is 20,
and gross partial-observability regret is 13. These values match the regenerated
normative result.

## Remaining boundary

Because each independent reviewer implemented the remediation it found, this
record does not self-approve the final patch. A third fresh independent
implementation reviewer must review the complete remediation before
publication, hosted exact-head assurance and merge. Independent scientific
review, Rust/R/Julia parity, stable promotion, release, parent #597 closure and
umbrella #318 closure remain pending.
