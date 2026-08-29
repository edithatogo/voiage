# Signed/social information reference and adjacent-surface review

## Normative boundary

The C18/M29 evaluator uses exact expectation on a complete finite world law.
For agent `i`, design `d` and selected nonanticipative policy `pi_d`, the
pre-transfer value is `E[u_i(pi_d(S), W)]`. Transfers are zero-sum ledger
entries and declared costs are subtracted to obtain post-transfer value. The
comparator-specific signed value is the post-transfer difference between `d`
and its named comparator. Social value applies the declared nonnegative
weighted-sum aggregator at the declared pre- or post-transfer ledger stage.
Cardinal comparability is therefore an explicit modelling assumption.

## Independent references

- Shuo Li and Matteo Pozzi, *Information Avoidance and Overvaluation in
  Sequential Decision Making under Epistemic Constraints*, arXiv:2106.04984,
  demonstrates that information value can be negative when epistemic
  constraints and the decision maker's economic perspective are misaligned.
  The VOIAGE fixture adopts only that signed-value phenomenon; it does not
  claim to reproduce the paper's POMDP or finite-state-controller model.
- Junyu Cao, *Collaborative Learning and Decision Making on Pricing and
  Recommendation: A Simple Framework for Planning*, Management Science,
  DOI 10.1287/mnsc.2023.00320, distinguishes centralized and decentralized
  team planning. VOIAGE uses that distinction only to motivate named roles and
  bounded policy-selection modes; it does not implement the paper's learning
  algorithms.

## Applicability and exclusions

Blackwell-style nonnegativity is applied only to aligned centralized catalogs
with the same selector, verified signal refinement, unchanged preferences and
constraints, and an actually embedded comparator catalog. The checked value is
gross pre-transfer value for the declared selector. Fixed policies, declared
responses, incomplete catalogs, strategic conflict, transfer/cost accounting,
rights constraints or failed refinement assurance are not silently treated as
the theorem's domain.

The existing `strategic_behavior`, federated/privacy and equity helpers do not
establish this estimand: they neither freeze this complete joint-world law nor
return comparator-specific recipient/controller/stakeholder/social ledgers.
Bayesian persuasion, mechanism design, rational inattention and general game
solving remain adjacent. The finite-equilibrium mode accepts only a declared
policy selected from a receipt-verified complete finite catalog.

## Fixture assurance

The synthetic Li-Pozzi-inspired construction has two equiprobable worlds. The
baseline safe policy gives every agent zero. Selective sharing chooses `expand`
after a positive signal and `restrict` after a negative signal, giving the
recipient `-2`, the stakeholder `4`, the controller `0`, and the declared
weighted social aggregator `2`. The negative recipient value is retained, so
the fixture exercises information avoidance, a policy switch, winners/losers
and a positive applicable Blackwell check on the aligned social objective.
