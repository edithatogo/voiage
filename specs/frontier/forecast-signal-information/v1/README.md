# Forecast and signal information v1

This experimental contract values a declared finite probabilistic forecast by
the downstream decision it changes. It consumes, but never trains, the
forecast artifact. The joint law is `P(outcome) P(signal | outcome)`; deployed
actions are selected using the forecast's reported conditional probabilities
and evaluated under that joint law.

The result separates timely-oracle signal value, signed deployed value,
calibration loss, acquisition cost, maximum price, and regret avoided.
Predictive accuracy is a diagnostic, not the value estimand. Late or stale
information has zero operational value, while its counterfactual timely value
remains visible.

The v1 scope is exact finite enumeration with static feasible actions and a
single decision. Continuous signals, model fitting, sequential recourse and
polyglot execution are unsupported.
