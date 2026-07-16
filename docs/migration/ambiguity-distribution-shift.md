# Ambiguity and distribution-shift VOI migration note

`value_of_ambiguity_distribution_shift` is a new fixture-backed method family
for source-sample reweightings that represent plausible target or drift
scenarios. It reports radius-penalized robust net benefit, scenario regret,
shift sensitivity, and the expected value of resolving the scenario.

The method is separate from model-validation and causal-transportability
surfaces. It does not claim stable promotion until cross-language parity and a
licensed real source-target drift snapshot are available.
