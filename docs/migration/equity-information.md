# Equity-information VOI migration note

`value_of_equity_information` is a new fixture-backed method family. It is
separate from the experimental `value_of_distributional_equity` API because it
quantifies information value from resolving uncertain equity weights, rather
than only measuring subgroup tailoring.

The method accepts subgroup net benefits, baseline equity weights, plausible
post-acquisition weight scenarios, scenario probabilities, and information
cost. Existing distributional-equity calls are unchanged. The method remains
fixture-backed until cross-language parity and licensed open-data attribution
are completed.
