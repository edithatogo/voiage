# Mermaid design — planned v1.2.0

```mermaid
flowchart LR
    Problem["Payoffs + probabilities + units"] --> Current["Optimize current policy"]
    Utility["Named utility + wealth/reference"] --> Current
    Utility --> Informed["Optimize informed policy"]
    Problem --> Informed
    Information{"Information structure"} -->|clairvoyant| States["One signal per state"]
    Information -->|finite signal| Signals["Joint signal-state probabilities"]
    States --> Informed
    Signals --> Informed
    Scope["Stakeholder scope + information structure"] --> Current
    Scope --> Informed
    Current --> EUI["Expected utility increase"]
    Informed --> EUI
    EUI --> CEI["Certainty-equivalent increase"]
    Current --> Buy["Buying-price indifference root"]
    Informed --> Buy
    Current --> Sell["Selling-price indifference root"]
    Informed --> Sell
    EUI --> PPI["Anchored probability price"]
    Buy --> Diagnostics["Bracket + residual + iterations + policy switches"]
    Sell --> Diagnostics
    Informed --> Kind{"Clairvoyant result?"}
    Kind -->|yes| VoC["VoC alias / presentation"]
    Kind -->|no| SignalValue["Finite-signal utility value"]
    Affine{"Positive-affine utility?"} -->|yes| EVPI["Verified monetary EVPI reduction"]
    Affine -->|no| Distinct["Keep utility and money scales distinct"]
```

```mermaid
flowchart TD
    C16["Canonical C16 / M16-M17"] --> Track["risk_adjusted_information_pricing_20260731"]
    Track --> Parent["#313 > #318 > #595"]
    Parent --> Children["Native delivery subissues"]
    Children --> Project["Project 28"]
    Track --> Rust["Rust numerical authority"]
    Rust --> Python["Python facade + VoC presentation"]
    Python --> Users["CLI + reports + fixtures + docs"]
    Users --> Bindings["R / Julia / Mojo dispositions"]
    Bindings --> Gate{"Scientific promotion and hosted gates satisfied?"}
    Gate -->|no| Experimental["Experimental repository evidence"]
    Gate -->|yes| Promotion["Separate stable-promotion decision"]
```
