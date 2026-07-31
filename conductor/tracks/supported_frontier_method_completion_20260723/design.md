# Mermaid design — planned v1.2.0

```mermaid
flowchart LR
    Utility["Utility + wealth/reference state"] --> Current["Current policy EU"]
    Information["Perfect information"] --> Clairvoyant["Clairvoyant policy EU"]
    Current --> Delta["Expected utility increase"]
    Clairvoyant --> Delta
    Current --> CEI["Inverse-utility CEI"]
    Clairvoyant --> CEI
    Current --> Prices["BPI / SPI indifference roots"]
    Clairvoyant --> Prices
    Delta --> PPI["Anchored PPI"]
    Clairvoyant --> VoC["VoC alias/presentation"]
    Affine{"Affine utility?"} -->|yes| EVPI["Monetary EVPI reduction"]
    Affine -->|no| Distinct["Retain utility-scale distinction"]
```
