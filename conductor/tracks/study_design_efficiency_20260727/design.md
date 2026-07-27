# Mermaid design — planned v1.2.0

```mermaid
flowchart LR
    Designs["Declared feasible designs"] --> Evaluate["EVSI + cost per design"]
    Evaluate --> Curve["Signed ENBS curve"]
    Curve --> Select["Deterministic argmax + tie policy"]
    Select --> Optimum["Optimum + boundary + uncertainty"]
    Curve --> Plot["Versioned plotting inputs"]
    EVPI["Commensurate EVPI"] --> Ratio["EVSI / EVPI"]
    Evaluate --> Ratio
    Ratio --> Guard["Zero-EVPI + bounds diagnostics"]
```
