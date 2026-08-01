# Mermaid design — planned v1.3.0

```mermaid
flowchart LR
    Point["Declared point-estimate functional"] --> EV["Deterministic EV problem + solution"]
    EV --> EEV["Evaluate induced policy over scenarios"]
    States["Finite states + probabilities"] --> EEV
    Stages["Stages + shared histories"] --> Policies["Nonanticipative policy class"]
    Recourse["State outcomes + feasibility"] --> Policies
    Policies --> RP["Exact recourse/stochastic optimum"]
    States --> WS["Statewise wait-and-see optimum"]
    EEV --> VSS["Direction-aware VSS = EVIU in v1"]
    RP --> VSS
    RP --> EVPI["Direction-aware EVPI"]
    WS --> EVPI
    VSS --> Result["Ties + audit + solver assurance"]
    EVPI --> Result
```

```mermaid
flowchart TD
    Issue["#313 > #318 > #594"] --> Children["#774 contract / #775 runtime / #776 assurance"]
    Children --> Track["uncertainty_modelling_value_20260801 / C18-M26"]
    Track --> Python["Experimental exact finite Python API + CLI"]
    Python --> Gate{"Scientific + hosted + parity + promotion gates?"}
    Gate -->|pending| Experimental["Remain experimental; issues open"]
    Gate -->|separately approved| Promotion["Future promotion decision"]
```
