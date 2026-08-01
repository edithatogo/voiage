# Mermaid design — planned v1.3.0

```mermaid
flowchart LR
    Worlds["Finite joint worlds: probability + action values + all source observations"] --> Baseline["Optimize current action"]
    Sources["Sources: cost, latency, privacy, rights, freshness, SLA, coverage"] --> Enumerate["Enumerate bounded ordered sequences"]
    Limits["Budget, time, privacy, coverage, cardinality and order limits"] --> Enumerate
    Enumerate --> Feasible{"Feasible and rights-cleared?"}
    Feasible -->|no| Reject["Reject input or prune sequence with reason"]
    Feasible -->|yes| Condition["Condition joint worlds on observation tuple"]
    Worlds --> Condition
    Condition --> Policy["Re-optimize action with complete ties"]
    Baseline --> Value["Gross VOI"]
    Policy --> Value
    Value --> Net["WTP and net VOI after source and delay costs"]
    Net --> Select["Exact optimum and complete sequence ties"]
    Select --> Marginal["Order-conditional marginal value"]
    Select --> Shapley["Decision-value Shapley attribution"]
    Select --> Diagnostics["Exact-search diagnostics and switches"]
```

```mermaid
flowchart TD
    Issue["#313 > #318 > #582"] --> Track["information_source_portfolio_voi_20260801"]
    Track --> Contract["Strict v1 schemas and fixtures"]
    Contract --> Python["Experimental exact Python evaluator + CLI"]
    Python --> Review{"Scientific, hosted and parity gates?"}
    Review -->|pending| Experimental["Remain experimental"]
    Review -->|separately approved| Promotion["Future stable-promotion decision"]
```
