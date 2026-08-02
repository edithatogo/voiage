# Mermaid design — planned v1.3.0

```mermaid
flowchart LR
    Design["Candidate action d, including explicit comparator d0"] --> Joint["Design-indexed state, observable history, information and harm law"]
    State["Prior state and downstream decision problem"] --> Joint
    Joint --> EVSI["Gross decision EVSI"]
    Joint --> Harm["Harm distribution by affected party"]
    Cost["Ordinary research cost"] --> Gate{"Separable and commensurate?"}
    EVSI --> Gate
    Harm --> Gate
    Gate -->|"Yes"| Net["Signed EVSI - cost - valued harm"]
    Gate -->|"No"| Constrained["Constrained or vector result"]
    Harm --> Safety["Expected, chance, tail or lexicographic safety criterion"]
    Safety --> Feasible{"Declared mathematical safety constraints satisfied?"}
    Net --> Feasible
    Constrained --> Feasible
    Feasible -->|"Infeasible"| Excluded["Exclude design; d0 remains an evaluated comparator"]
    Feasible -->|"Indeterminate"| Unsupported["Fail closed; obtain more evidence"]
    Feasible -->|"Feasible"| Candidate["Nondominated candidate set, complete ties and uncertainty"]
    Candidate --> Authorization{"Accountable ethics and regulatory authorization?"}
    Authorization -->|"No or pending"| Unsupported
    Authorization -->|"Yes"| Human["Candidate-bound scientific/domain and named human review"]
    Human --> Runtime{"Runtime approved?"}
    Runtime -->|"No"| Unsupported["C18/M32 unsupported research scope"]
    Runtime -->|"Yes"| Future["Separately implemented Rust-authoritative family"]
```
