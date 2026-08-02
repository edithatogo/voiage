# Mermaid design — planned v1.2.0

```mermaid
flowchart LR
    Target["Target g(theta): scalar/vector + units"] --> Functional["Variance/covariance functional"]
    Prior["Prior + parameter subset"] --> EVPPI["EVPPI_var"]
    Prior --> EVSI["EVSI_var"]
    Functional --> EVPPI
    Functional --> EVSI
    Functional --> VectorGate{"Scalar target?"}
    VectorGate -->|"No"| Reserved["Reserved vector vocabulary: fail closed"]
    Reserved --> HumanReview["Candidate-bound independent scientific and named human review"]
    Sampling["Sampling model + design"] --> EVSI
    Conditioning["Conditioning + averaging convention"] --> EVPPI
    Conditioning --> EVSI
    EVPPI --> Result["Reduction + uncertainty + diagnostics"]
    EVSI --> Result
    Result --> Surfaces["Schema + CLI + report + bindings"]
```
