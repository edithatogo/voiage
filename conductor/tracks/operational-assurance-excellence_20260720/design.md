# C15 design: Operational Assurance Excellence

```mermaid
flowchart LR
    VOP["Content-addressed VOP bundle"] --> Matrix["Current / N-1 / incompatible consumer matrix"]
    Matrix --> Evidence["C15 evidence manifest"]
    DiffCov["Changed-line + critical coverage"] --> Evidence
    Mutation["Mutation cohort + debt density"] --> Evidence
    Linux["Linux normalized build"] --> Compare["Digest comparator"]
    Windows["Windows normalized build"] --> Compare
    Compare --> Evidence
    Samples["Repeated benchmark samples"] --> Stats["Confidence interval + runner cohort"]
    Stats --> Evidence
    App["Redacted correlated telemetry"] --> Collector["Ephemeral OTel collector"]
    Collector --> Privacy["Export privacy scan"]
    Privacy --> Evidence
    Oracles["Boundary / tie / tail / high-dimensional references"] --> Evidence
    Evidence --> Human{"Authorized review"}
    Human -->|approved only| Merge["Merge / trusted baseline / publication"]
```

## Invariants

Evidence is schema-validated and content-addressed; missing evidence fails closed.
VOIAGE never imports VOP runtime code. Stable dependencies remain frozen, and
approval-bearing operations are staged but not executed by pull-request CI.
