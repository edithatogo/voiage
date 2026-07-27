# Comprehensive VOI software landscape design

```mermaid
flowchart TD
    Search["Recorded discovery channels and queries"] --> Candidate["Candidate product/version"]
    Candidate --> Evidence["Version-pinned evidence observations"]
    Evidence --> Product["Product identity, maintenance, rights, availability"]
    Evidence --> Schema["Data, model, decision, information, result schemas"]
    Evidence --> Capability["Feature, subfeature, option, default, diagnostic"]
    Evidence --> Adoption["Workflow, UX, report, integration, governance lesson"]
    Schema --> Normalize["Canonical capability normalization"]
    Capability --> Normalize
    Adoption --> Normalize
    Normalize --> Parity["Native, equivalent, adapter, planned, excluded, not reproducible"]
    Parity --> Matrix["Deterministic comparison views"]
    Parity --> Gaps["Evidence-linked gap records"]
    Gaps --> Proposal["MoSCoW improvement proposal"]
    Proposal --> Analyst{"Named analyst review"}
    Analyst -- Approve --> Route["Later roadmap and owning-track proposal"]
    Analyst -- Reject or defer --> Decision["Preserved decision ledger"]
    Analyst -- Revise --> Gaps
```

```mermaid
erDiagram
    PRODUCT ||--o{ PRODUCT_VERSION : has
    PRODUCT_VERSION ||--o{ ARTIFACT : evidenced_by
    PRODUCT_VERSION ||--o{ SCHEMA_SURFACE : exposes
    PRODUCT_VERSION ||--o{ CAPABILITY : exposes
    CAPABILITY ||--o{ SUBFEATURE : contains
    SUBFEATURE ||--o{ OPTION : configures
    ARTIFACT ||--o{ EVIDENCE_OBSERVATION : supports
    CAPABILITY ||--o{ PARITY_DISPOSITION : maps_to
    CAPABILITY ||--o{ ADOPTION_LESSON : teaches
    PARITY_DISPOSITION ||--o{ GAP : opens
    GAP ||--o{ IMPROVEMENT_PROPOSAL : proposes
    IMPROVEMENT_PROPOSAL ||--o{ REVIEW_DECISION : receives
```

The source observations are append-only evidence. Generated matrices and
proposals may be regenerated; analyst scientific notes and review decisions are
preserved separately and cannot be overwritten by refresh automation.
