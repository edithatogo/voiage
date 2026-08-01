# VOIAGE design

```mermaid
flowchart LR
    VOP[VOP canonical contract]
    Mirror[Digest-pinned VOIAGE mirror]
    Perspective[Perspective method API]
    Arrow[Arrow IPC and Parquet]
    Polars[Polars consumer]
    Fixtures[Golden fixtures and manifest]
    Harness[Repository and benchmark harness]
    CI[Python 3.12–3.14 CI and security gates]
    External[External maturity and publication gates]

    VOP --> Mirror
    Mirror --> Perspective
    Perspective --> Arrow
    Arrow --> Polars
    Arrow --> Fixtures
    Fixtures --> Harness
    Polars --> Harness
    Harness --> CI
    CI --> External
```

```mermaid
sequenceDiagram
    participant V as VOIAGE method
    participant F as Fingerprint
    participant W as Arrow writer
    participant X as Fresh-process consumer
    participant T as Hosted tests

    V->>F: Describe ordered logical fields
    F->>F: Hash canonical JSON
    V->>W: Add shared contract metadata
    W->>X: IPC or Parquet artifact
    X->>T: Values, types, metadata and fingerprint
    T-->>V: Pass or fail closed
```

## Specialized VOI v1.2.0

```mermaid
flowchart LR
    Target["Scalar/vector target + component units"] --> Functional["Variance/covariance functional"]
    Conditioning["Prior + conditioning convention"] --> Estimation["EVPPI_var / EVSI_var"]
    Sampling["Sampling model + design"] --> Estimation
    Functional --> Estimation
    Estimation --> Assurance["Estimator uncertainty + diagnostics"]

    Designs["Evaluated feasible designs"] --> EVSI["Decision EVSI"]
    EVSI --> ENBS["Signed ENBS curve"]
    Costs["Study + opportunity costs"] --> ENBS
    ENBS --> COSS["Optimum + tie/boundary state + uncertainty"]
    COSS --> Plot["Accessible plotting inputs"]
    EVSI --> Ratio["EVSI / EVPI"]
    EVPI["Commensurate EVPI"] --> Ratio
    COSS --> Candidate["Governed portfolio candidate"]
    PortfolioSemantics["Metrics + heterogeneity + delay + interference + multiplicity + stopping"] --> AssuranceDeclaration["No-effect or already-in-COSS assurance + provenance"]
    AssuranceDeclaration --> Candidate
    DisjointCosts["Incremental costs excluded from COSS + provenance"] --> Candidate
    Candidate --> Portfolio["Exact signed-ENBS subset allocation"]
    PortfolioConstraints["Capacity + dependencies + exclusions + guardrails"] --> Portfolio
    Portfolio --> GlobalTieSet["Fixed global maximum + one tolerance tie set"]

    Payoffs["Payoffs + state/signal probabilities"] --> CurrentEU["Optimize current-policy EU"]
    Utility["Named utility + wealth/reference state"] --> CurrentEU
    Utility --> InformedEU["Optimize informed/clairvoyant-policy EU"]
    Payoffs --> InformedEU
    CurrentEU --> EUI["EUI on utility scale"]
    InformedEU --> EUI
    CurrentEU --> CEI["Inverse-utility CEI"]
    InformedEU --> CEI
    CurrentEU --> Roots["BPI/SPI indifference roots + diagnostics"]
    InformedEU --> Roots
    EUI --> PPI["Anchored probability price"]
    InformedEU --> VoC["VoC alias/presentation governed by #595"]
    Affine{"Positive-affine utility?"} -->|yes| Monetary["Verified EVPI/EVSI reduction"]
    Affine -->|no| DistinctScale["Keep utility and monetary scales distinct"]

    DSABaseline["DSA baseline + direction + units"] --> DSAAdapter{"Callback or normalized records"}
    DSAAdapter --> DSAEvaluator["Shared deterministic evaluator"]
    DSAEvaluator --> DSASurfaces["One-way / two-way / scenarios"]
    DSASurfaces --> DSASwitches["Observed ties + bracketing switch intervals"]
    DSASurfaces --> DSARanking["Deterministic ranges + ranking"]
    DSARanking --> DSATornado["Accessible tornado plot"]
    DSASwitches --> DSABoundary["Not PSA, EVPPI, global sensitivity or VoI"]

    ModelFamilies["Declared model-family index + P(M|D)"] --> FamilyValues["Within-family conditional expected values"]
    FamilyValues --> CurrentFamilyPolicy["Current mixture-optimal policy"]
    FamilyValues --> ResolvedFamilyPolicies["Family-resolved policies"]
    CurrentFamilyPolicy --> FamilyVDI["Gross and signed net VDI"]
    ResolvedFamilyPolicies --> FamilyVDI
    FamilyVDI --> FamilyBoundary["Discrete-index EVPPI; not full structural EVPI"]

    QualDecision["Decision + accountable human reviewers"] --> QualAssessment["Versioned qualitative assessment"]
    QualGaps["Information questions + evidence gaps"] --> QualAssessment
    QualJudgements["Ordinal impact, feasibility, timeliness, equity/ethics, burden and confidence"] --> QualAssessment
    QualAssessment --> QualPriority["Deterministic priority classes + complete ties"]
    QualDissent["Dissent + conflict + missingness"] --> QualPriority
    QualAI["AI provenance + unverified state"] --> QualHuman{"Human verified?"}
    QualHuman -->|no| QualIncomplete["Incomplete/unverified"]
    QualHuman -->|yes| QualResult["Recommendation classes + rationale"]
    QualPriority --> QualResult
    QualResult --> QualAudit["Immutable audit history + accessible rendering"]
    QualAudit --> QualBoundary["Not numerical VOI, MCDA, Delphi or evidence grading"]
```

```mermaid
flowchart TD
    C16["Canonical C16 / v1.2.0"] --> Projection["Versioned public projection"]
    Projection --> Planner["Three-way conflict-safe planner"]
    Planner --> Issues["#313 > #318 > #571/#595/#619"]
    Planner --> Consumers["Other registered repositories"]
    Issues --> Project["Project 28"]
    Consumers --> Project
    Project --> Fields["MoSCoW + Contract Version + Track ID + Record ID + Sync State"]
    Planner --> Guard{"Conflict, private data, or missing credential?"}
    Guard -->|yes| Stop["Fail closed"]
    Guard -->|no and authorized| Managed["Update managed fields only"]
```

The archived Conductor registry documents historical implementation. GitHub
issues and the shared project provide the public ledger; local specifications,
fixtures, and CI evidence remain authoritative for technical completion.

```mermaid
flowchart TD
    Cargo[Cargo workspace version] --> Maturin[Maturin dynamic metadata]
    Tag[Git release tag] --> Validate[Fail-closed tag validation]
    Cargo --> Validate
    Maturin --> Package[Dynamic package version]
    Validate --> Package
    Settings[Pydantic v2 LoggingSettings] --> Logs[Human or JSONL logs]
    Run[Run and command context] --> Logs
    Pixi[Pixi tasks] --> UV[uv lock and execution]
    UV --> Fast[Ruff + ty + BasedPyright + test matrix]
    Fast --> Build[Build, install and release gates]
    Scheduled[Scheduled/manual frontier] --> Scalene[Scalene artifact]
    Scheduled --> Mutation[Mutation evidence]
    Scheduled --> Experimental[Experimental/free-threaded probes]
```

Pixi delegates Python environment resolution to uv, so the repository retains
one dependency lock. Expensive evidence is scheduled or manually requested;
stable pull requests keep deterministic correctness, typing, security,
interchange, coverage, and package gates.

```mermaid
flowchart LR
    Concern --> Risk
    Evidence --> Decision
    Risk --> Decision
    Decision --> IssueLink
    IssueLink --> Project
    AnalysisSpec --> NumericalPolicy
    AnalysisSpec --> Kernel
    Kernel --> BackendCapabilities
    Kernel --> AnalysisResult
    RunContext --> AnalysisResult
    AnalysisResult --> Arrow
```

```mermaid
sequenceDiagram
    participant S as Analysis specification
    participant D as Capability dispatcher
    participant B as Backend
    participant K as Calculation kernel
    participant R as Result envelope
    S->>D: Requirements and numerical policy
    D->>B: Check explicit capabilities
    alt Unsupported
      D-->>S: Fail closed or disclosed fallback
    else Supported
      D->>K: Execute with backend and run context
      K->>R: Typed payload, diagnostics and provenance
      R-->>S: Versioned serializable result
    end
```
