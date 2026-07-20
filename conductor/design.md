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
