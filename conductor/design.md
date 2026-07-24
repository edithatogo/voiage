# VOIAGE design

```mermaid
flowchart LR
    Literature[Literature census] --> Registry[Canonical method registry]
    Libraries[External library census] --> Registry
    Registry --> Rust[Rust numerical authority]
    VOP[Shared VOP C01/C02] --> Perspective[Perspective contracts]
    Registry --> Perspective
    Rust --> ABI[Versioned C ABI]
    Perspective --> ABI
    ABI --> Python
    ABI --> R
    ABI --> Julia
    ABI --> Mojo
    Registry --> ML[ML, LLM, and agent VOI]
    ML --> ABI
    Data[Rights-cleared evidence] --> Conformance[Cross-language conformance]
    Python --> Conformance
    R --> Conformance
    Julia --> Conformance
    Mojo --> Conformance
    Conformance --> Releases[v1.1, v1.2, v1.3]
```

GitHub issue #313 is the public programme record. Native subissues #314--#323
map one-to-one to active Conductor child tracks. Local specifications,
versioned registries, fixtures, and evidence remain authoritative for technical
completion; Project 28 is the synchronized public projection.

```mermaid
flowchart TD
    subgraph Discovery["Recorded discovery channels"]
        CRAN[CRAN and R-universe]
        PyPI[PyPI]
        Rust[crates.io]
        Julia[Julia General]
        Mojo[Mojo channels]
        Git[GitHub and GitLab]
        Web[Web and commercial documentation]
        Papers[Guidelines and software literature]
        Adjacent[Bayesian design and active learning]
    end

    Discovery --> Candidates[Candidate records]
    Candidates --> Evidence{Authoritative evidence?}
    Evidence -- No --> Reject[Record search limitation or reject hit]
    Evidence -- Yes --> Features[Feature-level inventory]
    Features --> Classify{Classify capability}
    Classify --> Estimand[Estimand]
    Classify --> Estimator[Estimator]
    Classify --> Workflow[Workflow or application]
    Classify --> Visual[Visualization]
    Classify --> Related[Related analysis]
    Estimand --> Methods[Canonical method IDs]
    Estimator --> Methods
    Workflow --> Methods
    Visual --> Methods
    Related --> Methods
    Methods --> Disposition[VOIAGE parity disposition]
    Disposition --> Matrix[Generated public feature matrix]
    Matrix --> Tests[Schema, traceability and freshness tests]
```

```mermaid
stateDiagram-v2
    [*] --> planned
    planned --> native: VOIAGE-owned implementation and tests
    planned --> equivalent: independent equivalent evidence
    planned --> adapter: migration or interchange only
    planned --> excluded: reviewed scientific, legal or architectural reason
    planned --> not_reproducible: behavior cannot be independently pinned
    adapter --> native: numerical authority moves into VOIAGE
    not_reproducible --> planned: new public evidence appears
    excluded --> planned: reviewed rationale expires or changes
```

The source registry is authoritative. The Markdown matrix is deterministic
derived output. Refresh automation may update machine-readable registry
metadata, but feature interpretation, exclusions, scientific maturity, and
license decisions remain reviewed changes.

```mermaid
flowchart LR
    Problem["Canonical Decision Problem"] --> Alternatives["Alternatives"]
    Problem --> States["Uncertain states"]
    Problem --> Actions["Information actions"]
    Problem --> Utility["Utility or loss"]
    Problem --> Context["Perspective, population, horizon, units"]
    Problem --> Provenance["Data and model provenance"]
    Draws["Posterior or predictive draws"] --> Problem
    Problem --> Estimator["Registered estimand and estimator"]
    Estimator --> Assurance["MC error, convergence, RNG, budget, stopping"]
    Assurance --> Result["Versioned Arrow and JSON result"]
    Result --> Bindings["Rust, Python, R, Julia, Mojo"]
```

The Decision Problem is the portable semantic boundary. Inference systems may
produce draws, but stable VOI calculations do not require their runtimes. Each
result carries estimator assurance rather than presenting a point estimate
without its numerical uncertainty.

```mermaid
flowchart TD
    Registry["Canonical method and capability registries"] --> Code["Rust facade and ABI"]
    Registry --> Matrix["Feature and maturity matrices"]
    Registry --> Docs["Astro documentation"]
    Registry --> Packages["Binding capability manifests"]
    Code --> Check{"Claim conformance"}
    Matrix --> Check
    Docs --> Check
    Packages --> Check
    Check -- mismatch --> Fail["Fail release"]
    Check -- aligned --> Evidence["Fixture-linked release evidence"]
    Drift["Quarterly and pre-release drift proposals"] --> Review["Human scientific review"]
    Review --> Registry
```

Machine updates may propose dependency and landscape changes. They do not
approve a method, exclusion, maturity promotion, or architecture decision.

```mermaid
flowchart TD
    Advisory["GitHub dependency graph and Dependabot alerts"] --> Renovate["Renovate"]
    OSV["OSV vulnerability feed"] --> Renovate
    Registries["Python, Cargo, npm, Actions, submodules"] --> Renovate
    Renovate --> Dashboard["Dependency and security dashboard"]
    Renovate --> PR["Immutable update PR"]
    PR --> Stability["Release-age and artifact checks"]
    Stability --> Protected["Maximal-quality required checks"]
    Protected --> Review{"Human review required?"}
    Review -- "Security, major, numerical, lock or submodule" --> Human["Maintainer review"]
    Review -- "Eligible ordinary non-major" --> Auto["Protected automerge"]
    Human --> Merge["Merge"]
    Auto --> Merge
    Merge --> Posture["Live alert and security-posture reconciliation"]
    Posture --> Release{"Release gate"}
```

Deleting `dependabot.yml` disables duplicate Dependabot version updates, not
GitHub's advisory alerts. Dependabot security updates remain a temporary
fallback until the Renovate App demonstrates a dashboard and checked PR; only
then are they disabled to ensure one update owner without a coverage gap.

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
flowchart LR
    Source["VOIAGE public source and docstrings"] --> Extract["Commit-pinned astro-polyglot extractors"]
    Griffe["Griffe Python analysis"] --> Extract
    Extract --> Guard{"Public members and safe paths?"}
    Guard -- No --> Fail["Fail closed"]
    Guard -- Yes --> MDX["Ignored generated MDX"]
    MDX --> Astro["Astro 7 and Starlight"]
    Astro --> Links["Link and content validation"]
    Astro --> LLMSTxt["Offline llms.txt output"]
    Links --> Build["Static production build"]
    LLMSTxt --> Build
    Build --> Pages["GitHub Pages artifact"]
```

The initial production extractor is Python because it has a deterministic,
CPU-only Griffe path in the repository docs environment. Rust, R, Julia, and
Mojo enter the same pipeline only after their native toolchains, public-symbol
filtering, generated-page contracts, and failure semantics have fixture-backed
evidence. The plugin is a source-pinned submodule until it has a reviewed
registry release; this prevents a local workspace link from being mistaken for
an independently installable package.

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
