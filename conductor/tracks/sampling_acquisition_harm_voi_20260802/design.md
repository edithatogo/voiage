# Mermaid design — planned v1.3.0

```mermaid
flowchart LR
    Design["Candidate action d, including explicit comparator d0"] --> Joint["Design-indexed state, observable history, information and harm law"]
    State["Prior state and downstream decision problem"] --> Joint
    Joint --> EVSI["Gross incremental design value G(d; d0)"]
    EVSI --> EVSICheck{"Only information changes under matched model?"}
    EVSICheck -->|"Yes"| OrdinaryEVSI["Ordinary EVSI interpretation"]
    EVSICheck -->|"No"| DesignValue["Design value, not ordinary EVSI"]
    Joint --> Harm["Harm distribution by affected party"]
    Cost["Ordinary research cost"] --> Gate{"Unchanged state/actions, policy-independent harm, separable, commensurate and non-overlapping?"}
    EVSI --> Gate
    Harm --> Gate
    Gate -->|"Yes"| Net["Signed incremental design value - cost - valued harm"]
    Gate -->|"No"| Constrained["Joint-welfare, constrained or vector result"]
    Harm --> Safety["Expected, chance, tail or lexicographic safety criterion"]
    Safety --> Feasible{"Declared mathematical safety constraints satisfied?"}
    Net --> Feasible
    Constrained --> Feasible
    Feasible -->|"Infeasible"| Excluded["Exclude design; d0 remains an evaluated comparator"]
    Feasible -->|"Indeterminate"| InsufficientEvidence["Fail closed; obtain more evidence"]
    Feasible -->|"Feasible"| Candidate["Nondominated candidate set, complete ties and uncertainty"]
    Candidate --> Freeze["Verified #850 packet: Git OIDs, SHA-256 manifest, sources, claims and findings"]
    Freeze --> AutomatedPanel["Automated challenge panel: five role-shaped reports"]
    AutomatedPanel --> AutomatedSynthesis["Separate automated synthesis: all findings and dissent retained"]
    AutomatedSynthesis --> Readiness["#867 source observations and 19-finding readiness partition"]
    Readiness --> Commissioning["#870 candidate decision, reviewer screening, receipt contracts and privacy-safe handoff"]
    Commissioning --> CandidateChoice{"#873 exact accountable candidate-context choice recorded?"}
    CandidateChoice -->|"No"| ChallengeOnly
    CandidateChoice -->|"Yes: option 1"| SelectedExclusion["Proposed generic-kernel reviewed exclusion; narrower non-authorizing research preserved"]
    SelectedExclusion --> PacketGate{"#876 independent retrieval, rights, applicability and eligible humans complete?"}
    PacketGate -->|"No"| ChallengeOnly["Preparation only; no replacement freeze and H8-D/H8-E remain false"]
    PacketGate -->|"Yes, with accountable evidence"| Freeze
    Freeze --> Panel["Eligible independent reviewers plus domain/ethics specialist"]
    Panel --> Orchestrator["Separate orchestrator: findings, dissent, options, contingencies, rationale and recommendation"]
    Orchestrator --> Human{"Named scientific and domain/ethics humans confirm exact candidate?"}
    Human -->|"No, expired or conflicted"| UnsupportedRuntime
    Human -->|"Disputed"| Chair["Independent chair adjudication and fresh evidence"]
    Chair --> Freeze
    Human -->|"Yes"| Maintainer["Separate maintainer product/maturity decision"]
    Maintainer --> Runtime{"Runtime implementation approved separately?"}
    Runtime -->|"No"| UnsupportedRuntime["C18/M32 unsupported research scope"]
    Runtime -->|"Yes"| Future["Separately implemented Rust-authoritative family"]
    Candidate -. "real-study deployment only" .-> Authorization{"Accountable ethics and regulatory authorization, where applicable?"}
    Authorization -->|"No or pending"| AuthorizationPending["No real-study deployment authorization"]
    Authorization -->|"Yes or not applicable with authority"| AuthorizedStudy["Separately authorized study scope"]
```
