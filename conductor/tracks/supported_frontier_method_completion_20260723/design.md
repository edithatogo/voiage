# Mermaid design — planned v1.2.0 and v1.3.0

## Implementation-information decomposition

```mermaid
flowchart LR
    States["Uncertain states + net benefit by action"] --> Cells["Four joint value cells"]
    Current["State- and policy-dependent current implementation"] --> Cells
    Perfect["Perfect implementation"] --> Cells
    Specific["Specific implementation intervention"] --> Extra["EVSIM cells"]
    Signals["Sampling likelihood + signal-dependent implementation"] --> Sample["IA-EVSI cell"]
    Cells --> Matrix["Current/perfect information x current/perfect implementation matrix"]
    Extra --> Components["EVPIM, EVSIM, realizable EVPI, EVP"]
    Sample --> Components
    Matrix --> Components
    Components --> Interaction["Interaction + exact identity residuals"]
    Costs["Action-specific aggregate costs"] --> Net["Signed net components"]
    Components --> Net
    Interaction --> Assurance["Complete ties, switches, provenance and language dispositions"]
    Assurance --> Boundary["EVEIm/EVSEIm are candidate labels; no independence or stable claim"]
```

## Additive MCDA information value

```mermaid
flowchart LR
    Alternatives["Named alternatives"] --> Kernel["Fixed additive-value kernel"]
    Criteria["Raw units + directions + fixed value anchors"] --> Kernel
    Preferences["Nonnegative normalized weights"] --> Kernel
    JointLaw["Finite correlated outcome/preference states"] --> Current["Baseline expected scores, ranking and complete choice ties"]
    Kernel --> Current
    Actions["Criterion, preference or joint perfect-resolution actions"] --> Conditional["Conditional scores and optimal choices"]
    JointLaw --> Conditional
    Kernel --> Conditional
    Current --> Value["Gross and signed net information value"]
    Conditional --> Value
    Value --> Decomposition["Criterion/preference/joint interaction + no-double-counting"]
    Conditional --> Diagnostics["Regret + rank acceptability + expected/statewise Pareto"]
    Decomposition --> Assurance["Exact enumeration, invariants, provenance and language dispositions"]
    Diagnostics --> Assurance
    Assurance --> Boundary["Not AHP elicitation, outranking, veto, post-information normalization or EVSI"]
```

## Qualitative value of information

```mermaid
flowchart LR
    Decision["Decision + accountable reviewers"] --> Assessment["Versioned qualitative assessment"]
    Gaps["Information questions + evidence gaps"] --> Assessment
    Judgements["Impact, feasibility, timeliness, equity/ethics, burden and confidence"] --> Assessment
    Sources["Sources + missingness + redaction markers"] --> Assessment
    Assessment --> Validate{"Portable contract valid?"}
    Validate -->|no| Incomplete["Incomplete or unverified result"]
    Validate -->|yes| Prioritize["Deterministic ordinal priority + complete ties"]
    Dissent["Dissent + conflict declarations"] --> Prioritize
    Prioritize --> Recommendations["Ordinal recommendation classes + rationale"]
    AI["AI contribution + model/version provenance"] --> Human{"Human verified?"}
    Human -->|no| Incomplete
    Human -->|yes| Recommendations
    Recommendations --> Audit["Append-only versioned audit history"]
    Audit --> Render["Deterministic accessible serialization and rendering"]
    Render --> Boundary["Not quantitative VOI, MCDA, Delphi or evidence grading"]
```

```mermaid
flowchart LR
    Utility["Utility + wealth/reference state"] --> Current["Current policy EU"]
    Information["Perfect information"] --> Clairvoyant["Clairvoyant policy EU"]
    Current --> Delta["Expected utility increase"]
    Clairvoyant --> Delta
    Current --> CEI["Inverse-utility CEI"]
    Clairvoyant --> CEI
    Current --> Prices["BPI / SPI indifference roots"]
    Clairvoyant --> Prices
    Delta --> PPI["Anchored PPI"]
    Clairvoyant --> VoC["VoC alias/presentation"]
    Affine{"Affine utility?"} -->|yes| EVPI["Monetary EVPI reduction"]
    Affine -->|no| Distinct["Retain utility-scale distinction"]
```

## Deterministic sensitivity analysis

```mermaid
flowchart LR
    Contract["Fixed baseline, coordinates, direction and units"] --> Adapter{"Input surface"}
    Adapter -->|callback| Evaluator["Shared deterministic evaluator"]
    Adapter -->|normalized records| Evaluator
    Evaluator --> OneWay["One-way evaluated grid"]
    Evaluator --> TwoWay["Two-way evaluated grid"]
    Evaluator --> Scenarios["Declared scenario table"]
    OneWay --> Switches["Observed ties and bracketing switch intervals"]
    OneWay --> Ranking["Ranges and deterministic ranking"]
    TwoWay --> Result["Versioned DSA result"]
    Scenarios --> Result
    Switches --> Result
    Ranking --> Tornado["Accessible tornado plot"]
    Result --> Boundary["Not PSA, EVPPI, global sensitivity or VoI"]
```

## Value of Distribution-Family Information

```mermaid
flowchart LR
    Evidence["Evidence D"] --> Probabilities["Declared P(M=m | D)"]
    Families["Candidate family index M"] --> Conditional["Within-family E[U(a,Y) | M=m,D]"]
    Alternatives["Common alternatives + direction + unit"] --> Conditional
    Probabilities --> Current["Optimize probability-weighted current value"]
    Conditional --> Current
    Conditional --> Resolved["Optimize separately after learning M"]
    Probabilities --> Resolved
    Current --> VDI["Gross VDI = resolved - current"]
    Resolved --> VDI
    VDI --> Net["Signed net VDI = gross VDI - information cost"]
    VDI --> Assurance["Exact enumeration + ties + invariants + bounds"]
    Assurance --> Boundary["Discrete-index EVPPI; not structural EVPI or discrimination EVSI"]
```
