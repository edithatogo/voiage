# Mermaid design — planned v1.2.0 and v1.3.0

## Orchestrated scientific review

```mermaid
flowchart TD
    Candidate["Freeze exact candidate, tree and artifact hashes"] --> Packet["Immutable review packet"]
    Packet --> Orchestrator["Orchestrating agent"]
    Orchestrator --> Estimand["Estimand and domain subagent"]
    Orchestrator --> Assurance["Estimator-assurance subagent"]
    Orchestrator --> Parity["Cross-language and API subagent"]
    Orchestrator --> Governance["Governance and publication subagent"]
    Estimand --> Reports["Independent structured reports"]
    Assurance --> Reports
    Parity --> Reports
    Governance --> Reports
    Reports --> Register["Finding and disagreement registers"]
    Register --> Blocked{"Critical, High or scientific dissent?"}
    Blocked -->|yes| Remediate["Issue-backed remediation"]
    Remediate --> Rebind["Freeze new candidate and re-review affected roles"]
    Rebind --> Packet
    Blocked -->|no| Synthesis["Orchestrator synthesis and family verdict matrix"]
    Synthesis --> OwnerScience["Accountable owner scientific decision"]
    OwnerScience --> Maintainer["Separate owner maintainer maturity decision"]
    Maintainer --> Gates["Parity, promotion, hosted, release, publication and closure gates remain separate"]
```

## Risk-sensitive and constrained information value

```mermaid
flowchart LR
    States["Finite states and probabilities"] --> Current["Matched current feasible policy problem"]
    Risk["Declared expectation, utility, CVaR or regret functional"] --> Current
    Constraints["Budget, capacity, fairness, regulation and service constraints"] --> Current
    States --> Informed["Matched post-information feasible policy problem"]
    Risk --> Informed
    Constraints --> Informed
    Current --> Value["Gross and signed net information value"]
    Informed --> Value
    Value --> Diagnostics["Complete ties, infeasibility, switches and risk/constraint diagnostics"]
    Diagnostics --> Boundary["C18/M22 experimental capability; sampling-acquisition harm is separately scoped"]
```

## Static and dynamic heterogeneity value

```mermaid
flowchart LR
    Partition["Prespecified subgroups + weights"] --> Current["Current-information policies"]
    Eligibility["Eligibility + fairness/privacy constraints"] --> Current
    Effects["Finite subgroup effect-state law"] --> Current
    Current --> C0["C0: population-common current value"]
    Current --> Cf["Cf: subgroup-policy current value"]
    Effects --> Perfect["Perfect effect-state information"]
    Perfect --> P0["P0: population-common perfect value"]
    Perfect --> Pf["Pf: subgroup-policy perfect value"]
    C0 --> Static["Static value = Cf - C0"]
    Cf --> Static
    P0 --> Dynamic["Dynamic value = Pf - P0"]
    Pf --> Dynamic
    Static --> Identity["dynamic - static = EVPIf - EVPI0"]
    Dynamic --> Identity
    Sample["Optional finite signal + study cost"] --> EVSI["S0/Sf + separate EVSI and net diagnostics"]
    EVSI --> Boundary["Experimental exact Python; no subgroup discovery or validity claim"]
    Identity --> Boundary
```

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

## Forecast and signal information value

```mermaid
flowchart LR
    Prior["Outcome prior"] --> Joint["P(outcome) × P(signal | outcome)"]
    Likelihood["Signal likelihood"] --> Joint
    Joint --> Posterior["Signal marginal + posterior"]
    Reported["Reported probabilities"] --> ReportedPolicy["Feasible deployed choice"]
    Payoffs["Frozen action consequences + objective unit"] --> ReportedPolicy
    Payoffs --> Oracle["Posterior-optimal timely choice"]
    Posterior --> Oracle
    Posterior --> ReportedPolicy
    Timing["Horizon + freshness + latency + lead time"] --> Usable{"Usable at decision?"}
    ReportedPolicy --> Usable
    Usable -->|yes| Deployed["Signed deployed value"]
    Usable -->|no| Baseline["Baseline policy; operational value zero"]
    Oracle --> Loss["Oracle value + calibration loss"]
    Deployed --> Economics["Cost + signed net + maximum price + regret avoided"]
    Loss --> Diagnostics["Calibration L1 + Brier + signal coverage"]
    Diagnostics --> Boundary["C18/M23; exact finite experimental Python; accuracy is not value"]
```

## Additive MCDA information value

## Signed and social information value

```mermaid
flowchart LR
    Worlds["Complete finite joint-world law"] --> Policies["Nonanticipative finite policies"]
    Roles["Decision maker + recipient + controller + stakeholders"] --> Designs["Baseline + selective-sharing designs"]
    Topology["Private, public or team signal topology"] --> Designs
    Policies --> Designs
    Designs --> Selector{"Centralized, fixed, declared response or verified finite equilibrium"}
    Selector --> Ledger["Pre-transfer + transfer + cost + post-transfer ledgers"]
    Welfare["Declared cardinal comparability + weighted welfare"] --> Ledger
    Rights["Rights, consent + purpose receipts"] --> Ledger
    Ledger --> Values["Signed agent, role + social comparator values"]
    Values --> Diagnostics["Harm + avoidance + switches + winners/losers + externalities"]
    Diagnostics --> Blackwell{"Aligned verified centralized refinement?"}
    Blackwell -->|yes| Check["Gross selector value must be nonnegative"]
    Blackwell -->|no| Reasons["Return theorem-inapplicability reasons"]
    Check --> Boundary["C18/M29 experimental Python; adjacent strategic methods excluded"]
    Reasons --> Boundary
```

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

## Event-localized information value

```mermaid
flowchart LR
    States["Finite states + probabilities + coordinates"] --> Baseline["Baseline-optimal reference action a*"]
    Values["Action values + declared unit"] --> Baseline
    States --> Conditional["Grouped conditional g_a(x)"]
    Baseline --> PolicyDensity["i(x) = f(x) [max g_a(x) - g_a*(x)]"]
    Conditional --> PolicyDensity
    Baseline --> Centered["Signed j(x) = f(x) [max g_a(x) - V0]"]
    Conditional --> Centered
    PolicyDensity --> Integral["Coordinate information value + modes + directions"]
    Centered --> Integral
    Event["Declared event and complement"] --> Perfect["Exact perfect-event VOI"]
    Event --> Channel["Symmetric binary channel over accuracy grid"]
    Channel --> Curve["Gross/net VOI + complete signal-policy ties"]
    Perfect --> Assurance["Partition, p versus 1-p symmetry and 0.5 limit"]
    Integral --> Assurance
    Curve --> Assurance
    Assurance --> Plot["Result-only density and accuracy plots"]
    Plot --> Boundary["C18/M27 experimental Python; BPI delegated to #595"]
```

## Outcome-conditional sample-information value

```mermaid
flowchart LR
    Prior["Finite state prior"] --> Predictive["Predictive P(x) = sum_s P(s) P(x | s)"]
    Likelihood["Finite measurement likelihood"] --> Predictive
    Values["Action utility/maximize or loss/minimize"] --> Baseline["Baseline values, complete ties and reference action"]
    Predictive --> Posterior["Posterior P(s | x) and action values"]
    Baseline --> Posterior
    Posterior --> Metrics["delta-EV_x and nonnegative VSI_x"]
    Metrics --> Weighted["Predictive-probability-weighted outcome distribution"]
    Weighted --> Mean["EVSI = E[VSI_x] = E[delta-EV_x]"]
    Weighted --> Dispersion["Equation 10 weighted population sigma-VSI; ddof 0"]
    Weighted --> Low["rVSI_delta + weighted quantiles/tails"]
    Mean --> Assurance["Expectation-only tower scope"]
    Dispersion --> Assurance
    Low --> Assurance
    Assurance --> Ties["rVSI0 kept distinct from reference exclusion, mandatory switch and tie-set change mass"]
    Ties --> Boundary["C18/M31 exact finite experimental Python; continuous/scientific/parity/promotion/release gates open"]
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
