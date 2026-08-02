# Primary-source and estimand review

## Review question

Should harms caused by the act of obtaining information be represented by an
existing VOIAGE method, a distinct sampling-design estimand, or an unsupported
research boundary?

## Source findings

### Research value and ordinary cost

Heath, Baio, Manolopoulou and Welton (2024, DOI
`10.1007/s40273-024-01372-0`) describe VOI trial
design as valuing information in the full downstream decision context and
subtracting trial cost to obtain net economic value. This supports #571's
ordinary EVSI/ENBS comparator. It does not establish that physical,
psychological, privacy or community harm is interchangeable with ordinary
research expenditure.

### Safety-constrained information acquisition

Camilleri et al. (NeurIPS 2022, DOI `10.52202/068431-2406`) formulate active
learning as identification of the best arm subject to unknown safety
constraints. Bottero et al. (NeurIPS 2022, DOI `10.52202/068431-2226`) select
informative evaluations while restricting them to regions considered safe
with high probability. These are adjacent safe-exploration methods rather than
health-economic EVSI, but they show that information gain and acquisition
safety are separate design objects and need not collapse into one scalar.

### Human-subject risk and affected parties

The Belmont Report distinguishes the probability and magnitude of harm and
identifies physical, psychological, legal, social and economic harms. It also
requires systematic consideration of alternative ways of obtaining the
benefit and gives risks to immediate subjects special weight. HHS 45 CFR 46
provides the applicable US human-subject protection framework and additional
protections for specified populations. The ICH E6(R3) Step 4 final guideline,
including Principles 1, 2, 3, 6 and 7, requires participant rights, safety and
well-being, informed consent, quality by design and proportionate processes.

These authorities are domain and jurisdiction specific. Any future candidate
must declare its domain and jurisdiction, select applicable primary
authorities, and record conflicts or limitations. A generic runtime cannot
infer ethics, consent, rights or regulatory authorization from this register.

These sources are ethical/regulatory authorities, not numerical VOI formulas.
They rule out any software claim that positive ENBS alone authorizes a study.

## Estimand disposition

Let `d0` denote the explicit no-sampling comparator and `d` a declared
sampling action. For each design, declare design-indexed potential outcomes:
state `theta_d`, acquisition harm `H_d`, observation and reporting process,
decision-time observable history `O_d`, admissible action set `A_d(O_d)`, and
downstream outcome `B_down,d(a, theta_d)`, which excludes every acquisition-
harm consequence assigned to `L_d`. A policy `pi_d` must be measurable
with respect to `O_d`; it cannot use latent harm or information observed only
after the decision. Interference, spillovers, harm-induced dropout and
sampling-induced downstream changes must be explicit.

Define the optimized downstream value

```text
W_B(d) = sup over admissible pi_d E[B_down,d(pi_d(O_d), theta_d)]
G(d; d0) = W_B(d) - W_B(d0).
```

`G` is a gross incremental design value. It is ordinary EVSI only when `d`
changes the information in `O_d` but not the state, outcome mapping, admissible
actions or other downstream mechanisms. Let incremental ordinary cost be
`Delta C(d; d0) = C(d) - C(d0)`.

When a declared stakeholder valuation `L_d(H_d)` is policy-independent,
separable from downstream decision value and in the same cardinal units, a
candidate signed scalar is

```text
NIV_H(d; d0) = G(d; d0) - Delta C(d; d0)
                - {E[L_d(H_d)] - E[L_d0(H_d0)]}.
```

Separability here is a declared additive welfare decomposition, not a claim of
statistical independence. Commensurability requires a named perspective,
benefit recipients and burden bearers, cardinal scale, numeraire, valuation
source and date, horizon, discount base and uncertainty. Ordinal harms are not
scalarizable merely by assigning numeric labels.

This is not a universal definition. When acquisition changes the state or
action set, harm depends on the downstream policy, or harms are heterogeneous,
incommensurate, catastrophic, absorbing or protected by non-compensatory
constraints, the correct candidate is a constrained or vector design problem,
for example

```text
maximize W_B(d) - Delta C(d; d0)
subject to P(catastrophic H_d for party p over horizon T) <= alpha_p
           rho(H_d) <= budget
           d in the mathematically admissible candidate set.
```

The risk functional `rho`, probability threshold, harm budget, affected-party
scope and statistical assurance method are user- and domain-supplied. Software
reports mathematical status as `feasible`, `infeasible` or `indeterminate`.
Accountable ethics and regulatory authorization remains a separate external
gate. A finite penalty cannot replace a hard or lexicographic prohibition.

For positive loss `L`, adverse-tail CVaR at confidence `q` uses the upper tail:

```text
CVaR_upper_q(L) = inf_eta {eta + E[(L - eta)+] / (1 - q)}, 0 <= q < 1.
```

For signed welfare `W`, lower-tail expected shortfall at mass `beta` uses
`sup_eta {eta - E[(eta - W)+] / beta}`, `0 < beta <= 1`. These optimization
forms define behavior at atoms; an implementation must also declare its
quantile/interpolation convention. An exact zero-catastrophe threshold needs a
structural exclusion or logically sufficient bound: observing zero events is
not proof of zero probability.

### No-double-counting rule

The scalar contract requires an outcome-component ledger that partitions every
valued outcome exactly once between `B_down,d`, ordinary cost `C(d)` and
acquisition-harm valuation `L_d`. `B_down,d` excludes health, action, cost or
other consequences carried by `L_d`; `L_d` includes every direct and downstream
consequence attributed to acquisition. If acquisition changes the state or
action set, or a consequence cannot be partitioned without changing the policy
problem, the additive scalar is undefined. Define total joint welfare once as
`J_d(a, theta_d, H_d)` and optimize its increment net of `C(d)` subject to harm
constraints, without separately subtracting `L_d`. Constrained and vector
results use the same mutually exclusive component ledger.

### Affected-party and authorization record

Every candidate must retain an unaggregated role matrix with participants,
researchers, carers, bystanders, communities, data subjects, payers and other
relevant groups; identify benefit recipients, burden bearers and transfers;
and report group-specific distributions, tails and constraints. Consent,
autonomy, rights, justice and external authorization are noncompensatory gates,
not utilities emitted by the estimator.

| Role dimension | Required candidate record |
|---|---|
| Parties | participants, researchers, carers, bystanders, communities, data subjects, payers and domain-specific groups |
| Incidence | benefit recipients, burden bearers, transfers and spillovers |
| Distribution | party/subgroup and, where applicable, participant-level or defensible risk-unit ceilings; expectation, quantiles, adverse tail, catastrophe probability and uncertainty |
| Noncompensatory boundary | consent, autonomy, rights, justice, legal prohibitions and accountable ethics/regulatory authorization |
| Aggregation | perspective, cardinal scale, numeraire, weights, valuation source/date, horizon and discount base, while retaining unaggregated results |

## Adjacent-method exclusions

- #571 supplies EVSI, cost, ENBS, no-sampling and commissioning semantics but
  does not currently model a stochastic acquisition-harm law.
- #570 supplies risk-sensitive matched current/perfect-information policy
  problems, not harm caused by gathering sample information.
- #595/VoC changes utility-based presentation of clairvoyant policy value; it
  does not model the physical or social act of sampling.
- #598 can report signed/private/social consequences of information sharing,
  not the acquisition process unless a future contract explicitly models it.
- Safe active learning and safe exploration are methodological analogies and
  possible future estimator references, not implementation evidence for this
  family.

## Conclusion

The concept warrants its own governed C18/M32 research track. The evidence is
insufficient for a generic executable kernel because there is no universal
harm unit, risk criterion, ethical feasibility rule or scalar aggregation.
The current capability remains `unsupported_research_scoping` until a narrow
domain and estimand receive candidate-bound independent scientific/domain
review and a named human verdict.

## Source register

| Source | Stable identifier | Used for | Not used for |
|---|---|---|---|
| Belmont Report | HHS/OHRP official record | harm kinds, probability/magnitude, alternatives, affected parties | numerical aggregation |
| 45 CFR 46 | HHS/OHRP regulation index | accountable human-subject protection boundary | universal cross-jurisdiction law or formula |
| ICH E6(R3) | Step 4 final guideline, 2025-01-06, Principles 1, 2, 3, 6 and 7 | rights, consent, quality and participant-risk proportionality | scalar VOI estimand or cross-domain authorization |
| Heath et al. | DOI `10.1007/s40273-024-01372-0`; PharmacoEconomics 42 (2024) | full decision context and ordinary net research value | acquisition-harm equivalence |
| Camilleri et al. | DOI `10.52202/068431-2406` | safety-constrained active experimental design | health-economic EVSI implementation |
| Bottero et al. | DOI `10.52202/068431-2226` | information-directed safe exploration | human-subject ethics approval |

The content-addressed retrieval, version, locator and limitation record is
`primary-source-manifest-20260802.json`. It records the HHS Belmont CLI 403 as
an unavailable automated retrieval rather than pretending to archive bytes.
