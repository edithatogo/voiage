# Primary-source and estimand review

## Review question

Should harms caused by the act of obtaining information be represented by an
existing VOIAGE method, a distinct sampling-design estimand, or an unsupported
research boundary?

## Source findings

### Research value and ordinary cost

Strong et al. (2024, DOI `10.1007/s40273-024-01372-0`) describe VOI trial
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
protections for specified populations. ICH E6(R3) Principle 7 requires trial
processes to be proportionate to risks to participants and to the importance
of the information, with attention to rights, safety, well-being and
unnecessary burden.

These sources are ethical/regulatory authorities, not numerical VOI formulas.
They rule out any software claim that positive ENBS alone authorizes a study.

## Estimand disposition

Let `d0` denote no sampling and `d` a declared sampling action. Let `Y_d` be
the information produced, `H_d` the acquisition-harm outcome, `C_d` ordinary
research cost and `V(d)` the downstream value improvement from using `Y_d`.
All quantities must be defined on a declared joint probability space because
information, harm and state may be dependent.

When a declared stakeholder valuation `L(H_d)` is separable from downstream
decision value and is in the same cardinal units, a candidate signed scalar is

```text
NIV_H(d) = E[V(d)] - C_d - E[L(H_d)].
```

This is not a universal definition. When harms are heterogeneous,
incommensurate, catastrophic, absorbing or protected by non-compensatory
constraints, the correct candidate is a constrained or vector design problem,
for example

```text
maximize E[V(d)] - C_d
subject to P(catastrophic H_d) <= alpha
           rho(H_d) <= budget
           d in ethically and legally feasible designs.
```

The risk functional `rho`, probability threshold, harm budget, affected-party
scope and feasibility authority are user- and domain-supplied. A finite
penalty cannot silently replace a hard or lexicographic prohibition.

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
| ICH E6(R3) | Step 4 presentation, 2025-01-23, Principle 7 | participant-risk proportionality and burden | scalar VOI estimand |
| Strong et al. | DOI `10.1007/s40273-024-01372-0` | full decision context and ordinary net research value | acquisition-harm equivalence |
| Camilleri et al. | DOI `10.52202/068431-2406` | safety-constrained active experimental design | health-economic EVSI implementation |
| Bottero et al. | DOI `10.52202/068431-2226` | information-directed safe exploration | human-subject ethics approval |
