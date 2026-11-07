# Value of Perspective and Value of Heterogeneity in Voiage

Date: 2026-11-08. Inspected Voiage baseline: `ce9ea7da53ed302cc57664068a53a1ede72b9e5e`.
Status: qualification requirements; no runtime or scientific-maturity change.

## Ownership and approach

**Voiage is the production home for reusable Value of Perspective (VOP) and
Value of Heterogeneity (VOH), alongside VOI. `vop_poc_nz` is a proof of concept.**
The POC retains NZ data, institutional assumptions, empirical scenarios and its
research compendium. Qualified reusable methods belong in Voiage's Rust core,
with thin language interfaces. Voiage must not import the POC at runtime.

Extend the existing [VP03 handoff](innovation-paignia-vop-handoff.md),
[issue #1286](https://github.com/edithatogo/voiage/issues/1286) and scientific-alignment
owner, coordinated with Innovate E04. These requirements are not new task IDs,
a second promotion ledger, or additional O01-O08 completion counters. Inventory
and qualification of either method can proceed without an unrelated game adapter.

Chosen approach: reconcile and extend existing methods, verify a small method slice,
then migrate the POC consumer. A wholesale POC copy would import application-specific
assumptions; leaving production algorithms in the POC would create the wrong dependency.
Do not create another evaluator or require Paignia, Innovate or Kairos for ordinary VOP/VOH.

## Existing source to reuse, not duplicate

| Existing source | Inspected responsibility | Remaining qualification boundary |
| --- | --- | --- |
| `voiage/methods/perspective.py` and `voiage/contracts/perspective.py` | Perspective dimension, expected net benefits, directional regret and decision summaries; method contract 1.1.0 | Reconcile every summary with the primary directional estimand; Rust authority, weights, ties, units and installed parity must be evidenced per method |
| `voiage/methods/perspective_diagnostics.py` and `voiage/plot/perspective.py` | Existing diagnostics/presentation paths, including prior POC-derived work | Inventory existing promotion rather than claim a new wholesale migration; numerical error is not normative uncertainty |
| `voiage/methods/heterogeneity.py` and `rust/crates/voiage-numerics/src/heterogeneity.rs` | Existing descriptive subgroup strategy gain; inspected Rust kernel uses subgroup row frequencies as weights | Preserve the supported semantics. Population-weighted joint-state analysis needs an explicit contract, not silent reinterpretation of rows |
| `voiage/methods/heterogeneity_value.py`, `voiage/contracts/heterogeneity_value.py` and `specs/frontier/heterogeneity-value/v1/` | Experimental finite C0/Cf/P0/Pf decomposition and optional signal model | No assumption of Rust/R/Julia parity, stable promotion or valid subgroup estimation from fixture existence |
| `vop_poc_nz/src/vop_poc_nz/perspective.py` | Expected-value directional regret, tie-aware acceptability and two-perspective mixture frontier | POC reference and migration source, not independent ground truth or a production dependency |

Source presence, documentation and old completed track labels do not establish
current method qualification. Retain existing maturity labels until the appropriate
method and binding evidence is accepted.

## VOP: required semantics and outputs

For perspective p, let a_p maximise its current expected net benefit. Directional
VOP for choosing under p and evaluating under q is:

`VOP(p -> q) = E[NB_q(a_q)] - E[NB_q(a_p)]`.

Both terms use the target perspective q, the same population and units, and a
specified common feasible policy set. Report direction, chosen/evaluating perspective,
strategy, tie policy, population scaling and decision rule. Reversing direction
need not yield the same value. Never substitute `E[NB_q(a_q)] - E[NB_p(a_p)]`:
that subtracts different evaluation scales, not directional regret.

The primary result uses decisions selected from current expected values. Per-draw
optimisation is a separately labelled diagnostic/information regime, not the same
estimand. Qualify directional expected VOP, disagreement/regret matrices, tie-aware
perspective acceptability probabilities, the Perspective Acceptability Frontier,
and exact regime-switch thresholds before broader fitted regime-discovery claims.
Perspective probabilities of optimality describe uncertainty in model outcomes,
not a probability that a normative perspective is morally or scientifically true.

Mixture/frontier weights require declared commensurability, units and normative
interpretation. Do not silently treat them as Bayesian probabilities. A reporting
summary over perspectives, consensus or maximin policy needs its own assumptions;
keep it distinguishable from the directional headline result. Harmonisation is a
subordinate decision/reporting analysis, not a replacement estimand or separate
mandatory publication programme. MCDA remains an explicit export/integration boundary.

## VOH: policy tailoring and information value are different quantities

For fixed perspective p, a prespecified partition and declared population weights,
reuse the finite contract's four policy values:

- C0: best common policy under current information.
- Cf: best subgroup-specific feasible policy under current information.
- P0: expected best common policy after perfect state information.
- Pf: expected best subgroup-specific feasible policy after perfect state information.

Preserve the existing contract terminology: static VOH = Cf-C0; its dynamic
heterogeneity contrast = Pf-P0; EVPI0 = P0-C0; EVPIf = Pf-Cf. Verify
`(Pf-P0) - (Cf-C0) = EVPIf - EVPI0` independently. Do not equate every publication's
use of "dynamic value" to one of these outputs without an explicit estimand mapping.
The difference between the two VOH contrasts may be negative; it is not itself EVPI.

With an admissible signal, return S0 and Sf plus EVSI0=S0-C0 and EVSIf=Sf-Cf;
verify `(Sf-S0)-(Cf-C0)=EVSIf-EVSI0`. Study cost changes signed net research value,
not gross VOH or EVSI. Report gross tailoring gain and net implementation/classification
value separately. Nonnegativity of gross tailoring relies on nested feasible policy
sets and compatible valuation; different constraints or added costs need signed reporting.

Distinguish acting on known subgroup membership, learning subgroup effects/prevalence,
measuring/classifying membership, discovering a partition and making later adaptive
decisions. A subgroup is not a perfectly observed random parameter by default.
Declare eligibility, measurement accuracy, implementation costs, resource constraints,
fairness/privacy limits, selection and multiplicity rules. Held-out/prespecified
validation is required for discovered subgroups; finite enumeration does not establish
sparse-group validity, causal effects or freedom from selection bias.

## Joint VOP × VOH × VOI contract

Preserve separate strategy, perspective, subgroup, joint-parameter-state and conditional
simulation-replicate identities through existing O01-O04/O06 contracts. For the finite
references use `NB[state, subgroup, strategy, perspective]`, explicit state probabilities
and population shares. The labels describe distinct roles, not statistically independent
variables. Uncertain or state-dependent shares require their declared joint model.
Do not infer population weights from simulation counts; aggregate conditional simulation
noise at the correct decision stage before maximisation.

Evaluate common and tailored policies separately under each perspective. Give directional
VOP under each policy class and VOH under each perspective; show interaction rather than
adding VOP, VOH and VOI as independent gains. Preserve correlated outcomes and valuation
components without double-counting effects, transfers or implementation costs.
Normative perspective/equity weights remain declared application inputs, not estimates
of a uniquely correct objective.

A Paignia player is not automatically a perspective, and a subgroup is not automatically
a player. Optional strategic evaluation adds recipient information, response policies,
solution concept and selection provenance through SI01-SI04; it must not overwrite the
fixed-policy VOP/VOH estimands or their conditions. Signed strategic effects retain their
own interpretation. Existing independent release paths and the HPC deferral remain intact.

## Qualification within the existing owners

These are acceptance requirements for existing work, not a new execution registry.

| Existing owner/work | Bounded implementation and verification requirement |
| --- | --- |
| #1286/VP03 scientific alignment; Q01 where admitted | Inventory POC and Voiage methods as duplicate, reusable-correct, domain-specific or unsupported. Record exact source versions and scientific differences in the existing promotion record |
| Rust consolidation/Q02 and method owner | Reuse `voiage-domain` and `voiage-numerics`. Qualify one finite VOP or VOH slice at a time; proposed native perspective kernels must not become a parallel Python implementation. Preserve existing API/ABI semantics or version the migration |
| O01-O04/O06 | Test perspective/group/state identity, explicit population versus sampling weights, units, constraints and paired Arrow/Parquet interchange in their existing tests |
| O07/Q03 and method owner | Admit the reference inputs below into existing numerical fixtures after independent review; add a failing distinguishing test before changing an estimator, then verify native results |
| O08/Q07 | Test a standalone installed Rust consumer without Python/POC/game providers and thin supported bindings; unsupported methods must not silently fall back while claiming native execution |
| POC migration owner; Innovate E04 | After provider qualification, pin the accepted package/hash, migrate the POC adapter and reproduce a named NZ example with rollback and retained historical artefacts |
| Q09/method research owners | Compare directional VOP, PAF and VOH decomposition on exact/held-out problems; document independent and same-author evidence separately; retain current software/evaluation publication paths |

Before native implementation, inventory exact existing test modules and lock the
function/result signatures in the canonical method contract. Candidate additions
`rust/crates/voiage-numerics/tests/perspective_reference.rs` and
`rust/crates/voiage-numerics/tests/heterogeneity_decomposition.rs` are proposed targets,
not asserted existing tests or a new universal gate. Use the existing frontier
registry and installed-consumer checks; do not create a competing validator.

## Independent finite references

The adjacent [reference-input file](vop-voh-reference-cases.json) contains seven
illustrative decision problems, not a runtime schema, accepted dataset registry or
a claim of package conformance. Their expected values are derived by finite sums,
maximisation and explicit policy enumeration, not by calling Voiage or the POC.

| Case | Required distinguishing result |
| --- | --- |
| directional | p=(6,4), q=(1,9): VOP(p→q)=8, VOP(q→p)=2; mixture switches at q-weight 0.2 |
| current-versus-draw | Paired p/q samples choose A under current p and B under current q: expected directional regret 3.5, but draw-wise regret 0 |
| population-weighted | Shares (0.75,0.25), group values (8,2)/(1,9): C0=6.25, Cf=8.25, static gain 2; equal row counts are not a substitute for shares |
| information-interaction | Symmetric hidden effect reversal: C0=Cf=P0=5, Pf=10; informative 0.75-accurate signal gives S0=5, Sf=7.5 |
| signed-difference | C0=2.75, Cf=5, P0=Pf=5.5: static 2.25, dynamic contrast 0, EVPIf-EVPI0=-2.25 |
| joint-perspective-group | Directional p→q regret is 4.5 for common policies and 7.5 for tailored policies; this interaction is not an additive independent benefit |
| no-heterogeneity | Identical subgroup payoffs give zero static VOH despite different declared population weights |

Also require translation/positive-scaling invariance where appropriate, complete ties,
identity/permutation controls, identical-perspective zero, empty/invalid/negative
weights, missing groups, unsupported constraints and unit mismatches. Do not silently
repair invalid input or hide numerical failures by clipping. Retain the existing
O07 simulation-noise negative control instead of copying its implementation here.

## Evidence sources, applications and release boundaries

- Primary VOH framework: Espinoza, Manca, Claxton and Sculpher (2014),
  [PMID 24944196](https://pubmed.ncbi.nlm.nih.gov/24944196/), DOI
  `10.1177/0272989X14538705`. It distinguishes current-evidence and further-evidence
  value. Source-specific equations and applicability require review before equivalence claims.
- VOP source semantics: inspected POC `perspective.py`, blob
  `6540312b2d46ca75f9f5dcd7e0e56b2f80d8dd88`; inspected Voiage `perspective.py`,
  blob `81d6be47f1bbd3beeffd2be68ae3c8fb4bffdf63` (contract 1.1.0).
- VOH source semantics: native `heterogeneity.rs`, blob
  `193f7dcc0ebd286ee482cdee8dfe347f83a974bd`; finite-contract README,
  blob `3f227f6c533899d1a2bf501df0547689577a9e0e`.

Start with synthetic exact cases; then select a pinned, licensed POC application
with paired perspective/subgroup outcomes and documented uncertainty. The POC is an
application demonstration, not an independent validation oracle or independent adoption.
Observed health/diffusion outcomes do not validate normative welfare assumptions.
No patient-level publication, Hub upload or manuscript submission is authorised here.

Planning and reference arithmetic can be checked separately, but stable/native claims
need the real method tests, clean installed consumers, documentation/registry agreement
and exact-head hosted evidence. Full repository verification uses the current guarded
tox policy on its supported interpreter. This amendment implements no kernel, changes
no provider pin, advances no task status and claims no completed POC migration.
