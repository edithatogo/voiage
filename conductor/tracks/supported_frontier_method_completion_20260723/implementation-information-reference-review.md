# Implementation-information reference review

## Scope

This review freezes the experimental #593 finite-enumeration contract. It does
not claim that one terminology convention is universal, that uptake is causally
identified, or that implementation is independent of information.

## Primary definitions checked

- Johannesen et al. (2020), *Subcategorizing the Expected Value of Perfect
  Implementation to Identify When and Where to Invest in Implementation
  Initiatives*, defines EVPIM as the upper bound from replacing current
  implementation with perfect implementation and EVSIM as the value of a
  specified implementation improvement. DOI: 10.1177/0272989X20907353;
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC7488812/>.
- Heath (2022), *Calculating Expected Value of Sample Information Adjusting for
  Imperfect Implementation*, defines implementation-adjusted EVSI through
  sample-dependent market shares and explicitly explains why assuming uptake
  is unrelated to future data is unrealistic. DOI: 10.1177/0272989X211073098;
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC9189720/>.
- The 2024 ISPOR presentation, *Health Technology Decision-Making in an
  Imperfect World*, is treated as a taxonomy/reconciliation input, not as an
  independent numerical validation source:
  <https://www.ispor.org/heor-resources/presentations-database/presentation/euro2024-4016/144524>.

## Contract interpretation

The evaluator uses a joint state/action model. Each implementation scenario is
a conditional distribution over realized actions given both the uncertain
state and the intended action. The post-sample scenario is additionally
conditional on the observed signal. This directly represents the dependence
highlighted by Heath rather than multiplying an information value by a fixed
uptake fraction.

The four primary value cells are current/perfect information crossed with
current/perfect implementation. The signed interaction is
`C11 - C10 - C01 + C00`; consequently `EVP = realizable EVPI + EVPIM +
interaction` under this explicit cell convention. A specific implementation
scenario yields EVSIM, and a declared signal likelihood with signal-dependent
implementation yields IA-EVSI. Costs are subtracted only after the gross
components have been identified.

`EVEIm` and `EVSEIm` remain review candidates and presentation labels. They do
not create additional estimands or numerical kernels.

## Assurance boundary

The normative fixture is a finite exact enumeration with state-dependent
current uptake, a specific implementation intervention, sample likelihoods,
signal-dependent implementation, population/time scaling, complete ties and
decomposition residuals. Continuous integration, causal estimation of uptake,
dynamic diffusion and external scientific approval remain future gates. Python
is experimental; Rust, R, Julia and Mojo are explicitly unimplemented.
