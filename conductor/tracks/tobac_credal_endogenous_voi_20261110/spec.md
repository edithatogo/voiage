# Specification — Credal LP dominance and endogenous censored EVSI for tobac

## Scope and baseline

`edithatogo/tobac` evaluates tobacco and vaping regulatory policies where key illicit-supply, border-seizure, and wastewater biomarkers are partially identified (producing polyhedral/interval credal sets $\Theta_I$ rather than point-identified priors) and where surveillance precision depends endogenously on policy and enforcement actions (`BF1`, `BF2`, `F03`, `F05` in `tobac`).

`voiage` owns generic value-of-information, value-of-perspective, and credal/imprecise-probability decision rules (`rust/crates/voiage-domain`, `rust/crates/voiage-numerics`). `tobac` owns tobacco epidemiology, customs/wastewater partial-identification bounds (`src/tobac/partial_identification.py`), and NZ/AU regulatory policy specifications.

## Proposed capabilities

1. **Polyhedral credal-set dominance and E-admissibility with duality witnesses (`CE01`, `CE02`)**:
   - Given a linear or affine mapping from partially identified parameter bounds $\theta \in \Theta_I = \{\theta : A\theta \le b\}$ to strategy net monetary benefits $U(a, \theta)$, classify actions under $\Gamma$-maximin, $\Gamma$-maximax, interval dominance, and E-admissibility ($\exists \theta \in \Theta_I$ or prior $P \in \mathcal{P}$ such that $a \in \arg\max_{a'} \mathbb{E}_P[U(a', \theta)]$).
   - Emit explicit primal witness points $\theta^* \in \Theta_I$ for E-admissible actions and Farkas/dual convex-combination weights $\lambda \ge 0$ certifying strict dominance over $\Theta_I$.
2. **Action-dependent and Tobit-censored sequential EVSI (`CE03`, `CE04`)**:
   - Extend sequential EVSI to observation channels whose signal variance, detection probability, or lower limit of quantification (`<LOQ` left-censoring in wastewater; seizure-rate conditioning in border enforcement) depends on the chosen policy/enforcement action $a$.
   - Guarantee Blackwell garbling monotonicity ($\text{EVSI}(Y_{\text{garbled}}) \le \text{EVSI}(Y)$) and non-negativity across all censored/endogenous channel configurations.
3. **Zero-copy Polars/Arrow credal and paired-draw handoff (`CE05`)**:
   - Round-trip polyhedral/interval constraint tables and draw-keyed `dapane`/`tobac` net-benefit frames through `voiage`'s Polars/Arrow interchange without losing `draw_id`, `strategy_id`, `perspective`, or `subgroup_id`.
