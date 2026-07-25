# Round 5 domain and health-economics review

Date: 24 July 2026
Reviewed head: `07cf3f83098952439f7b94773c62a8cbf9c176c1`
Score: 747/1000
Disposition: major revision

This is an independent AI-assisted domain review.

## Independent numerical findings

- The generating incremental net benefit has mean zero and standard deviation
  1,634.778 value units.
- The theoretical preferred-strategy probability is 50%.
- Theoretical EVPI is 652.182 value units per person.
- Theoretical health-gain EVPPI is 598.413.
- Theoretical programme-cost EVPPI is 259.312.
- All six EVSI and ENBS rows reproduce to numerical precision; EVSI at
  \(n=200\) is 124.179 value units per person.
- Discounted opportunities are 11,089.264 immediately and 5,161.052 under the
  delayed/60%-realisation scenario.
- The reproduction hashes and focused scientific tests pass.

## Deductions

| Area | Maximum | Awarded |
| --- | ---: | ---: |
| Scientific correctness and calculations | 250 | 226 |
| Study model, uncertainty, and interpretation | 200 | 159 |
| Need, significance, and research impact | 200 | 115 |
| Related work and cross-domain usefulness | 130 | 91 |
| Reproduction and release evidence | 120 | 78 |
| Clarity and precision | 100 | 78 |

## Blocking findings

1. No qualifying research use is documented.
2. Public v1 does not contain the software described by the paper.
3. `paper/SUPPLEMENTARY_METHODS_AND_FORMULAE.md` incorrectly called the
   plug-in EVPI estimator unbiased and claimed methods without executable
   support.

## Required scientific edits

- State the complete conjugate normal study model and preposterior variance.
- Explain that the normal QALY endpoint is an illustrative continuous-endpoint
  approximation.
- Distinguish evaluated sample-size brackets from a continuous optimum.
- Report generating-distribution benchmarks beside finite-PSA estimates.
- Give EVSI and study costs in value units.
- Say that the study does not recommend a real design or perform optimisation.
- Narrow cross-domain claims unless a non-health research application is
  demonstrated.
