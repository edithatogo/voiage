# Completed developer research workflow

On 27 July 2026, an isolated environment for the public
[`vop_poc_nz`](https://github.com/edithatogo/vop_poc_nz) revision
`2c46db2fe5f907d894bb07f1127c008f10ee462e` installed and imported published
`voiage==2.0.0` from `site-packages`. It did not import the `voiage` checkout.

The run used the VOP health-system HPV-vaccination Markov scenario, generated
500 deterministic cost and QALY uncertainty draws (seed `20260727`), formed
the two-strategy net-benefit matrix at NZD 50,000/QALY, and calculated EVPI
with the released package. Its result was NZD 0 per cohort. That is a valid
result for the stated model: the sampled scenarios did not change the preferred
strategy. It is not a clinical recommendation or a policy estimate.

The machine-readable receipt is
[`paper/joss-developer-research-use.json`](../../../../paper/joss-developer-research-use.json),
including input/output hashes, release version, command boundary and scope.
This is completed **developer/same-author research use**, not independent
adoption. It therefore clears only the demonstrated-research-use gate.
