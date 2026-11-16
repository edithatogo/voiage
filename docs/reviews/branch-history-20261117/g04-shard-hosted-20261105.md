
## Hosted evidence reconciliation — 5 November 2026

PR #1236's final head `a8760eed7707e09965ad344759befb30e6c4d170`
passed hosted run [40847354264](https://github.com/edithatogo/voiage/actions/runs/40847354264):
both shard jobs and the aggregate succeeded. Its tree
`981de629838599d4acadfb7e0fc1e390077a71ba` matches merged commit
`5af86b3cca4e3bd33cca35104eb3c00517f764d0`. Earlier hosted run
[40834939734](https://github.com/edithatogo/voiage/actions/runs/40834939734)
failed both shard jobs and the aggregate on a different candidate. These runs
provide a positive hosted result and one actual failed-matrix outcome; they
do not establish actual cancellation handling, current-head acceptance, or a
measured saving. Preserve the remaining G04 gates.
