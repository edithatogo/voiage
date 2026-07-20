# Mutation baseline promotion

Mutation baselines cannot approve themselves. The cohort workflow compares the
exact bytes of `voiage-cohort.json` with the protected repository variable
`VOIAGE_MUTATION_BASELINE_SHA256`.

Promotion is an explicit maintainer action:

1. Download `mutmut-universe.txt` and `mutation-cohort.json` from a trusted
   hosted mutation run.
2. Review every added and removed mutant ID, the score, absolute debt, debt
   density, source/configuration digest, locked Mutmut version, and lock digest.
3. Update the candidate baseline through normal reviewed repository governance.
   Do not add an approval Boolean to the baseline.
4. After the reviewed baseline commit is authoritative, calculate the SHA-256 of
   the exact `voiage-cohort.json` bytes and set that digest as the protected
   GitHub repository variable `VOIAGE_MUTATION_BASELINE_SHA256`.

Until both the reviewed universe and the external digest exist, the cohort gate
retains candidate evidence and fails closed.
