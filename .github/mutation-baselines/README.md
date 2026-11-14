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
4. After the human review, the maintainer may stage that approval in the
   maintainer-controlled repository variable `VOIAGE_MUTATION_STAGED_REVIEW`:

   ```json
   {"baseline_sha256":"<64 lowercase hex>","current_sha256":"<64 lowercase hex>","pull_request":1313,"head_sha":"<40 lowercase hex>","base_sha":"<40 lowercase hex>"}
   ```

   Use the actual pull request number, exact head and current base commits, the
   reviewed baseline digest and the current main anchor. The staged record is
   accepted only for the matching `pull_request` event and exact checkout. It
   does not apply to pushes or to another PR, head, base or baseline. The main
   anchor must still be configured. A changed head or base needs a fresh scope
   check and updated staged record. All identity, universe, score and debt gates
   still run; a missing or malformed record fails closed.
5. After the reviewed baseline commit is authoritative, set the digest of its
   exact `voiage-cohort.json` bytes as the protected repository variable
   `VOIAGE_MUTATION_BASELINE_SHA256`, then remove the staged variable. Confirm
   the authoritative commit's mutation workflow using the current anchor.

Until the reviewed universe and a matching external digest exist, the cohort
gate retains candidate evidence and fails closed. Staging does not change the
main anchor or constitute review; only the maintainer's explicit review does.
