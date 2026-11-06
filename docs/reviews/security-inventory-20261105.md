# Security inventory — 5 November 2026

GitHub's paginated open code-scanning API returned 109 alerts: 103 CodeQL,
three zizmor and three OpenSSF Scorecard. This is an inventory across recorded
instances, not a claim that all findings apply to the current commit or that
every finding is a security vulnerability. The exact API snapshot is retained
locally under `.conductor/local/colab-20261105/diagnosis/`.

Two zizmor findings, #1326 and #1327, identify the pinned PyPI publishing action
in `release.yml` as affected by GHSA-5xjx-3w67-w9x9. Commit `b2057fcf9` updates
the pin to the verified upstream v1.14.3 commit; 16 release-workflow tests pass.
This local change requires review, full applicable checks and protected delivery
before the alerts can be expected to clear. No release was performed.

Finding #1309 concerns Julia TagBot's `issue_comment` trigger. The workflow
restricts the job to the JuliaTagBot actor, but this inventory does not establish
whether the warning is exploitable or whether changing the trigger preserves
registry behavior. The upstream [canonical example](https://github.com/JuliaRegistries/TagBot/blob/master/example.yml)
also uses `issue_comment`, with an actor check; this repository limits the job to
the `JuliaTagBot` actor and passes fixed inputs to a pinned action. No untrusted
comment text is interpolated into shell commands in this workflow. These are
bounded observations, not proof of the action or credential security. The
alert remains open; no trigger or suppression was changed.

The largest groups include import-style, possibly uninitialized local variable,
unused-import and empty-except findings. Sampled uninitialized-local findings
point to optional-import tests using `pytest.skip`; each needs context-aware
triage. Do not bulk-dismiss alerts or relax scanning gates. Scorecard, secret
scanning and push protection are configured; Dependabot security updates were
reported disabled on the repository readback. Renovate dependency PRs are open.

Two sampled runtime-severity alerts are stale relative to inspected source:
#798 references `EnhancedJaxBackend.__init__`, which currently calls
`super().__init__()`. #776 was last seen at commit `7992dcee` and reports an
uninitialized `df`; the current format dispatch assigns it for CSV/Parquet,
returns for Excel/JSON, and raises on the guarded remainder. These observations
are triage, not a substitute for a current scan; neither alert was dismissed.

Scorecard alert #1331 points to the deferred native capacity workflow's local
reusable-workflow reference at line 25. It contains `$/` rather than `./`;
`actionlint .github/workflows/native-build-capacity-observation.yml` reports an
invalid workflow-call format. The finding is actionable, but repair and hosted
qualification belong to the deferred HPC lane. No alert was dismissed.

Scorecard alert #854 reports SAST on 24 of 30 sampled commits at its last
observed revision `888e071a`. The current CodeQL workflow is configured for
main pushes, pull requests and merge groups; that configuration alone does not
prove every historical commit was analyzed. Alert #851 reports the separate
OpenSSF Best Practices badge as in progress. The project's public JSON endpoint
confirmed `badge_level: in_progress` on 5 November; its questionnaire is not an
accepted assessment. These findings were not dismissed or represented as
completed.

CodeQL alerts #1277, #1314 and #1315 identify ellipsis statements in
`Protocol` method bodies in `voiage/metamodels.py` at the observed main
revision `888e071a`. They are interface declarations, not ordinary executable
statements missing an implementation. This is a scoped source assessment; no
alert was dismissed and it does not assess other CodeQL findings.

After PR #1271 merged as `f7c23b10b77e20d8cbce30736eaff779c88b7e61`,
GitHub's native code-scanning readback marked alerts #1326 and #1327 fixed at
2026-11-05 12:43:53 UTC. The [main Zizmor run 42029537586](https://github.com/edithatogo/voiage/actions/runs/42029537586)
completed successfully on that merge commit. This readback is specific to the
publishing-action findings and does not close other security alerts.
