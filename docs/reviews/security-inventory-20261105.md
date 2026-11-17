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

On 17 November, the maintainer chose to retain automatic registry sync.
The TagBot job now requires the existing `JuliaTagBot` actor and, for comment
events, the current bot account ID (`50554310`) for the sender, comment author,
and issue author. The issue must be closed and must not be a pull request.
This matches the [checked General notifier](https://github.com/JuliaRegistries/RegistryCI.jl/blob/4d0f4fff4b137758e23b62470bcb8bdd645ba73d/AutoMerge/src/TagBot/TagBot.jl#L86-L117),
which creates or reuses a closed issue owned by its authenticated account. The
workflow preserves the existing bot-only manual-dispatch condition; ordinary
maintainer dispatch remains unavailable. A protocol or account change can
therefore stop automatic sync until the guard is reviewed.
The pinned action reads dispatch-event inputs before its configured inputs;
this workflow declares none, and adding them requires a separate review.
The policy test asserts the trigger, job condition, permissions, and action
inputs. Its synthetic `act` simulation runs only where the required offline
dry-run controls are available, so hosted environments may skip that simulation.

The `dangerous-triggers` exception is limited to this `issue_comment` key; the
trigger still exists and Zizmor reports it when ignores are disabled. The
guard checks notifier identity and issue metadata, not a particular release. TagBot
still receives a contents-write token and `DOCUMENTER_KEY`, and checks all
registered, untagged versions. The pinned action's
[subdirectory tree lookup](https://github.com/JuliaRegistries/TagBot/blob/6b7c22e7bc2b8f4d1c56b7199a63421cf2667ed1/tagbot/action/repo.py#L611-L636)
can select a different parent commit with the same Julia subtree before its
registry-PR lookup. The downstream shared-manifest check runs after tag creation.
This is a conditional future risk, not an observed Voiage release failure.
No native registration, TagBot run, tag, release, or deploy-key pairing was
verified as part of this change.

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
