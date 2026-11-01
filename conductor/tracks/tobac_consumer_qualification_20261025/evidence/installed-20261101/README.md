# Installed tobac qualification, 1 November 2026

Two independent wheels were built from the producer and consumer main revisions in
[identity.json](identity.json) on macOS arm64 using Python 3.15.1. The verifier
compared installed files to each wheel, excluding the installer-generated RECORD.
All calculations ran outside the source trees with isolated Python imports.

| Consumer behavior | Result |
| --- | --- |
| Uniform, weighted and partitioned EVPI | 90 comparisons, maximum absolute error 2.842170943040401e-14, plus six buffer ownership checks. |
| Near-tied decisions | The exact finite-state oracle gives 2^-34 for full information and zero for an uninformative partition. All six state orders and two common payoff offsets matched exactly on the explicit installed adapter and Voiage JAX float64 CPU backend. |
| Dependent sources | Duplicate correlated reports retained gross value 3. Two sources cost 0.8 when the shared asset was new and 0.3 when already owned. Reversing sources and states preserved those values. |
| R11 scenario isolation and replay | Two 4,096-state regulatory scenarios were repeated in reverse order with unchanged inputs. The twelve current ecosystem-contract tests passed, including bounded extension, schema rejection, replay and rollback witnesses. |
| Covariance and evidence identity | A joint likelihood, labeled covariance, units and provenance survived Arrow/Polars interchange. Conditioning preserved weights and rejected reuse of the same evidence. |

The two installed-consumer test files returned **47 passed and one failed**.
The failing `test_real_installed_producer` uses tobac's default adapter. It
requires `2.2.0+tobac3151.2` and rejected the tested `2.2.0` candidate.
The explicit-candidate probe exercised the same conditioning route successfully.
This packet qualifies the candidate for the listed operations. Default promotion,
published-release acceptance, general migrations and scientific calibration remain open.

## Reproduction

Build both source revisions with `uv build --wheel`, install both wheel files into
a fresh Python 3.15.1 environment and verify installed member digests. Install the
consumer’s interchange dependencies and the two legacy tabular distributions
queried by the upstream oracle. Run the consumer's `scripts/qualify_voiage_consumer.py`
under isolated Python with the producer wheel, `--expected-version 2.2.0`,
`--buffer-ownership` and a fresh output file. Run the consumer's
`tests/engineering/test_evidence_interchange.py` and
`tests/engineering/test_ecosystem_contracts.py` with pytest importlib mode
and an empty pythonpath override.

Replay the three [October probes](../installed-20261029/README.md) in a fresh
directory with the November producer tree and consumer revision substituted
for their historical identity literals. For the exact near-tie probe, install
JAX and jaxlib 0.12.0.1. Copy `near-tie-jax.py.txt` to that directory with a
`.py` suffix and run it with the installed interpreter's `-I` flag. The output
records the device, comparison count and exact reference. All runs used the
repository resource guard with a 10 GiB reserve.
