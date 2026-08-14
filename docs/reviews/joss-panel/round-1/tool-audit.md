# Round 1 tool audit

Date: 24 July 2026

Reviewed revision: `1dfcd3d42af591ca15fcc03ad958123cc153dbbf`

## Repository preflight

Commands:

```console
uv run --extra ci python scripts/validate_joss.py
uv run --extra ci pytest tests/test_joss_readiness.py --no-cov -q
uv run --extra dev tox -e joss
git diff --check
```

Result: pass. The focused test suite reported seven passing tests. The only
messages were two upstream `xarray` deprecation warnings about generic NumPy
timedeltas.

This proves that the repository-owned structural contract passes. It does not
assess the strength of the scientific argument, the accuracy of every claim, or
JOSS eligibility.

## SourceRight

Pinned submodule revision:
`dde39b3bb334f79f12e395a5317b21e036336bdd`.

The audit used a disposable directory. It did not apply citation changes:

```console
pandoc -f bibtex -t csljson paper.bib \
  -o /tmp/voiage-joss-sourceright/references.csl.json
cargo run --quiet --manifest-path .repo-tools/sourceright/Cargo.toml \
  --bin sourceright -- validate-csl --json \
  /tmp/voiage-joss-sourceright/references.csl.json
```

The unnormalised Pandoc conversion produced seven structural diagnostics:

- five DOI values retained mixed case, while SourceRight's provider-matching
  contract requires lower-case canonical DOI values;
- the three `@misc` records converted to empty CSL types.

The disposable CSL was normalised to lower-case DOI values and `software`
types. SourceRight then returned:

```json
{"ok":true,"path":"/tmp/voiage-joss-sourceright/references.csl.json","diagnostics":[]}
```

The integrity report contained no CSL errors, conflicts, or queued decisions.
It reported missing verification sidecars for all eight records and no DOI for
the GitHub release and Software Heritage snapshot. Those two sources use
authoritative persistent URLs rather than DOI identifiers.

SourceRight's current citation extractor reported zero occurrences after
Pandoc rendered the paper to author-date text. Manual inspection confirmed that
the rendered text contains all eight citations. The extractor result is
therefore a documented tool-coverage limitation, not evidence that the
manuscript lacks citations. The repository-owned JOSS validator independently
confirmed that every Pandoc citation key resolves and that no bibliography
record is uncited.

## Authentext

Pinned submodule revision:
`7f70dad5b6deab1af92faf037ef2638e7f3aea05`.

Round 1 applies the professional core, academic, and reasoning modules as an
editorial checklist. Suggestions are evaluated sentence by sentence and are
not applied when they would remove necessary qualification, alter scientific
meaning, or replace a precise technical term. Particular checks are:

- vague or promotional claims;
- formulaic section and paragraph structure;
- unsupported quantitative or impact statements;
- filler and artificial signposting;
- abstraction-level changes between domain and implementation language;
- internal contradictions and unverified reasoning;
- preservation of identifiers, versions, commands, and citation keys.

## Official criteria snapshot

The rubric was reconciled on 24 July 2026 against the current JOSS submission,
paper-format, review-criteria, and editorial-guide pages linked from
`../rubric.md`. The key current requirements used in this review are:

- 750–1,750 words;
- substantive Summary, Statement of need, State of the field, Software design,
  Research impact statement, AI usage disclosure, Acknowledgements, and
  References sections;
- a non-specialist summary without API documentation;
- a build-versus-contribute justification;
- concrete research-use or near-term-significance evidence;
- meaningful design choices and trade-offs;
- complete AI-use, funding, and conflict disclosures;
- a reviewer-installable, documented, tested research package.

The separate JOSS pre-review gate for demonstrated research use and the strong
positive signal from non-author engagement remain outside the manuscript score.
