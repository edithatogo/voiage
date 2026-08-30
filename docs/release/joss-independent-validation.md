# JOSS independent validation protocol

## Purpose

The Journal of Open Source Software (JOSS) requires demonstrated research use,
at minimum by the developer, as a pre-review gate. Its detailed
collaborative-effort criterion separately classifies a single-author project
with no community engagement, external use, or collaborative input as not
acceptable, although the pre-review and editorial guides call non-author
engagement a strong positive signal rather than a hard gate. Non-author issues,
pull requests, discussions, installation reports, and research use can provide
that human evidence. Automated agents, dependency bots, fallback-only adapters,
and aspirations in another repository maintained by the same author cannot.

This protocol separates a completed developer research-use record from a
non-author validation exercise. It does not ask for an endorsement. Problems,
confusion, and unsuccessful installation attempts are useful evidence and
should be reported accurately.

Tracking issue: [#471](https://github.com/edithatogo/voiage/issues/471).

## Developer research-use record

Record a completed analysis that uses the exact public release in a research
workflow. The record should identify the research question, competing options,
uncertain quantities, units, release version, commands, and retained results.
It may be public or available confidentially to the JOSS editorial team. The
paper's synthetic demonstration and a project that only plans or falls back
from a `voiage` integration do not establish this use.

## Clean installation

Use Python 3.12, 3.13, or 3.14 in a new environment:

```console
python -m venv voiage-joss-review
source voiage-joss-review/bin/activate
python -m pip install --upgrade pip
python -m pip install voiage==2.2.0
python -c "import voiage; print(voiage.__version__)"
```

On Windows, activate the environment with
`voiage-joss-review\Scripts\activate` instead.

The final command should report `2.2.0`.

## Core calculation

Run this example without cloning the repository:

```python
import numpy as np

from voiage.analysis import DecisionAnalysis
from voiage.schema import ValueArray

net_benefit = ValueArray.from_numpy(
    np.array(
        [
            [10.0, 12.0],
            [11.0, 9.0],
            [13.0, 14.0],
        ]
    ),
    strategy_names=["Standard care", "New treatment"],
)

analysis = DecisionAnalysis(net_benefit)
print(f"EVPI: {analysis.evpi():.3f}")
```

The expected result is:

```text
EVPI: 0.667
```

## Study-value exercise

The participant should check out the reviewed release and run the paper's
declared health example:

```console
git clone https://github.com/edithatogo/voiage.git
cd voiage
git checkout --detach v2.2.0
uv run --locked --extra plotting python scripts/generate_paper_health_example.py --verify-tracked
shasum -a 256 --check paper/reproduction.sha256
```

The participant should compare
`paper/data/synthetic_health_example_summary.csv` and
`paper/data/synthetic_health_example_results.csv` with the worked example in
`paper.md`. In particular, they should report whether they can identify:

- the uncertain health effect and programme cost;
- which quantity the proposed study informs;
- the outcome variance, allocation, and candidate sample sizes;
- the population, time horizon, discount rate, value realisation, delay, and
  study costs;
- the meaning of EVPI, EVPPI, EVSI, and ENBS; and
- why the two ENBS scenarios cross zero at different sample sizes.

The public `v2.2.0` tag resolves to
`7af563c8cb373057d30662650b3f332f39e05b83`. The retained worked-example manifest
identifies the original v2.0.0 reproduction; its portable outputs are also
rechecked under v2.2.0. The historical VOP use record has a separate automated
two-environment replay in `paper/research-use/v2.2.0/`, not new human use.
The verification command regenerates in a temporary directory and compares the
portable CSV outputs. The checksum command checks the retained artifacts;
cross-platform byte-identical rendering of newly generated figures is not claimed.

## Evidence to report

Please comment on issue #471 or open a linked issue with:

- participant role or research context, without personal information that they
  do not want made public;
- operating system, processor architecture, Python version, and installation
  command;
- whether installation and the core calculation succeeded;
- whether the study-value exercise ran, which outputs were inspected, and
  whether each listed assumption and result was understandable;
- any unclear terminology, assumptions, warnings, or documentation;
- any defect, unexpected result, or missing prerequisite;
- whether author intervention was needed.

If the participant cannot report publicly, the author may instead retain an
editor-verifiable record and tell the JOSS editor that confidential evidence is
available. The manuscript should mention external use only when the evidence
supports that exact statement.

## pyOpenSci-to-JOSS partner handoff

Independent validation evidence supports review readiness but does not replace
either venue's process. The selected sequence is fail-closed:

1. Refresh the current
   [pyOpenSci author guide](https://www.pyopensci.org/software-peer-review/how-to/author-guide.html),
   package-scope guidance, submission template, and JOSS requirements.
2. Freeze the candidate revision and rerun the submission-readiness, wheel and
   sdist identity, documentation, test, security, and JOSS paper checks.
3. The maintainer has selected and authorized the pyOpenSci-first route and
   JOSS partnership. Complete all repairs, personal declarations, survey,
   contact-capacity clarification and human communication review before posting.
4. Respond to pyOpenSci editors and reviewers personally. Record the public
   review issue, exact reviewed revision, findings, changes, and final external
   decision without inferring acceptance from local checks.
5. If pyOpenSci accepts the package, create the requested reviewed-version
   release and archive evidence. Add a pyOpenSci badge only after the external
   acceptance record authorizes it.
6. Confirm again that the package meets JOSS scope. Use the already selected
   partner route only after its eligibility and author prerequisites are met,
   identifying the accepted pyOpenSci review issue.
7. Treat the JOSS paper check, editorial screening, acceptance, DOI, and badge
   as JOSS-controlled outcomes. Add or claim them only from authoritative
   receipts.

The current partnership guidance says JOSS may accept the pyOpenSci software
review and focus its review on `paper.md`; it does not guarantee JOSS scope or
acceptance. No step above authorizes an agent or repository validator to open
an issue, submit a form, communicate with editors, or add an acceptance badge.

Authoritative route guidance:

- <https://www.pyopensci.org/software-peer-review/partners/joss.html>
- <https://www.pyopensci.org/software-peer-review/how-to/editors-guide.html>

## Completion boundary

The demonstrated-use gate is complete when attributable or editor-verifiable
evidence records completed research-workflow use of the released package. The
author-selected engagement prerequisite is complete when separate evidence
records genuine non-author community engagement, external use, or collaborative
input. The exercise above is one route, not the only possible route. A locally
repeated demonstration, an AI-agent run, or an automated continuous-integration
result does not satisfy either evidence class.
