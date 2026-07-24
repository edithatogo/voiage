# JOSS independent validation protocol

## Purpose

The Journal of Open Source Software (JOSS) screens for demonstrated research
use. Non-author issues, pull requests, discussions, installation reports, and
research use are strong positive signals for a single-author project.
Automated agents, dependency bots, and another repository maintained by the
same author are not independent evidence.

This protocol gives a non-author researcher or research-software practitioner a
small, reproducible exercise. It does not ask for an endorsement. Problems,
confusion, and unsuccessful installation attempts are useful evidence and
should be reported accurately.

Tracking issue: [#471](https://github.com/edithatogo/voiage/issues/471).

## Clean installation

Use Python 3.12, 3.13, or 3.14 in a new environment:

```console
python -m venv voiage-joss-review
source voiage-joss-review/bin/activate
python -m pip install --upgrade pip
python -m pip install voiage==2.0.0
python -c "import voiage; print(voiage.__version__)"
```

On Windows, activate the environment with
`voiage-joss-review\Scripts\activate` instead.

The final command should report `2.0.0`.

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
git checkout --detach v2.0.0
uv run --locked --extra plotting python scripts/generate_paper_health_example.py
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

Run this exercise only after the public `v2.0.0` tag resolves to the revision
identified in the JOSS paper and its release evidence.

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

## Completion boundary

This gate is complete only when a non-author has performed the exercise and
reported attributable or editor-verifiable evidence. A locally repeated test,
an AI-agent run, or an automated continuous-integration result does not satisfy
it.
