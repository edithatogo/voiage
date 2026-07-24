# JOSS independent validation protocol

## Purpose

The Journal of Open Source Software (JOSS) evaluates whether research software
has credible research use and, for a single-author project, evidence of
community engagement. Automated agents, dependency bots, and another repository
maintained by the same author are not independent evidence.

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
python -m pip install voiage==1.0.0
python -c "import voiage; print(voiage.__version__)"
```

On Windows, activate the environment with
`voiage-joss-review\Scripts\activate` instead.

The final command should report `1.0.0`.

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

The participant should then follow one worked example relevant to their
interests from the [voiage documentation](https://edithatogo.github.io/voiage/)
and record whether the assumptions, units, inputs, result, and limitations were
understandable.

## Evidence to report

Please comment on issue #471 or open a linked issue with:

- participant role or research context, without personal information that they
  do not want made public;
- operating system, processor architecture, Python version, and installation
  command;
- whether installation and the core calculation succeeded;
- the second example attempted and its outcome;
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
