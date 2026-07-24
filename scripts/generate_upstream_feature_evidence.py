#!/usr/bin/env python3
"""Generate reviewed upstream-artifact evidence for external feature claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
LANDSCAPE = ROOT / "specs" / "software-landscape"
REGISTRY = LANDSCAPE / "registry.json"
OUTPUT = LANDSCAPE / "upstream-feature-evidence.json"

# Exact upstream revisions observed during the review. Artifact URLs are built
# from these immutable commits rather than mutable default-branch HEADs.
TOOL_REVISIONS = {
    "r-voi": "b43b89593b2c8c8e5e8024a236a77f9a0dda85ed",
    "r-bcea": "f8fadf5353967d4c7db35d7e9e47238f8157b0ce",
    "r-dampack": "0430fee8b8b3736cdc7dda353bd39ffa8b8480a2",
    "r-heemod": "98c7da0e78d358f71d866c6a748c596217c5a78b",
    "r-hesim": "676630b6247c8fedec5deed8f23a157e722b392c",
    "savi": "4875b0c1fe6e361536875f32182ab99d38aea399",
    "pyro-oed": "359dc54bc03631743379b83423ac4a5e00ef17e6",
    "botorch": "db15bbc49e347bdf226a3f47db8c6d3a8506934d",
    "scikit-activeml": "18e1ea392418f12187e46ecb0fef1248c66b38f0",
    "emukit": "fe68d344383db21968bfc965bbbc3084e933436c",
    "vop-poc-nz": "2c46db2fe5f907d894bb07f1127c008f10ee462e",
    "decision-security": "1906bcd7797a35e2031f71f442f5a8318c36880e",
    "r-surveyvoi": "b29150bfc7a5599d5fe3ee6960ef3544243e77a4",
    "r-predtools": "4d90f59c22485177c65cfae3778ec16ec48e950a",
    "r-metanb": "289cccdc54b55e75d561a4477d1fa6177602ee60",
    "gaussian-voi-supplement": "756b15baf180a8f2f968314cc511fcbf6d689ec8",
    "bayescal-voi": "d9d399332aadb468e00e7f074bbd99f72a55fdcc",
    "metavoi": "95860bb0f4e524650a84003f0737f0da2c2553ed",
    "nrel-geothermal-voi": "054982097e1c969081880a5e6fe0c194e9d0aa5b",
    "kirstine": "ceec8f6f178ae9b241eb72e9f9bd6ea126e95dc3",
}

# Paths are evidence pointers only. VOIAGE never imports or executes upstream
# packages to provide its stable calculations.
FEATURE_PATHS: dict[tuple[str, str], dict[str, list[str]]] = {
    ("r-voi", "voi-core"): {
        "source": ["R/evpi.R", "R/evppi.R", "R/evsi.R", "R/enbs.R"],
        "tests": [
            "tests/testthat/test_evpi.R",
            "tests/testthat/test_evppi.R",
            "tests/testthat/test_evsi.R",
            "tests/testthat/test_enbs.R",
        ],
        "examples": ["vignettes/voi.Rmd"],
    },
    ("r-voi", "voi-evppi-estimators"): {
        "source": [
            "R/evppi_bart.R",
            "R/evppi_earth.R",
            "R/evppi_gam.R",
            "R/evppi_gp.R",
            "R/evppi_inla.R",
            "R/evppi_mc.R",
        ],
        "tests": [
            "tests/testthat/test_evppi.R",
            "tests/testthat/test_evppi_mc.R",
        ],
        "examples": ["vignettes/voi.Rmd"],
    },
    ("r-voi", "voi-evsi-estimators"): {
        "source": ["R/evsi.R", "R/evsi_is.R", "R/evsi_mm.R"],
        "tests": [
            "tests/testthat/test_evsi.R",
            "tests/testthat/test_evsi_is.R",
            "tests/testthat/test_evsi_mm.R",
        ],
        "examples": ["vignettes/voi.Rmd"],
    },
    ("r-voi", "voi-tidy-plots"): {
        "source": ["R/evppivar.R", "R/evsivar.R"],
        "tests": [
            "tests/testthat/test_evppivar.R",
            "tests/testthat/test_evsivar.R",
        ],
        "examples": ["vignettes/voi.Rmd"],
    },
    ("r-bcea", "bcea-economic-evaluation"): {
        "source": ["R/prepare_ceac_params.R", "R/ceaf.plot.R"],
        "tests": ["tests/testthat/test-ceac_plot_ggplot.R"],
        "examples": ["vignettes/ceac.Rmd"],
    },
    ("r-bcea", "bcea-voi"): {
        "source": ["R/evppi.R", "R/evppi.default.R"],
        "tests": ["tests/testthat/test-evppi.R"],
        "examples": ["vignettes/evppi.Rmd"],
    },
    ("r-bcea", "bcea-interactive-plots"): {
        "source": [
            "R/ceac_plot_graph.R",
            "R/evppi_plot_base.R",
            "R/evppi_plot_ggplot.R",
        ],
        "tests": ["tests/testthat/test-ceac_plot_ggplot.R"],
        "examples": ["vignettes/ceac.Rmd", "vignettes/evppi.Rmd"],
    },
    ("r-dampack", "dampack-voi"): {
        "source": ["R/evpi.R", "R/evppi.R", "R/evsi.R", "R/exp_loss.R"],
        "tests": [
            "tests/testthat/test_evpi.R",
            "tests/testthat/test_evppi.R",
            "tests/testthat/test_evsi.R",
        ],
        "examples": ["vignettes/voi.Rmd"],
    },
    ("r-dampack", "dampack-psa"): {
        "source": ["R/ceac.R", "R/icers.R", "R/psa.R", "R/run_psa.R"],
        "tests": [
            "tests/testthat/test_ceac.R",
            "tests/testthat/test_calculate_icers.R",
            "tests/testthat/test_psa.R",
        ],
        "examples": ["vignettes/psa_analysis.Rmd"],
    },
    ("r-hesim", "hesim-cea"): {
        "source": ["R/cea.R", "src/cea.cpp"],
        "tests": ["tests/testthat/test-cea.R"],
        "examples": ["vignettes/cea.Rmd"],
    },
    ("r-heemod", "heemod-evpi"): {
        "source": ["R/evpi.R"],
        "tests": [],
        "examples": [],
    },
    ("r-heemod", "heemod-modeling"): {
        "source": ["R/calibration.R"],
        "tests": ["tests/testthat/test_calibration.R"],
        "examples": ["vignettes/k_calibration.Rmd"],
    },
    ("savi", "savi-analysis"): {
        "source": ["scripts.R", "scripts_GAMfunctions.R", "scripts_GPfunctions.R"],
        "tests": [],
        "examples": ["test_data/", "report.Rmd"],
    },
    ("savi", "savi-burdens"): {
        "source": ["scripts_tables.R", "scripts_text.R"],
        "tests": [],
        "examples": ["test_data/"],
    },
    ("savi", "savi-reports"): {
        "source": ["report.Rmd", "scripts_plots.R"],
        "tests": [],
        "examples": ["report_files/"],
    },
    ("r-surveyvoi", "surveyvoi-survey-value"): {
        "source": [
            "src/rcpp_expected_value_of_decision_given_current_info.cpp",
            "src/rcpp_expected_value_of_decision_given_survey_scheme.cpp",
        ],
        "tests": [
            "tests/testthat/test_rcpp_expected_value_of_decision_given_current_info.R",
            "tests/testthat/test_rcpp_expected_value_of_decision_given_survey_scheme.R",
        ],
        "examples": ["vignettes/surveyvoi.Rmd"],
    },
    ("r-surveyvoi", "surveyvoi-optimization"): {
        "source": [
            "R/optimal_survey_scheme.R",
            "R/approx_optimal_survey_scheme.R",
            "R/approx_near_optimal_survey_scheme.R",
        ],
        "tests": [
            "tests/testthat/test_optimal_survey_scheme.R",
            "tests/testthat/test_approx_optimal_survey_scheme.R",
        ],
        "examples": ["vignettes/surveyvoi.Rmd"],
    },
    ("r-predtools", "predtools-validation-evpi"): {
        "source": ["R/voipred.R"],
        "tests": [],
        "examples": ["docs/reference/evpi_val.html"],
    },
    ("r-metanb", "metanb-cluster-voi"): {
        "source": ["R/MA_NB_tri_voi.R"],
        "tests": [],
        "examples": ["vignettes/"],
    },
    ("gaussian-voi-supplement", "gaussian-approximation-voi"): {
        "source": ["R/"],
        "tests": [],
        "examples": ["README.md"],
    },
    ("bayescal-voi", "bayescal-gaussian-process-voi"): {
        "source": ["src/bin/voi.rs", "src/bin/voi_selection.rs"],
        "tests": [],
        "examples": ["data/"],
    },
    ("metavoi", "metavoi-meta-analysis"): {
        "source": ["src/"],
        "tests": [
            "tests/test_evpi.py",
            "tests/test_evppi.py",
            "tests/test_evsi.py",
            "tests/test_importance_evsi.py",
        ],
        "examples": ["examples/"],
    },
    ("vop-poc-nz", "vop-directional"): {
        "source": [
            "src/vop_poc_nz/perspective.py",
            "src/vop_poc_nz/value_of_information.py",
        ],
        "tests": [
            "tests/test_perspective.py",
            "tests/test_perspective_conformance.py",
        ],
        "examples": ["examples/perspective_manifest.example.yml"],
    },
    ("decision-security", "security-evpi"): {
        "source": ["src/voi.py"],
        "tests": [],
        "examples": ["README.md"],
    },
    ("nrel-geothermal-voi", "geothermal-imperfect-information"): {
        "source": ["app.py"],
        "tests": [],
        "examples": ["sample_jupyternotebook.ipynb"],
    },
    ("pyro-oed", "pyro-eig"): {
        "source": ["pyro/contrib/oed/eig.py", "pyro/contrib/oed/search.py"],
        "tests": [
            "tests/contrib/oed/test_finite_spaces_eig.py",
            "tests/contrib/oed/test_linear_models_eig.py",
        ],
        "examples": ["examples/contrib/oed/"],
    },
    ("botorch", "botorch-kg"): {
        "source": ["botorch/acquisition/knowledge_gradient.py"],
        "tests": ["test/acquisition/test_knowledge_gradient.py"],
        "examples": ["docs/acquisition.md"],
    },
    ("botorch", "botorch-bald"): {
        "source": [
            "botorch/acquisition/active_learning.py",
            "botorch/acquisition/bayesian_active_learning.py",
        ],
        "tests": [
            "test/acquisition/test_active_learning.py",
            "test/acquisition/test_bayesian_active_learning.py",
        ],
        "examples": ["docs/acquisition.md"],
    },
    ("botorch", "botorch-pes"): {
        "source": [
            "botorch/acquisition/predictive_entropy_search.py",
            "botorch/acquisition/preference.py",
        ],
        "tests": [
            "test/acquisition/test_predictive_entropy_search.py",
            "test/acquisition/test_preference.py",
        ],
        "examples": ["docs/acquisition.md"],
    },
    ("scikit-activeml", "skactiveml-query"): {
        "source": ["skactiveml/pool/_bald.py"],
        "tests": ["skactiveml/pool/tests/test_bald.py"],
        "examples": ["docs/examples/1-pool-classification/"],
    },
    ("emukit", "emukit-design"): {
        "source": [
            "emukit/experimental_design/",
            "emukit/bayesian_optimization/acquisitions/entropy_search.py",
            "emukit/multi_fidelity/",
        ],
        "tests": [
            "tests/emukit/experimental_design/",
            "tests/emukit/bayesian_optimization/test_entropy_search.py",
            "tests/emukit/multi_fidelity/",
        ],
        "examples": [
            "notebooks/Emukit-tutorial-experimental-design-introduction.ipynb"
        ],
    },
    ("kirstine", "kirstine-design"): {
        "source": ["src/designproblem.jl", "src/designmeasure.jl"],
        "tests": ["test/designproblem.jl", "test/designmeasure.jl"],
        "examples": ["docs/"],
    },
}


def _artifact_urls(repository: str, revision: str, paths: list[str]) -> list[str]:
    """Resolve reviewed repository paths without executing upstream code."""
    if not repository or not revision:
        return []
    return [
        f"{repository}/{'tree' if path.endswith('/') else 'blob'}/{revision}/{path}"
        for path in paths
    ]


def render() -> str:
    """Render one upstream evidence record for every external feature."""
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []
    for tool in registry["tools"]:
        if tool["scope"] != "external":
            continue
        for feature in tool["features"]:
            revision = TOOL_REVISIONS.get(tool["id"], "")
            paths = FEATURE_PATHS.get((tool["id"], feature["id"]), {})
            source_paths = paths.get("source", [])
            test_paths = paths.get("tests", [])
            example_paths = paths.get("examples", [])
            source_artifacts = _artifact_urls(
                tool["repository"], revision, source_paths
            )
            test_artifacts = _artifact_urls(tool["repository"], revision, test_paths)
            example_artifacts = _artifact_urls(
                tool["repository"], revision, example_paths
            )
            if feature["parity_state"] == "not-reproducible":
                state = "documentation-only"
            elif source_artifacts and test_artifacts:
                state = "source-tests-and-docs-reviewed"
            elif source_artifacts:
                state = "source-and-docs-reviewed"
            elif tool["repository"]:
                state = "repository-and-docs-reviewed"
                source_artifacts = [f"{tool['repository']}/tree/{revision}"]
            else:
                state = "documentation-only"
            missing = [
                name
                for name, values in (
                    ("source", source_artifacts),
                    ("tests", test_artifacts),
                    ("examples", example_artifacts),
                    ("schemas", []),
                )
                if not values
            ]
            records.append(
                {
                    "tool_id": tool["id"],
                    "feature_id": feature["id"],
                    "reviewed_version": tool["version"],
                    "reviewed_revision": revision or None,
                    "reviewed_on": registry["searched_on"],
                    "extraction_state": state,
                    "source_artifacts": source_artifacts,
                    "test_artifacts": test_artifacts,
                    "documentation_artifacts": feature["evidence"],
                    "example_artifacts": example_artifacts,
                    "schema_artifacts": [],
                    "limitations": (
                        f"No independently verified upstream {', '.join(missing)} "
                        "artifact was identified in this snapshot."
                        if missing
                        else "None recorded."
                    ),
                }
            )
    payload = {"schema_version": "1.0.0", "records": records}
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    """Write or verify the checked-in upstream evidence projection."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = render()
    if args.check:
        return 0 if OUTPUT.read_text(encoding="utf-8") == rendered else 1
    OUTPUT.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
