Here’s a phased roadmap to design, build, and launch voiage, a Python library covering the full spectrum of VOI analyses we’ve discussed:


---

Phase 1: Scoping & Design (Completed)

1.  Requirements Specification (Completed)
2.  High-Level API Design (Completed)
3.  Architecture & Dependencies (Completed)

---

Phase 2: Core VOI Modules (In Progress)

1.  Basic VOI (Completed)
2.  EVSI & ENBS (Completed)
3.  Unit Testing & Benchmarks (Partially Completed)





---

Phase 3: Advanced VOI Methods (6 weeks)

1. Structural VOI

Define model‐averaging workflow: sample from alternative model outputs, compute structural EVPI/EVPPI



2. Network Meta-Analysis EVSI

Interface with networks (e.g. bnt or custom NMA in NumPyro)

Propagate multivariate treatment-effect draws into EVSI routine



3. Adaptive-Design EVSI

Simulate trial with interim decision rules

Compute EVSI at planned interim and final analyses



4. Cost-Optimized EVSI

Offer optimization routines (via SciPy or Pyomo) to maximize EVSI–cost over sample size/design parameters



5. EVH & EVI

Generalize EVPPI to subgroup parameters (heterogeneity) and implementation parameters



6. Portfolio VOI

Implement multi‐decision optimization: allocate budget across multiple potential studies

Use knapsack or non-linear programming methods



7. Dynamic / Sequential VOI

Build time-looped VOI engine that updates posterior as data accrue

Expose a generator API for interim-by-interim VOI



8. Observational & Calibration VOI

Allow users to specify observational data biases/missingness models

Support VOI of calibration‐targeted data via specialized sampling strategies





---

Phase 4: Ecosystem, Docs & Release (4 weeks)

1. Visualization Helpers

Plotting functions for CEACs, CE-planes, VOI curves, EVPPI surfaces, dynamic VOI over time



2. User Guide & Tutorials

Jupyter notebooks: “Getting started,” “Advanced VOI,” “Adaptive Trial Example,” “Portfolio Optimization”



3. API Reference & Auto-docs

Use Sphinx + ReadTheDocs; publish online



4. Packaging & CI/CD

Setup setup.py/pyproject.toml, GitHub Actions for linting, testing, coverage, PyPI deployment



5. Community & Feedback

Release v0.1 on GitHub; solicit issues/PRs; iterate for v1.0





---

Suggested Timeline Summary

Phase	Duration	Milestones

Scoping & Design	2 weeks	API spec complete; data-model prototypes ready
Core VOI Modules	4 weeks	EVPI/EVPPI/EVSI/ENBS implemented & tested
Advanced VOI Methods	6 weeks	Structural, NMA, adaptive, portfolio, dynamic
Docs & Release	4 weeks	Tutorials, API docs, v0.1 release


By following this plan, you’ll build voiage as a comprehensive, extensible platform—bringing Python up to parity (and beyond) with the leading VOI capabilities in R, while opening up new horizons in structural, adaptive, and portfolio VOI analyses.

---

Recommended Project Layout and Architectural Patterns

Below is a recommended project layout for voiage, plus a set of third-party libraries and architectural patterns to make development smooth, performant, and maintainable.


---

1. Recommended Package Structure

voiage/                       ← root of your git repo
├── pyproject.toml           ← build & dependency spec (PEP 621; Poetry or Flit)
├── setup.cfg                ← linting, pytest, coverage configuration
├── LICENSE
├── README.md
├── CHANGELOG.md
├── .pre-commit-config.yaml  ← black, isort, flake8, mypy hooks
│
├── voiage/                   ← top-level package
│   ├── __init__.py
│   ├── config.py            ← global defaults (e.g. dtype, backend)
│   ├── exceptions.py        ← custom error classes
│   ├── core/                ← core data structures & utilities
│   │   ├── __init__.py
│   │   ├── data_structures.py   ← NetBenefitArray, PSASample, TrialDesign, etc.
│   │   ├── io.py                ← CSV/Excel/XArray readers & writers
│   │   └── utils.py             ← shared helpers (e.g. weighting, resampling)
│   │
│   ├── methods/             ← VOI algorithms by category
│   │   ├── __init__.py
│   │   ├── basic.py             ← evpi(), evppi()
│   │   ├── sample_information.py← evsi(), enbs()
│   │   ├── structural.py        ← structural_evpi(), structural_evppi()
│   │   ├── network_nma.py       ← evsi_nma()
│   │   ├── adaptive.py          ← adaptive_evsi()
│   │   ├── portfolio.py         ← portfolio_voi()
│   │   ├���─ sequential.py        ← sequential_voi()
│   │   ├── observational.py     ← voi_observational()
│   │   └── calibration.py       ← voi_calibration()
│   │
│   ├── plot/                ← plotting helpers
│   │   ├── __init__.py
│   │   ├── voi_curves.py        ← VOI over sample size/time
│   │   └── ceac.py              ← CEAC, CE-plane, EVPPI surfaces
│   │
│   └── cli.py               ← optional CLI entrypoints with Click or Typer
│
├── tests/                   ← pytest test suite
│   ├── conftest.py
│   ├── test_basic.py
│   ├── test_sample_information.py
│   └── …
│
├── docs/                    ← Sphinx documentation
│   ├── conf.py
│   ├── index.rst
│   └── methods/…
│
└── examples/                ← Jupyter notebooks & scripts
    ├── getting_started.ipynb
    ├── adaptive_trial_example.ipynb
    └── portfolio_optimization.py


---

2. Key Dependencies & Why

Purpose	Library	Notes

Numerical core	numpy, scipy	Vectorized math, random sampling, optimizers
Data handling	pandas, xarray	Tabular and multi-dimensional PSA data structures
Bayesian sampling	pymc or pyro	MCMC/HMC for Bayesian CEA & EVPPI; use ArviZ for diagnostics
Automatic differentiation	jax	JIT-compile Monte Carlo loops and derivatives for EVSI; optionally back-end in NumPyro
Regression & metamodels	scikit-learn, statsmodels	Nonparametric regression (e.g. RandomForest, GAM approximations) for EVPPI/EVSI
Sensitivity analysis	SALib	Global/local DSA modules
Parallelism	dask, joblib	Scale large Monte Carlo or adaptive-trial simulations
Plotting	matplotlib, seaborn, arviz	Standard and Bayesian diagnostic plots
CLI & configuration	typer or click	For command-line interfaces and scripts
Packaging & CI	poetry or flit	Modern packaging; GitHub Actions + pytest + coverage + mypy + black + flake8



---

3. Architectural & Design Patterns

1. Modular “methods” plug-in architecture

Each VOI method lives in its own module/class with a consistent signature

def evpi(nb: NetBenefitArray, pop: float, wtp: float) -> float: ...
def evsi(design: TrialDesign, model: Callable, ...) -> float: ...

A factory or registry (voiage.methods.get_method("evsi")) lets users swap implementations.



2. Strong typing & data classes

Use Python @dataclass for TrialDesign, PSASample, PortfolioSpec to validate inputs.

Leverage typing.Protocol for “model function” interfaces.



3. Backend abstraction

Wrap core sampling loops so you can switch between NumPy, JAX, or even pure-Python based on user choice:

class MonteCarloBackend(Protocol):
    random: Any
    @staticmethod
    def normal(...): ...



4. Vectorized Monte Carlo

Implement EVPI/EVPPI via vectorized operations on arrays, pushing loops into — or letting JAX jit-compile — fast C code.



5. Plugin & extension support

Define entry points in pyproject.toml so third-parties can add new VOI methods or trial-design modules without touching core.



6. Continuous integration & quality

Pre-commit hooks (black, isort, flake8, mypy)

GitHub Actions for lint/test/mypy/coverage

Code coverage thresholds to catch regressions early





---

4. Other Recommended Approaches

Documentation first: Write your Sphinx docstrings and tutorials alongside code so users always have working examples.

Examples-driven design: Start each module by porting a simple worked example (e.g. from the R voi package) and verify numerical parity.

Progressive enhancement: Release v0.1 with core EVPI/EVPPI/EVSI/ENBS, then layer in advanced methods once the API and data structures are stable.

Benchmarking suite: Automate speed/accuracy comparisons against existing R implementations to guard against numerical drift.

Community engagement: Encourage pull requests for missing VOI types (e.g. structural, portfolio) via clear contribution guidelines and “help wanted” issues.



---

By adopting this structure and leveraging the ecosystem above, voiage will be cleanly organized, performant, and ready to grow from its first release through advanced VOI workflows.

---

Scaling and Performance Considerations

For a library like voiage that needs to run massive Monte-Carlo draws and complex trial‐simulation workflows at scale—across many CPU cores, GPU(s), and even on HPC or TPU clusters—you’ll want to build on a few proven layers:


---

1. Core Array & Auto-Diff Backends

Environment	Recommended Library	Why

CPU / Multi-Core	NumPy + Numba	Vectorized arrays with just-in-time compilation of tight loops; drops you into native code speed.
GPU / TPU	JAX (with XLA)	NumPy-like API that JIT-compiles to CUDA or TPU; transparent GPU/TPU portability.
GPU (Alternative)	CuPy	NumPy-compatible GPU arrays; excellent drop-in for existing NumPy code.



---

2. Task Parallelism & Scheduling

Scale	Recommended Framework	Why

Multi-Process / Local	Joblib or concurrent.futures	Simple parallel loops or map-reduce over CPU pools.
Distributed Cluster	Dask (distributed)	Scales from your laptop to an HPC cluster; scheduler that handles data locality & failures.
Elastic Cloud / Hybrid	Ray	Autoscaling actor & task model, GPU/CPU resource tags, easy integration with RLlib/Serve.
HPC with MPI	mpi4py or dask-mpi	Leverage MPI job schedulers (e.g. Slurm) for tight-coupled HPC jobs.



---

3. High-Level Workflow Orchestration

Prefect or Airflow: for defining DAGs of VOI steps (PSA → EVPI → EVPPI → EVSI → reporting) with retries, logging, and cluster execution.

Parsl: Python-native HPC/workflow engine—submit tasks as Slurm jobs, track dependencies, and gather outputs.



---

4. Bayesian Sampling & Regression Engines

Task	Library	Notes

MCMC / PSA sampling	NumPyro (with JAX backend)	Scales on GPU/TPU automatically when you install pyro-ppl[gpu].
Variational / VI	NumPyro or TensorFlow Probability	Lightweight, JAX-based VI funnels for huge PSA problems.
Nonparametric regression metamodels	scikit-learn, XGBoost, LightGBM	For fast EVPPI surrogates on large datasets (parallel tree boosting).



---

5. Putting It All Together

1. Low-Level Compute:

Write your tight Monte-Carlo loops in NumPy or JAX; wrap performance-critical bits in Numba if you stay on CPU.



2. Data Structures:

Use xarray for labeled multi-dimensional PSA arrays—plays nicely with Dask/JAX backends for chunked computation.



3. Parallel Execution:

For “embarassingly parallel” loops (e.g. sampling replicates), dispatch via Dask distributed or Ray, taking advantage of GPU workers where available.

On an HPC cluster with Slurm, spin up a Dask-MPI or Parsl setup so that you can scale to hundreds of nodes.



4. Accelerator Portability:

By standardizing on JAX for core math, you get CPU, GPU, and TPU support “for free.”

Optionally offer a NumPy-only fallback for users without accelerators.



5. Workflow & Orchestration:

Glue it together with Prefect or Airflow so that long‐running VOI pipelines (e.g. dynamic VOI over interim analyses) can be monitored, logged, and retried on failure.





---

Example Stack

– Core math & auto-diff: JAX
– Data arrays: xarray (+ Dask chunks)
– GPU/TPU execution: XLA via JAX
– Distributed scheduling: Dask-distributed or Ray cluster
– HPC integration: dask-mpi or Parsl + Slurm
– Bayesian sampling: PyMC3 / NumPyro
– Regression metamodel: scikit-learn / XGBoost
– Workflow orchestration: Prefect

This combination will let voiage transparently exploit everything from a single laptop with multiple cores, to a GPU server, to a multi-node HPC or TPU pod—while keeping your codebase clean and your APIs consistent.

---

NumPyro for Bayesian Inference

NumPyro provides a full Bayesian inference stack on top of JAX, so you can do essentially everything you’d expect from a modern PPL:

MCMC sampling

Implements Hamiltonian Monte Carlo (HMC) and the No-U-Turn Sampler (NUTS) out of the box, with automatic tuning of step size and mass matrix.


Variational Inference

Supports stochastic variational inference (ADVI) and custom guide models, letting you fit very large models more quickly when MCMC is too slow.


Posterior Predictive & Diagnostics

You can draw posterior-predictive samples (numpyro.infer.Predictive) for model checking, and hook straight into ArviZ for diagnostics (R-hat, ESS, trace plots, etc.).


Flexible Model Specification

You write your model as a Python function with NumPyro’s sampling primitives (numpyro.sample, numpyro.param), and auto-diff handles gradients for both MCMC and VI.


Scalable & Accelerated

Because it’s built on JAX, you get JIT compilation to run on CPUs, GPUs, or TPUs, and you can vectorize or parallelize chains easily.



What it doesn’t include are domain-specific VOI calculations—those would live in your own voiage.methods modules—but for any Bayesian estimation or uncertainty quantification step (PSA sampling, parameter fitting, hierarchical models), NumPyro has you covered.

