# Dependency frontier

Python: `>=3.15.1,<3.16`
Strict dependency policy: **pass**

| Scope | Package | Minimum supported | Locked | Latest compatible | Latest overall | Upper-bound preview | Policy |
|---|---|---:|---:|---:|---:|:---:|:---:|
| `core` | `click` | `8.3.3` | `['8.5.0']` | `8.5.0` | `8.5.0` | no | pass |
| `core` | `numpy` | `2.5.4` | `['2.5.4']` | `2.5.4` | `2.5.4` | no | pass |
| `core` | `scipy` | `1.18.1` | `['1.18.1']` | `1.18.1` | `1.18.1` | no | pass |
| `core` | `pandas` | `3.1.0` | `['3.1.0']` | `3.1.0` | `3.1.0` | no | pass |
| `core` | `xarray` | `2026.9.0` | `['2026.9.0']` | `2026.9.0` | `2026.9.0` | no | pass |
| `core` | `scikit-learn` | `1.9.1` | `['1.9.1']` | `1.9.1` | `1.9.1` | no | pass |
| `core` | `pyarrow` | `26.0.0` | `['26.0.0']` | `26.0.0` | `26.0.0` | no | pass |
| `core` | `polars` | `2.1.0` | `['2.1.0']` | `2.1.0` | `2.1.0` | no | pass |
| `core` | `pydantic` | `2.13.4` | `['2.14.1']` | `2.14.1` | `2.14.1` | no | pass |
| `core` | `jsonschema` | `4.26.0` | `['4.26.0']` | `4.26.0` | `4.26.0` | no | pass |
| `core` | `typing_extensions` | `4.16.0` | `['4.16.0']` | `4.16.0` | `4.16.0` | no | pass |
| `core` | `typer` | `0.9` | `['0.27.3']` | `0.27.3` | `0.27.3` | no | pass |
| `optional:test-acceleration` | `pytest-testmon` | `2.2.0` | `['2.2.0']` | `2.2.0` | `2.2.0` | no | pass |
| `optional:test-acceleration` | `pytest-gremlins` | `1.9.0` | `['1.12.2']` | `1.12.2` | `1.12.2` | no | pass |
| `optional:test-acceleration` | `coverage` | `7.16.0` | `['7.17.0']` | `7.17.0` | `7.17.0` | no | pass |
| `optional:test-acceleration` | `vcrpy` | `8.3.0` | `['8.5.0']` | `8.5.0` | `9.2.0` | yes | pass |
| `optional:ecosystem` | `defusedxml` | `0.7.1` | `['0.7.1']` | `0.7.1` | `0.7.1` | no | pass |
| `optional:excel` | `openpyxl` | `3.1.5` | `['3.1.5']` | `3.1.5` | `3.1.5` | no | pass |
| `optional:docs` | `griffe` | `1.14.0` | `['1.15.0']` | `1.15.0` | `2.4.1` | yes | pass |
| `optional:jax` | `jax` | `0.12.0.1` | `['0.12.0.1']` | `0.12.0.1` | `0.12.0.1` | no | pass |
| `optional:ci` | `defusedxml` | `0.7.1` | `['0.7.1']` | `0.7.1` | `0.7.1` | no | pass |
| `optional:ci` | `docutils` | `0.23` | `['0.23']` | `0.23` | `0.23` | no | pass |
| `optional:ci` | `fastapi` | `0.139.2` | `['0.162.0']` | `0.162.0` | `0.162.0` | no | pass |
| `optional:ci` | `httpx` | `0.28.1` | `['0.28.1']` | `0.28.1` | `0.28.1` | no | pass |
| `optional:ci` | `hypothesis` | `6.157.0` | `['6.170.12']` | `6.170.12` | `6.170.12` | no | pass |
| `optional:ci` | `jax` | `0.12.0.1` | `['0.12.0.1']` | `0.12.0.1` | `0.12.0.1` | no | pass |
| `optional:ci` | `matplotlib` | `3.4` | `['3.11.2']` | `3.11.2` | `3.11.2` | no | pass |
| `optional:ci` | `maturin` | `1.9` | `['1.15.0']` | `1.15.0` | `1.15.0` | no | pass |
| `optional:ci` | `psutil` | `5.9` | `['7.2.2']` | `7.2.2` | `7.2.2` | no | pass |
| `optional:ci` | `pytest` | `9.1.1` | `['9.1.1']` | `9.1.1` | `9.1.1` | no | pass |
| `optional:ci` | `pytest-benchmark` | `5.2.3` | `['5.3.0']` | `5.3.0` | `5.3.0` | no | pass |
| `optional:ci` | `pytest-cov` | `7.1.0` | `['7.1.0']` | `7.1.0` | `7.1.0` | no | pass |
| `optional:ci` | `pytest-integration` | `0.2.3` | `['0.2.3']` | `0.2.3` | `0.2.3` | no | pass |
| `optional:ci` | `pytest-xdist` | `3.8.0` | `['3.8.0']` | `3.8.0` | `3.8.0` | no | pass |
| `optional:ci` | `PyYAML` | `6.0.3` | `['6.0.3']` | `6.0.3` | `6.0.3` | no | pass |
| `optional:ci` | `rfc8785` | `0.1` | `['0.1.4']` | `0.1.4` | `0.1.4` | no | pass |
| `optional:ci` | `tomli` | `2.4.1` | `['2.5.0']` | `2.5.0` | `2.5.0` | no | pass |
| `optional:dev` | `maturin` | `1.9` | `['1.15.0']` | `1.15.0` | `1.15.0` | no | pass |
| `optional:dev` | `mutmut` | `3.6.0` | `['3.6.0']` | `3.6.0` | `3.8.0` | yes | pass |
| `optional:dev` | `scalene` | `2.3.0` | `['2.4.0']` | `2.4.0` | `2.4.0` | no | pass |
| `optional:dev` | `ruff` | `0.17.2` | `['0.17.3']` | `0.17.3` | `0.17.3` | no | pass |
| `optional:dev` | `bandit` | `1.9.4` | `['1.9.4']` | `1.9.4` | `1.9.4` | no | pass |
| `optional:dev` | `nox` | `2026.7.11` | `['2026.10.20']` | `2026.10.20` | `2026.10.20` | no | pass |
| `optional:dev` | `ty` | `0.0.61` | `['0.0.92']` | `0.0.92` | `0.0.92` | no | pass |
| `optional:dev` | `basedpyright` | `1.39.9` | `['1.40.2']` | `1.40.2` | `1.40.2` | no | pass |
| `optional:dev` | `types-setuptools` | `83.0.0.20260716` | `['84.0.0.20261006']` | `84.0.0.20261006` | `84.0.0.20261006` | no | pass |
| `optional:dev` | `types-requests` | `2.33.0.20260712` | `['2.33.0.20261006']` | `2.33.0.20261006` | `2.33.0.20261006` | no | pass |
| `optional:dev` | `jupyterlab` | `4.6.2` | `['4.6.4']` | `4.6.4` | `4.6.4` | no | pass |
| `optional:dev` | `ipykernel` | `7.3.0` | `['7.4.0']` | `7.4.0` | `7.4.0` | no | pass |
| `optional:dev` | `pre-commit` | `4.6.0` | `['4.7.0']` | `4.7.0` | `4.7.0` | no | pass |
| `optional:dev` | `vulture` | `2.16` | `['2.16']` | `2.16` | `2.16` | no | pass |
| `optional:dev` | `tox` | `4.67` | `['4.67.1']` | `4.67.1` | `4.67.1` | no | pass |
| `optional:dev` | `tox-uv` | `1.37` | `['1.37.0']` | `1.37.0` | `1.37.0` | no | pass |
| `optional:dev` | `tomli` | `2.4.1` | `['2.5.0']` | `2.5.0` | `2.5.0` | no | pass |
| `optional:experimental` | `duckdb` | `1.5.4` | `['1.5.6']` | `1.5.6` | `2.0.0` | yes | pass |
| `optional:experimental` | `orjson` | `3.11.9` | `['3.13.0']` | `3.13.0` | `3.13.0` | no | pass |
| `optional:experimental` | `msgspec` | `0.21.1` | `['0.22.0']` | `0.22.0` | `0.22.0` | no | pass |
| `optional:plotting` | `matplotlib` | `3.11.1` | `['3.11.2']` | `3.11.2` | `3.11.2` | no | pass |
| `optional:plotting` | `seaborn` | `0.13.2` | `['0.13.2']` | `0.13.2` | `0.13.2` | no | pass |
| `optional:performance` | `psutil` | `5.9` | `['7.2.2']` | `7.2.2` | `7.2.2` | no | pass |
| `optional:web` | `fastapi` | `0.139.2` | `['0.162.0']` | `0.162.0` | `0.162.0` | no | pass |
| `optional:web` | `httpx` | `0.28.1` | `['0.28.1']` | `0.28.1` | `0.28.1` | no | pass |
| `optional:web` | `uvicorn` | `0.51.0` | `['0.54.0']` | `0.54.0` | `0.54.0` | no | pass |
| `build` | `maturin` | `1.9` | `['1.15.0']` | `1.15.0` | `1.15.0` | no | pass |
