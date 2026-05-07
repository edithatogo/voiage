# Docstring Audit

Baseline audit of the public method and plotting surfaces found no missing
module, class, or top-level function docstrings in `voiage/methods/` or
`voiage/plot/`.

Scope checked:

- `voiage/methods/**/*.py`
- `voiage/plot/**/*.py`
- `voiage/schema.py`
- `voiage/analysis.py`

Result:

- No missing public docstrings were found in the inspected surface.
- The remaining work is format standardization and content expansion where the
  docstrings are present but still need to be brought fully in line with the
  NumPy-style documentation target.
