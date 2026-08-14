# Plan: Add test for RandomForestMetamodel's unfitted `rmse` method

1. **Understand the gap**: The task mentions "Missing test for rmse" at `voiage/metamodels.py:604`, which is right where `RandomForestMetamodel` is defined. While there is a `test_random_forest_metamodel` that tests the `rmse` method on a *fitted* model, there is no corresponding `test_random_forest_metamodel_unfitted` test to verify that calling `rmse`, `predict`, and `score` raises a `RuntimeError` on an *unfitted* `RandomForestMetamodel` (like there is for `GAMMetamodel`). We can easily add a test to verify this. The error raised in `RandomForestMetamodel.rmse` is at line 653: `raise_runtime_error("The model has not been fitted yet.")`. Wait, the prompt says "Missing test for rmse", maybe it just needs the unfitted check. Actually let me re-read the prompt. "Missing test for rmse".
2. **Review existing unfitted tests**: Looking at `test_gam_metamodel_unfitted`, we can use the same pattern for `RandomForestMetamodel`. We will add a `test_random_forest_metamodel_unfitted` function below `test_random_forest_metamodel`.
3. **Implement**:
   - Add `test_random_forest_metamodel_unfitted(sample_data)` to `tests/test_metamodels.py`.
   - Skip if `SKLEARN_AVAILABLE` is false.
   - Create an unfitted `RandomForestMetamodel`.
   - Assert `RuntimeError` is raised with message "The model has not been fitted yet." when calling `predict(x)`, `score(x, y)`, and `rmse(x, y)`.
4. **Pre-commit checks**: Run `tox -e py312`, `uv run ruff check .`, `uv run ruff format .` (and use `pre_commit_instructions` tool).
5. **Verify**: Ensure the new tests pass and test coverage is satisfactory.
