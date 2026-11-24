# Installed tobac M2 retail EVPI, 24 November 2026

The opt-in Voiage backend passed the frozen `source_shape_partial_replacement`
retail scenario at crime weight 1.0. The decision uses 20,480 equally weighted
states, four actions and a 50% lawful replacement assumption. The consumer built
from signed tobac main `fe06ef5958ba0f218317a738fe8de95973300a12` called the
published Voiage 3.1.0 wheel on macOS arm64 with CPython 3.15.2. Both wheels were
installed into a fresh environment and run from outside both source trees.
[Consumer source](https://github.com/edithatogo/tobac/commit/fe06ef5958ba0f218317a738fe8de95973300a12) · [Provider release](https://github.com/edithatogo/voiage/releases/tag/v3.1.0)

| EVPI calculation | Normalised decision units |
| --- | ---: |
| Frozen M2 result | 0.06181538352544022 |
| Installed tobac reference | 0.06181538352544015 |
| Installed Voiage through tobac | 0.06181538352543689 |
| Installed Voiage with reversed state order | 0.06181538352543689 |

The installed result differed from the frozen result by `3.33e-15` at the
existing `rtol=1e-10` and `atol=1e-10`. The input digest stayed the same after
execution. The opt-in M2 route rejected a nonuniform prior, and the separate
version-checked adapter rejected a mismatched producer version. The existing
[frozen case test](https://github.com/edithatogo/tobac/blob/fe06ef5958ba0f218317a738fe8de95973300a12/tests/engineering/test_voiage_installed.py)
passed independently. The [receipt](./receipt.json) records both wheel and
module hashes, the scenario inputs, the exact result, and exit status.

## Reproduce

1. Export the pinned tobac Git revision into a scratch source directory. Use
   `uv build --wheel --python` with the qualified Python 3.15.2 executable and
   setuptools 84.2.1 bound to the hash in `engineering/requirements.lock`. Set
   `SOURCE_DATE_EPOCH` to 1795358010 for the observed build.
2. Verify the published Mac arm64 Voiage wheel against the SHA-256 in the receipt.
   Install both local wheel files in a fresh environment with the hashes and pins
   from `engineering/voiage-published-requirements.txt`, replacing its Voiage
   URL with the verified wheel's file URL. Use the pytest 9.2.0 pins and hashes
   from `engineering/requirements.lock` and run `uv pip check`.
3. From a separate working directory, run the signed source's frozen case
   test using its absolute path. `SRC` points to the exported source and `PY`
   points to the new environment's Python. Use an empty pytest configuration
   to keep source paths out of imports.

   ```sh
   printf '[pytest]\n' > pytest.ini
   "$PY" -I -m pytest -c pytest.ini --noconftest --import-mode=importlib -q \
     "$SRC/tests/engineering/test_voiage_installed.py::test_installed_release_matches_frozen_m2_retail_evpi"
   ```

4. Use the consumer's
   [wheel-member verifier](https://github.com/edithatogo/tobac/blob/fe06ef5958ba0f218317a738fe8de95973300a12/scripts/qualify_voiage_consumer.py)
   to compare each installed artifact with its wheel. Imports resolve from the
   fresh environment, including the released Voiage native extension.

This qualifies the named, source-informed structural scenario and its opt-in
finite EVPI route. The reverse-order and input checks contribute to R11.2, the
installed reference comparison contributes to R11.3, and the artifact and
scenario hashes contribute to R11.5. The R11 parameter round trip and migration,
multi-scenario extension, and broader weighted, structure-only and shared-source
portfolio requirements remain in the parent plan. Empirical acceptance remains
under the consumer's [research gates](https://github.com/edithatogo/tobac/blob/fe06ef5958ba0f218317a738fe8de95973300a12/conductor/research-gates.md).
