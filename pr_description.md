🧪 [testing improvement] Add test for create_metamodel_config factory

🎯 **What:**
Added missing unit tests for the `create_metamodel_config` factory function in `voiage/config_objects.py`.

📊 **Coverage:**
- Tests the default behavior (method="gam") and ensures a `MetamodelConfig` is correctly instantiated.
- Tests creating a configuration with a custom valid method (e.g., method="rf").
- Tests validation logic to ensure a `ValueError` is raised when providing an invalid method name.

✨ **Result:**
Increased test coverage and confidence in the configuration generation code, preventing potential regressions around default arguments and input validation.
