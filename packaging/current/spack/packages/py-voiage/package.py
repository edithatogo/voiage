"""Current-runtime candidate recipe; native resolution/build remains gated."""

from spack.package import EnvironmentModifications, depends_on, version
from spack.package import license as spack_license
from spack.user_environment import environment_modifications_for_specs
from spack.util.environment import preserve_environment
from spack_repo.builtin.build_systems.python import PythonPackage


class PyVoiage(PythonPackage):
    """Build the immutable migrated source candidate, never the old release sdist."""

    homepage = "https://github.com/edithatogo/voiage"
    git = "https://github.com/edithatogo/voiage.git"
    spack_license("Apache-2.0")
    version("develop", commit="0a961dafd2e68a480c668b30600430d9ac71be38")
    depends_on("python@3.15.2", type=("build", "run"))
    depends_on("c", type="build")
    depends_on("rust@=1.99.1", type="build")
    depends_on("py-maturin@1.9:1", type="build")
    depends_on("py-click@8.3.3:8", type="run")
    depends_on("py-numpy@2.5.4:2", type="run")
    depends_on("py-scipy@1.18.1:1.18", type="run")
    depends_on("py-scikit-learn@1.9.1:1", type="run")
    depends_on("py-pyarrow@26.0.0:26", type="run")
    depends_on("py-polars@2.1.0:2", type="run")
    depends_on("py-pydantic@2.13.4:2", type="run")
    depends_on("py-jsonschema@4.26.0:4", type="run")
    depends_on("py-typing-extensions@4.16.0:", type="run")
    depends_on("py-typer@0.9:0", type="run")
    import_modules = ("voiage", "voiage._core")

    def setup_build_environment(self, env: EnvironmentModifications) -> None:
        """Set the qualified source identity for the Rust build."""
        # VCS mirrors can omit .git. Retain the qualified immutable source
        # identity; the Rust builder still rejects a conflicting Git identity.
        env.set("VOIAGE_SOURCE_REVISION", "0a961dafd2e68a480c668b30600430d9ac71be38")
        env.set("VOIAGE_SOURCE_TREE_GIT_OID", "6561ac4d928fa2bf75ff15ca19981c6e1b416018")
        env.set("VOIAGE_SOURCE_CLEAN", "true")

    def test_imports(self) -> None:
        """Run Spack's import test against the completed installation."""
        # The build environment omits the package's run-only dependencies.
        env = environment_modifications_for_specs(self.spec, set_package_py_globals=False)
        with preserve_environment(*env.group_by_name()):
            env.apply_modifications()
            super().test_imports()
