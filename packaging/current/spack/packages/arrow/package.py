# SPDX-License-Identifier: Apache-2.0
"""Current Arrow source with the pinned upstream builder and allocator."""

from pathlib import Path

from spack_repo.builtin.packages.arrow.package import Arrow as BuiltinArrow
from spack.package import EnvironmentModifications, depends_on, resource, version


class Arrow(BuiltinArrow):
    """Extend the qualified catalogue without replacing its build semantics."""

    version(
        "26.0.0",
        sha256="764adbb5601a4d0d9d67a10cf04892d33cfac9fd43012f08a2d7c9956a7a90d2",
        url="https://github.com/apache/arrow/archive/446167169a0dc547e00e1bc29f417a40c0ca267a.tar.gz",
    )
    depends_on("python@3.15.1", when="+python", type=("build", "run"))
    resource(
        name="mimalloc-3.4.1",
        url="https://github.com/microsoft/mimalloc/archive/v3.4.1.tar.gz",
        sha256="37107a52c16baa80c5f74861dddda7b27bb9949e41a6637691867a94c88ca446",
        expand=False,
        destination="spack-vendor",
        placement={"v3.4.1.tar.gz": "mimalloc-v3.4.1.tar.gz"},
        when="@=26.0.0+mimalloc",
    )

    def setup_build_environment(self, env: EnvironmentModifications) -> None:
        """Use the staged allocator archive without changing upstream defaults."""
        # The pinned upstream package has no package environment hook.
        if self.spec.satisfies("@=26.0.0+mimalloc"):
            archive = (
                Path(self.stage.source_path) / "spack-vendor/mimalloc-v3.4.1.tar.gz"
            )
            if not archive.is_file():
                raise RuntimeError("The checksum-bound mimalloc source was not staged")
            env.set("ARROW_MIMALLOC_URL", archive.as_uri())
