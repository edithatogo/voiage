# SPDX-License-Identifier: Apache-2.0
"""Current Arrow source with the pinned upstream builder and resources."""

from pathlib import Path

from spack.package import EnvironmentModifications, depends_on, resource, version
from spack_repo.builtin.packages.arrow.package import Arrow as BuiltinArrow


class Arrow(BuiltinArrow):
    """Extend the qualified catalogue without replacing its build semantics."""

    version(
        "26.0.0",
        sha256="764adbb5601a4d0d9d67a10cf04892d33cfac9fd43012f08a2d7c9956a7a90d2",
        url="https://github.com/apache/arrow/archive/446167169a0dc547e00e1bc29f417a40c0ca267a.tar.gz",
    )
    depends_on("python@3.15.2", when="+python", type=("build", "run"))
    resource(
        name="mimalloc-3.4.1",
        url="https://github.com/microsoft/mimalloc/archive/v3.4.1.tar.gz",
        sha256="37107a52c16baa80c5f74861dddda7b27bb9949e41a6637691867a94c88ca446",
        expand=False,
        destination="spack-vendor",
        placement={"v3.4.1.tar.gz": "mimalloc-v3.4.1.tar.gz"},
        when="@=26.0.0+mimalloc",
    )
    resource(
        name="simdjson-4.6.4",
        url="https://github.com/simdjson/simdjson/archive/refs/tags/v4.6.4.tar.gz",
        sha256="b091107844fe928158c5c2265c20360fff312889ddf7ebc4528a0f0f8f2ff9cd",
        expand=False,
        destination="spack-vendor",
        placement={"v4.6.4.tar.gz": "simdjson-v4.6.4.tar.gz"},
        when="@=26.0.0",
    )
    resource(
        name="uriparser-1.0.2",
        url="https://github.com/uriparser/uriparser/releases/download/uriparser-1.0.2/uriparser-1.0.2.tar.bz2",
        sha256="dd2e4843f43de6f5aedf430e530f1e2d159eba8ca4d64c2787af1c20f707a9e4",
        expand=False,
        destination="spack-vendor",
        placement={"uriparser-1.0.2.tar.bz2": "uriparser-1.0.2.tar.bz2"},
        when="@=26.0.0",
    )

    def cmake_args(self):
        """Translate the shared variant to Arrow 26's actual CMake switches."""
        args = super().cmake_args()
        if self.spec.satisfies("@=26.0.0"):
            args.extend(
                [
                    self.define_from_variant("ARROW_BUILD_SHARED", "shared"),
                    self.define("ARROW_BUILD_STATIC", "ON"),
                    self.define("simdjson_SOURCE", "BUNDLED"),
                    self.define("uriparser_SOURCE", "BUNDLED"),
                ]
            )
            if self.spec.satisfies("+lz4"):
                args.append(self.define("LZ4_ROOT", self.spec["lz4"].prefix))
        return args

    def setup_build_environment(self, env: EnvironmentModifications) -> None:
        """Point Arrow at the local archives Spack stages before configure."""
        if not self.spec.satisfies("@=26.0.0"):
            return
        # Spack sets up the build environment before it stages resources.
        vendor = Path(self.stage.source_path) / "spack-vendor"
        env.set("ARROW_SIMDJSON_URL", (vendor / "simdjson-v4.6.4.tar.gz").as_uri())
        env.set("ARROW_URIPARSER_URL", (vendor / "uriparser-1.0.2.tar.bz2").as_uri())
        if self.spec.satisfies("+mimalloc"):
            env.set("ARROW_MIMALLOC_URL", (vendor / "mimalloc-v3.4.1.tar.gz").as_uri())
