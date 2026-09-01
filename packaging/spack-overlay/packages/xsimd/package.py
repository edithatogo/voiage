# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack_repo.builtin.packages.xsimd.package import Xsimd as BuiltinXsimd

from spack.package import depends_on, version


class Xsimd(BuiltinXsimd):
    """Checksum-bound xsimd required by the selected Arrow 25 source."""

    depends_on("c", type="build")

    version(
        "14.2.0",
        sha256="21e841ab684b05331e81e7f782431753a029ef7b7d9d6d3ddab837e7782a40ee",
        url="https://github.com/xtensor-stack/xsimd/archive/14.2.0.tar.gz",
    )
