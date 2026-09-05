# Pre-Polars-runtime source URL evidence

This directory preserves the solver manifest, receipt, and exact concrete graph
from `main` at `9307d9ec` before the Polars runtime recipe bound its source to
the exact PyPI sdist URL. The graph concretized successfully, but the later
native installation failed because Spack derived a source filename without the
`_32` suffix. These files remain historical evidence and do not establish a
successful native installation.
