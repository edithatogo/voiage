# Installed qualification admission receipts

`manifest.json.gz` retains 26 structured JSON receipts from the immutable hosted
runs listed inside the manifest. The uncompressed canonical JSON SHA-256 is
`e23056cf13e58f9a3058b3b972feca1b319bc5703c2b491db763d12d3486b22c`.
It includes source/replay admission, effective resource limits and cleanup, the
failed full-tox report, and both first EasyBuild failures. These are terminal
receipts for their stated scope; the parent Spack run was still active at capture.

The first EasyBuild failures are the UCX OS dependency
`('libibverbs-dev', 'libibverbs-devel', 'rdma-core-devel')`, before compilation.
The retry installs Ubuntu's `libibverbs-dev`; checksum, OS-dependency, network
isolation and resource gates remain enabled. No SDK binaries or installed-prefix
content is included. Source replay and root cleanup do not prove installed
acceptance. The handoff records active retry identities separately.
