# Historical dependency audit inputs

These inputs preserve the 29 August audit
`specs/submission-readiness/dependency-frontier-audit-20260829.json` without
rewriting its recorded 216 packages and 60 declared requirements. The manifest
binds exact bytes from the commit that introduced that audit. Both files were
unchanged at the pre-acceleration base `27488e81`.

The lock is shared with the existing reproduction-environment snapshot because
the bytes are identical; this does not establish the paper's release identity
or certify a new reproduction. The historical configuration is retained here.
Tests verify both digests before checking the historical counts.

Current dependency changes use the root configuration and lock, independently
validated by frozen-lock CI and the strict dependency-frontier command. An
optional extra must not alter this historical receipt.
