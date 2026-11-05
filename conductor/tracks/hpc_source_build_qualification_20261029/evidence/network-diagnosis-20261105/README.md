# Native timeout and network admission evidence

`manifest.json.gz` retains eleven structured receipts. Its uncompressed canonical
JSON SHA-256 is `26408e1cd192723f6a7ebd4fbf800a649210f12f041e4f29d14d9c951d360262`.

All three installed cells stopped at the four-hour stage limit with successful
cleanup. Neither memory nor filesystem reserve triggered the stops. Source graph,
source replay and initial cleanup controls had passed; installed acceptance did not.

The new local-TCP regression fails against the original namespace command.
Colab's initial privileged cleanup control passes, but enabling loopback without
a private user namespace returns EPERM. A private user/network namespace can enable
loopback; exact root/runner UID mappings are then denied by newuidmap policy.
No mapping grants were changed. The two failed new-control tests and ten passing
controls remain recorded. Colab is not an accepted installed-build backend.

The helper's standard Linux/root configuration requires a new hosted control
receipt. It changes only isolated loopback, denies the parent network and any
non-loopback interface, then clears supplementary groups before dropping GID/UID.
No dependency test, external-denial check or resource gate is suppressed.
Private machine connection details are excluded from this public evidence.
