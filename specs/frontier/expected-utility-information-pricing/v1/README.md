# Expected-utility information pricing v1

Experimental Rust-authoritative contract for EUI, CEI, BPI, SPI, anchored PPI,
and the VoC presentation. Inputs are finite terminal-payoff decisions with a
named utility and either clairvoyant or finite-signal joint probabilities.
VoC is a presentation of the canonical result and is not a second kernel.
Power/CRRA utility includes its logarithmic limit at risk aversion one and uses
stable `expm1`/`log1p` evaluation and inversion near that limit. Every result
binds the presentation contract version, selected measure and canonical input
digest into a deterministic presentation digest; changing the VoC display
selection changes only this presentation provenance, never the Rust kernel.

The request and result schemas reject unknown fields. Normative fixtures freeze
independently calculated affine and nonlinear references before implementation.
They do not promote this family into the stable v1 API or imply binding parity.

`capabilities.json` is the machine-readable language boundary: Rust is the
execution authority, Python is executable through PyO3 and the public facade,
R and Julia are unsupported, and Mojo is an external boundary. The shared
frontier registry discovers this family's bundled request/result fixtures and
verifies each recorded SHA-256 digest. Scientific review, cross-language parity
and explicit stable-promotion approval remain pending.
