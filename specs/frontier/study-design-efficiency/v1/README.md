# Study-design efficiency contract v1

The experimental COSS contract preserves the signed ENBS curve and the best
evaluated design while reporting the no-study comparator separately. A best
evaluated design is a sampling recommendation only when its ENBS exceeds the
declared no-study ENBS beyond tolerance. Results state whether the supplied
designs are a complete feasible enumeration or only an evaluated subset; a
boundary optimum in an evaluated subset triggers an expansion/sensitivity
diagnostic rather than an unqualified global optimum claim.

The portable request/result schemas are generated from the authoritative
Pydantic models by `scripts/export_study_design_efficiency_contracts.py`.
`capabilities.json` preserves honest Rust/Python execution and unsupported
R/Julia states. It deliberately records installed-wheel and joint-replicate
uncertainty assurance as pending rather than promoting fixture-backed claims.
