# Event-localized information value v1

This experimental Python-only contract evaluates a declared binary event under
perfect revelation and a symmetric imperfect binary channel. It also evaluates
Hazen's policy-relative expected-utility information density on an exact finite
probability-mass support:

`i(x) = f(x) [max_a g_a(x) - g_a*(x)]`.

Here `a*` is a declared baseline-optimal reference action. Complete baseline and
conditional ties remain in the result. The optional centered diagnostic
`j(x) = f(x) [max_a g_a(x) - V0]` is explicitly signed; it is not substituted
for the nonnegative policy-relative density. Both sums recover the coordinate
information value within the declared tolerance.
When that value is zero, v1 returns no mode or direction rather than inventing
a direction of concern from tied zero-density atoms.

The v1 objective is higher-is-better and fixed to `maximize`; raw cost/loss
minimization must first be transformed to a higher-is-better score. The
reference action must attain the true baseline maximum. `tie_tolerance` is
bounded to `[0, 1e-6]` and `integral_tolerance` to `(0, 1e-6]`. Raw atoms drive
the integral assurance and cleanup is presentation-only.

The event result reports exact event/complement probabilities, conditional
actions and values, gross/net perfect-event VOI, and an accuracy curve for a
symmetric binary channel. Accuracy `0.5` is uninformative, while accuracies
`p` and `1-p` have identical gross value because the signal labels can be
inverted. Plots consume the evaluated result only.
The symmetry error is `null` when the requested grid contains no `p`/`1-p`
pair, so missing assurance is never presented as an exact zero.

The finite mass contract is source-grounded in Hazen, Borgonovo and Lu (2023),
DOI `10.1287/deca.2022.0465`, and Bakır (2025), DOI
`10.1287/deca.2024.0172`. It does not claim a continuous-density estimator or
monetary buying-price information. BPI remains delegated to issue #595. Rust,
R and Julia are not implemented; Mojo is an external upstream boundary.
