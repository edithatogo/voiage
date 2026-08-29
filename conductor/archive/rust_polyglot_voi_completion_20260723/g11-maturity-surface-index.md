# G11 maturity-surface index

This index is the documentation and capability-discovery projection for the
Rust/polyglot programme. It deliberately mirrors the evidence state rather
than implying that source packages or registry listings are installed parity.

| Surface | Canonical documentation/example | Advertised maturity |
|---|---|---|
| Python stable API | `docs/astro-site/src/content/docs/api-reference/methods.mdx`, `docs/astro-site/src/content/docs/user-guide/v1-release-readiness.mdx` | Stable Rust-backed contracts through the Python façade. |
| Rust core and C ABI | `docs/astro-site/src/content/docs/backends.mdx`, `docs/astro-site/src/content/docs/reference/c-abi.mdx` | Stable CPU authority; ABI symbols and layouts are versioned. |
| R binding | `r-package/voiageR/README.md`, `r-package/voiageR/vignettes/voiageR-getting-started.Rmd` | Narrow EVPI C-ABI consumer; installation and registry review remain external. |
| Julia binding | `bindings/julia/README.md`, `docs/astro-site/src/content/docs/reference/bindings.mdx` | Narrow EVPI C-ABI consumer; native library and registry evidence remain required. |
| Frontier methods | `docs/astro-site/src/content/docs/sota-voi-frontier.mdx`, `docs/astro-site/src/content/docs/api-reference/methods.mdx` | Experimental or fixture-backed unless the migration matrix says `verified`. |
| Mojo | `docs/astro-site/src/content/docs/backends.mdx`, `docs/astro-site/src/content/docs/reference/bindings.mdx` | External upstream boundary; no local executable or binding claim. |

## Discovery and example rules

1. Capability records and binding tables must use the statuses in
   `specs/rust/migration_matrix.json` and `specs/v1/binding-matrix.json`.
2. Examples may demonstrate Python checkout behaviour and documented narrow
   R/Julia source APIs, but must not present blocked installed runs as proof.
3. Experimental methods retain their maturity labels and unsupported language
   dispositions; no generated API surface may promote them implicitly.
4. Every future generated surface must link back to the versioned contract,
   fixture manifest and evidence packet before it is described as verified.

## Review result

The existing documentation and binding examples are consistent with the
current capability and runtime dispositions. No generated surface is changed
by G11; this index provides the auditable projection and prevents accidental
promotion while G10 installed evidence remains incomplete.
