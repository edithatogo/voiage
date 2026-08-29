# G1–G3 programme reconciliation and language dispositions

The root programme is GitHub issue #313, represented in Project 28 and linked
to native workstreams #314–#323 plus historical #416. The central manifest,
registry, metadata and plan use the same track identifier; PR #826 is governance
delivery evidence only. Child issue state is not runtime or release evidence.

## Frozen contract and dispositions

| Surface | Disposition | Boundary |
|---|---|---|
| Rust core | authoritative for stable numerical kernels | Versioned Rust contracts and ABI fixtures are required. |
| Python | supported façade/orchestration | Must preserve stable v1 shapes; Python-only experimental paths stay labelled. |
| R | supported native EVPI façade | Clean installed-package and registry evidence remain separate gates. |
| Julia | supported native EVPI façade | JLL/General registration remains external; unsupported methods stay explicit. |
| Mojo | external/upstream integration boundary | No repository parity claim without an approved toolchain and fixture run. |

Stable v1, supported extensions and experimental v1.3 work remain distinct
maturity lanes. Cross-language parity means installed bindings consume the same
versioned fixtures and agree within declared tolerances; source inspection or a
single local build is insufficient. Registry publication, hardware speedup,
scientific review, release signing and external acceptance remain independent.
