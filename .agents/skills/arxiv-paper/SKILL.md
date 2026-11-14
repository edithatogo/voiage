---
name: arxiv-paper
description: Build and audit voiage's non-submitting arXiv source package.
---

# arXiv paper skill

Build with `uv run python scripts/build_arxiv_submission.py`. Inspect the canonical PDF, LaTeX
source archive, SHA-256 digest, semantic HTML, and readiness evidence. Generate
and inspect the review-only Textstat report from the canonical PDF, but do not
treat readability scores as scientific-quality thresholds. Never upload or
infer an arXiv identifier. Category, license, author, and final-upload decisions
remain human gates.

## Manuscript contract and scientific acceptance

Before revising a manuscript, update its versioned contract. Link the current
official journal and article-type rules, including panel counting, formatting
exceptions, declarations, submission files, and unknown portal requirements.
Classify the graphical abstract as required, optional, or unconfirmed using the
journal's rules. Record the question, estimands, and original and co-author
feedback dispositions.
Maintain separate inventories for analysis production, the main paper, supplements,
and workbooks. Preserve the protocol chronology and source provenance. Scale the
contract to the study design and record when causal or subgroup analysis is not
applicable.

For headline and material causal or subgroup claims, record the exact wording,
estimand and population, frozen evidence location, assumptions, diagnostics,
sensitivities, limitations, permitted wording, downgrade or withdrawal triggers,
and final manuscript locations. Keep null and conflicting results visible.
Compare designs by their estimands and populations.

When subgroup results are applicable, name display priorities before revising
prose and record the rationale and results already seen. Separate descriptive
comparisons from supported effect modification. Record analytic support and
privacy dispositions separately and retain all reportable results in the
supplements or workbooks. Do not select subgroups by significance.

For the exact edition, record named and dated scientific, methods, domain, and
reporting review as applicable, including unresolved objections and the acceptance
decision. Distinguish claim blockers from permissible limitations and define when
to stop, downgrade, or withdraw a claim, including conditional and no-go
outcomes. Require a question-driven amendment before expanding methods. Do not
invent independence or sign-off. Package checks, scientific acceptance, author
approval, and authority to submit are separate decisions.

## Manuscript and supplement DOCX delivery

Inventory every final manuscript and supplement, including clean, tracked,
review, and journal variants. After the final save, scrub, Compare, conversion,
or package edit, validate the explicit list of final files. The portable
repository validator requires the .NET 10 SDK and restores the locked Open XML
SDK and SharpZipLib packages. Both dependencies use the MIT licence. From the
repository root, use a new receipt path:

```bash
dotnet restore scripts/docx_validator/DocxGate.csproj --locked-mode
dotnet run --project scripts/docx_validator/DocxGate.csproj \
  --configuration Release --no-restore -- \
  --report "$docx_receipt" "${final_docx_paths[@]}"
```

Require zero errors in the ZIP/CRC, XML, internal relationship, and Microsoft
Open XML schema and semantic checks. Retain hash-bound diagnostics and tool
and runtime versions. Verify each final hash against its receipt before delivery.
Changed bytes invalidate the receipt. Missing tools, incomplete checks,
exceptions, and validation errors block delivery. Preserve failed evidence and
repair versioned copies within scope.

On hosts with the maintained shared `manuscript-docx-delivery` skill, its
qualified read-only runner may also be used. Qualify any other equivalent with
valid manuscript and supplement fixtures and negative package, relationship,
XML, schema, and invocation cases before relying on it.

Verify content and property fidelity, fields, comments, tracked revisions, and
repeating table headers independently. Review every rendered page and retain
required native Word evidence separately. Structural validation carries no
scientific, privacy, author, or submission approval.
