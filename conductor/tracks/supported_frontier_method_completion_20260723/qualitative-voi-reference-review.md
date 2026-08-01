# Qualitative value-of-information reference and boundary review

Date: 2026-08-01

Scope: issue #558 and governed delivery subissues #738–#742

## Review question

Can VOIAGE provide an executable, portable and auditable qualitative workflow
for prioritizing information questions without presenting ordinal judgements as
cardinal value of information?

## Primary references

- The GRADE Evidence-to-Decision framework makes decision criteria, panel
  judgements, research evidence and additional considerations explicit before
  a recommendation is formed: <https://www.bmj.com/content/353/bmj.i2016> and
  <https://www.bmj.com/content/353/bmj.i2089>.
- W3C PROV distinguishes entities, activities and responsible agents and
  supports interoperable provenance and revision relationships:
  <https://www.w3.org/TR/prov-o/>.
- NIST AI RMF 1.0 treats AI as a socio-technical risk-management concern and
  emphasizes governance, documentation, provenance, accountability and human
  roles: <https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf>.
- WCAG 2.2 provides testable, technology-independent accessibility criteria
  and the perceivable, operable, understandable and robust principles:
  <https://www.w3.org/TR/WCAG22/>.
- JSON Schema Draft 2020-12 supplies the portable validation vocabulary used
  for the assessment, result, audit-event and rendering contracts:
  <https://json-schema.org/draft/2020-12>.

## Contract inference

The references support a structured and transparent judgement workflow,
provenance, explicit human responsibility and accessible presentation. They do
not define a standardized numerical "qualitative VOI" estimand. The repository
therefore uses the phrase as an issue-facing workflow label and makes no claim
that ordinal classes have cardinal distance, probability, utility or monetary
meaning.

The narrow executable contract should:

1. preserve entered ordinal judgements and rationale rather than average or
   weight them;
2. derive recommendation and priority classes only through declared logical
   rules;
3. retain complete tie groups, dissent, conflicts, missingness and unverified
   states;
4. identify sources, actors, revisions, redactions and AI-assisted content;
5. require accountable human verification before a result can be complete;
6. provide deterministic JSON and accessible text rendering.

## Exclusions

The workflow is not MCDA, Delphi consensus, expert elicitation, evidence
grading, risk-of-bias assessment, EVPI, EVPPI, EVSI or an economic study-design
calculation. It must not silently infer consensus, rank criteria by hidden
weights, expose redacted content, or allow an AI-generated event to approve its
own output.

## Review outcome

PASS for an experimental portable contract and deterministic Python reference
implementation under the above boundaries. Practitioner usability,
accessibility conformance, privacy/ethics review, scientific naming approval,
polyglot execution, stable promotion, release and issue closure remain external
or later gates.
