# GRAIN vector observer specification

2026-09-06. **DRAFT_RESEARCH_CONTRACT**. No new public scorer/API is released.
This specifies reproducible model observations, not a completed psychological
instrument. Acceptance requires the pending full-source/bridge evidence and final scoped synthesis
listed in GRAIN_VECTOR_REMAINING_REQUIREMENTS.md. The official GRAIN scorer and
historical release/lockbox provenance are unchanged.

## Object and input contract

Each observation identifies source account/participant, source event and raw
text fingerprint, role, condition/opportunity metadata, human timestamp basis,
within-document/within-conversation order, language metadata provenance, and
consent/access scope. Missing covariates remain missing, not inferred clinical
states. Account identity is not assumed to establish a unique human author.

User text, assistant context, system/task instructions and target candidates
are separate fields. Do not pool assistant output into user evidence by default.
Every transformation records exact source spans, selected/omitted material,
separators, prefix/template bytes, tokenizer identity, token budget and hashes.
Whole-document truncation must be explicit; reject unsupported material or use
documented lossless windows, never silently treat the first512 tokens as all text.

Raw text is an input to the observer; numeric vectors can retain sensitive
information and remain subject to the same local/privacy constraints. No source
ID, questionnaire value or answer-generated token is a pre-answer predictor.

## Model operator identity

Record checkpoint revision and every used file hash, tokenizer/template,
model implementation/library versions, precision/quantization, backend/device,
physical execution shape, mask/position IDs, layer/indexing convention, role,
pooling and normalization. No automatic fallback to a different model, device,
quantization, context length, prompt or extraction site.

Distinct object types:

| Type | Meaning | Not interchangeable with |
|---|---|---|
| Input token embedding | Fixed learned lookup at a token ID | Contextual state or human trait |
| Contextual hidden state | Model/site/position-conditioned computation | Arbitrary other-model coordinates |
| Independently pooled windows | Declared summary of separate forwards | Full-conversation contextual encoding |
| Application memory | Selected/retrieved/rewritten source material | Model weights, KV cache, faithful autobiography |
| Readout score | Declared function of observed states/output distribution | Calibrated personality scale score |
| Source-trained probe | Statistical decodability under training-only choices | Evidence the LLM causally uses a personality concept |

## Available research observation families

1. Context displacement: same current/body/site under actual, matched other,
   random, or absent context. Save full original values as well as contrasts.
2. Task/readout access: local derivative, norm-matched intervention and restoration,
   with non-target endpoints and code/semantic qualification separately recorded.
3. Final-head geometry: target-logit/partition change, conditional likelihood,
   KL and directional curvature. These describe the model distribution only.
4. Role-separated author correspondence: fixed/free or disjoint-text similarities,
   with same-material, length and condition controls. Percentile is not MBTI AUC.
5. External associations: source-only supervised heads using official axis/trait
   folds and inner selection; bridge-only evaluation after source choices freeze.
6. Geometry/function correspondence: standalone evidence similarity versus a
   fixed contextual readout on identical cases, with lexical/condition controls.
   Raw and adjusted associations stay distinct; adjustment is not noise deletion,
   causal mediation or a certificate of personality-specific information.

No observation is marked `personality_validated` merely because geometry,
likelihood, author retrieval, repeatability, or a model intervention is positive.
Externally linked constructs and behavioral interpretations require distinct
evidence status. Model output explanations are not verbatim internal reasoning.

## Output contract

Every accepted result must contain:

- `observation_id`, `material_identity`, `operator_identity`, `source_scope`.
- `observation_type`, `unit`, `role_scope`, `context_condition`, `site`, `pooling`.
- Per-case numeric values plus denominator, missing/refused cases and reason.
- `uncertainty_unit` and its conditioning: participant/scenario/corpus; whether
  model fitting and donor assignment are repeated or held fixed.
- Independent status fields: instrument, material, function, author correspondence,
  semantic/code fidelity, external association, psychological validity.
- Local artifact/checksum/replay references and report version. Aggregates alone
  do not establish a model's provenance or an individual assessment's validity.

Statuses are `observed`, `unsupported`, `underresolved`, `not_tested`, or
`refused`, each with a reason. A source-only ranking AUC with majority-class
threshold collapse must not become a calibrated classifier. NaN is never zero
evidence; absent psychological labels never become generated ground truth.

## Evaluation and release gates

Exact source/token checks and self-replay come before domain inference. Verify
padding/position/execution-shape sensitivity at the intended precision. Keep
failed technical packets and repair the instrument before private reruns.

All representation/PCA/scaler/head/threshold choices fit training or inner-fold
data only. Source-exclusive Big5, MBTI and bridge roles stay separate. Reusing
historically exposed reference authors requires an explicit sensitivity branch.
All target samples use the same text, eligibility and auxiliary-information
budget when claiming an improvement. Partial-source models cannot choose winners.

Corpora with different people do not supply paired cross-domain individuals;
within-essay parts are not time retests; source language compatibility does not
prove measurement invariance. No population-normalized score refits reference
mean/variance using the incoming target batch.

Final integration needs complete full-source/bridge results, domain scope,
audited serialization/replay and a reviewed evidence atlas. Until then, research
scripts may produce scoped observations, but the public GRAIN scorer must not
silently expose them as validated psychological measurements.
