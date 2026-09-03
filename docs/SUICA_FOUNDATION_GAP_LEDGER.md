# GRAIN Foundation Gap Ledger

Date: 2026-07-30  
Authority: `docs/SUICA_FOUNDATIONAL_CLOSURE_STANDARD.md`

## Status semantics

- `CONSTITUTIONALLY_CLOSED`: the object, assumptions, evidence class, and
  refusal path are formalized. This does not mean every empirical instance
  passes.
- `EMPIRICALLY_BOUNDED_OPEN`: the missing evidence is localized and may be
  tested without treating it as a premise.
- `DOWNSTREAM_INACTIVE`: intentionally unavailable until an application
  declares a construct or use.

## Ledger

| ID | Missing piece | Current status | What is already known | Evidence needed to close an instance | Refusal while open |
| --- | --- | --- | --- | --- | --- |
| F01 | Human process -> expression behavior | `EMPIRICALLY_BOUNDED_OPEN` | V8 does not claim direct access to cognition/personality. | Fixed/free opportunity designs, response-process evidence, assistance/audience manipulation. | No intention, effort, disclosure, or psychological-process claim. |
| F02 | Expression behavior -> stored record | `CONSTITUTIONALLY_CLOSED` | Provenance, raw-record version, AI/quote policy, missingness and attribution are required fields. | Per-study capture and attribution audit. | No complete-behavior or sole-authorship claim when capture is incomplete. |
| F03 | Record -> representation | `CONSTITUTIONALLY_CLOSED` | Encoder, tokenizer, slicing, fitted transforms, fit/apply split, and gauge sensitivity are operator-indexed. | Per-study perturbation and admissible-operator audit. | No universal coordinate or cross-version transport. |
| F04 | Representation -> anonymous technical object | `CONSTITUTIONALLY_CLOSED` | Estimand, alias worlds, support, dependence, nulls, uncertainty, and refusal are mandatory. | Source-disjoint recovery/replication for the declared object. | No object-existence or unique-mechanism claim after a failed gate. |
| F05 | Typed technical object -> typed reference-relative measurement | `EMPIRICALLY_BOUNDED_OPEN`; `E45-V` opened synthetic candidate | The current machine-enforced reference contract covers `E45-V`. V1 exposed omitted occasion uncertainty (coverage 0.714); V3 passed the repaired 30-repeat, two-noise, ten-world opened battery with minimum condition coverage lower bound 0.925 and all mandatory refusal rates 1.0. It does not cover `E45-R/P`. | Separately sealed fresh-randomness confirmation, then type-specific G-study/D-study or dependence analysis, reference/operator transport, and conditional uncertainty/MDD analogue. The schema must declare output type before an `E45-R/P` experiment. | No reusable human score, cross-type conversion, relation scalarization, ecological-to-individual inference, person change, cross-batch comparison, or psychological naming. |
| F06 | Measurement -> psychological construct | `DOWNSTREAM_INACTIVE` | External mapping is many-to-many and does not inherit technical validity. | Construct theory, rival constructs, external behavioral/psychological anchors, response process, invariance. | Anonymous labels only; no personality/state/emotion naming. |
| F07 | Construct -> use/decision | `DOWNSTREAM_INACTIVE` | Use is a separate evidence layer. | Prospective field validation, base rates, costs, fairness, authority, contestability. | No diagnosis, treatment, ranking, or forensic decision. |
| F08 | Residual semantics | `CONSTITUTIONALLY_CLOSED` | Residual defaults to unresolved mixture, not noise or personality. | Randomization, replicate, counterfactual, or validated component model for any stronger label. | Do not subtract or name unidentified residual components. |
| F09 | Generalization universe | `CONSTITUTIONALLY_CLOSED` | Ten facets must be declared, bounded-open, or explicitly inapplicable. | Per-study population/domain sampling and support audit. | No transport outside sampled facets. |
| F10 | State/trait distinction | `CONSTITUTIONALLY_CLOSED` as a definition; `EMPIRICALLY_BOUNDED_OPEN` per construct | State/trait depends on occasion universe and horizon, not feature appearance. | Matched repeated occasions and probabilistic consistency/occasion-specificity decomposition. | No trait claim from same-session split-half stability. |
| F11 | Event opportunity, choice, and response | `CONSTITUTIONALLY_CLOSED` | Choice and conditional response are separate channels; context may affect both. | Positivity, menu overlap, and channel-specific replication. | No preference claim without opportunity; no response claim from selection alone. |
| F12 | Slice/operator multiplicity | `CONSTITUTIONALLY_CLOSED` | Different cuts are operator-indexed views, not competing truths. | Invariant, interval, or explicit design-specific selection across admissible operators. | No single “true slice” claim. |
| F13 | Language/culture transport | `EMPIRICALLY_BOUNDED_OPEN` | Shared embedding space is not measurement invariance. | Within-language stability plus cross-language/culture invariance and response-process evidence. | No universal-human or bias-free comparison. |
| F14 | LLM observer/explanation | `CONSTITUTIONALLY_CLOSED` in role; `EMPIRICALLY_BOUNDED_OPEN` in fidelity | LLM cannot be final scorer or mutate evidence. | Model/rater perturbation, fidelity, calibration, and comprehension tests. | Renderer-only output with evidence and counterevidence. |
| F15 | High-stakes/forensic overlay | `DOWNSTREAM_INACTIVE` | Jurisdiction, response style, collateral sources, base rates, costs, contestability, and custody are mandatory overlays. | Application-specific prospective and legal validation. | No high-stakes use. |
| F16 | Frozen-source integrity | `EMPIRICALLY_BOUNDED_OPEN` in the current checkout | Source hashes are first-class governance evidence; a changed implementation is a new operator. | Reconstruct the originally locked source from its recorded commit or register a new study ID and seal. 2026-08-03 reconciliation: the five legacy H4C/R2D/R2E source-lock tests pass against the committed tree AND run-time artifact lock hashes match it — the earlier "currently detect hash drift" described a transient pre-commit working-tree state, since reverted; no re-bake occurred (see the dated note at the end of this document). | Do not relabel or rerun the old frozen protocol against changed source code. |
| GAP-F17 | L4 output-type to L5 interface | `CONSTITUTIONALLY_CLOSED` as a typing rule; `EMPIRICALLY_BOUNDED_OPEN` for relation- and population-valued instances | L4 author vectors/operators (`V`), dyadic relations/distances (`R`), and population relation fields (`P`) are distinct. E45-V, E45-R, and E45-P cannot inherit one another's license. The current L45 V3 candidate covers E45-V only. | Type-specific reference object, dependence model, confidence region, detectable-difference rule, transport audit, and separately sealed confirmation. | No relation-to-person scalarization, ecological-to-individual inference, or person ranking from a population field. |

## Closure ruling

The foundation is no longer allowed to hide an inferential gap in prose:
every known gap has a layer, a contract field, a closure test, and a refusal.
The open entries are empirical programs, not licenses to make the intended
claim early.

Any newly discovered gap must be added here before an experiment is designed.
If it cannot yet be typed, the study status is `REFUSE_EXPERIMENT_START`.

## 2026-08-03 F16 reconciliation note

Three-way verification (`docs/V8_PUSH_REMEDIATION_20260803.md`, item 4):
run-time `source_lock` expected/observed hashes persisted in the decision
artifacts were compared against the committed tree. R2E 9/9 and 9/9; R2E1
discovery 10/10 (the only mismatch anywhere is the superseded same-day
corrected-smoke runner state, replaced before the discovery run); R2E2 11/11
and 11/11; R2D 5/5 in all three runs; H4C recorded
`detector_source_lock: true` at run time and its config lock (3 hashes)
matches the tree. All five legacy lock tests pass. The 2026-07-30 row text
"currently detect hash drift" described a transient pre-commit working-tree
state that was reverted before commit `4eb0124`; the committed sources equal
the originally locked sources, so no re-bake occurred and the F16 refusal
rule was not violated in committed history. F16 remains
`EMPIRICALLY_BOUNDED_OPEN` as a program: per-study source-reconstruction
obligations are unchanged.

## Dated addendum (2026-08-10): identity-era annotations

The IDT arc (K/L lines, defense phase; see
`docs/SUICA_V8_IDT_INTEGRATION.md`) touches this ledger as follows —
statuses unchanged; annotations only:

- **F10 (state/trait):** the per-construct empirical side gains a
  measured MECHANISM: within one jurisdiction b and s̄ are not
  separately identified (IDT T4 typing); the deployed reader loses
  trait content quadratically and taxes person-state variance; the
  licensed state meter is the two-split probe. The refusal sharpens:
  same-session split-half stability is not merely insufficient for a
  trait claim — same-FRAME reproducibility is demonstrably FORGEABLE
  (T9, measured), so identity-flavored readings require frame
  refreshment (T6″).
- **F08 (residual semantics):** reinforced — S4's repetition
  specificity (H6) and the discriminator's rule (only
  frame-refreshed-stable components may be named) are two more measured
  instances of "residual defaults to unresolved mixture".
- **F03/F09 (coordinates/generalization):** T1's gauge trichotomy and
  T10's anchor impossibility give the "no universal coordinate / no
  frame-free transport" refusals theorem form.
- **PROPOSED rows (owner's call under the closure standard; NOT added
  here):** F12 frame-forgeability (identity-like statistics are
  forgeable by frame error; counter-operations refreshment/contrast);
  F13 anchor-dependence (every informative reading consumes issuer,
  gallery, or coordinates — T10). Proposed because both have proofs and
  measured instances; they await the closure-standard formalization
  (object, assumptions, evidence class, refusal path).
