# NAV02/NAV03 next bounded experiment: source-span expansion

2026-09-09. **PLANNED, NOT RUN.** This is the next active-goal action after
round4, not an invitation to reopen human-label tuning or restart old queues.

## Question and mathematical reason

Synthetic-only residual transport has natural-state error energy84%-91%
outside its construction span. Does adding a small, already permitted,
unlabelled reference-author span recover the missing operation on *different*
authors? Or does the remaining problem concern conditional coefficients?

For query q, the existing constructor is
y_hat=x+mean(R)+k_c(x,X)(K_c+I)^(-1)(R-mean(R)), R=Y-X.
Its correction is limited to a construction snapshot span. Extend the snapshot
source, not the questionnaire target or chosen prediction sign.

Compare the existing frozen synthetic-only operator with **one** expanded
operator. Keep four blocks, linear kernel, alpha1 and the same synthetic-reader
G/g fixed. Do not add a degree/alpha/reference-subset search.

## Construction and separation

- Synthetic source: same48 construction worlds per query; reuse existing early
  and teacher snapshots with identity/hash checks.
- Natural source: only12 reference authors from NAV08/author_panel_r1.
- Average the two reference views into one paired early/upper state per author;
  each natural author contributes once. Combined construction has60 snapshots
  per query,48 simulated scenes/combinations plus12 real-author aggregates.
- Upper P/Q targets are the already captured original frozen-model states.
  These are model-state supervision, not psychological questionnaire labels.
- Capture only missing four-block early states: at most72 new shallow passes.
- Never use the12 evaluation authors, MEPS text, human criterion columns or
  held synthetic scenes for fitting. The source manifest must list reference
  membership before any fitting or outcome comparison.
- This extension is **synthetic-plus-unlabelled-source** construction. Preserve
  pure-synthetic results separately; do not call the extension toy-only learning.

## Evaluation sequence

1. Freeze source IDs, tokens, state operators and source hashes.
2. On held worlds and all12 evaluation authors/eight natural edit conditions,
   reuse already captured early inputs. Generate states without teacher inputs.
3. Freeze predictions, then join original teacher states for fidelity, signed
   actor/endorsement/time/removal responses and state-geometry analysis.
4. Separate orthogonal outside-span error from inside-span coefficient error.
   Report error in the direction of the frozen reader as well as total energy.
5. If any natural query shows both useful positive signed-response recovery and
   improved function fidelity, evaluate the frozen candidate on existing external
   endpoints. Do not pick a sign or coefficients from human labels; score all
   three declared queries. Exclude construction-author overlap from PANDORA
   endpoint reports, and show comparable restricted old-candidate controls.
6. If span expansion lowers Euclidean error but not score/response error, keep
   that distinction and target the conditional coefficient map next.

A non-improving candidate is still a completed experiment. One new finite
candidate is enough to decide this question; no requirement to hit r>.5 or
outperform every historical predictor before interpreting the operation.

## Deliverables

Proposed new script: scripts/run_grain_source_span_extension.py.
Proposed output: results/grain_vector_mechanisms/navigation_v1/NAV-02/source_span_r1/.
Save manifest, construction membership audit, operators, generated predictions,
paired-response and error-geometry tables, source-overlap audit and report.
Runtime dependency class remains four extracted Transformer blocks plus a
fixed snapshot operator. Keep token-only T/F and contextual low-state candidates.

Source precedent is the same snapshot/operator-inference idea used in the
[state construction report](../reports/grain_vector_mechanisms/NAVIGATION_STATE_TRANSPORT_REPORT.md),
with an explicit domain-coverage experiment, not a new claimed universal theorem.
The experiment tests source coverage as an explanation for a measured failure.
