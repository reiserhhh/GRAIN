# P1C: evidence-to-query transport, not a larger persona-direction search

2026-09-07. Follow-up selected after complete P1B; explicitly exploratory.

P1B completed862 forwards/186.2s with zero self-copy/restoration discrepancy.
Canonical trait high-low gap1.564 coexists with gaps near0.064-0.110 under other
question/code cells. Late evidence-end direction effects are tiny, and whole
evidence replacement atblock16 does not recover the counterfactual donor score
(mean absolute discrepancy1.488). Do not conclude absent evidence information:
earlier layers may already have transferred information to later query positions.

## Mechanism contrast

Retain P1B paired role-swap text and exact original observer. Use block04/10/16
as three fixed locations, not a search grid. At each layer partition all positions
from evidence start onward into E=evidence span, Q=remaining suffix excluding the
last prompt token, D=last prompt token. The unchanged earlier prefix is checked.

Take saved activations from the opposite-behavior donor with identical token layout
outside E. Run all8 masks:000 baseline and100/010/001/110/101/011/111 replacement.
Every active set is copied at the same layer in one combined patch. This records
all pair/triple interactions, rather than calling joint-minus-one an indirect
effect. Full111 should reproduce the donor endpoint (<=1e-5); partial effects
need not add. Conditional-internal interventions establish effects in this model,
not human causation or a unique psychological mechanism.

60 source prompts:12 role-swap materials x event-target/event-companion,
positive-trait canonical/reversed-code, unrelated venue. All were already in P1B;
this is not a fresh blind validation sample.48? No: five variants x12=60.
60 baselines +60 x3 layers x7 patches=1320 forwards. Add6 self/restore checks and
36 factual opportunity questions to compare with the existing36 trait opportunity
scores; total1362 expected when candidates are single-token. Hard cap1800 actual
forwards and900seconds. No free generation, training, new data or extra layers.

## Opportunity diagnostic

Existing free/assigned/neutral trait gaps=3.979/6.050/2.826. Query the SAME action
records about the factual act of initiating, using Yes/No. Compare gap changes
within each query task, not their absolute psychometric scales. If both facts and
traits strengthen under assigned context, general contextual salience is a live
alternative; if only attribution changes, task-specific weighting remains a
candidate. Neither outcome alone identifies rational evidence diagnosticity.

## Readout and stopping

Save every raw margin, donor baseline, intervention mask, native high-low gap,
interaction contrast, absolute restoration error and signed recovery. Ratios to
donor-baseline gap are descriptive only; very small denominators remain explicitly
unstable, not discarded to improve the headline. Report squared aggregate error
relative to no intervention, separately by variant/layer, including venue control.

Trace how E/Q/D effects redistribute across layers; do not name a late answer-ready
state a personality axis. Stop after this finite decomposition and assess which
location genuinely warrants a functional map. Keep nulls, do not infer a unique
attention edge from a residual-stream patch, and do not restart the old queues.
