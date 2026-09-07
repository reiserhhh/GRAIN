# S27: locate when the useful configuration contrast becomes query-carried

2026-09-08. Planned next, not run. S25/S26 preserve the useful paired route
but show that the larger layer17 head vector adds dispersion with little N
covariance. Do not keep refining that head or launch a broad normalization grid.
The new question is where the full S9 configuration contrast C travels next.

## Fixed computational measurement

Use the24 N authors at indices24:48, view0, same raw12-comment prompts and S9
block16 body projection. Keep source membership explicit and match by author ID.
Collect native and projected full forwards, capturing the final-position residual
state after blocks16,17,20,23,26,27 (zero-based). This is2forwards/author.
Reproduce S9 native/projected endpoint scores; the block16 query states must be
identical because only earlier body positions are projected.

For each interior cut l in17,20,23,26, cross-patch only the last residual state
after block l into the other body condition, leaving all other positions intact.
Two crossed scores per cut give

Phi_l=.5[(F_native-F_native_with_projected_query_l)
         +(F_projected_with_native_query_l-F_projected)].

This is the same finite two-order allocation used in S24, but the carrier is
the residual query state at a declared boundary, not the pre-W_O head output.
Keep both directed effects, their interaction, and Other_l=C-Phi_l.
Measure coarse increments Phi_17-Phi_16, Phi_20-Phi_17, etc.; they telescope.
These are changes of query-carried allocation between intervention boundaries,
not uniquely identified local layer-creation effects.

Phi_16=0 follows from identical query states. At the final block27 the score
depends only on the last state through tokenwise final norm and LM head, so
Phi_27=C. Replay that final-state score using the original final norm/lm_head
without running another full decoder. These endpoint identities are numerical
checks of the declared carrier definition, not human causal assumptions.

The allocation follows the marginal-contribution/order averaging in
[Lundberg & Lee2017, Eq4](https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf).
The next possible compact readout is a transported differential at a source-
relevant query boundary, using the exact path formula of
[Sundararajan et al.2017](https://proceedings.mlr.press/v70/sundararajan17a.html).
Do not apply the layer17 head covector to different-layer residual coordinates.

## Budget, output and next decision

At most240full forwards (2endpoints+8cross patches per author),0backwards,
600seconds after loading. First8author numerical/resource checkpoint; complete
24 only if within the cap and endpoint replay holds. One neural process, no
questionnaire join or per-layer source-label search. Save all selected-cut
states, scalar cells, runtime, hashes and full signed per-author profiles.

Report C/Phi_l/Other_l magnitudes, means, r and author-bootstrap intervals;
retain the complete profiles instead of presenting only the best layer.
Inspect whether a low/intermediate boundary already carries a useful part of C
or whether substantial effects enter only late. A near-final cut recovers C
by construction, so high r there is not a discovery by itself.
Use the profile to choose the next finite functional-gradient transport question,
not a personality claim or a benchmark-winning layer. Do not expand to every
head/layer/token before this coarse source-path question is answered.

Proposed runner scripts/run_grain_query_carrier.py and a scoped residual-state
patch helper with algebra, cleanup, identity-cut and final-readout tests.
Output results/grain_vector_mechanisms/query_carrier_s27_r1/.
Reuse S24 source binding and exact projected_body context manager. No frozen
script edits, old queue, LoRA, protected bridge/MEPS/lockbox, cloud upload or push.
