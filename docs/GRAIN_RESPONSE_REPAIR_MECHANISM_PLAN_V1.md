# NEXT54: explain the response repair, then leave the fixed-basis family

2026-09-10. Planned, not executed. NEXT53 both joint maps and matched value-only
controls pass the new8 component gate. Source4 endpoints remain weak. No refitting.

## Question

Joint source teaching changes both the current estimated state X and its tangent d.
Is its endpoint improvement carried by the state adjustment, the tangent adjustment,
or their pairing inside attention? This determines what should be preserved when
the retained component is connected to a fuller scoring path, rather than refining
its correlation again.

## Fixed numerical crossing

Same12 authors/two queries/four GRAIN directions, both pointwise/causal families.
Reconstruct X_V,d_V and X_J,d_J from the frozen NEXT53 factorized packets. Reuse
Qbar/rbar/actual attention22 matrices, RMSNorm/differential, RoPE/mask exactly.
Let A_q be the same K-component program. For each material/family calculate:

    f_VV=A_q(X_V,d_V), f_VJ=A_q(X_V,d_J)
    f_JV=A_q(X_J,d_V), f_JJ=A_q(X_J,d_J).

VV/JJ are cached diagonal candidates; independently check their numerical parity.
Only two off-diagonal evaluations are new. These crosses are **mechanism probes**,
not derivatives of a newly deployable state map. Each evaluates the normalization
differential at its chosen primal state. Do not bypass norm dependence.

Freeze all four scalar cells before reading endpoint targets.96 new cross-vector
evaluations, each four directions; all768 diagonal/cross scalar rows retained.
0 native/backward/fit/API/Y calls,60s CPU plus one cached replay. Source assets
and all current inputs are existing H16-derived candidates; no current pre22/native
gradient input in prediction. No extra seeds/queries/alpha/rank/gain rescue.

## Analysis and completion

Use the exact symmetric allocation:

    state=((f_JV-f_VV)+(f_JJ-f_VJ))/2
    tangent=((f_VJ-f_VV)+(f_JJ-f_JV))/2
    interaction=f_JJ-f_JV-f_VJ+f_VV
    change=state+tangent=f_JJ-f_VV.

Keep the interaction separately, not counted twice in the allocated change.
For e_V=f_VV-kappa and delta=state+tangent, verify:

    MSE_J-MSE_V = 2 mean(e_V delta)+mean(delta^2).

Report each term, their covariance, signs, source4/new8, both queries, query
contrasts and all four factors. Compare *error reduction* and signed response
change; correlation of a correction alone is not the completed effect. Check
whether source4 weak endpoints reflect an ineffective correction here, without
reweighting either cohort or fitting its outcomes.

Completion is an exact, replayed mechanism account, not an imposed preferred
state/tangent winner. Preserve the original NEXT53 retention decisions. After this
one crossing family, move to a specified whole-path assembly experiment using the
retained map; do not continue local operator or loss-weight searches on these data.

New `response_repair.py`, `run_grain_response_repair.py`, minimal identity/2x2/MSE
tests and cached audit. Private packet
`results/grain_vector_mechanisms/logic_response_repair_v1/`; public
`RESPONSE_REPAIR_REPORT_V1.md`, aggregate allocation/error tables and an updated
whole-path dependency recipe. Local commits; no scorer/release/lockbox/MEPS edits.
