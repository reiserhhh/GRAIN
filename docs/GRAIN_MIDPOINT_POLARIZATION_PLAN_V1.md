# HM19: Midpoint Polarization and Observable Self-Interaction

2026-10-08. Planned after verified HM18, not executed. A bounded CPU,
zero-forward/zero-fit packet; not a new model or rank/alpha/precision search.

## Decision

HM18 source symmetric observation rank increases182 to264 when moving from
receiver to midpoint dyads, while skew rank stays182. Determine exactly
which self-quadratic observation directions are added, whether the frozen
query operators use them, and which native response remains outside the
compiled source-supported joint structure. Keep psychological interpretation
and natural-author responses as separately open links.

## Fixed Inputs and Boundaries

Use verified HM18r1/HM17r2/HM16r1 frozen inputs, all2172 query addresses/
724 states/7arms/3queries/source1080-valid528-app564/8-4-4families. Preserve
all actions, source units, exactnoops, A/T/order/tokens and shuffled virtual
contexts. No new material/model/API/private reads, source target fits,
neural training, Claude, automation, publication, old queues or lockbox.

## Source Geometry Before Held Responses

Define U_i=sym(x_i c_i^T), V_i=kappa*x_i x_i^T/2 and W_i=U_i+V_i.
Compute explicit Gram blocks:

  G_UU = T_S(receiver)
  G_VV = kappa^2*(X X^T)^2/4
  G_UV[i,j] = kappa*(x_i.x_j)*(c_i.x_j)/2
  G_WW = G_UU+G_UV+G_UV^T+G_VV.

Check W against HM18 midpoint symmetric Gram and unchanged skew Gram.
Report ranks of U,V,W and stacked observation rows [U;V]; distinguish
stacking row functionals from concatenating different coefficient spaces.

With the same fixed machine tolerance100, project V onto the row span of U:
G_V_perp=G_VV-G_UV^T G_UU^+ G_UV. Use a checked pseudoinverse with only
numerical-null truncation, not a selected ridge or accuracy rank. Explicit
orthogonal projection/QR must replay the result and report cancellation
error, small negative numerical eigenvalues and both ranks.

Freeze source full-design and declared full-property diagnostics and
family-balanced weights before held native scores. No held choice of a
preferred direction, threshold, dimension or query interpretation.

## Fixed-Operator Action, Not New Fitting

For each frozen receiver and endpoint operator, evaluate U, V, W actions;
decompose the V functional into source-U-supported and orthogonal remainder.
Use original ambient covectors for tiny normalized action statistics;
source-coordinate Gram compression must carry its reconstruction error.
Preserve both separately fitted endpoint coefficients and algebraic
receiver-potential coefficients; do not swap them as if identical.

On all roles/arms, report RMS/mean and source-supported/remainder sharing,
native residual covariance, and the exact action recombinations. Use
existing native responses only for interpretation, not choosing a model.
Explicitly record whether gains arise only from observations not available
to full-property single-token cases. No claim that gain82 is a factor count.

## Tests and Alternate Verification

Test midpoint symmetric increment/skew invariance, Gram block identity,
row-span projection, degenerate/zero states, orthogonal gauge and a synthetic
design with a genuinely new quadratic direction. Include a design where
V adds nothing, so novelty is not hard-coded. Verify all numerical rows,
source projection residuals and original addresses using different arithmetic.
Preserve any failed verification attempts and frozen prior results.

## Output and Next Decision

Fresh result directory results/grain_math_response_v1/midpoint_polarization_r1,
alternate verifier, mathematical companion, parent report/reading ledger/
checkpoint. No existing frozen files edited.

If V adds a reproducible observable direction with a distinct query action,
carry it as a measurement component in the text-to-target route. If its
actions collapse into U despite rankgain, distinguish design capacity from
this reader's used action, then target a missing joint observation rather
than increasing predictive precision. New native interventions require a
specific scientific decision not answered by this finite algebra.
