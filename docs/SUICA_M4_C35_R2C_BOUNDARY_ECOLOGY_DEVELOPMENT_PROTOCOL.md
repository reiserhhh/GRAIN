# GRAIN M4-C.3.5-R2C Boundary Ecology Development Protocol

## Status

This is a fresh-seed mechanism-development experiment. It cannot produce a
confirmation `GO`.

R2B established a finite-synthetic structural advantage for the RCCA chart
but failed its zero-refusal gate. R2C proved that a pre-response RCCA/B0
routing policy can be replayed exactly, but all 16 fresh cells were accepted,
so routing value was not identified.

## Question

Let

\[
d_i=\Gamma_{R,i}-\Gamma_{B0,i},
\qquad
e_i=\Gamma_{Oest,i}-\Gamma_{R,i},
\]

and let \(A_i\) be the frozen pre-response support decision. The routed chart
is

\[
\Pi_i=A_iR_i+(1-A_i)B0_i.
\]

The experiment asks whether lower pre-response support predicts:

1. a harmful forced-`R` effect \(d_i<-.01\);
2. a larger oracle approximation error \(e_i\); or
3. neither, making the gate an applicability warning rather than a utility
   safety valve.

The two development estimands are

\[
U_Q
=
\mathbb E_Q[\Gamma_\Pi-\Gamma_R]
=
-\mathbb E_Q[(1-A_i)d_i]
\]

and

\[
V_Q
=
\mathbb E_Q[\Gamma_\Pi-\Gamma_{B0}]
=
\mathbb E_Q[A_id_i].
\]

`U_Q` is routing value relative to always using `R`; `V_Q` is full-population
policy value relative to always using `B0`.

## Paired intervention

Within each repetition and world, response truth, author mechanisms, ecology,
condition identities, and every non-evaluation panel are held fixed.

The RCCA chart is fit once from native pre-response data. The least-supported
still-in-domain evaluation conditions are then moved coherently across both
sources and all authors, away from the calibration support. No response byte
is read or changed.

With 16 evaluation conditions, frozen target counts are:

- `13/16 = .8125`: accepted side immediately above `.80`
- `12/16 = .75`: refused side immediately below `.80`
- `11/16 = .6875`
- `10/16 = .625`
- `native`: unmodified reference cell

The `.80` threshold is fixed and is not optimized from outcomes.

## Worlds and sampling

Eligible worlds:

- `endogenous_creation_expansion`
- `endogenous_source_partition_matched`
- `source_rotated_feedback`

Safety and boundary worlds:

- `history_gated_ecology`
- `selection_creation_compensation`

Null:

- `linear_null_ecology`

This first boundary-enriched development uses two independent repetitions.
It is intended to test intervention controllability and discover the shape of
the support-error relation, not estimate a publishable interval.

## Phase separation

Phase A:

1. Generate pre-response tensors only.
2. Fit native `B0` and RCCA maps once per repetition/world.
3. Apply all registered evaluation-support interventions.
4. Verify exact target coverage and both accepted/refused cells.
5. Serialize only `B0` and forced `R` bases.
6. Seal config, protocol, source, runtime, pre-response digests, and bundles.

Phase B:

1. Require the Stage-A manifest hash.
2. Verify every source, runtime, digest, intervention, basis, and decision.
3. Open response and oracle truth only after all intervention cells are
   frozen.
4. Score `B0`, forced `R`, same-estimator `Oest`, and routed `Pi`.

Any replay error invalidates the run.

## Development classifications

- `USEFUL_SAFETY_GATE_CANDIDATE`: `U_Q` LCB and `V_Q` LCB are positive and
  coverage discriminates harmful forced-`R` cells with AUC LCB above `.65`.
- `SUPPORT_BOUNDARY_PROXY_CANDIDATE`: support predicts oracle approximation
  error out of repetition, but `U_Q` is not positive.
- `FALSE_REFUSAL_RISK`: refused cells are not more harmful than accepted cells
  or routing loses to always using `R`.
- `INCONCLUSIVE`: the intervention works but none of the above mechanisms is
  resolved.

These labels are development outputs only.

## Stop rules

- Stop before outcomes if any world lacks both sides of the `.80` boundary.
- Stop before outcomes if native non-evaluation coverage is below `.875`.
- Stop on any non-`SUPPORT_SHIFT` RCCA refusal.
- Stop on any source, runtime, hash, digest, or basis replay error.
- Do not change `.80`, remove refused cells, or select intervention levels
  after opening outcomes.

## Boundary

This experiment concerns finite-synthetic response-safe chart applicability.
It licenses no natural-text, personality, behavioral, clinical, or M4-D
claim.
