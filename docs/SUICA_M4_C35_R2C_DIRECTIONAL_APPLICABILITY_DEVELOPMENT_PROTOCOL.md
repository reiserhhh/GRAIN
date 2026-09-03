# GRAIN M4-C.3.5-R2C Fixed-Coverage Directional Applicability

Status: **prospective development protocol**

## Question

Can a response-safe vector signature identify when the frozen RCCA chart `R`
is preferable to fallback chart `B0` after scalar support amount is held fixed
and only the direction of support departure changes?

This follows the falsification of the historical `.80` scalar gate. It does
not retune that threshold.

## Frozen estimand

For each cell,

\[
d = G(R)-G(B0),
\]

where `G` is the frozen downstream author-relation geometry score. A routed
policy has

\[
U=E[d\,1\{A(q)=R\}], \qquad
V=E[-d\,1\{A(q)=B0\}],
\]

relative to always using `B0` and always using `R`, respectively.

## Design

- Eight independent repetitions.
- Five non-null M4 ecology worlds and one linear-null world.
- One native cell plus exact evaluation support counts `12/16` and `13/16`.
- At each non-native count, four outcome-blind departure geometries:
  - `radial_dispersed`
  - `radial_concentrated`
  - `tangential_rotation`
  - `source_asymmetric`
- Total: `8 x 6 x 9 = 432` cells, including `360` main cells.

Only `mechanism_evaluation.pre_context` may change. Responses, oracle truth,
all other panels, and the frozen chart maps must remain byte-identical.

## Competing policies

- `S0`: historical `.80` support threshold.
- `S1`: fold-local standardized ridge/logistic models using scalar minimum
  coverage only.
- `Q`: the same fixed estimators using the preregistered 24-component
  pre-response applicability signature.

All scaling and model fitting occur inside:

1. leave-one-world-out (`LOWO`), and
2. leave-one-repetition-out (`LORO`).

The route uses `R` only when the fold-local predicted relative gain is at
least zero. Harm classification uses the frozen `d < -.01` definition.

## Development gates

`Q` is a finite-synthetic vector-domain candidate only if both LOWO and LORO
meet all of the following:

1. `U_Q > 0` and repetition-bootstrap 95% lower bound `> 0`.
2. `V_Q > 0` and repetition-bootstrap 95% lower bound `> 0`.
3. `U_Q - U_S1 > 0` with lower bound `> 0`.
4. Harm AUC lower bound `>= .65` and point AUC at least `.05` above `S1`.
5. Out-of-fold gain-prediction `R2 > .10`.

If LORO passes but LOWO fails, the result is `WORLD_DEPENDENT_DOMAIN`: the
signature may interpolate repeated mechanisms but cannot route an unseen
world. Any replay, source-hash, response-identity, oracle-identity, or null
failure stops interpretation.

## Information order

Phase A freezes config, protocol, source hashes, runtime, pre-response
interventions, B0/R bases, signatures, digests, and decisions. Phase B opens
response/oracle truth only after the Phase-A manifest hash is supplied
explicitly. No source edit is allowed between phases.

## Claim boundary

This is finite-synthetic applicability development. Even a full pass would
authorize only an independently sealed routed-policy confirmation. It cannot
support natural-text transport, personality, behavioral, diagnostic,
clinical, or M4-D claims.
