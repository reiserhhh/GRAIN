# GPT Pro V3 F13 recovery 3: binary next-p uncertainty for B1/B2/M under missingness (2026-09-26)

Worker: Claude Opus 5.5 (TASK-CLAUDE-GRAIN-FULL-LOOP-20260926). This closes the
census gaps F13-02 (B2/M next-p uncertainty) and F13-04 (binary missingness).
Protocol `docs/GRAIN_F13_RECOVERY3_PROTOCOL_V1.md` was frozen before the run.
Output: `results/grain_gptpro_foundation_v3/F13/recovery3_r1`. No reader
forwards, API, training or private data.

## Design

The C2 xor and iid worlds share one transition law across people, so the true
next probability is known given the true history.

- **Source fits:** the canonical source fit plus the 200 archived source
  bootstrap refits.
- **Filter:** the frozen QualifiedHistory filtered update, vectorised across
  fits; it equals the original to 1.1e-16.
- **Components per prediction:** lag-state, personal and source variance.
  - Personal: none for B1; ridge working posterior for B2; Beta variance of
    the raw table projected onto M's subspace for M.
  - Source: variance across the bootstrap refits.
- **Masks:** none, MCAR .4, MAR (.2/.6 by the last observed value) and MNAR
  (.2/.6 by the current value). Missing events keep their time position with
  qualification 0.
- **Check:** with no mask, the run replays the F08 complete-panel losses
  exactly.

## Findings

- **MNAR enters through personal parameters.** At a matched missing rate,
  MNAR bias relative to MCAR is:

  | model | xor | iid |
  |---|---|---|
  | B1 (population table) | -.019 | -.001 |
  | B2 | -.124 | -.158 |
  | M | -.066 | -.071 |

  MAR is effectively ignorable: B2/M are inconclusive, and B1 xor differs by
  only +.003.
- **Interval components are target-specific.**
  - Complete-history B1 intervals hold only source variance and under-cover
    (.91-.92).
  - With missing data, the lag state dominates and coverage rises to
    .96-.99.
  - B2's working posterior is too conservative (width .58-.86).
  - M's coverage falls to .929 under MNAR in xor.
- **Missing information costs +.028 to +.041 squared error where history
  matters (xor).** Where history does not matter (iid), it slightly helps
  B2/M, by reducing adaptation to noise.

## Production rules (feed F13-05 / F16)

1. Publish next-p intervals with separate state, personal and source
   components.
2. Do not publish B2's working-posterior interval as a reliability statement.
3. Under possibly response-dependent missingness, prefer population-model
   predictions or model the selection before personal adaptation.
4. Report missingness cost per target.

## Open

- Worlds with person-specific laws.
- Calibrated B2 personal variance.
- Reader and material components, which belong to F16 text transport.
- No human interval validity (H01).

## Companion: F11 source-refit uncertainty (`F11/source_uncertainty_r1`)

Protocol: `docs/GRAIN_F11_SOURCE_UNCERTAINTY_PROTOCOL_V1.md`, with amendment
A1 recorded before the full run. Run: 200 source refits x 3 seeds x 3 worlds;
the replay is exact.

- **Source component:** it is 1.4-2.9% of trait interval variance and lifts
  coverage by only .001-.004.
- **Conditional-loading under-coverage** (plug-in .929, oracle .936): source
  uncertainty does not explain it. A post-hoc check with 200,000 fresh
  authors shows the oracle is exact (.9498), so the shortfall is sampling
  variation in this application sample.
- **Robustness defect:** REML fails on 6.6% of refits. That is 85
  non-convergences, plus 33 eigenvalue-tolerance exceptions concentrated in
  one conditional cell (flagged unreliable).
- **Tests:** new variance-0 boundary tests; 4 pass.
