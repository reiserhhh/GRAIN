# NEXT41: complete GRAIN transport to a source-fixed readout

2026-09-10. Pass30 complete; whole-system goal ACTIVE. Corrected source capture
and fixed prediction/analysis were committed in `24aaaee` before new predictions;
original implementation/source pilot is `1ac8b8d`. No current-author target tuning.

## 1. Finding

The source-fixed reader now recovers the **full block16 GRAIN local action**, not
only the last normalized-BODY interface. New8 same-query source r=.9981 for N
and .9961 for confirmation; squared error relative to zero prediction=.00389/.00803.
The major action is already carried into the last block, with its attention
acting as a smaller signed correction. This changes the next experiment from
last-head refinement to earlier-layer accumulation and usable early exits.

These numbers measure native-model directional derivative fidelity. Current
material still runs through native upstream layers; no human personality
accuracy, clinical construct, whole-model replacement or speedup is asserted.

## 2. All new-author comparisons

Each row is8 authors x4 fixed directions, not32 independent subjects. Factor-
centered r removes the across-author mean separately for each direction before
pooling. SSE/zero is sum squared error / sum squared native action, not centered R2.

| Query | Source t | Readout route | Action r | Centered r | SSE/zero | Predicted RMS | Native RMS |
|---|---|---|---:|---:|---:|---:|---:|
| N | own | full | .99814 | .99826 | .00389 | .77646 | .76501 |
| N | own | before last attention | .99688 | .99698 | .00701 | .78528 | .76501 |
| N | own | last attention only | -.19120 | -.20826 | 1.02250 | .03480 | .76501 |
| N | pooled | full | .99848 | .99859 | .00803 | .70939 | .76501 |
| N | pooled | before last attention | .99693 | .99707 | .00881 | .72213 | .76501 |
| N | pooled | last attention only | -.33632 | -.35286 | 1.03212 | .03172 | .76501 |
| Confirmation | own | full | .99610 | .99589 | .00803 | .40723 | .42074 |
| Confirmation | own | before last attention | .98816 | .98921 | .02320 | .43252 | .42074 |
| Confirmation | own | last attention only | -.49670 | -.63658 | 1.11644 | .04538 | .42074 |
| Confirmation | pooled | full | .99672 | .99654 | .01136 | .44996 | .42074 |
| Confirmation | pooled | before last attention | .98890 | .99009 | .04019 | .47576 | .42074 |
| Confirmation | pooled | last attention only | -.45597 | -.60082 | 1.12046 | .05041 | .42074 |

All24 cohort/query/source/route rows are in [comparison.csv](complete_transport_tables_v1/comparison.csv).
Own and pooled are separately frozen programs, not a query-wise selected blend.
Old4 LOO uses three other sources, new8 uses all four; they are different protocols.
Old4 own full r is N=.99848,confirmation=.99113. For old4 N, adding last attention
raises own-source error .02084->.03074 despite a slightly higher r. Keep amplitude
and error, not only the best correlation. No new significance claim on this small,
previously reused author panel.

New8 own full per-factor r: N .99903/.99973/.99476/.99977;
confirmation .99672/.99866/.99307/.99543. All query/factor/route results are in
[factor_actions.csv](complete_transport_tables_v1/factor_actions.csv).

## 3. What the branch comparison means

`direct` in the code is the final-position response **after blocks17..26**, before
attention27. It includes all earlier attention, MLP and query changes. It is not
an attention-free text path. Initial perturbations touch BODY only; no initial
final-query perturbation is introduced.

With r=X_final+attention27_final and fixed source t, full action is the signed
sum of both routes. Relative to full predicted energy, new8 own-source N gives
1.022857 direct +.002008 last-attention -.024865 cross; confirmation gives
1.128070+.012419-.140489. Opposite signs occur18/32 and17/32 times, respectively.
These are action-energy accounts, not fractions of personality information.
See [branch_accounting.csv](complete_transport_tables_v1/branch_accounting.csv).

Thus NEXT39/40's useful last-head value/key mechanism is retained, but it is not
mistaken for the whole H16 mechanism. Comparing .998 to NEXT40 .809 is not a
controlled precision improvement: the local derivative and input plane changed.

## 4. Construction, execution and correction

The source-only pilot moved the last MLP into a one-token computation, changing
float32 rounding. Its two L replay errors were9.54e-7 and4.77e-6; the latter
failed the fixed2e-6 tolerance. That directory remains untouched. Correction used
a hook in the original full-sequence source forward, not a wider threshold.

Corrected8 source captures give L replay error exactly0 and maximum W_O^Tt versus
stored g relativeL2 error7.45e-7. Source t means are frozen before current scoring.
New-author predictions use forward-mode JVPs, not output scores or backwards.
Target NPZ/JSON denial probes both pass; no unexpected denied data reads.

-24 completed units;96 JVPs,4 prescribed finite-difference forwards.
-8 corrected source forwards/backwards plus2 failed-pilot forwards/backwards.
-Total bounded execution172.890s including the pilot; prediction165.625s,
 corrected source6.026s, pilot1.239s. Model loading is additional, outside the cap.
-Two fixed vector finite-difference relative errors=.01315/.01638 at epsilon.001,
 passing the.05 relative bound; absolute maxima .0190/.0711. These vector tests
 do not claim the1e-7 parity of the previous analytic last-layer formula.
-24 prediction packets hash-verified, all saved source contractions replay with
 error0; three analysis CSVs reproduce byte-for-byte. No second96-JVP run.
-Vector branch addition max9.54e-7; scalar signed addition max1.77e-8.
-57 scoped tests pass, including real-Qwen2 synthetic CPU/MPS JVP/VJP, full-source
 capture and residual closure.18 existing PyTorch/Python3.14 JIT deprecation
 warnings, no test failures. Ruff/diff checks pass; no full-suite claim.

No fit, human label read, external API, new prefix capture, generated text,
lockbox opening, release scorer change or live long run.

## 5. Artifacts and next action

Private accepted run: `results/grain_vector_mechanisms/logic_complete_transport_jvp_v1_r2/`.
Preserved source-only failed pilot: `logic_complete_transport_jvp_v1/`.
Each contains manifests, exact source/code references and explicit status. Accepted
run includes split forward inputs, source means, restricted prediction freeze,
per-unit responses and evaluator targets; public outputs are aggregate tables only.

Audit without new native calls:

```bash
"../project persona/.venv/bin/python" scripts/audit_grain_complete_transport.py
"../project persona/.venv/bin/python" -m pytest -q tests/test_grain_complete_transport.py
```

Runner stages are positional: `prepare`, `sources`, `predict`, `evaluate`, with
`--name logic_complete_transport_jvp_v1_r2`. Completed source/predict stages return
without recollecting; do not overwrite existing analysis directories. Use
`evaluate --analysis-name analysis_another_replay` only for a fresh numeric replay.
The original r1 pilot is not resumable with corrected code; retain its committed
implementation for historical reproduction.

Code: `grain_research/vector_mechanisms/complete_transport.py`,
`scripts/run_grain_complete_transport_jvp.py`, `scripts/audit_grain_complete_transport.py`.
[Theory](../../docs/GRAIN_COMPLETE_TRANSPORT_THEORY_V1.md),
[pre-outcome plan and source-only correction](../../docs/GRAIN_COMPLETE_TRANSPORT_JVP_PLAN_V1.md).

[NEXT42](../../docs/GRAIN_LAYER_ACCUMULATION_PLAN_V1.md) tests fixed exits20/23/26
and signed layer updates with unchanged source readout. The purpose is a usable
smaller transport program, not another fit to this near-perfect target.
