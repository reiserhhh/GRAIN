# GRAIN response teaching mainly repairs the state-dependent reader

2026-09-10. NEXT54 complete. Primary construction/analysis freeze `8540e24`.
Same12 author materials, two queries, four GRAIN directions; source4 leave-one-out
and reused new8 evaluation materials remain separate. No new fits or model calls.

## 1. Finding and decision

The NEXT53 joint-teaching gain has two routes. On new8, symmetric error accounting
attributes **80.4-83.4% of the endpoint error reduction to the changed state**, and
16.6-19.6% to the changed tangent. Both help on both queries and both map families.
Thus teaching GRAIN responses affects not only the transported direction, but
also the state at which attention and normalization read that direction.

This is an explanation of the existing14.8-19.1% matched-control gain, not a new
accuracy improvement. The target is one attention22 K-channel contribution to
the final fixed-logit derivative. These percentages are a declared computational
allocation, not causal/personality variance shares or questionnaire accuracy.

Retain the NEXT53 compatible joint maps and matched controls. Close this
fixed-basis refinement family. [NEXT55](../../docs/GRAIN_WHOLE_PATH_ASSEMBLY_PLAN_V1.md)
will connect a frozen map and its derivative to the complete downstream logit
function, instead of optimizing this component again.

## 2. Four frozen cells, no new learning

V is NEXT53's normalization-matched value-only map; J its joint value/direction
map. Both retain the same source features, Qbar/rbar and attention22 matrices.
For A_q(X,d), recompute normalization and its differential at the chosen X:

    VV=A_q(XV,dV), VJ=A_q(XV,dJ)
    JV=A_q(XJ,dV), JJ=A_q(XJ,dJ).

All24 packets froze before native endpoint targets were opened.96 off-diagonal
vector evaluations plus96 diagonal checks, four directions each:768 scalar cells,
384 raw query contrasts,192 paired effects and96 contrast effects. Diagonals match
the prior frozen vectors/scalars exactly. Off-diagonal pairs are diagnostic
crossings; they are not the derivative of an eligible new fitted function.

The symmetric response allocation is

    s=((JV-VV)+(JJ-VJ))/2
    t=((VJ-VV)+(JJ-JV))/2
    i=JJ-JV-VJ+VV, delta=s+t=JJ-VV.

With e=VV-actual, error change is 2 mean(e delta)+mean(delta^2). Split its cross
term equally: C_s=mean(2es+s^2+st), C_t=mean(2et+t^2+st). Positive entries below
are -C_s/MSE_V and -C_t/MSE_V. Their sum is the total relative error reduction.
The interaction i is retained separately; it is already half allocated to s/t.

## 3. New8 error account

| Family / query | State contribution | Tangent contribution | Total reduction | State share of reduction |
|---|---:|---:|---:|---:|
| Pointwise / neuroticism | 15.9200% | 3.1796% | 19.0996% | 83.35% |
| Pointwise / confirmation-seeking | 11.9321% | 2.9059% | 14.8380% | 80.42% |
| Causal / neuroticism | 14.7979% | 2.9841% | 17.7820% | 83.22% |
| Causal / confirmation-seeking | 12.4654% | 2.6385% | 15.1039% | 82.53% |

Changed-state/old-tangent JV already captures most of the gain. Full cell results
(SSE divided by squared native target, not by a fitted baseline):

| Family / query | VV | VJ | JV | JJ |
|---|---:|---:|---:|---:|
| Pointwise / neuroticism | .478022 | .461602 | .400735 | .386722 |
| Pointwise / confirmation-seeking | .736096 | .713173 | .646770 | .626874 |
| Causal / neuroticism | .464702 | .449678 | .394814 | .382068 |
| Causal / confirmation-seeking | .728110 | .707451 | .635938 | .618137 |

The state contribution also has81.6-83.7% of the signed projection on the response
change. Interaction RMS is0.973-1.026% of change RMS. Covariance of s/t is positive
in all four aggregates; opposite signs occur in3-9 of32 paired actions. Both
linear error-alignment terms are negative and outweigh positive correction-energy
terms. This is error correction, not an improvement inferred only from correlation.

For query contrast, first subtract paired N/C predictions and targets, then
calculate errors. Pointwise state/tangent contributions are12.5745%/2.2365%, total
14.8110%; causal10.4988%/2.1621%, total12.6609%. Every new8 factor improves in
both per-query and contrast accounts. These dependent direction rows are not
treated as independent human samples, and no significance/promotion claim is added.

## 4. Source4 and interpretation

Source4 aggregate single-query improvements are only0.66-1.69%. F01/F02 worsen
under both queries/families while F03/F04 improve. Query contrast worsens1.44%
pointwise and0.90% causal; all four source contrast factors worsen. The state
contribution is negative for both source contrast aggregates. Source4 endpoint
weakness therefore is not repaired by the same crossing mechanism.

Keep these results alongside new8. Source4 uses three-person leave-one-out maps
and readers, new8 all four; cohort amplitudes also differ. The present data do
not separate those causes, and no new fit or source reweighting is authorized.

NEXT53 global state error increased slightly while endpoint error decreased.
Together with the crossing, this shows why Euclidean state fidelity and functional
fidelity differ: a small change at a task-sensitive location can improve the
reader despite slightly worse average state reconstruction. Do not simplify
this to a pure tangent correction, or to a demonstrated single attention-head cause.

See [theory](../../docs/GRAIN_RESPONSE_REPAIR_THEORY_V1.md) for the exact
state-dependent linear functional and the whole-function chain rule. It preserves
the same-material GRAIN/pathway interface, with no new clinical interpretation.

## 5. Reproducibility and artifacts

Preparation .224085s, primary crossing12.371018s, total12.595102s; one cached
replay12.330552s. The two local CPU prediction processes briefly overlapped;
these are observed wall costs, not isolated throughput benchmarks. Each stayed
within60s.0 fits/neural/API/Y calls, four denial probes per prediction, zero
unexpected target opens. No original asset or primary NEXT53 packet changed.

Independent audit:24 packets and12 CSV tables replay exactly, prior diagonal
error0; metric error<=1.377e-14, accounting error<=1.383e-13. The direct allocation
identity error is0 and aggregate MSE identity error<=1.735e-18.34 focused tests
pass,18 existing torch/Python deprecation warnings; Ruff passes. A new synthetic
test's hand-written expected squared error was corrected from3 to4 before the
primary freeze. No scientific outputs were changed or recomputed for that correction.

- Runner: `scripts/run_grain_response_repair.py` (`prepare`, `predict`, `evaluate`).
- Audit: `scripts/audit_grain_response_repair.py`; no fitting or neural execution.
- Private packet: `results/grain_vector_mechanisms/logic_response_repair_v1/`.
- Public [eight aggregate tables and audit](response_repair_tables/).
- [NEXT55 whole-path plan](../../docs/GRAIN_WHOLE_PATH_ASSEMBLY_PLAN_V1.md), planned.

No release/scorer/lockbox/MEPS changes, raw-text export or push. Whole-system goal
remains active. A frozen source map can replace this component's current
blocks17-21, while H16 still needs the original prefix; full-score assembly is next.
