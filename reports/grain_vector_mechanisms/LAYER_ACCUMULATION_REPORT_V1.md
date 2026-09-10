# NEXT42: where GRAIN directional responses become readable

2026-09-10. **Complete finite experiment; whole-system goal ACTIVE.**

## Finding and decision

The response is formed progressively rather than simply copied through the
last blocks. For the eight source-disjoint reader authors, fixed reading after
block23 already follows the full local action at r=.952 (N) and .823
(confirmation), but has about half its amplitude. Block25's MLP then adds the
largest positive projected update in both queries. This identifies a concrete
next mechanism to unpack: context-dependent gated feed-forward transformation.

Keep after26 as a useful reduced-suffix candidate. Do not promote after20:
its N response generalizes poorly despite a strong old-source correlation.
After23 is a useful intermediate signal, not yet an amplitude-faithful substitute.
No release scorer, original lockbox or human-criterion model is changed.

## Fixed comparison

Same4 old source authors and8 source-disjoint reader authors as NEXT41, two
queries and four frozen GRAIN BODY directions on the same H16 material. Source4
uses other-three-author means; new8 uses the fixed four-source means. Every
query's32 new8 actions come from8 people x4 directions, not32 independent people.
These materials were previously explored, not a fresh blind cohort.

Own-query source t; all exits fixed before outcomes:

| New8 endpoint | N action r | N SSE/zero | Confirmation action r | Confirmation SSE/zero |
|---|---:|---:|---:|---:|
| after20 | .091902 | .996928 | .343391 | .924480 |
| after23 | .951756 | .315353 | .822903 | .441064 |
| after26 | .996878 | .007006 | .988156 | .023199 |
| full post-attention27 | .998140 | .003893 | .996104 | .008028 |

SSE/zero = sum(predicted-actual)^2 / sum(actual^2), not centered R-squared.
The target is the model's H16 directional logit action, not a questionnaire or
the author's actual neuroticism. The source t is fixed; no current gradient or
score is used by the prediction process.

New8 after23 RMS is .375597 versus actual .765006 for N, and .202763 versus
.420743 for confirmation. Factor-centered r remains .949120/.814998; the
relationship is not just the mean difference between four factor IDs. Per-factor
after23 r ranges .877740-.983279 for N and .734493-.885503 for confirmation.
All four factor rows, sign counts and RMSE are saved, including worse comparisons.

Pooled-query sources give the same qualitative depth pattern: new8 after20
r=.073877/.328191, after23=.951893/.822109, after26=.996934/.988904,
full=.998482/.996723 (N/confirmation). No best-per-query composite was chosen.

Old4 after20 own-source N r=.876052 contrasts with new8 .091902, with small
predicted amplitude in both. Old4 N full SSE/zero .030744 is worse than after26
.020844, so depth is not universally monotonic in this source-approximation metric.

## What the signed updates show

Twenty-one attention/MLP updates sum to the full response. All earlier attention
and MLP effects are carried inside later tangents; the accounting is not a
collection of independent module ablations.

| New8 own-source stage | N signed projection share | Confirmation signed projection share |
|---|---:|---:|
| block22 MLP | .139543 | .126195 |
| block23 MLP | .154458 | .120434 |
| block25 MLP | .371310 | .508022 |
| block26 MLP | .121364 | .045025 |
| block27 attention | -.010424 | -.057825 |

Shares are sum(update * full_prediction) / sum(full_prediction^2); all21 signed
shares sum to one. They are not percentages of personality, variance explained,
or causal importance. The full signed table is provided, not only positive rows.

Immediately before block25 MLP, cumulative N r=.959689, SSE/zero=.274621;
after it, .995854/.021729. Confirmation changes from .903201/.314404 to
.984654/.028023. This directly locates a useful computation in the observed
trajectory. Source4 block25 projected shares are smaller (.144652/.185894),
so its contribution is material-dependent rather than a fixed universal fraction.

Attention can route and configure the input on which an MLP acts. A large MLP
increment does not establish an attention-free explanation. Neither does a
negative late increment mean the block is noise: it can correct earlier excess.

## Numerical and execution checks

- Code and primary analysis frozen in local commit `c30f792` before predictions;
  design was already fixed in `92539d0`.
-96 augmented suffix JVPs plus4 genuine after20 truncated JVPs;167.352s after
  loading (3.209s loading), within the300s cap. Zero new source backwards,
  current score calls, current backwards, model fits, labels or API calls.
- The truncated function reproduces all4 captured after20 directions exactly
  on the fixed source unit. Synthetic forbidden hooks also prevent later blocks
  and the output head from running. No per-exit wall-time speedup was benchmarked.
- All24 full response arrays match NEXT41 exactly. Independent sum of each
  exit's updates differs by at most9.168e-7; full relative error<=7.698e-8.
-24 prediction packets hash-verified;5 analysis CSVs byte-identical on replay;
  scalar contractions reproduce exactly. Independent metric recomputation differs
  by<=5.274e-16; signed share sum differs from1 by<=1.039e-8.
- Four deliberate forbidden-target read probes pass, zero extra denials.
-65 scoped tests pass, including new CPU/MPS telescoping and actual truncated
  execution.20 existing PyTorch/SWIG deprecation warnings, no full-suite claim.
- One display-only command requested wrong RMS column names; corrected to the
  saved schema. No experiment, model, selection or result was rerun for that typo.

After20 still needs original blocks0..16 to construct H16, then17..20 to
transport D. After26 skips only the last attention in this post-attention target
program. Native computation remains; this is not an independent text predictor.

## Next mechanism, not another accuracy grid

[NEXT43](../../docs/GRAIN_GATED_UPDATE_PLAN_V1.md) will separate block25's
SwiGLU input-variation and gate-variation terms, including RMS normalization,
then compare the current operator with fixed source operators. The block is
chosen from this discovery and its follow-up is explicitly exploratory.
This tests whether conditional multiplicative coefficients are needed beyond
the incoming contextual direction. No questionnaire tuning or naming is involved.

The connection to [GLU variants](https://arxiv.org/abs/2002.05202) is the actual
gated architecture, not a new invention of multiplication. The interpretive
precedent is [Geva et al.2021](https://aclanthology.org/2021.emnlp-main.446/):
feed-forward outputs can be viewed through learned activation/value combinations.
Our next test concerns derivatives of GRAIN-linked directions, not a claim that
a discovered unit is a psychological memory.

## Artifacts and reproduction

Public aggregate tables:
- [Every exit/cohort/source/query](layer_accumulation_tables_v1/comparison.csv)
- [Every factor](layer_accumulation_tables_v1/factor_actions.csv)
- [All signed stages](layer_accumulation_tables_v1/stage_summary.csv)

Private root: `results/grain_vector_mechanisms/logic_layer_accumulation_v1/`:
manifest, copied forward inputs/source models, prediction freeze, numerical
packets, `analysis/`, `analysis_replay/`, `replay_audit.json`. Never publish those
author-level inputs, targets or caches.

```bash
PY="../project persona/.venv/bin/python"
# Original collection; prepare requires a new, empty output directory.
"$PY" scripts/run_grain_layer_accumulation.py prepare
"$PY" scripts/run_grain_layer_accumulation.py predict
"$PY" scripts/run_grain_layer_accumulation.py evaluate
"$PY" scripts/run_grain_layer_accumulation.py evaluate --analysis-name analysis_replay
"$PY" scripts/audit_grain_layer_accumulation.py
"$PY" -m pytest -q tests/test_grain_layer_accumulation.py tests/test_grain_complete_transport.py
```

Read [the signed-update formulas](../../docs/GRAIN_LAYER_ACCUMULATION_THEORY_V1.md)
and [pre-outcome protocol](../../docs/GRAIN_LAYER_ACCUMULATION_PLAN_V1.md).
