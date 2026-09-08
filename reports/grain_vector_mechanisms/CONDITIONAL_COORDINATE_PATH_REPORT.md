# GRAIN M23-M24B: readable scores and operative configuration paths

2026-09-09. All four bounded packets COMPLETE. The five-piece research goal
remains ACTIVE. No background neural collector remains from this packet.

## Main finding

**The old frozen functional matrix reads the newly isolated query-configuration
effect at T/F r=.9533 and N r=.9615.** A source-score-trained linear reader is
not interchangeable with that functional matrix, despite both being affine.
The useful research distinction is now the fitting/construction target and
representation, not a linear-versus-nonlinear model competition.

This connects earlier GRAIN mechanisms rather than replacing them. Query-token
configuration is an operative carrier on this panel. Pooled means also contain
readable author information, but can track scores through shared covariation
without explicitly responding to a mean-preserving configuration intervention.

## M23: where the source information can be read

Same M10 teacher, author membership and five folds; 64 T/F and213 N authors,
three views each. Six representations, Ridge only, with each fold's previously
chosen M22 P/Q Ridge alpha reused for every arm. PCA/scales fit on training users.
Human labels joined after1,662 OOF rows were saved and hashed; no label tuning.

| Representation | T/F teacher r | T/F source AUC | N teacher r | N source r |
|---|---:|---:|---:|---:|
| P token mean | .6152 | .5796 | .6137 | .1682 |
| Q final state | .5227 | .5996 | .6787 | .1289 |
| P mean + Q | .6005 | .5926 | .7134 | .1629 |
| P configuration | .6986 | .6276 | .5937 | .1763 |
| Full P + Q | .6889 | .6266 | .5818 | .1743 |
| P configuration + Q | .6984 | .6296 | .5937 | .1772 |

T/F configuration improves teacher MSE over P mean/Q by .002280
[.000875,.003636]. N P mean/Q retains the strongest teacher fidelity here;
small source-r differences are not used to choose a new head. Full-P feature
blocks contain many more coordinates than Q; the PCA/regularization geometry
therefore changes. These comparisons do not prove that Q is intrinsically useless.

P configuration means the zero-token-mean contrast C=P-1*m, not removal of
psychological noise. All components remain in the exact full-P/Q accounting:

```
prediction = training_reference + P_mean_part + P_configuration_part + Q_part
```

The full reader's configuration has most signed source covariance in this fit,
but terms are correlated and are not independent explained-variance percentages.
Across the three existing material views, score correlations range roughly
.37-.60; this is material-view association, not an independent temporal retest.

Historical M22 means replay within5.55e-17. Canonical float64 view/token averaging
changes the refitted P-mean/Q score by at most1.63e-7, not the scientific finding.

## M23B: score distillation only partly tracks actual evidence edits

All six frozen source readers were averaged across five folds and applied to
twelve different M8 authors, same observer and question tokens, one E1 deletion
per query/person. No fit, labels, neural calls or family selection.

N P-mean response r=.3025 [.0501,.6604]; P-mean/Q=.2529; configuration=.1138.
T/F configuration response r=.4754; P-mean/Q=.3057. Response magnitude/sign
remain imperfect: for example T/F configuration MSE/zero=2.716 and sign match
4/12. These numbers motivated actual suffix intervention, not amplitude tuning.

The same twelve authors give native T/F-N deletion-response r=.3052. P-mean/Q
readers give .6358, configuration readers .3819: high predicted cross-query
association alone is not evidence that the corresponding actual path is recovered.

## M24: configuration carries an actual suffix effect

Original evidence-region states E and layout stay fixed. Decompose the cached
natural P/Q displacement into P mean M, zero-mean configuration C, and final Q:

```
dP = dM + dC; mean_token(dC)=0
f_abc = F(E_full, P_full+a*dM+b*dC, Q_full+c*dQ)
```

Eight endpoints per author/query give three main, three pairwise and one triple
anchored term. Their sum equals f111-f000. This is conditional P/Q action; the
whole natural deletion additionally changes E, layout and tokenization.

| Effect comparison,12 authors/query | T/F r | N r | T/F MSE/zero | N MSE/zero |
|---|---:|---:|---:|---:|
| Configuration main vs conditional P/Q total | .9247 | .9573 | .1722 | .2759 |
| Conditional P/Q total vs natural deletion | .9147 | .8429 | .1719 | .3172 |

Configuration changes alone agree with the conditional total's direction for
12/12 T/F and11/12 N authors. This is a concrete evidence-change -> configuration
-> model-response connection, not a questionnaire correlation of .95.

Native RMS effects (score units) show why this is not an interaction-only story:

| Query | Mean | Configuration | Final Q | Largest pair term | Triple |
|---|---:|---:|---:|---:|---:|
| T/F | .01563 | .07555 | .02498 | .00531 | .00034 |
| N | .00466 | .02028 | .01235 | .00274 | .00014 |

These amplitudes are not additive shares. Main terms can cancel. Interactions
exist, but on these endpoints their amplitudes are smaller than the leading
configuration effect; no need to reopen a nonlinear-head grid to study the path.

## M24B: earlier functional coefficients recover the isolated mechanism

The two-author P/Q reference matrix was frozen before these twelve evaluation
authors. Apply it to the same M/C/Q cube, with no new fit or neural call:

| Frozen functional matrix vs actual effect | T/F r | N r | T/F MSE/zero | N MSE/zero |
|---|---:|---:|---:|---:|
| Configuration effect | .9533 | .9615 | .1035 | .0930 |
| Mean effect | .9146 | .9341 | .2348 | .2185 |
| Final Q effect | .8678 | .8252 | .2387 | .2214 |
| Conditional P/Q total | .9193 | .9392 | .1401 | .3250 |

Configuration-r bootstrap intervals are [.9172,.9919] and [.9261,.9919].
Both conditional-total directions match12/12. Natural-change r=.8857/.6814
reproduces the earlier M8 result; it is not a newly independent success. The
new finding is its decomposition into separately observed operative components.

By contrast, source-score-trained full-P/Q Ridge reads the configuration effect
at r=.4840/.3168. P-mean/Q Ridge is exactly blind to dC (response around1e-16).
Thus similar score accessibility does not imply equal intervention behavior.

Across queries on the same authors, actual configuration responses correlate
.4688 (frozen matrix .3760), whereas actual mean responses correlate-.0182 and
final responses .0524. This points to a candidate shared configuration function;
it does not identify a universal shared personality factor from12 cases.

## Mathematical update and next construction

Keep a linear functional readout, but preserve its spatial structure:

```
s_hat = b_ref + <G_q,P-P_ref> + g_q'(Q-Q_ref)
      = b_ref + (sum_t G_qt)'(m-m_ref)
        + <G_q,C-C_ref> + g_q'(Q-Q_ref)
```

A mean-only reader sets the coefficient on C to zero by construction. It can
still predict cross-person scores if m and C co-vary in the source population.
The effect-defined G_q avoids conflating population proxy with operative response.
Construction target matters at least as much as estimator nonlinearity.

**Next M25:** use the already isolated configuration effects and frozen G_q to
build a same-material shared/private functional response map and fixed-reference
author profiles. Separate group score position from response configuration. Use
the existing twelve-author panel first, then acquire only the comparison whose
answer changes the construction. Do not repeat score fitting or numerical
integration of the already functioning P matrix.

The five-piece goal stays active: local carrier and effect links advanced;
group-referenced shared/private assembly and compact encoder-to-function mapping
are the next pieces. E5 vs Qwen still does not isolate the effect of a query from
the encoder change. Same-model query contrast is a later targeted comparison,
not a reason to erase the configuration result or restart all earlier experiments.

## Reproduction and verification

- M23 cached CPU:277 authors,1,662 OOF scores,4,986 view scores; six arms, no grid.
- M23B:3.28seconds, zero fit/neural. M24:192 forwards/129.12seconds, zero gradients.
- M24B:zero new fit/neural; existing two-author reference remains unchanged.
- Cube accounting error1.11e-16; affine interactions at most3.33e-16.
- Fifteen new numerical tests pass. Full GRAIN selection:776 passed,1908
  deselected, two existing SWIG warnings. CPU LightGBM tests run separately.
- No release-model changes, no new human-label training, API, lockbox or old queue.
- M24 acquisition used the pre-lint script/hash retained in its manifest; final
  code changes only sort imports, replace equivalent map syntax, make pandas axis
  explicit and strengthen future resume hash checking. No neural data rerun.

Private result directories under `results/grain_vector_mechanisms/`:
`conditional_coordinates_m23_v1`, `conditional_edits_m23b_v1`,
`query_coordinate_cube_m24_v1`, `cube_reference_m24b_v1`.

Research Python is the sibling Persona `.venv/bin/python`. From this repo:

```bash
python -m pytest tests/test_grain_conditional_coordinates.py tests/test_grain_coordinate_cube.py -q
python scripts/run_grain_conditional_coordinates.py --output-dir results/grain_vector_mechanisms/m23_replay
python scripts/analyze_grain_conditional_edits.py --readout-dir results/grain_vector_mechanisms/m23_replay --output-dir results/grain_vector_mechanisms/m23b_replay
python scripts/run_grain_coordinate_cube.py prepare --readout-dir results/grain_vector_mechanisms/m23_replay --output-dir results/grain_vector_mechanisms/m24_new
python scripts/run_grain_coordinate_cube.py run --output-dir results/grain_vector_mechanisms/m24_new
python scripts/run_grain_coordinate_cube.py analyze --output-dir results/grain_vector_mechanisms/m24_new
python scripts/analyze_grain_cube_reference.py --cube-dir results/grain_vector_mechanisms/m24_new --output-dir results/grain_vector_mechanisms/m24b_new
```

Outputs must be new directories. Running M24 `run` acquires new neural endpoints;
do not execute it merely to reread existing results. The other replay paths use
authorized local caches. All individual artifacts remain ignored/private.
