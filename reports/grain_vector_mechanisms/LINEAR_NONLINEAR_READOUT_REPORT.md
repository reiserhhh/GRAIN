# GRAIN M22/M22B: where nonlinear readouts help, and where representation matters

Started2026-09-08; closed2026-09-09 JST. COMPLETE. No background collector, new neural calls, API calls,
questionnaire fitting, bridge/MEPS outcome access or lockbox opening.

## Findings and mainline decision

**The most useful result is a representation change, not a universal nonlinear
reader.** On the same213 N authors and teacher target, query-conditioned prefix
vectors with Ridge retain substantially more source-label association than E5
means with Ridge. T/F's existing linear success remains. Nonlinear tree readers
are useful comparison tools, but do not replace the linear baseline in this packet.

The mainline stays real text -> conditional vector evidence -> trait judgment.
Next investigate how query/context makes evidence linearly accessible, rather
than extend tree grids or treat teacher-score fit as the research endpoint.

## M22: same-vector reader comparison

Training supervision is ONLY cached LLM scores. All three views are averaged per
author. Same5 outer author folds (`s4_outer`),3 inner author folds for teacher MSE,
training-only scaling/PCA64. Fixed alternatives: Ridge, boosted stumps, depth3
trees, Ridge+stump residual and Ridge+tree residual. The inner-selected reader is
reported separately from each fixed arm. No selection uses external labels.

| Source-all E5 mean, query | Authors | Reader | Teacher r | External association |
|---|---:|---|---:|---:|
| T/F | 2328 | Ridge | .6668 | AUC .7173 |
| T/F | 2328 | Stumps | .6159 | AUC .6951 |
| T/F | 2328 | Trees | .6489 | AUC .7120 |
| T/F | 2328 | Ridge+trees | .6656 | AUC .7159 |
| N | 213 | Ridge | .4614 | r .0673 |
| N | 213 | Stumps | .4459 | r .0834 |
| N | 213 | Trees | .4581 | r .1357 |
| N | 213 | Ridge+trees | .4538 | r .0433 |

T/F trees improve over stumps (teacher MSE gain .000727, paired95%CI
[.000538,.000917],5/5 positive folds; external AUC delta+.0169 [.0090,.0244]).
They do not improve over Ridge. This compares finite estimators: interaction trees
may approximate relationships already accessible linearly, so this is not proof
of an additional independent nonlinear personality circuit.

N mean-tree external delta versus Ridge is+.0684 [-.0298,.1634]. Preserve it as a
local candidate, not the basis for further tuning. Mean-plus-query pooling gives
N inner-selected teacher r=.4749 and external r=.0908, without a decisive paired
gain. All candidates and negative controls remain in the complete tables.

Permutation control uses one fixed permutation of teacher targets, not labels.
Held-author teacher r for fitted readers is -.0085..+.0201 (T/F) and
.0112..+.0442 (N). This checks target leakage; it is not a permutation p-value or
a claim that external associations from one randomly trained reader must be zero.

## M22: matched representation comparison

The M10 prefix cohort uses a separately recorded cached teacher. Comparisons below
use exactly the same authors, target, folds, head class and source labels. Prefix
features are three-view means of the P query-token mean and Q final-token state;
they are not full token matrices or a replacement for the frozen P/Q API.

| Query | Authors | Representation + Ridge | Teacher r | External association |
|---|---:|---|---:|---:|
| T/F | 64 | E5 mean | .5634 | AUC .6727 |
| T/F | 64 | Prefix P/Q mean | .6005 | AUC .5926 |
| N | 213 | E5 mean | .5726 | r .0400 |
| N | 213 | Prefix P/Q mean | .7134 | r .1629 |

N external delta is **+.12294 [.02492,.22039]**; paired teacher MSE reduction is
.0002319 [.0001026,.0003948], about27.8% of the E5 Ridge MSE. The source-label
covariance retained by the reader grows from about15.4% to68.9%, with the remaining
label correlation falling from .1786 to .0772. These are signed covariance shares,
not explained-variance percentages. No human outcome was used to fit either head.

The unchanged M10 reference-matrix readout still has N teacher r=.8053, source
r=.1008; its two-author construction differs from these trained CV readers. It
remains a valid previous result, not an identically trained competitor.

For T/F, the prefix-versus-E5 external AUC delta is-.0801 [-.1936,.0277]. Thus the
N finding does not prescribe a universal representation for every trait. These
are previously exposed discovery authors; bootstrap intervals condition on the
frozen OOF predictions and are not repeated-training or multiplicity-adjusted CIs.

## M22B: return to the evidence-to-judgment mechanism

The frozen source-all E5 heads were applied to S1 authors excluded from every
training fold. No new fit or neural call. Level/E1/E2 use24 authors, averaging two
views; joint effects use eight authors with only their original view0 available.
Each head prediction averages the five already-trained outer models.

| E5 mean reader | Query | Level r | E1 change r | E2 change r |
|---|---|---:|---:|---:|
| Ridge | T/F | .4593 | .2799 | .3444 |
| Trees | T/F | .4940 | .1405 | .2595 |
| Ridge | N | .5115 | -.0789 | .0861 |
| Trees | N | .5799 | -.0555 | -.0895 |

T/F Ridge E2 change r=.3444 [.0898,.6171]; the mean-plus-query Ridge gives
.3551 [.0851,.6329]. These retain useful evidence-response links. Higher N level
fit in the small transfer panel does not carry over to E1/E2 response. Hence
score-level distillation alone is not an adequate substitute for a response model.
We did not choose a new head from this panel's outcomes.

The natural mixed effect is decomposed exactly:

```
I_natural = f(x12) - f(x1) - f(x2) + f(x0)
I_reader  = f(x1+x2-x0) - f(x1) - f(x2) + f(x0)
I_feature = f(x12) - f(x1+x2-x0)
I_natural = I_reader + I_feature
```

Maximum identity error1.11e-16. Ridge's reader-interaction RMS is approximately
1e-16, while its mean-feature natural interaction is nonzero (joint T/F RMS .00936).
Thus even a linear head can inherit nonlinear natural-text effects from changing
pooling denominators and representations. Stump additivity is in PCA coordinates:
two arbitrary input displacements may both move the same PCA coordinate and then
produce curvature, even without cross-coordinate interaction. No observed tree
gain is labeled a neural-path discovery by itself.

## What changes in the theory

Use the composition, not a forced two-circuit story:

```
h_q = T_theta(text, context, query)
s_hat_q = ell_q(h_q) + g_q(h_q)
```

`T_theta` can organize evidence nonlinearly so that a subsequent `ell_q` reads it
linearly. `g_q` is an optional approximation remainder relative to the fixed
representation and fitting rule, not all leftover personality information.
The retained labels in the N prefix readout make this distinction useful to the
real-author mainline; a larger nonlinear head on E5 means was not the key change.

M23 should explain which query-conditioned coordinates account for that gain and
test varying coefficients/conditional interactions on the existing paired states.
Then compare predicted finite effects with actual LLM edits, borrowing the
interaction analysis in Integrated Hessians. Avoid new general-purpose grid sweeps.
The literature and deferred tasks are recorded in
[the research plan](../../docs/GRAIN_LINEAR_NONLINEAR_READOUT_PLAN_V1.md).

## Execution, reproducibility and verification

- M22:118.67s total, cached CPU; M22B:11.21s, zero-fit CPU. LightGBM4.7.0 installed
  only in the research environment; release requirements/weights unchanged.
- Initial starts retained: dataframe `query` attribute collision, macOS AppleDouble
  sidecar parsing, and an OpenMP collision between PyTorch and Homebrew libomp.
  No trial result existed before these failures. Pure-CPU helpers remove neural
  imports; filenames explicitly exclude sidecars. M22B's first analysis correctly
  stopped at the unavailable second joint view; v2 uses the actual frozen design.
- Old S4 author membership/targets/folds agree for5082 author-family rows. Three
  query/family cells reproduce scores within6.11e-16. T/F mean differs at most
  .001303 because old randomized PCA used `fit_transform` for training while the
  new shared geometry uses `transform` consistently; identical bases and this
  sole change reproduce the difference. Teacher r changes only .00000568.
- 12 new tests passed in a CPU-only process.761 existing GRAIN tests passed
  separately, with1908 non-GRAIN/new tests deselected and two old SWIG warnings.
  Keep the CPU tests separate on macOS to avoid mixing neural/tree OpenMP runtimes.
- Post-fit audit:60009 unique OOF rows,277 matched representation authors,
  identical targets/folds, additive readout error2.05e-16, covariance error3.58e-18.
  No final artifact was selected by external labels. Ruff/diff checks pass.

Private artifacts are all below `results/grain_vector_mechanisms/`:

- `nonlinear_readout_m22_v4/`: manifest, saved fold models, private OOF before label
  join, selected configurations, metrics, paired comparisons and covariance ledger.
- `nonlinear_edit_connection_m22b_v2/`: frozen-model natural edits and exact
  interaction accounting, private view/author predictions, aggregate metrics.
- `nonlinear_readout_m22_audit_v1/`: representation comparison CIs, S4 replay and
  independently recomputed arithmetic. Raw text, author IDs and models are ignored.

Run from the independent repo using the research Python environment:

```bash
python -m pip install -r requirements-readout-research.txt
python -m pytest tests/test_grain_nonlinear_readout.py -q
python scripts/run_grain_nonlinear_readout.py --output-dir results/grain_vector_mechanisms/m22_replay
python scripts/analyze_grain_nonlinear_edits.py --readout-dir results/grain_vector_mechanisms/m22_replay --output-dir results/grain_vector_mechanisms/m22b_replay
python scripts/audit_grain_nonlinear_packet.py --readout-dir results/grain_vector_mechanisms/m22_replay --edits-dir results/grain_vector_mechanisms/m22b_replay --output-dir results/grain_vector_mechanisms/m22_audit_replay
```

Output directories must be new. Existing authorized local caches are required;
do not restart old collectors just to satisfy a missing file in another checkout.
