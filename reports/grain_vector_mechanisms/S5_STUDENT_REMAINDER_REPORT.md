# S5: functional projection and distribution students

2026-09-08. Complete, two finite cached-data experiments, no neural calls.

## Finding and decision

Teacher-only T/F compression remains successful across the new candidates.
For neuroticism, improving teacher-score fidelity does not recover the teacher's
external association. PLS and nonlinear comment-distribution features therefore
do not resolve the remainder in this experiment. Stop enlarging these ordinary
student families; directly test query-conditioned token configurations next.

This is a useful narrowing: the failed S4 transfer was not rescued merely by
changing PCA to response coordinates, removing dimensionality reduction, using
broader regularization, or replacing first-moment pooling with a finite kernel
mean. It is not an information-theoretic impossibility claim about E5.

## Design

Existing, exposed D1 source authors: T/F n=2,328; neuroticism n=213. Same frozen
cue-mask comments, three views of12 comments, same `s4_outer` five author folds.
Three inner folds select teacher squared error only. Individual psychological
labels are joined only after all candidate OOF predictions are saved and hashed.
These are exploratory connections on historically exposed outcomes, not new
lockbox tests. Existing teacher and label-trained E5 controls use identical users.

S5A: train-standardized E5 means; PCA64/Ridge, full Ridge, PLS/Ridge.
PLS ranks2/4/8/16/32; alpha1/10/100. Directions are orthonormal; coordinates are
not subsequently whitened. `inner_chosen` selects the complete method in each
outer fold using only inner teacher MSE. The PLS implementation agrees with
the standard scikit-learn PLS subspace on a synthetic test.

S5B: fixed256-dimensional Gaussian random Fourier maps. Compare linear mean,
phi(mean h), mean(phi(h)), and equal-trace linear-plus-distribution blocks.
Bandwidth uses training median distances only. Centering and scalar kernel
trace normalization are training-only, not per-frequency whitening. Ridge
minimizes average loss plus lambda times squared coefficient norm, with
lambda=.001/.01/.1/1/10. This broader regularization is also tested on the linear
control because S5A frequently selected its largest alpha.

## Complete results

| Query | S5A method | Teacher r | Teacher MSE | External r | External AUC |
|---|---|---:|---:|---:|---:|
| T/F | PCA64 | .6668 | .007827 | .3632 | .7173 |
| T/F | Full Ridge | .6205 | .009259 | .3144 | .6938 |
| T/F | PLS | .6713 | .007755 | .3442 | .7070 |
| T/F | Inner chosen | .6675 | .007815 | .3577 | .7150 |
| N | PCA64 | .4614 | .001916 | .0673 | n/a |
| N | Full Ridge | .3760 | .002369 | .0494 | n/a |
| N | PLS | .5059 | .001750 | .0467 | n/a |
| N | Inner chosen | .5059 | .001750 | .0467 | n/a |

PLS was selected in all five N folds, normally at rank2; T/F selected PCA in
four folds and PLS in one. External results did not determine those choices.

| Query | S5B method | Teacher r | Teacher MSE | External r | External AUC |
|---|---|---:|---:|---:|---:|
| T/F | Linear mean | .6799 | .007573 | .3573 | .7158 |
| T/F | Mean RFF | .6389 | .008392 | .3524 | .7112 |
| T/F | Bag RFF | .6464 | .008198 | .3498 | .7111 |
| T/F | Mean + bag | .6758 | .007651 | .3570 | .7154 |
| T/F | Inner chosen | .6799 | .007573 | .3573 | .7158 |
| N | Linear mean | .5028 | .001712 | .0593 | n/a |
| N | Mean RFF | .4452 | .001841 | .0709 | n/a |
| N | Bag RFF | .5032 | .001711 | .0682 | n/a |
| N | Mean + bag | .5096 | .001695 | .0663 | n/a |
| N | Inner chosen | .5079 | .001699 | .0572 | n/a |

Controls: native cue-mask teacher T/F AUC=.6591, N r=.1765. Existing supervised
E5 T/F AUC=.7537; none of these teacher-only students replaces that stronger
label-trained result. T/F S5B inner-chosen minus native AUC=.0567, paired-author
95% CI [.0379,.0760]. N corresponding delta r=-.1193, CI[-.2375,.0136]. These CIs
condition on the frozen OOF estimates; they do not repeat the full training.

## Mathematical interpretation

PLS targets a response-related subspace rather than a large-variance subspace;
kernel means represent more than an original embedding mean. These are genuinely
different operations, but neither guarantees that the additional teacher detail
is the part associated with a psychological label.

For N in S5A, the exact score-scale covariance split is:

| Student | Cov(Y,teacher) | Cov(Y,student) | Cov(Y,remainder) | Remainder-label r |
|---|---:|---:|---:|---:|
| PCA64 | .002663 | .000693 | .001969 | .1428 |
| PLS | .002663 | .000456 | .002206 | .1674 |

The residual correlation alone is not an amount of unexplained personality.
The covariance identity holds to numerical precision and keeps that distinction
explicit. See the [active mathematical plan and source map](../../docs/GRAIN_ACTIVE_RESEARCH_GOAL_V3.md)
for Cook/Helland/Su, Fukumizu/Bach/Jordan, Szabo et al., and Rahimi/Recht.

## Resource value and reproducibility

S5A total15.37seconds; S5B21.55seconds, including reporting; zero new LLM calls.
The original S4 mean PCA and S5A rerun share every outer fold. N predictions agree
to floating precision; T/F differs by at most .001304 in individual score because
the randomized PCA projection is applied explicitly rather than using its
approximate training `fit_transform` scores. Headline AUC differs by .000015.
Do not describe this as a bit-identical S4 replay.

Private local output directories under `results/grain_vector_mechanisms/`:

- `supervised_student_s5_r1/`
- `distribution_student_s5b_r1/`

Each contains the source/script hashes, frozen OOF scores, selected inner models,
portable coefficient artifacts, metrics, covariance accounting, paired intervals
and completion summary. No original text, author IDs or individual scores are
included in tracked reports. No bridge/MEPS labels, API, or LoRA training used.

From the independent GRAIN repo, with the parent project's Python environment:

```bash
python scripts/run_grain_supervised_student.py \
  --connection results/grain_vector_mechanisms/response_student_source_connection_r2 \
  --panel results/grain_vector_mechanisms/multiquery_sensitivity_s1_r1 \
  --output-dir results/grain_vector_mechanisms/supervised_student_s5_replay
python scripts/run_grain_distribution_student.py \
  --connection results/grain_vector_mechanisms/response_student_source_connection_r2 \
  --panel results/grain_vector_mechanisms/multiquery_sensitivity_s1_r1 \
  --output-dir results/grain_vector_mechanisms/distribution_student_s5b_replay
```

Use a new output directory; never overwrite frozen OOF artifacts. S6's mean-null
token-state experiment is the next mechanism question, not another student grid.
