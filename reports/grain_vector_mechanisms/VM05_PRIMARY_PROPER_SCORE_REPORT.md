# VM05: near-majority classification is not zero probabilistic signal

2026-09-07, round32. **COMPLETE post-hoc diagnostic of36 primary MBTI heads;
not completion of VM05 or a new selected scorer.** All three arms and all
three primary observers are included. Source choices, thresholds and coefficients
are unchanged; no new source label table, bridge outcome or LLM forward is used.

## Question and method

The primary AUCs quantify ordering, not calibrated probabilities. We ask whether
the same saved OOF predictions improve proper prediction losses over the class
proportion available on the training side. For author i in official fold f:

```
pi_f = mean(y_j for j outside f)
log_loss_i(p) = -y_i log(p_i) - (1-y_i) log(1-p_i)
gain = mean(loss_i(pi_f) - loss_i(p))
```

Positive gain means better probabilistic prediction on the saved held-out
authors. Brier loss is `(y_i-p_i)^2`. Fold priors exclude the held-out fold;
all36 heads use the same6,294 reference-free MBTI-only source authors, their
task-specific official folds and retain-thread training. This baseline formula
would not be valid unchanged for a purge/count head with different training
membership. There is no added probability calibration or threshold optimization.

All observers within a task/arm share1,000 author-within-fold bootstrap draws,
seed731. These are pointwise intervals conditional on saved OOF predictions and
fold priors, not refit uncertainty, simultaneous inference or independent mask
replication. Probability clipping at1e-15 is declared; no prediction required it.
The diagnostic was added after primary AUC results, not preregistered as a new
primary endpoint. AUC-based inner selection is not changed to log-loss selection.

## Results

Pooled raw log loss, in nats per author; smaller is better:

| Axis | Fold-prior baseline | BGE | E5 | Qwen10 | Qwen10 gain and pointwise95% CI |
|---|---:|---:|---:|---:|---|
| EI |.510918|.496109|.491268|.494408|+.016511 [.011768,.021535]|
| SN |.346437|.332597|.331826|.334991|+.011446 [.007139,.015536]|
| TF |.649744|.557920|.556814|.567570|+.082174 [.071990,.091197]|
| JP |.681074|.661557|.660892|.663375|+.017699 [.012808,.023016]|

All36 conditions improve both log loss and Brier over the training-prior
baseline, with positive pointwise interval lower bounds for both pooled and
mean-official-fold estimands:144 rows, not144 independent replications.
TF has the largest descriptive gain. This does not establish a superior
observer, probability calibration for each individual, or transport performance.

For raw EI, the observed E rate is.207658, while BGE/E5/Qwen predict E at.003495/
.002701/.002542 of authors using the unchanged0.5 threshold. For SN, the N rate
is.890054 and predicted positive rates are1/1/.999682. Nevertheless each head
improves the prior's probability losses. Thus almost-all-majority decisions
can coexist with real, modest predictive differentiation within the majority
side of the threshold. Mean probability close to prevalence is not sufficient
evidence of individual or subgroup calibration.

The earlier generative200-iteration collapse is a different model/readout and
cohort. These results neither retroactively validate that experiment nor
prove that every near-chance balanced-accuracy result contains usable signal.

## Implication for the mechanism program

Keep three questions separate: how well a head orders authors, how well its
probabilities predict labels, and whether the LLM itself uses the corresponding
information. The first two now have positive but axis-dependent source evidence;
the third still needs the declared task/history/direct and functional controls.
Do not interpret this loss reduction as bits of human personality or mutual
information. A finite fitted predictor and a fold-trained constant baseline
are not the unknown true conditional distribution and true marginal entropy.
See [readability versus information](../../docs/GRAIN_VECTOR_READABILITY_THEORY_NOTE_V1.md).

## Reproduction and checks

Entrypoint: `scripts/analyze_grain_source_proper_scores.py`, with `--primary-dir`,
`--audit-dir`, `--resource-dir` and a fresh `--output-dir`. It validates the
complete45-head/225-replay primary parent before processing all36 binary heads.
The9 Big5 primary heads are not assigned binary log loss. No new head is fitted;
only20 distinct training-fold class proportions are derived from saved source
outcomes, reused across equivalent arms/observers.

Packet: `results/grain_vector_mechanisms/vm05_primary_proper_scores_r1`.
Identity: `d4bd4024152e7843a33b2c8b9a900c1d78946d87042b0768e863539f01071f1d`.
`proper_scores.csv` SHA256:
`6655ebe9295ee54cb19d6851c8770aee3c54e63b340270b51f62333ea934a2cb`.
`ranking_diagnostics.csv` SHA256:
`16c6c14c39298e77658b5b997e36022c00833db2b2811800e33b296cce403efa`.

Three focused tests independently reproduce scalar scores and bootstrap
endpoints with sklearn, check unchanged held-out priors after changing that
fold's labels, identical-prediction zero intervals, invalid probability/support
rejection and clipping counts. The combined proper-score/layer/projection
suite passed10 tests; this is not a full-repository test run.
An additional independent real-data calculation loaded all36 saved prediction
CSVs directly and recomputed AUC and144 loss/prior/gain rows with sklearn.
Maximum absolute discrepancy was3.45592e-14. This real-data audit covers scalar
estimates; bootstrap endpoints were independently checked in the synthetic tests,
not recomputed a second time for all real-data draws.
