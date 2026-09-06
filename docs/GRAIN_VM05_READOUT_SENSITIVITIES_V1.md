# VM05 cached readout sensitivities and missing context contrast

2026-09-06. This specifies the already planned equal-view, post-question and
cross-view sensitivities before full-source neural labels/readouts are analyzed.
It does not change Windows collection, the primary block10 body candidate,
reference geometry, source cohorts, statistical heads or the released scorer.

## Observable objects

Each of three disjoint12-comment views has a pre-question body mean mu_v and
population standard deviation s_v at each frozen Qwen site. Let n_v be its actual
body-token count. For weights w_v summing to one, define

```
m(w) = sum_v w_v mu_v
W(w) = sum_v w_v s_v^2
B(w) = sum_v w_v (mu_v - m(w))^2
T(w) = W(w) + B(w)
```

Operations on variance vectors are elementwise. This is the ordinary law of
total variance, not a new personality formula. Counts n_v/N produce the existing
token-pooled candidate [m,sqrt(T)]. Weights1/3 produce an equal-view sensitivity
with the same2d dimension. Neither equals the unweighted mean of standard
deviations. The decomposition [m,sqrt(W),sqrt(B)] is a separate3d sensitivity;
its higher input dimension and identical downstream PCA cap must be reported.
Three views are still one author; B is dispersion of these texts, not established
trait variability, reliability or within-person temporal change.

At the same site, d_v^behavior and d_v^disposition are the pre-answer final
positions for the frozen questions about concrete social initiation versus a
general tendency to initiate. Report each task as [mean_v d_v,sd_v d_v], and the
paired task contrast [mean_v delta_v,sd_v delta_v], with
delta_v=d_v^disposition-d_v^behavior. All are2d and use equal view weights.
The contrast includes wording, question-token length/position and task demands.
It is not automatically a personality direction or proof of model reasoning.
Body states precede the question and are validated task-invariant in the frozen
collector; no task-specific body features are invented after collection.

The two stored Yes-minus-No logit margins can supply a fixed four-scalar
diagnostic [mean_behavior,sd_behavior,mean_disposition,sd_disposition]. These are
neither calibrated probabilities nor direct Big5/MBTI scores. The existing
behavioral-code failures remain relevant; statistical readability cannot repair
them retroactively. No new answers, prompts, layers or trained LLM weights here.

## Readout family and controls

- Sites: block04,block10,block16,final_norm, the original four cached sites.
- Main existing readout: body_token. New declared sensitivities: body_equal_view,
  body_variance_parts, decision_behavior, decision_disposition, decision_task_delta.
- Logit-margin diagnostic is site-independent and must not be duplicated as four
  different models. Preserve its scope as the fixed social-initiation questions.
- Each feature uses all36 comments. All raw/cue/placebo arms remain available;
  author subsets are never chosen by successful masks or observed label scores.
- Full-source refits and exact same-ready comparisons keep source-exclusive
  official outer folds/inner3seed731 and the existing scaler/PCA/head grids.
  New features are sensitivities, not a larger pool from which to select the
  best layer or direction using outer/bridge outcomes.
- Qualification first uses the original12 technical authors without target values.
  Reconstruct token-pooling and variance identities, shape/count/task completeness,
  task-delta direction and aliases; independently replay saved derived arrays.
  This is instrumentation acceptance only. Actual neural source comparisons
  still require the full72 verified shards and complete author union.

## Genuine remaining gap: internal history effects

The full-source collector saved complete12-comment prefixes and two questions.
It did NOT encode the same current excerpt under own-history/other-history/none.
Consequently neither a task difference nor between-view variance is the required
internal context counterfactual. Existing smaller VM03/VM06 context packets are
functional evidence with their own samples; they are not a full-source VM05
context-effect representation or a source-frozen bridge head.

The original requirement for an internal-context-effect/GRAIN+context-effect
external comparison remains open. It needs a separately frozen, source-safe
same-current design and bounded additional forwards after the running collection,
or an explicit evidence-based limitation if matched material is insufficient.
Do not pretend existing cached moments identify unobserved history contrasts.
Direct LLM Big5/MBTI scoring is also not supplied by these two Yes/No questions.

## Deliverables

`external_readouts.py` defines the pure transformations. A label-free qualification
script writes a source-bound private derived-feature NPZ, aggregate numerical
diagnostics, protocol/source snapshots and checksums. Tests and an independent
reconstruction validate exact axes, weights and task pairing. Later full-source
execution must expose the readout identity, not reuse a body-only feature name.

## Implemented qualification and execution path

The existing12-author technical packet passed all75 feature transformations and
an independent reconstruction of900 author-channel rows. These operators are
implemented in`external_readouts.py`; the verified full-cache loader is
`external_readout_cache.py`. The same-ready runner now has an explicit`--readout`
argument. It leaves default body_token untouched and refuses incomplete observer
queues before any target conversion or fitting. It still uses the same regularized
head, not a new representation learner or calibrated direct LLM scorer.

Full-source secondary readouts now have an independent bounded entrypoint,
`run_grain_full_source_readout.py`, with its own execution contract. Synthetic
source-only Big5/MBTI fits and saved-head replay passed; the real incomplete queue
was correctly refused before labels. Actual full-corpus fits remain pending.
True same-current history-effect collection is separate pending work. Neither
runner invents that contrast from task differences. Qualification is not
full-corpus feature evaluation, and a technical case is not another independent
experiment.
