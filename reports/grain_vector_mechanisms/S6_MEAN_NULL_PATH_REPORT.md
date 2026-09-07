# S6: a trait-query response inside the kernel of comment averaging

2026-09-08. Complete. Eight fixed S1 authors, one view each, three queries,
two common selected-comment sets. No questionnaire labels used.

## Finding

Keeping each selected comment's block16 mean unchanged does not keep the
teacher's judgment unchanged. Token-level configuration has a direct functional
effect in the fixed decoder suffix. The same interventions couple neuroticism
and the narrower confirmation-seeking query: edit-centered r=.9717,
author-bootstrap95% CI [.8094,.9928]. This is a shared internal response, not a
correlation between two measured psychological scales.

The integrated directional derivative reconstructs the measured endpoint effect
at r=.9980-.9998. First-order gradients track direction but misestimate magnitude.
Thus the experiment supplies both a nonzero mean-null effect and an explicit
mathematical way to account for it along the declared internal path.

## Intervention and equation

At block16, take each selected comment's complete body-token states H_t and
apply H_t(a)=H_t+a(mean_comment(H)-H_t), a in[0,1]. All other states, including
the last query state, remain unchanged at the intervention site. Token IDs,
positions and attention-mask length do not change. Within-comment means are
preserved up to float32 arithmetic; maximum observed error2.253e-6.

This is a controlled hidden-state collapse, not a natural text edit. It changes
within-comment dispersion and arrangement together. S6 therefore identifies
configuration beyond the first moment, not specifically word order.

Let D be that mean-null displacement. The exact fixed-suffix identity is

\[
s(H+D)-s(H)=\int_0^1\langle\nabla s(H+aD),D\rangle_F da.
\]

Three-node Simpson integration is extended to five nodes only when its endpoint
gap exceeds max(.001,.05*abs(effect)). The measured endpoint is not replaced by
the numerical integral. Local comparison uses the original derivative at a=0.
See [the mathematical development](../../docs/GRAIN_AGGREGATE_CONFIGURATION_THEORY_V3.md)
and the cited integrated-gradient literature there.

## Results

Each query has16 paired edit paths from8 authors. Score units are the native
candidate-conditioned score in[0,1], not raw questionnaire units.

| Query | Effect RMS | Mean absolute effect | Local r | Integral r | Local RMSE | Integral RMSE |
|---|---:|---:|---:|---:|---:|---:|
| T/F | .01355 | .00936 | .7962 | .9980 | .01892 | .000894 |
| Neuroticism | .02471 | .01016 | .9842 | .9998 | .01283 | .000874 |
| Confirmation seeking | .00900 | .00532 | .9540 | .9994 | .01294 | .000355 |

Absolute effects exceed.001 on14/16,13/16 and13/16 paths, respectively. The
neuroticism RMS includes one large effect (-.0963); its mean absolute effect
is therefore also reported rather than presenting RMS as a typical subject.

Five-node refinement was needed on12 T/F,2 N and3 confirmation paths. Two T/F
paths still exceeded the integration tolerance, none in the other queries.
These gaps are retained; a high overall correlation is not exact completeness.

Shared edit-centered responses:

| Query pair | r | Author-bootstrap95% CI |
|---|---:|---|
| T/F - N | .4318 | [-.3939,.7594] |
| T/F - confirmation | .4886 | [-.2873,.8231] |
| N - confirmation | .9717 | [.8094,.9928] |

Within this same panel, collapse-effect versus existing natural-deletion effect
r is T/F=-.0059, N=.8283 and confirmation=.4666. These interventions are
different, so their correlation links two response patterns; it does not make
internal collapse an equivalent operation to deleting a comment.

## What changes next

The lightweight mean/distribution students in S5 did not recover the N teacher
association. S6 now gives a concrete configuration response to investigate;
it does not establish that this particular response explains the missing
teacher-label covariance on the separate D1 sample.

Next distinguish ordering from dispersion by permuting intact token-state rows
within a comment. That preserves the full unordered distribution, unlike S6
collapse. Group averaging gives an exact algebraic split into original-versus-
orbit-average response and orbit-average-versus-collapsed response. S7 uses a
bounded Monte Carlo estimate of the orbit average, not a new optimization grid.

## Reproduction and cost

Runner: `scripts/run_grain_mean_null_path.py`.
Private output: `results/grain_vector_mechanisms/mean_null_path_s6_r1/`.
Contains frozen manifests, source hashes, per-author packets, path derivatives,
endpoint scores, paired-query comparisons and run status. Raw texts remain in
the original ignored S1 material; no external data or cloud API used.

154 forward/backward pairs,318.04seconds of acquisition/analysis after model
loading; below the216-pair/30-minute cap. No process from S6 remains running.
The related regression suite passed185 tests before the S7 extension.

```bash
python scripts/run_grain_mean_null_path.py \
  --panel results/grain_vector_mechanisms/multiquery_sensitivity_s1_r1 \
  --output-dir results/grain_vector_mechanisms/mean_null_path_s6_replay
```

The runner resumes complete author packets only when code, model, material and
device match. Keep original results immutable and use a new directory for replay.
