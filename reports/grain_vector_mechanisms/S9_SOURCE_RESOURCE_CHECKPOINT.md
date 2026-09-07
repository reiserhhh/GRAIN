# S9 resource checkpoint:48 authors, no outcome inspection

2026-09-08. The planned checkpoint completed288 forwards in421.3seconds, with
24 authors per query. No labels inspected; no source or student reselected.
At this rate, the277-author cohort costs about40.5minutes total, within its
one-hour cap. All48 complete-author packets are resumable.

| Query | Paired old/new origin r | Old/new RMS difference | Configuration RMS | Configuration SD |
|---|---:|---:|---:|---:|
| T/F | .8267 | .07519 | .02485 | .02475 |
| Neuroticism | .6773 | .03869 | .00867 | .00709 |

The two origins differ appreciably despite identical model revision and exact
prompt-token hashes. The old CUDA/BF16/padding3072 and new MPS/FP32/compact
execution conditions are known to differ, but their individual contributions
have not yet been localized. All831 original candidate-logit records reconstruct
their cached scalar scores within2.7e-15; the new differentiable score uses the
same conditional-softmax formula. No label-direction or source-selection change
has been found.

Before finishing the cohort, run a bounded four-author view0 observer check:
the first four prepared authors, two queries, selected by existing order.
Compare MPS FP32/BF16 crossed with compact/fixed3072 execution using the native
full-head path. Existing S9 compact/last-logit scores supply the implementation
comparison. Maximum16 forwards, five-minute cap, no labels, no fitting. This is
a scientific identity check prompted by the observed discrepancy, not a new
unbounded hardware-qualification queue. Preserve both sources regardless of result.

Runner: `scripts/check_grain_source_observer.py`.
Private results: `source_remainder_s9_r1/observer_check/`.
Interpret after completion, then resume the fixed cohort if the intervention
remains interpretable and costs remain useful. An observer-dependent association
can still be accounted for explicitly; it must not be silently assigned to
token-configuration geometry or dismissed as pure noise.

## Check completed and continuation

All16 forwards completed in57.2seconds. FP32 compact/full-head agrees with S9's
last-logit path within2e-6; FP32 padded/full-head agrees within5e-6. Thus those
implementation/shape choices do not explain the large origin difference here.
Switching to BF16 changes scores substantially (RMS vs S9 .13835/.13054 for
compact/padded, four views). BF16 MPS is closer to old CUDA BF16 on these four
views (RMS .05600/.07489 versus FP32 .13223) but does not exactly reproduce it.
No claim that this small check isolates every backend contribution.

Decision: complete the fixed277-author S9 cohort using its unchanged FP32 paired
origin/intervention protocol. The internal paired intervention is interpretable;
the observer difference is real and explicitly separated. About34minutes remain
at the checkpoint rate. No outcome-informed stopping or modification occurred.
