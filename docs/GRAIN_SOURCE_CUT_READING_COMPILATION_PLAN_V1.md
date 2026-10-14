# Source cut-reading compilation

HM45 planned, not implemented or executed, 2026-10-09 JST. Goal ACTIVE.
Next mainline construction after HM44, not an old queue restart or oracle sweep.

## Question and fixed scope

Can a source-compiled logit potential of the new block17 state preserve the
original complete mixed H/P action without executing blocks18 and above,
final normalization or LM head? HM44 shows query carriers can reinforce or
cancel body routes. Application teacher gradients are not compiler inputs.

Retain all24 HM44 addresses, both H arms, wordings0/2, corners000/111 and
three queries. No new texts/targets/layers/prompts. Source12 is the construction
set, not12 independent people. The lower encoder remains a dependency; this
is not an LLM-free raw-text predictor or new psychological-validity experiment.

## Explicit source programs

Use ONLY source ten logits l_s, complete cut states S_s and cut gradients
G_s=D_S l_s. Verify both source wordings share body length120, ordered query
length93 or82 and one final-Q slot; query-slot token identities must also
match application. Failure stops preparation, not invented alignment.

Phi_T linearly maps current S to the source grid. Query/final slots use their
actual token identity/order. Body uses fixed piecewise-linear nodal interpolation:
current knots i/(T_body-1), target knots j/119; adjacent weights, one-token input
constant. Source length120 gives identity; application length119 uses this
explicit approximation. Save Phi and its adjoint. No claim of semantic/RoPE
invariance. Phi uses layout/length only, not scores, labels or gradients.

Per query/corner, equally average its two source wordings:

```text
S0=mean_s S_s, l0=mean_s l_s, G0=mean_s G_s
lhat_all(S)=l0+G0[Phi_T(S)-S0]
lhat_final(S)=l0+G0_final[S_final-S0_final]
Fhat=sum_j value_j softmax(lhat)_j.
```

Two affine logit potentials, no application fit/rank/degree/alpha grid. Keep
original ten-digit gauge and test common-shift invariance. Use float64 CPU
potentials with differentiable copies from the original float32 MPS cut;
casting is explicit. Freeze12 assets: six query/corner pairs times two
potentials. State-blind action controls are source mean complete mixed action
for the same query/corner/H arm and zero; they are not psychological zero scores.

## Cold dependency separation

Prepare an input-only asset with original IDs/layout and ONLY declared HM35
receiver/donor h/p/q/delta_h/delta_p fields. These historical activation inputs
are disclosed; no held logits/scores/C/Jacobians/mixed gradients are exported.
Compiler construction can read source teacher arrays. Scoring cannot read
HM39/HM43/HM44 held teacher directories or corresponding native job files:
test denied reads in its restricted process, not just an assertion in a report.

One new frozen eager/MPS forward per node patches block16, captures block17
then deliberately exits. Upper-block/final-norm/head guards must never fire.
Do not run a full forward to prepare the cut. Source12 acquired/sealed before
application12. Seal all predictions before evaluator opens held native data.
Both potentials share the same lower graph.

For each potential, ten logit-to-leaf gradients extract H and P together,
create_graph=True. Add one Fhat-to-leaf gradient and one gradient of
detached-ghat-weighted logits. Differentiate each H-gradient pairing in both
H directions to P:24 first/8 second per node; maximum24 forward/576 first/
192 second,600 cumulative model seconds and180 CPU analysis/audit seconds,
including imports/failures/readbacks. Record attempts/completions. No retry,
cap escalation, upper-model fallback or application-dependent refit.

```text
ghat=gradient_lhat m, Chat=Hessian_lhat m
linkhat=(Jhat_H d)^T Chat(Jhat_P e)
decoderhat=sum_j ghat_j D_P<D_H lhat_j,d> . e
fullhat=D_P<D_H Fhat,d> . e
fullhat=linkhat+decoderhat+reported numerical remainder.
```

Chat comes from predicted logits, never held native scores. Detach ghat for
decoder differentiation to avoid double-counting link. Direct full and
chain-contracted actions must agree under original1e-8 physical gate. New S
must replay HM44 cut exactly; scoring arithmetic/casts are separate checks.
Prediction error against native is a scientific outcome, not implementation
failure to be repaired. Retain decoder; link-only success is not full success.

## Outputs, tests and decision

24 input/call/cut captures;12 frozen source assets, coefficients/anchors/Phi
hashes; cut replay, denied-read and zero-upper-call logs; potential logits,
scores, H/P Jacobians, direct/link/decoder actions. Four candidates times24
nodes times two H arms:192 action rows/192 mean-contrast rows/144 metric groups.
Mean/zero controls do not fabricate logits or derivatives.

After prediction seal evaluate against HM39 complete mixed actions: absolute
RMSE/MS/signs/amplitudes for word/mean/half-contrast, link and decoder errors
and signed crosses, all three targets and both arms. HM44 nine-cell matrices
are evaluator references, never runtime inputs. No tiny-sample winner/trait
claim or unstable normalized ratios when totals cancel.

Tests: affine upper exactness, curved-upper counterexample, causal P-body zero,
Phi identity/adjoint/constants/linear fields/variable lengths, gauge/alphabet,
detached chain versus direct mixed derivative, cancellation, source barriers,
denied reads, upper guards and caps. Alternate verifier reconstructs all source
programs, potentials, score derivatives, tables and hashes,0 new model calls.
Same-agent arithmetic is not independent neural replication.

If full-cut useful and final-only loses action, preserve distributed reading.
If both useful, advance the simpler endpoint while retaining omission accounts.
If both lose it, identify input-conditioned upper-jet variation or the explicit
resampling approximation; do not start rank/alpha grids. This changes the next
compiler dependency, not the psychological definition of an author.

## Mathematical lineage

Affine potentials, pullback/adjoint and second-order composition are mature
mathematics. Unlike the historical last-attention cotangent compiler, this
replaces all upper reading blocks, retains ordered query slots, predicts logits
and readout, and measures both link and lower internal curvature. Unlike HM26's
property-only Hermite field, its object is the complete receiver cut.

OpenAI/math curvature-change.tex motivates preserving domains when transporting
representations, not automatic resampling invariance. HM44 theory records the
actual198-line reading. RCD assumptions/theorems are not transferred to this
Transformer. No general approximation guarantee from the two source wordings.

Original18/100/44/natural-author B_u/psychological interpretation/fully standalone
measurement remain open. No API/private text/Claude/automation/neural training/
commit/push/tag/lockbox/old queue. Save failures; no automatic scientific rescue.
