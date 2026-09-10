# NEXT19/20: shared readouts for measured actions

2026-09-10. Ordered exploratory construction, source outcomes already exposed.
Preserve the release scorer, old lockboxes, all earlier successes and failures.

## NEXT19: N mixed logits from paired distribution features

Reuse NEXT16/r2_replay fixed32 projections, old four author folds, and NEXT17
view0 endpoints for all24 N authors/two fixed pairs. No new neural call.
Training geometry uses only the18 training authors' three cached views, exactly
as NEXT16. Each comment feature is its Fourier odd/even contrast, weighted by
its token share of the full material. Keep two fixed families: bag and position.
No frequency, penalty, seed, rank or Y search.

Target is the centered candidate-logit mixed difference, not a label and not
unobserved separate odd/even effects. Positive-only pair endpoints cannot
identify the latter. For comment features a,b define symmetric tensor feature
Psi(a,b)=(a tensor b+b tensor a)/2. Its kernel is

    k((a,b),(c,d))=((a.c)(b.d)+(a.d)(b.c))/2.

It vanishes if either feature is zero and is invariant to swapping the two
comment arguments with their position metadata attached. Normalize the kernel
by its training mean diagonal; fit mean-loss Ridge penalty.01, no intercept.
Four folds,36 training pairs/12 evaluation pairs each. Eight joint fits total.
Controls: zero mixed logits (additive latent), and training-only pair-type mean.

Report centered-logit error energy/zero, norm fidelity and paired-author CIs;
also reconstruct S(z_add+mixed_hat) with cached z_add for a conditional action
comparison. This assembly has oracle single-comment logits and is explicitly
NOT a standalone raw-text personality predictor. Source Y only joins after
prediction freeze; keep all arms and both pairs.

The architecture motivation is rho(sum Phi) in
[Deep Sets](https://papers.nips.cc/paper/2017/file/f22e4747da1aa27e363d86d40ff442fe-Paper.pdf)
and [distribution regression](https://www.jmlr.org/papers/v17/14-510.html).
This is a finite symmetric tensor readout, not their universal/statistical guarantee.

## NEXT20: T/F R action by a reference-only functional matrix

Reuse NEXT18's136-position role-aligned R, ordinal body reference M0, original
M, fixed P/Q and target layout. Same four old reference authors, view0;20 other
source authors never supply gradients or fitting targets to the reader.

For each reference u and M in {M0,Mu}, integrate grad_R F(M,R0+t(Ru-R0),L_u)
at t=0,.5,1. Use exact cached endpoints, not rounded reconstruction. Average
the two path matrices, then average the four reference matrices:

    G_ref=mean_u .5*(integral grad_R F(M0,R(t))+integral grad_R F(Mu,R(t)))
    Phi_R_hat(u)=<G_ref,Ru-R0>.

Supervised quantity is the symmetric retained allocation from NEXT18, not
complete B. Controls fixed before collection: zero, reference target mean,
token-mean matrix, rank1,rank4,full matrix. No rank/Y selection. Report allocation
fidelity/error/signs and then criterion linkage. Diagnostic complete-B assembly
uses cached baseline/body allocation plus predicted R; explicitly label these
oracle components, do not imply all inputs have been compressed.

Costs:24 suffix forward/backward pairs initially. If an individual path's
Simpson action differs from its exact endpoint difference by more than
max(1e-4,.02*abs(action)), add nodes.25/.75 for that path only. Maximum40 pairs,
180seconds, no new prefix/API/model training. Save nodes before author closure.
Adaptive refinement is numerical integration, not Y tuning; report all failures.

This follows the existing P-region construction and the path-calculus basis of
[Integrated Gradients](https://proceedings.mlr.press/v70/sundararajan17a.html).
Reference transport is an empirical hypothesis, not implied by exact integration.

## Evidence and acceptance

- Fit/read/freeze before source Y. Hash old endpoints, projections, layout,
  reference arrays, source states, model and current code. Public aggregates only.
- Test tensor kernel versus explicit features, PSD, swap/zero, held-label
  isolation, reference-only averaging, and bilinear path integration.
- Replay model-only numerical predictions; no second gradient collection just
  to fill a table. Preserve current input encoding dependency in every report.
- If N joint readout fails, do not expand a feature grid. Inspect whether joint
  magnitudes/directions require missing context before any new architecture.
- If R matrix transports, first close a complete conditional-baseline input
  contract; do not rename136 template positions as136 personality factors.
- These close two construction cards, not all scientific logic possibilities.
