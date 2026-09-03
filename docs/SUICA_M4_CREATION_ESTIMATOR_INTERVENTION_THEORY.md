# GRAIN M4-C.3.2 Creation-Estimator Intervention

## 1. Question

M4-C.3.1 showed that the relation-space composition loss is mathematically
coherent and allocated about 80% of the symmetric loss budget to the creation
edge. That allocation is not causal. M4-C.3.2 therefore asks whether changing
only the creation estimator improves the frozen physical loop:

\[
L_u^{(j)}=A_u^{(j)}R_u^DD_u^D.
\]

The response and choice edges, the author-relation endpoint, and all prior
M4 results remain fixed.

## 2. Relation-kernel development

The first intervention replaced the discovered chart basis with a centered
RBF kernel-PCA basis fitted on calibration-condition relations only. Nyström
extension transported that basis to selection and evaluation conditions.
Twelve development arms crossed ranks `2/4/8/12` with bandwidth multipliers
`.5/1/2`.

This route stopped:

- chart baseline loop geometry: `.6702`;
- oracle creation-swap geometry: `.9199`;
- oracle creation headroom: `.2497`, clustered LCB `.1515`;
- best kernel arm: rank `12`, bandwidth multiplier `1`;
- best creation-only loop geometry: `.5316`;
- gain over baseline: `-.1386`, clustered LCB `-.2168`;
- baseline versus candidate creation geometry: `.7141` versus `.5081`;
- recovered oracle headroom: `-.5552`; and
- positive repetitions: `0/4`.

Every tested kernel arm degraded the author-relation endpoint. The degradation
was present in both independently fitted path views, so it is not adequately
explained as ordinary held-out overfitting.

## 3. Licensed interpretation

The result does not reject nonlinear creation mechanisms or local condition
smoothness. It rejects this narrower repair:

> A globally shared, stationary, isotropic RBF remapping of the discovered
> calibration chart, followed by independent per-author hazard regression, is
> not a useful generic creation repair in the registered finite worlds.

The kernel altered condition similarity while leaving the main estimation
problem untouched: recovery of the author-specific response-to-creation
derivative. A deterministic transform of a lossy discovered chart also cannot
restore directional or source information absent from that chart.

Positive oracle-swap headroom proves only that a better creation estimate
could improve the frozen loop. It does not show that the headroom is observable
through relation-kernel smoothing.

## 4. Next estimator object

The next and only registered estimator family is a cross-fitted
Fisher-Wiener shrinkage of author-specific creation derivatives. The chart is
held fixed. Calibration occasions are split into two independent technical
views:

\[
\widetilde b_u^{(1)},\qquad \widetilde b_u^{(2)}.
\]

After author centering, stable signal and split noise are:

\[
S=
\Pi_+\left[
\frac{
(HB_1)^\top HB_2+(HB_2)^\top HB_1
}{
2(n-1)
}
\right],
\]

\[
N=
\frac{
[H(B_1-B_2)]^\top H(B_1-B_2)
}{
4(n-1)
}.
\]

For author \(u\), leave-one-author-out shrinkage is:

\[
W_{-u}=S_{-u}
\left(S_{-u}+N_{-u}+\epsilon I\right)^{-1},
\]

\[
\widehat b_u^{FW}
=
\bar b_{-u}
+
W_{-u}\left(\widetilde b_u-\bar b_{-u}\right).
\]

The shrinkage operator may read response state as the creation hazard's
declared input. It may not read response-head loss, oracle creation, oracle
loop, or author-relation endpoint. No rank, kernel, bandwidth, or endpoint
search is allowed.

## 5. Confirmation boundary

An independent confirmation must require:

- oracle creation headroom at least `.05` with clustered LCB above zero;
- loop-geometry gain at least `.03` with clustered LCB above zero;
- final loop geometry at least `.70`;
- at least 50% oracle-headroom recovery;
- at least `6/8` positive repetitions also reaching `.70`;
- held-out creation-hazard log-loss degradation no greater than 2%;
- orthogonal-gauge error at most `1e-6`;
- author-label permutation and no-creation specificity controls; and
- condition/support alias refusal at least `.95`.

Passing would support only a finite synthetic claim: some creation-relation
loss arises from independent per-author derivative-estimation noise and is
recoverable through a response-safe stable-author subspace. Failure would stop
further kernel/embedding replacement and redirect the theory to opportunity
budget or designed intervention. M4-D remains blocked either way.

## 6. Fisher-Wiener confirmation result

The sealed estimator produced a partial but insufficient recovery:

- baseline/final loop geometry: `.7198/.7699`;
- gain: `.0501`, clustered LCB `.0051`;
- baseline/final creation geometry: `.7347/.8054`;
- oracle headroom: `.1974`, clustered LCB `.1359`;
- recovered headroom: `.2537`;
- qualifying repetitions: `5/8`;
- held-out hazard degradation: `1.30%`;
- author-permuted split-half gain: `-.4363`; and
- no-creation false success: `0`.

Thus a stable cross-author derivative subspace is observable and useful, but
it does not recover the registered 50% of oracle headroom or pass `6/8`
repetitions. The sealed result is:

```text
M4_C32_NO_GO_FISHER_WIENER_CREATION
```

Two additional failed checks were audit-definition issues rather than
substantive estimator failures. Common-shift geometry was undefined only in a
zero-variance null world, and `condition_alias_ecology` requires the separate
truth-open proper-loss alias audit rather than the ordinary chart refusal
interface. They remain failed in the frozen decision. The main NO-GO is
already determined by headroom recovery and repetition stability. See
`reports/SUICA_M4_FISHER_WIENER_CONFIRMATION_AUDIT_NOTE.md`.

## 7. Opportunity-excitation information frontier

M4-C.3.3 held the chart, Fisher-Wiener estimator, author population,
response/choice edge, and natural evaluation endpoint fixed while increasing
creation information through nested passive occasions and balanced orthogonal
response excitation:

\[
K\in\{1,2,4,8\},\qquad
r^{exc}=r^{natural}+\delta s,
\qquad s\in\{+e_1,-e_1,+e_2,-e_2\}.
\]

Across five main worlds and eight repetitions, the high/low Fisher information
ratio was `13.0378` (LCB `12.4051`). Relation geometry increased with
log-information (slope `.02935`, LCB `.02223`), and the frozen endpoint
contrast was `+.08022` (LCB `.06111`). High-information geometry reached
`.75232` without material hazard degradation, permutation gain, null false
success, gauge error, or alias leakage.

The result nevertheless failed sufficiency. Only `.44026` of oracle headroom
was recovered and `5/8` repetitions passed the joint gain/geometry gate. The
frozen decision is:

```text
M4_C33_NO_GO_INFORMATION_LIMIT
```

Post-hoc decomposition shows that repeated passive opportunity provides most
of the gain (`+.0611`, 95% cluster interval `[.0364,.0824]`), while orthogonal
excitation adds a smaller conditional increment at `K=8` (`+.0192`,
`[.0033,.0372]`). Geometry peaks at `K=4 excitation`, so the frontier
saturates. More importantly, recovered headroom varies from `-.216` in the
history-gated world to `.810` in selection-creation compensation.

The retained formula is therefore conditional:

\[
\Delta\Gamma
=
f\!\left(
\mathcal I_{\mathrm{creation}},
\mathcal M_{\mathrm{world}},
H_{\mathrm{latent}},
\mathcal C_{\mathrm{chart}},
\mathcal B_{\mathrm{author}}
\right),
\]

not \(\Delta\Gamma=f(\mathcal I_{\mathrm{creation}})\) alone. Observable
creation information is causally useful, but it cannot by itself close
chart-covariant ecology transport. Increasing `K`, adding embedding
dimensions, or repeating kernel replacement is stopped. The next bounded
question is whether a declared heterogeneous creation/history family can
explain the world-specific residual without oracle access.

## 8. Creation residual attribution

M4-C.3.4 fixed the high-information `K=8 excitation` endpoint and the original
gate-zero creation derivative, then crossed:

\[
C\in\{\text{discovered},\text{oracle}\},\qquad
S\in\{\text{joint},\text{history/source stratified}\},\qquad
P\in\{\text{Fisher-Wiener pooled},\text{author local}\}.
\]

The eight-repetition result was:

```text
M4_C34_NO_GO_THREE_FACTOR_INCOMPLETE
```

The full cell recovered `.3405` of oracle headroom. The proposed
history/source likelihood did not explain the residual: its target-world
effect was `-.0844` (clustered LCB `-.1581`) and was positive in `0/8`
repetitions. Its Shapley contribution was `-.0434`. Author-local pooling was
small (`+.0090`). The truth-open chart factor was the only separated positive
source (`+.0727`, leading-margin LCB `+.0396`) and its `C`-only cell recovered
`.7854` of oracle headroom.

Therefore the information frontier and the residual cube jointly imply:

\[
\Delta\Gamma
\not\approx
f(\mathcal I_{\mathrm{creation}})
\quad\text{and}\quad
\Delta\Gamma
\not\approx
f(\mathcal S_{\mathrm{history/source}}),
\]

while

\[
\Delta\Gamma_{\mathrm{recoverable}}
\approx
f(\mathcal C_{\mathrm{chart}})
\]

is a supported **attribution hypothesis**, not yet an estimator. The
truth-open chart cannot be deployed, and its alias audit passed only `7/8`.
The next bounded experiment must discover a response-safe, truth-blind chart
that approaches the frozen `C`-only relation benefit on unseen worlds. It may
not reopen shared isotropic RBF remapping or any further
`K`/kernel/embedding/excitation sweep. The candidate cross-view
errors-in-variables object and its information restrictions are defined in
`SUICA_M4_C35_RESPONSE_SAFE_EIV_CHART_DESIGN.md`; it is not yet frozen or
tested.
