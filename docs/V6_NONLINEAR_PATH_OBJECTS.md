# V6 Nonlinear Path Objects: Definition, Evidence, and Refusal Boundary

Status: `SUPPORTED-SYNTHETIC / NOT_DETECTED-AS-ORDER-STABLE-IN-CURRENT-PANDORA`
(2026-07-14). This document defines a measurement object. It does not name a
new personality dimension.

## 1. Why a new object is needed

The rejected V6 dynamic-factor family summarizes paths through finite linear
statistics: covariance spectrum, one-step linear transition, roughness, and
change-point magnitude. A failure of those axes is not a theorem that no
author-specific path structure exists. It is only a failure of that projection.

Let a frozen text representation be `psi(T)` and let `Q` and `C` denote measured
expression opportunity and condition. The existing author-cross-fitted residual
is

\[
X_{uork}=\psi(T_{uork})-
\widehat{\mathbb E}_{-u}[\psi(T)\mid Q_{uork},C_{uor},e].
\]

For a preregistered history order `L`, define the observable history

\[
H^{(L)}_{uork}=(X_{k-L+1:k},Q_{k-L+1:k},C_{uor},\Delta t_{k-L+1:k}).
\]

The candidate is not a scalar axis. It is the author-specific conditional law

\[
\Pi_u^\psi(dy\mid h)=\Pr(X_{k+1}\in dy\mid H_k^{(L)}=h,U=u).
\]

With a characteristic output kernel `ell`, its conditional mean embedding is

\[
\mu_u(h)=\int \ell(y,\cdot)\Pi_u^\psi(dy\mid h)\in\mathcal H_\ell.
\]

On a fixed common-support reference distribution `nu`, the proposed unknown
dynamic configuration is the function-valued deviation

\[
G_u=[\mu_u-\mu_0]_\nu\in L^2(\nu;\mathcal H_\ell),
\qquad
\mu_0(h)=\mathbb E[\ell(X_{k+1},\cdot)\mid H_k=h].
\]

The natural comparison is therefore a functional distance, not a factor loading:

\[
d_\nu^2(u,v)=\int\|\mu_u(h)-\mu_v(h)\|^2_{\mathcal H_\ell}\,d\nu(h).
\]

Because a Gaussian kernel is characteristic, the infinite-dimensional object can
distinguish conditional means, variance, skewness, multimodality, and nonlinear
switching. It is strictly richer than a linear AR coefficient, but its finite
sample estimator is still fallible.

## 2. Practical estimator used here

For each complete same-condition run `r`, a pooled conditional embedding
`mu_0` is fitted from discovery authors only. Confirmation authors receive the
equal-run-weighted residual embedding

\[
\widehat G_{u,h}=
\frac{1}{|R_{u,h}|}\sum_{r\in R_{u,h}}
\frac{1}{m_r-1}\sum_{t=1}^{m_r-1}
\left[\phi(X_{t+1})-\widehat\mu_0(H_t)\right].
\]

The code uses fixed-seed random Fourier features (RFF) for both input and output
Gaussian kernels. This is an approximate conditional kernel mean embedding, not
a per-author kernel operator: PANDORA is too sparse for the latter.

Full path geometry is screened separately by the truncated, time-augmented path
signature

\[
S_{\leq m}(X)=\left(\int dX,\ \int_{t_1<t_2}dX_{t_1}\otimes dX_{t_2},\ldots\right).
\]

At degree two, the antisymmetric component

\[
\mathsf A_{ij}(X)=\tfrac12\left(S^{ij}(X)-S^{ji}(X)\right)
\]

is the signed area / ordered cross-coordinate coupling. It is zero-information
to a mean-only object and changes when the same observations are traversed in a
different order.

## 3. Necessary identification conditions

`G_u` is an observable transition-law object only when the representation and
scorer are frozen, the output kernel is characteristic, histories have common
support, and complete runs are locally stationary or sufficiently mixing.

Interpreting `G_u` as a stable author parameter additionally requires repeated
condition support, measured or randomized opportunity, independent technical
replicates, and epochs in which short-term state can genuinely vary. Without
those, these two worlds are observationally equivalent:

\[
X_{k+1}=\rho X_k+\theta_u+\epsilon_k
\]

and

\[
X_{k+1}=\rho X_k+S_{uo}+\epsilon_k,\qquad S_{uo}\equiv\theta_u.
\]

No kernel, neural network, or factor rotation can distinguish a stable author
parameter from a persistent state when the state never independently changes.

## 4. What the present evidence says

### Synthetic counterexample to linear sufficiency

In the phase-coupled world,

\[
W_{t+1}=2Z_t+\theta_u+\eta_{t+1}\pmod{2\pi},
\]

all authors have the same state mean, covariance, and linear lag covariance.
The circular conditional witness

\[
M_u=\mathbb E[e^{i(W_{t+1}-2Z_t)}]
=\frac{I_1(\kappa)}{I_0(\kappa)}e^{i\theta_u}
\]

recovers the planted author configuration. Across 100 worlds, linear summaries
gave mean AUC `0.503` (2.5--97.5% quantiles `0.459--0.559`); the nonlinear
conditional witness gave `0.985` (`0.981--0.988`). Endpoint-preserving interior
shuffling reduced the witness to `0.548` (`0.484--0.600`).

This licenses `SUPPORTED-SYNTHETIC`: linear factor-like summaries do not exhaust
possible path information. It does **not** establish this mechanism in human text.

### Current PANDORA screen

The all-author path-signature exploration initially found a small degree-two
order effect, but it did not survive author-disjoint replication:

| Object | All-author weighted screen | Confirmation-only weighted screen | Reading |
|---|---:|---:|---|
| Degree 1 signature | `0.0000 [0.0000, 0.0000]` | `0.0000 [0.0000, 0.0000]` | Endpoint-only null behaves exactly as expected. |
| Degree 2 signature | `0.00745 [0.00050, 0.01344]` | `0.00199 [-0.00714, 0.01150]` | Discovery microtrace; not detected in unseen authors. |
| Degree 3 signature | `0.00729 [0.00031, 0.01365]` | `0.00477 [-0.00381, 0.01402]` | Not detected in unseen authors; no material increment. |

The held-out confirmation-author conditional-kernel screen does **not** replicate
an order-specific effect. The weighted opportunity match gives AUC `0.5826`
before and `0.5815` after endpoint-preserving internal shuffle, with delta
`0.0011 [-0.0004, 0.0027]` across 669 paired confirmation authors. Its mean
subreddit Jaccard overlap is only `.035`, so it is not a strong same-condition
control. The stricter sensitivity curve agrees: at Jaccard `.05`, coverage is
99.1% and delta is `.00057 [-.00117, .00218]`; at `.10`, delta is
`.00087 [-.00131, .00286]`, but coverage drops to 72.5% and fails the planned
80% adequacy rule.

The all-author path-signature microtrace also uses only the low-overlap weighted
match (mean Jaccard `.036`), not a caliper-confirmed same-condition contrast.
The confirmation-only result is already null, so it is retained solely as a
discovery artefact record rather than a surviving geometric effect.

### Frozen four-subepoch feasibility gate

Stage V1 then replaced the coarse two-half screen with four disjoint whole-run
subepochs per author. It froze two partitioning operators before endpoint
fitting: a transition-balanced cut and a time-balanced cut. The criterion was
at least 300 fixed-hash endpoint authors plus 80% early/late pair-level
condition coverage at Jaccard `>= .10`.

The transition-balanced operator retained 1,903 structurally eligible authors
and 382 endpoint authors, but coverage was only `.593`. The time-balanced
operator retained 1,047 eligible authors and 197 endpoint authors, with `.548`
coverage. Both operators therefore fail the gate. No nonlinear endpoint model,
event detector, or text correspondence analysis was fitted after this result.

This makes the limitation more specific: PANDORA can supply many within-run
transitions, but it cannot supply enough repeated condition support for this
registered cross-period dynamic comparison. The correct action is refusal, not
more model capacity or a weaker matching rule.

Thus the current data support neither a stable nonlinear transition variable nor
a nameable dynamic construct. Both the confirmation-only path-signature screen
and the broader conditional-distribution estimator are null for order-specific
stability under their frozen PANDORA splits.

## 5. Binding language and next design

Permitted now:

- "A nonlinear conditional transition object is mathematically well-defined and
  distinguishable from linear summaries in a planted world."
- "The all-author PANDORA screen contained a small degree-two order trace that
  did not replicate in held-out authors."
- "Neither frozen nonlinear estimator detected an order-stable effect in
  held-out PANDORA authors."

Not permitted now:

- "GRAIN discovered a dynamic personality dimension."
- "The residual is a stable author dynamics parameter."
- Any individual score, clinical interpretation, or cross-language transport claim.

A future human study must pre-freeze `L`, kernel family, RFF seed/dimension,
regularization, conditioning fields, and matching rule; use at least two epochs
by two independent technical replicas, repeated common conditions, and explicit
opportunity metadata. If multiple kernels/history orders are searched, final
confirmation requires stratified author-block permutation (at least 1,999 draws)
with max-T familywise control, plus leave-condition/community and scorer/tokenizer
perturbation checks. Only then can `G_u` be tested as a repeatable author object.

## Artifacts

- `scripts/run_suica_v6_path_signature_probe.py`
- `reports/V6_PATH_SIGNATURE_PROBE.md`
- `reports/V6_PATH_SIGNATURE_CONFIRMATION_PROBE.md`
- `scripts/run_suica_v6_conditional_transition_probe.py`
- `reports/V6_CONDITIONAL_TRANSITION_PROBE.md`
- `scripts/run_v6_path_geometry_simulation.py`
- `reports/V6_PATH_GEOMETRY_SIMULATION.md`
