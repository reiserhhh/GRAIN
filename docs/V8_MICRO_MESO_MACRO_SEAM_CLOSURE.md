# GRAIN V8 Micro-Meso-Macro Seam Closure

Status:
`SUPPORTED_SYNTHETIC_TYPED_CLOSURE__PSYCHOLOGICAL_MAPPING_OPEN`

Date: 2026-07-29

## 1. What is closed

The registered synthetic theory now contains one typed path from observable
event sequences to a population relation field:

\[
\mathcal N^{f,r}_{u,s}
\xrightarrow{\Phi_{\mathrm{M3}}}
\bar\Phi^{f,r}_{u,s}
\xrightarrow{Q_f}
\widehat A^{f,r}_{u,s}
\xrightarrow{\mathcal R_s}
\widehat J_s
\xrightarrow{\Gamma}
(\widehat J_W,\widehat J_B,\widehat J_T,H,\kappa)
\xrightarrow{\mathcal L}
\text{license/refusal}.
\]

Every arrow has an observable input, a frozen operator, an uncertainty or
adequacy check, and a refusal path. No arrow requires a questionnaire label or
synthetic truth during licensing.

This is an L3-to-L4 technical relation-procedure closure in the registered
synthetic worlds. It is not an L5 measurement-score closure. The external map

\[
\Psi:
(\widehat A,\widehat J,\widehat J_W,\widehat J_B,H,\kappa)
\to
\text{psychological or behavioral construct}
\]

remains closed.

## 2. Micro layer

For each author \(u\), context \(s\), family \(f\), and replicate \(r\), the
observable is an ordered event path \(\mathcal N^{f,r}_{u,s}\). A fixed block
map records means, variances, lag-one covariance, and empirical
characteristic-function features of centered events and transitions:

\[
\Phi(B)=
\left[
\bar x,\operatorname{Var}(x),\operatorname{LagCov}_1(x),
\widehat\varphi_x(\omega),
\widehat\varphi_{(x_t,x_{t+1})}(\nu)
\right].
\]

The frequencies and block rule are frozen before confirmation. The estimator
does not know which registered generator produced the events.

## 3. Mesoscopic author object

Same-family technical repetitions identify replicated support:

\[
\widehat S_f
=
\frac12[
\operatorname{Cov}(\bar\Phi_f^{(1)},\bar\Phi_f^{(2)})
+
\operatorname{Cov}(\bar\Phi_f^{(2)},\bar\Phi_f^{(1)})^\top].
\]

The positive leading support \(Q_f\), center \(c_f\), and replicated whitener
\(W_f\) are frozen on D0. The mesoscopic author coordinate is

\[
\widehat A_f^{(r)}
=
(\bar\Phi_f^{(r)}-c_f)Q_f.
\]

This coordinate is defined only up to transformations that preserve
observables. Gauge-equivalent hidden mechanisms therefore share one technical
object; the system never licenses a unique micro mechanism.

## 4. Support-comparability gate

Before comparing D0 and confirmation coordinates, confirmation support
\(\widehat Q_{f,d}\) is estimated in the raw feature space. Its worst
principal-angle alignment is

\[
A_{f,d}
=
\sigma_{\min}^2(Q_{f,0}^{\top}\widehat Q_{f,d}).
\]

The second positive eigenvalue and its normalized eigengap establish whether
the support is estimable. D0 author splits calibrate the lower-tail thresholds.

The taxonomy is:

- `SUPPORT_ALIGNED`: comparison may continue;
- `SUPPORT_NONINVARIANT`: the calibrated feature support changed, so the
  relation is not comparable;
- `SUPPORT_UNDERRESOLVED`: the support itself is too weak or degenerate to
  decide invariance.

A rotation inside the same support is accepted. A support replacement is
refused. Refusal does not claim that no information exists; it says the
registered coordinate system cannot carry it without a new transport study.

## 5. Relation layer

For a resolved context \(s\), the technical cross-family relation is

\[
\widehat J_s
=
W_L\frac12[
\operatorname{Cov}(\widehat A_L^{(1)},\widehat A_R^{(2)}\mid s)
+
\operatorname{Cov}(\widehat A_L^{(2)},\widehat A_R^{(1)}\mid s)
]W_R.
\]

The cross-replicate construction removes registered same-occasion
cross-family measurement shocks that contaminate a same-view covariance. A
relation license additionally requires independent confirmation agreement,
context resolution, a family-wise local null, a material floor, and no
registered nuisance or causal-role veto.

An out-of-family periodic signal may be stable while the M3 quotient is not.
That case is classified as `MECHANISM_FAMILY_MISMATCH`, not erased as noise.

## 6. Macro layer

The macro relation is formed in covariance space before whitening:

\[
C_T
=
\underbrace{\sum_s p_sC_s}_{C_W}
+
\underbrace{
\sum_s p_s
(\mu_{L,s}-\mu_L)(\mu_{R,s}-\mu_R)^\top
}_{C_B}.
\]

Because the same linear maps are applied after decomposition,

\[
\widehat J_T=\widehat J_W+\widehat J_B
\]

is an auditable identity. \(\widehat J_W\) is within-context relation and
\(\widehat J_B\) is ecological between-context structure. They use separate
D0 null distributions because one is a max-context family and the other is a
single aggregate matrix.

Heterogeneity and cancellation are

\[
H=
\frac{\sum_sp_s\|\widehat J_s-\widehat J_W\|_F^2}
{\sum_sp_s\|\widehat J_s\|_F^2+\epsilon},
\qquad
\kappa=
1-
\frac{\|\widehat J_W\|_F}
{\sum_sp_s\|\widehat J_s\|_F+\epsilon}.
\]

A large \(\widehat J_B\) cannot become an individual relation license.
Opposing local relations cannot become a global invariant relation.

## 7. Uncertainty and estimands

Two truth audits are deliberately separate:

- sample-latent fidelity measures whether the estimated local field recovers
  the direction and shape of the same generated authors;
- author-cluster bootstrap coverage targets the analytic population relation.

Mixing the first target with the second interval caused conservative
overcoverage during development and was corrected before formal confirmation.
A nested author-plus-block bootstrap remains a sensitivity diagnostic.

## 8. Empirical closure sequence

| Experiment | Formal result | What it established |
| --- | --- | --- |
| SEAM-1 | `PARTIAL`, 9/10 | Raw event-to-relation path worked, but local and between nulls were incorrectly conflated. |
| SEAM-1A | targeted repair supported, overall `PARTIAL` | A statistic-specific \(J_B\) null restored ecological classification; hidden support drift remained. |
| SEAM-1B | `PASS`, 13/13 | Aligned support is accepted; gauge rotation is accepted; support drift, underresolution, ecological-only structure, and out-of-family mechanisms are separately refused. |

SEAM-1B's 60x999 confirmation obtained clean seam license 1.000, relation
fidelity q05 .992, relative-error q95 .174, population coverage .924,
attenuation correction win .967, cancellation and ecological detection
1.000, both support-drift detections 1.000, support-underresolution detection
1.000, within-support-gauge false refusal .017, and zero truth use by
licensing.

## 9. Closure proposition

Within the registered synthetic family:

1. if the raw replicated support is resolved and D0-compatible, context and
   nuisance gates pass, and the relation exceeds its statistic-specific null
   and material floor, the event path admits a reproducible technical
   context-relation field with calibrated population uncertainty;
2. if support is noninvariant, support is underresolved, the signal lies
   outside the M3 family, context is unresolved, nuisance conditioning is
   unstable, or only an ecological component exists, the procedure returns
   the corresponding refusal rather than a stronger claim;
3. observationally equivalent hidden mechanisms remain one equivalence class
   and cannot receive unique-mechanism names.

This proposition is closed for the registered simulators because every
antecedent has a positive, negative, or alias world and an explicit decision
gate. It is not a theorem about all text-generating processes.

## 10. What remains open

- M3-V4 still requires its separately sealed external-randomness confirmation.
- Support drift has a refusal rule but no learned transport map.
- Real-text PANDORA and Essays now establish corpus-local, scale-free
  replicated-density resolution for anonymous M and K families. A
  shared-gauge replay did not establish either registered cross-corpus
  spectral order. Local resolution must therefore remain separate from
  cross-corpus orientation and transport.
- The K family has now been split into event-set permutation susceptibility
  \(G^{set}\) and observed-order phase \(C^{phase}\). Cross-corpus set geometry
  survives a block-Haar null, whereas two complete statistic-level order
  tests do not detect phase excess. A subsequent context/slot/length-matched
  event reallocation reproduces the set overlap, so the supported common
  object is an operator-plus-event-marginal background \(G^0\), not natural
  within-author composition, opportunity response, or a sequence-sensitive
  author operator.
- A final corpus-local signed residual test, run without positive-part
  projection, also refused stable composition residuals in both corpora
  (PANDORA maxT \(p=.865/.800\); Essays \(p=.995/.995\)). The current K
  branch therefore terminates at the conditional marginal
  orbit-susceptibility operator; it does not enter the author-response seam.
- The order-free M family now has a separate corpus-local route. Conditional
  pseudo authors define a marginal-background quotient; normalized signed
  cross-replicate concordance survives in PANDORA, Essays, and X-market under
  an exchangeability-corrected null. After D0-frozen declared-opportunity
  filtration, no corpus retains a stable global linear spectrum. The earlier
  X axes are therefore historical opportunity-mediated directions, not
  current author factors.
- Axis-free author-relation correspondence survives at one short RBF scale in
  PANDORA and two scales in X, while Essays remains scalar-only. This relation
  object is distinct from a factor chart and from a nonlinear claim.
- PANDORA same-author cross-context transport is underresolved. The pooled
  relation geometry does not close inside any individual context, and the
  corpus lies on an information-support frontier: three context pairs are
  admissible at 8 events, one at 12, and none at 16/24. A matched-support
  8-versus-12 diagnostic supports decreasing finite-budget error but does not
  close transport.
- The static real-text seam is therefore branched:
  \(x\to m(\mathcal X)\to q_{\mathbb B}\to(B,W)\), followed either by a
  resolved linear spectrum or by an axis-free relation matrix
  \(\widetilde K\). Its observable form is
  \(\widehat{\mathcal G}_{u,c,b}=\mathcal T_c(\mathcal G_u)+
  \mathcal E_{u,c,b}\), so context and information budget are part of the
  measurement object. It may enter a macro relation field only after support,
  nuisance, within-context precision, and transport gates; psychological
  naming remains a later external-linkage problem.
- The macro test uses four fixed contexts; random-context superpopulation
  inference requires more independent contexts and context-level
  random-effects/bootstrap methods.
- Real text must establish that the registered supports and relations occur
  outside simulation.
- Psychological, behavioral, state, emotion, diagnostic, and clinical names
  require independent external evidence.

Therefore the mathematical seam is closed by either identification or
refusal, while psychological interpretation remains intentionally open.
