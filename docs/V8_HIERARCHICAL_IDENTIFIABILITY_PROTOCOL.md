# V8-HJIC-1 Hierarchical Joint Identifiability and Commutation

Status: `PROSPECTIVE_SYNTHETIC_PROTOCOL__NO_HUMAN_LABELS`

## 1. Question

GRAIN V8 must not be represented as one chain ending in a questionnaire
score. The object is a typed graph:

\[
\begin{array}{ccc}
\mathcal N_u & \xrightarrow{\mathcal C} & L_{2,u}^{*} \\
\downarrow\mathcal O_{\theta_R} && \downarrow J_{\theta_R} \\
L_{0,u}\rightarrow L_{1,u}
& \xrightarrow{\mathcal E_m} &
\widehat{[L_{2,u}^{*}]}_m .
\end{array}
\]

\(\mathcal N_u\) is a latent event process. \(L_{2,u}^{*}\) contains only
latent technical author objects:

\[
L_{2,u}^{*}
=
(\pi_u,[G_u],[K_u],[C_u],[P_u]).
\]

The measurement result is a different type:

\[
\widehat L_{2,u}^{(m)}
=
(\widehat{[L_{2,u}^{*}]},
\mathcal U_{u,m},\delta_{u,m},E_{u,m};
P_0,Q_0,\theta_R).
\]

Uncertainty \(\mathcal U\), refusal \(\delta\), and evidence/provenance \(E\)
are not latent author properties.

Population text geometry and external calibration also branch:

\[
L_{3,P}^{\mathrm{text}}
=
\Gamma(P_{\widehat L_2}),
\]

\[
R_{P}^{BM,\mathrm{pred}}
=
\operatorname{Corr}_P
\left(
h_B(\widehat L_2),
h_M(\widehat L_2)
\right).
\]

The second object uses external heads and is not a natural \(L_2\to L_3\)
coarse-graining.

## 2. Formal restrictions

### Frozen-map information order

If \(\theta_R\) is estimated independently and frozen, and

\[
L_{k+1}=f_{\theta_R}(L_k),
\]

then under the declared Markov condition:

\[
I(Y;L_{k+1}\mid Z,\theta_R)
\le
I(Y;L_k\mid Z,\theta_R).
\]

Equivalently, the Bayes risk of the richer object cannot be worse for every
decision problem. A finite-sample contraction may generalize better because
it changes estimation variance; this cannot be interpreted as new
psychological information.

### Identification set

\[
\mathcal I_u(D)
=
\left\{
A:P_A(O_{\mathrm{obs}})=P_D(O_{\mathrm{obs}})
\right\}.
\]

A component is point-identified only if its requested functional is constant
over \(\mathcal I_u(D)\). Otherwise V8 must return an equivalence class,
interval, or refusal.

### Route non-ontology

Different tasks selecting different finite-sample routes is evidence about
readout behavior only. It is not evidence that those routes are distinct
psychological levels.

## 3. Registered worlds

| World | Identified or attacked object |
|---|---|
| `IDENTIFIABLE` | visible menus, randomized common conditions, multiple occasions, full-rank observation, and ordered events |
| `STATE_ALIAS` | stable position and constant state have the same observation |
| `MENU_ALIAS` | availability and preference produce the same realized choices |
| `KERNEL_ALIAS` | response operators differ only in the observation-map kernel |
| `ORDER_ALIAS` | occupancies agree while transition order differs |
| `ROUTE_NULL` | every task depends on the same sufficient core, while finite samples may select different routes |
| `REFERENCE_DRIFT` | external origin and population relation structure change with the declared reference population |

Two relational-lift arms are nested in the battery:

- `SHARED_LATENT` plants a common cross-anchor structure plus large
  target-private variation. It tests whether relation patterns can be stable
  while individual correlations remain modest.
- `COMMON_NUISANCE` plants a shared non-construct nuisance. Raw relation
  recovery may be high, but nuisance-conditioned structure must refuse.

## 4. Estimators

The identifiable free phase fits a menu-conditioned choice kernel:

\[
\Pr(c_t=j\mid M_t,c_{t-1})
\propto
\mathbf 1(j\in M_t)
\exp(\beta_{u,j}+\kappa_u\mathbf 1[j=c_{t-1}]).
\]

The fixed phase uses balanced conditions, symmetric cues, repeated occasions,
and sum-to-zero occasion coding:

\[
\xi_{uocr}
=
a_u+B_u\phi(c)+C_u q_r+Hs_{uo}+\epsilon.
\]

The commutation defect compares:

\[
J_{\theta_R}\circ\mathcal E_m
\quad\text{against}\quad
\mathcal E_m\circ\mathcal O_{\theta_R},
\]

not an estimate against its truth. Recovery error and commutation error are
reported separately.

The Gaussian arm computes exact conditional covariance and mutual
information under a frozen contraction. Alias arms use paired
observationally equivalent worlds. Route-null reports both a naive
route-heterogeneity claim and the licensed claim after checking planted
incremental information.

## 5. Frozen gates

- Every identifiable component has repetition-level 5th-percentile truth
  correlation at least `.80`.
- The 95th-percentile standardized commutation defect is at most `.10`.
- Pooled nominal 95% coverage lies in `[.90,.98]`.
- Every alias family has refusal rate at least `.95`, false point
  identification at most `.05`, and observed-world AUC in `[.48,.52]`.
- Frozen-map DPI violation is at most `.01` bit and the conditional-covariance
  Loewner-order minimum eigenvalue is at least `-1e-8`.
- `ROUTE_NULL` licensed false-ontology rate is at most `.05`; naive route
  heterogeneity must occur in at least `.20` of repetitions so the attack is
  non-vacuous.
- `SHARED_LATENT` relation-pattern 5th percentile is at least `.80`, mean
  individual-correlation 95th percentile is at most `.50`, and structural
  license rate is at least `.95`.
- `COMMON_NUISANCE` structural license rate is at most `.05`.
- Fixed-reference shift direction 5th percentile is at least `.95`, amplitude
  error 95th percentile is at most `.10`, population relation shift 5th
  percentile is at least `.25`, and reference mismatch refusal is at least
  `.95`.

## 6. Required outputs

- `component_recovery.csv`
- `commutation_defects.csv`
- `refusal_calibration.csv`
- `uncertainty_coverage.csv`
- `information_order.csv`
- `route_nonontology.csv`
- `relational_lift.csv`
- `reference_drift.csv`
- `decision.json`
- `run_manifest.json`
- `artifact_inventory.json`

## 7. Boundary

Passing licenses a synthetic typed-graph implementation: recovery when
design identifies an object, refusal when two latent worlds are observationally
equivalent, and separation of population structure from individual prediction.
It cannot establish construct validity, clinical meaning, invariance across
real language/embedding systems, or same-person stability across real
contexts.

The mathematical alignment is with Blackwell comparison of experiments,
information bottleneck/data processing, canonical relation analysis, and the
ecological-correlation warning. These precedents justify the restrictions;
the GRAIN contribution is their combination in one evidence-carrying
natural-language measurement graph.
