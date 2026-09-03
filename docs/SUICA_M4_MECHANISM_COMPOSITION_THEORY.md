# GRAIN M4-A Mechanism Composition Grammar

Date: 2026-07-31  
Status: finite synthetic discovery supported with scope correction; gate
naming heuristic remains open; no joint-kernel, human-text, or psychological
interpretation

## 1. Development question

M3 established that several author-associated mechanisms can be recovered
selectively:

\[
\mathcal A_u
=
(\mu_u,\mathcal R_u,\mathcal K_u,\mathcal C_u,\mathcal P_u).
\]

That result did not define how mechanisms combine. Concatenating five
estimators cannot distinguish:

- two independent additive mechanisms;
- a genuine joint mechanism;
- two redundant views of the same source;
- a suppressor that becomes informative only conditionally;
- a directional gate; or
- an order-sensitive decomposition created by dependent drivers.

M4-A introduces the missing composition grammar.

## 2. Three mathematical layers

### 2.1 Joint event kernel

Let the augmented event state be

\[
Y_t=(O_t,S_t,C_t,I_t,X_t,H_t),
\]

where the components denote opportunity, state, condition, interaction,
expression response, and history. For a declared module order \(\sigma\),

\[
K_u^{\mathrm{joint},\sigma}
=
K_{u,\sigma_1}\circ\cdots\circ K_{u,\sigma_6}.
\]

Given measurable conditional kernels and an initial distribution, the
Ionescu--Tulcea construction defines a unique finite path law for this
declared order. The order is part of the model, not a causal fact.

### 2.2 Function decomposition

Relative to a declared reference \(K_0\), write

\[
\ell_u=\log\frac{dK_u^{\mathrm{joint}}}{dK_0}.
\]

For independent mechanism coordinates, conditional projection and Möbius
inversion yield the ordinary functional-ANOVA components. For dependent
coordinates, non-nested components need not be orthogonal and projection
order matters. The general target is therefore a hierarchical-orthogonal
decomposition under the joint measure:

\[
\ell_u=\sum_{A\subseteq\mathcal M}\psi_{u,A},
\qquad
\langle\psi_{u,A},g_B\rangle_{L^2(K_0)}=0
\quad(B\subsetneq A).
\]

M4-A V1 does not claim to solve general dependent-input HOFD. It implements a
finite multilinear approximation and exposes the dependence problem rather
than hiding it.

### 2.3 Contribution lattice

For a frozen coalition value \(v_u(A)\), define the Harsanyi/Möbius dividend

\[
d_u(A)
=
\sum_{B\subseteq A}
(-1)^{|A|-|B|}v_u(B),
\]

so that

\[
v_u(A)=\sum_{B\subseteq A}d_u(B).
\]

The truncated Shapley allocation is

\[
\phi_{u,m}
=
\sum_{A\ni m}\frac{d_u(A)}{|A|}.
\]

Shapley values distribute shared contribution; they do not identify its
mechanism. The full lattice is retained.

## 3. Two games, not one

V1 estimates two coalition games for every author from independent event
panels.

### Observational game

\[
v_u^{\mathrm{obs}}(A)
=
1-
\frac{
\operatorname{MSE}_{\mathrm{heldout}}
[\widehat f_{u,A}(X_A),Y]
}{
\operatorname{MSE}_{\mathrm{heldout}}[\bar Y_{\mathrm{train}},Y]
}.
\]

It preserves the observed dependence among mechanism drivers. Its dividends
therefore combine response structure and input dependence.

### Product-reference game

The full response law is fitted on training occasions. Evaluation then
independently permutes each mechanism coordinate, preserving its marginal but
breaking cross-mechanism dependence:

\[
Q_u^\otimes
=
\bigotimes_{m\in\mathcal M}Q_{u,m}.
\]

The product-reference value is the variance share of the projected fitted
law under \(Q_u^\otimes\). Its dividend estimates functional-ANOVA interaction
of the fitted multilinear law under this product measure. It approximates a
planted structural interaction only in the matched V1 synthetic family.

The dependence distortion is

\[
\delta_u^{\mathrm{dep}}(A)
=
d_u^{\mathrm{obs}}(A)-d_u^\otimes(A).
\]

This is the central M4-A distinction. A high observational interaction is not
automatically a joint response mechanism.

## 4. Direction and non-commutation

Symmetric interaction energy cannot distinguish a gate from ordinary
synergy. For candidate gate \(g\) and target mechanism \(i\), V1 compares
target slopes on the two gate regions:

\[
G_{g\rightarrow i}
=
\frac{
|\widehat\beta_i^{g+}|-|\widehat\beta_i^{g-}|
}{
|\widehat\beta_i^{g+}|+|\widehat\beta_i^{g-}|+\epsilon
},
\]

and accepts a gate direction only when the slope orientation does not reverse.
A symmetric product reverses its slope and is therefore not called a gate.

For dependent projection spaces, V1 also estimates

\[
\Omega_{ij}
=
\frac{
\|\widehat P_i\widehat P_jY-\widehat P_j\widehat P_iY\|_2
}{
\|Y-\bar Y\|_2
}.
\]

\(\Omega_{ij}>0\) means the two projection operations do not commute. It does
not identify temporal or causal order. Dynamic kernel order remains an M4-D
problem.

## 5. Author mechanism-composition signature

The new mesoscopic object is

\[
\Lambda_u
=
\left(
\{d_u^\otimes(A)\},
\{\delta_u^{\mathrm{dep}}(A)\},
\{G_{g\rightarrow i}\},
\{\Omega_{ij}\},
\{\phi_{u,m}\}
\right).
\]

The theory chain is upgraded to:

\[
K_u^{\mathrm{joint}}
\longrightarrow
\Lambda_u
\longrightarrow
\mathcal A_u
\longrightarrow
J_{uv}(z)
\longrightarrow
\mathcal T_{\mathrm{population}}.
\]

Two authors may have similar marginal mechanism scores but different
composition signatures because their mechanisms are redundant, mutually
revealing, gated, or order-sensitive in different ways.

## 6. V1 discovery worlds

The finite battery uses six named precursor drivers, including an
`emission_drive` proxy, and a separate scalar response. This is a seven-object
engineering system rather than the complete six-component joint kernel above.
It truncates the multilinear lattice at order three:

\[
{6\choose1}+{6\choose2}+{6\choose3}=41
\]

candidate blocks.

| World | Planted distinction |
| --- | --- |
| `additive_dependent` | dependent unused drivers plus an additive response |
| `synergy` | genuine state-by-condition product |
| `redundancy` | positively correlated same-direction predictors |
| `suppression` | positively correlated opposite-direction predictors |
| `gate` | history opens condition response without reversing its direction |
| `projection_order` | a three-driver dependence chain with noncommuting predictive projections |
| `composite` | two pair mechanisms plus one signed third-order hyperedge |
| `alias` | perfect opportunity/condition equivalence |
| `null` | no author response law |

Each repetition generates independent train/test occasions with shared author
parameters. The estimator never receives the world label, planted parameters,
or active hyperedges.

## 7. Discovery result

Eight repetitions across all nine worlds produced:

| Diagnostic | Result |
| --- | ---: |
| mechanism-type macro F1 | 0.941 |
| active hyperedge support F1 | 1.000 |
| interaction sign accuracy | 1.000 |
| mean planted-parameter geometry | 0.736 |
| gate direction margin | 0.357 |
| alias refusal | 1.000 |
| null hyperedge false-positive rate | 0.000 |

The decisive separations were:

| World | Product-reference dividend | Observational dividend | Additional signal |
| --- | ---: | ---: | --- |
| synergy | +0.794 | +0.707 | target geometry 0.726 |
| redundancy | -0.003 | -0.522 | target geometry 0.978 |
| suppression | -0.004 | +0.433 | target geometry 0.764 |
| gate | +0.306 | +0.259 | correct direction 0.357; reverse 0 |
| projection order | approximately 0 | -0.268 | commutator 0.306; geometry 0.777 |

The composition signature also carried independent-panel author structure:

| World | Same-author AUC |
| --- | ---: |
| synergy | 0.640 |
| redundancy | 0.638 |
| suppression | 0.633 |
| gate | 0.608 |
| projection order | 0.682 |
| composite | 0.711 |
| null | 0.492 |

This is evidence that authors can differ in how mechanisms combine, not only
in their marginal levels.

## 8. Interpretation and open mechanism work

The finite synthetic result supports four development claims:

1. structural interaction and dependence-induced interaction are different
   mathematical objects;
2. redundancy and suppression have opposite observational dividends while
   both disappear under the product reference;
3. directional gate information is not contained in a symmetric interaction
   score; and
4. dependent predictive projections can carry a stable non-commuting author
   signature.

The coarse `gate` versus `synergy` label heuristic classified only 4/8 gate
replications at its registered diagnostic threshold. The directional gate
estimand itself remained positive in the correct direction and zero in the
reverse direction. The threshold is therefore not retuned on this discovery
sample; future work should use a continuous gate score or an independent
calibration panel.

Remaining boundaries:

- product-reference effects depend on the fitted response family and common
  support;
- the V1 multilinear basis is not a general RKHS-HOFD;
- projection non-commutation is not generating-kernel, causal, or temporal
  order;
- the six named drivers are planted simulator channels, not discovered human
  variables;
- sparse order-three recovery does not prove that real composition is
  low-order; and
- no component is licensed as personality, state, or emotion.

The static grammar alone was sufficient to start M4-B interface development,
but not to claim transition-kernel composition. A second development world
therefore replaces contemporaneous drivers with actual condition/history
transition kernels:

\[
O_{t+1}
\sim
K_u^O(\cdot\mid H_t,C_t,X_t,\text{others}),
\]

and whether authors differ in creating, maintaining, avoiding, and returning
to their own opportunity ecology.

## 9. Dynamic kernel composition result

The dynamic extension supplies three designed regimes:

1. condition-only transitions identify \(T_C\);
2. history-only transitions identify \(T_H\); and
3. held-out joint transitions compare \(T_HT_C\) with \(T_CT_H\).

For each author, the estimator composes independently fitted affine kernels
in both orders. The held-out order score is

\[
\Delta_u^{\mathrm{order}}
=
\frac{
\operatorname{MSE}(T_CT_H)-\operatorname{MSE}(T_HT_C)
}{
\operatorname{MSE}(T_CT_H)+\operatorname{MSE}(T_HT_C)+\epsilon
}.
\]

After selecting the better order, the transition residual estimates the
temporal gate \(H_t\to(C_t\to X_{t+1})\). Eight repetitions of forward,
reverse, commuting, and role-alias worlds produced:

| Diagnostic | Result |
| --- | ---: |
| order accuracy above the planted resolution floor | 0.828 |
| commutator truth-geometry Spearman | 0.940 |
| gate-strength Spearman | 1.000 |
| gate direction margin | 0.936 |
| held-out path log-score gain | 1.347 |
| active dynamic-signature same-author AUC | 0.856 |
| commuting-world same-author AUC | 0.622 |
| active AUC increment over commuting | 0.234 |
| alias refusal | 1.000 |

The commuting world exposed an important theoretical distinction. Commutation

\[
T_CT_H=T_HT_C
\]

removes composition order, but it does not remove author-specific marginal
kernels \(T_C\) or \(T_H\). Its author AUC therefore need not be chance. The
correct null properties were recovered instead: order margin 0.0004, gate
margin 0.0018, and a small finite-sample commutator 0.0317.

This dynamic result upgrades \(\Omega_{ij}^{\mathrm{proj}}\) to a designed
transition-kernel counterpart:

\[
\Omega_{CH}^{\mathrm{kernel}}
=
\|T_HT_C-T_CT_H\|.
\]

It remains a designed synthetic identification result. Natural text usually
does not provide condition-only and history-only calibration regimes, so an
observational estimator cannot inherit this identification for free.

The reported path log-score gain is the total gain of the calibrated
condition/history kernels, their selected order, and the fitted temporal gate
over a pooled affine transition. It is not an isolated estimate of the order
contribution. V1 also selects the candidate order and reports its error on the
same held-out half-panel. This leaves a small selection-optimism channel even
though fitting remains separated from evaluation. M4-B must therefore assign
three disjoint roles: kernel calibration, mechanism selection, and untouched
final evaluation.

The gate-strength rank correlation of 1.000 is valid only inside the matched
linear temporal-gate family. It is not evidence that an arbitrary nonlinear
or natural-text gate is fully identified. The static generic
gate-versus-synergy naming heuristic remains open even though the narrower
designed temporal gate is recovered.

M4-A now closes a finite synthetic static/dynamic mechanism space and is
sufficient to enter M4-B development. It still does not supply a universal
dependent-input HOFD, arbitrary nonlinear kernel composition, a discovered
human mechanism basis, or a general causal interpretation.

## 10. Reproduction

```bash
python scripts/run_suica_m4_composition_discovery.py
python scripts/run_suica_m4_dynamic_kernel_discovery.py
python -m pytest -q tests/test_m4_mechanism_composition.py
python -m pytest -q tests/test_m4_dynamic_kernel.py
```

Primary artifacts:

- `configs/m4_composition_discovery.json`
- `results/m4_composition_discovery/decision.json`
- `results/m4_composition_discovery/metrics.csv`
- `results/m4_composition_discovery/mechanism_lattice.csv`
- `reports/SUICA_M4_MECHANISM_COMPOSITION_DISCOVERY.md`
- `configs/m4_dynamic_kernel_discovery.json`
- `results/m4_dynamic_kernel_discovery/decision.json`
- `reports/SUICA_M4_DYNAMIC_KERNEL_GATE_ORDER_DISCOVERY.md`
