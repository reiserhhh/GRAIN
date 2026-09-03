# GRAIN M4-B Endogenous Opportunity Ecology

Date: 2026-07-31  
Status: finite synthetic discovery passed with scope correction; spectral
radius ranking remains open; no natural-text or psychological interpretation

## 1. Development question

The original rind idea treated topic, situation, platform, and prompt as
conditions of expression. M4-B adds a missing mechanism: authors do not only
respond inside an opportunity set. They may select, create, maintain, avoid,
and return to opportunities differently.

The central falsifiable statement is

\[
P(Q=k)\text{ is equal}
\quad\not\Rightarrow\quad
\text{selection, opportunity generation, feedback, and return are equal}.
\]

Thus a stable topic or situation proportion is not itself a mechanism.

## 2. Path law

For author \(u\), category \(k=1,\ldots,K\), and event \(t\), define

\[
Y_{u,t}
=
(E_t,M^{ext}_{u,t},M^{gen}_{u,t},M_{u,t},
Q_{u,t},X_{u,t},H_{u,t},D_{u,t}).
\]

- \(E_t\) is the external environment;
- \(M^{ext}\) and \(M^{gen}\) are externally supplied and author-generated
  opportunity menus;
- \(M=1-(1-M^{ext})(1-M^{gen})\) is the observable menu;
- \(Q_t\) is the selected category or the outside option;
- \(X_t\) is the expression state;
- \(H_t\) is observable path history; and
- \(D_{t,k}\) is time since the last selection of category \(k\).

The declared event order is

\[
E_{t+1}
\to
M^{ext}_{t+1},
\qquad
(M_t,X_t,H_t,D_t)
\to
Q_t
\to
X_{t+1}
\to
M^{gen}_{t+1}
\to
(H_{t+1},D_{t+1}).
\]

The choice kernel is an availability-restricted softmax:

\[
P(Q_t=e_k\mid M_t,X_t,H_t,D_t)
=
\frac{
M_{t,k}\exp\eta_{u,k}(X_t,H_t,D_t)
}{
1+\sum_jM_{t,j}\exp\eta_{u,j}(X_t,H_t,D_t)
}.
\]

The response transition is

\[
X_{t+1}
=
F_uX_t+G_uQ_t+R_uH_t+\epsilon^X_{t+1}.
\]

The generated-opportunity hazard is

\[
\begin{aligned}
\operatorname{logit}P(M^{gen}_{t+1,k}=1)
=\;&
\alpha^{gen}_{u,k}
+r_{u,k}^{\top}M_t
+b_{u,k}^{\top}X_{t+1}
+c_{u,k}^{\top}Q_t\\
&+
g_{u,k}(H_t)b_{u,k}^{H\top}X_{t+1}
+d_{u,k}(D_{t,k}).
\end{aligned}
\]

This ordering defines a finite path law without simultaneous equations.

## 3. Local ecology operators

Because menu coordinates are binary, the choice response to opportunity is a
finite difference:

\[
B_u^{C\leftarrow O}[:,k]
=
E[
m_C(O_k=1)-m_C(O_k=0)
].
\]

The remaining local operators are

\[
B_u^{X\leftarrow C}
=
\frac{\partial E[X_{t+1}]}{\partial Q_t},
\qquad
B_u^{O\leftarrow X}
=
\frac{\partial E[M_{t+1}]}{\partial X_{t+1}}.
\]

Their closed-loop composition is

\[
L_u
=
B_u^{O\leftarrow X}
B_u^{X\leftarrow C}
B_u^{C\leftarrow O}.
\]

With direct opportunity persistence \(R_u^O\),

\[
J_u^O=R_u^O+L_u.
\]

The proposed mesoscopic ecology signature is

\[
\Lambda_u^{eco}
=
\left(
K_u^C,K_u^O,L_u,J_u^O,\rho(J_u^O),
\tau_u^{return},\tau_u^{recovery},G_u^{history}
\right).
\]

The full matrix \(L_u\) is primary. The scalar spectral radius
\(\rho(J_u^O)\) is a lossy stability summary and must not replace the loop
geometry.

## 4. Three disjoint data roles

M4-B corrects a selection-optimism boundary left by the M4-A dynamic battery.
Every independent author view is split into:

1. **calibration**: fit \(K^O,K^C,K^X\), scaling, and all candidate families;
2. **selection**: choose among registered `base`, `return`, `feedback`, and
   `gate` opportunity hazards; and
3. **evaluation**: compute final path score, mechanism signature, return,
   recovery, and replication metrics once.

No model family, regularizer, gate position, or threshold may be selected on
the evaluation panel.

## 5. Finite discovery worlds

| World | Planted distinction |
| --- | --- |
| `null_exogenous` | common external menu and no author ecology difference |
| `exogenous_selection` | authors differ only in conditional choice |
| `endogenous_creation_matched` | authors create opportunities through expression feedback |
| `fast_return_equal_marginal` | low persistence and fast return |
| `slow_hysteresis_equal_marginal` | high persistence and slow return |
| `history_gated_ecology` | history opens expression-to-opportunity feedback |
| `selection_creation_compensation` | selection and generation oppose one another |
| `hidden_opportunity_alias` | source menus are observationally identical |

The primary matched pair is `exogenous_selection` versus
`endogenous_creation_matched`. The raw category-availability vectors must
satisfy

\[
D_M=\frac12\|\bar M^{(1)}-\bar M^{(2)}\|_1\le 0.03,
\]

and the categorical choice distributions must have total-variation distance
at or below 0.03 before mechanism discrimination is interpreted. \(D_M\) is a
half-\(L_1\) discrepancy, not a probability TV, because a multi-label menu
mean need not sum to one. Fast and slow return worlds provide a second
equal-marginal contrast.

## 6. Identification and refusal

Selection and opportunity generation can be separated only when:

- source menus are observed;
- each category has common support and enough transitions;
- innovations follow the declared event order;
- the fitted hazard and transition families are adequate locally; and
- the process remains inside the registered stability region.

If external and generated menus are observationally identical, M4-B refuses
source decomposition. Predicting the union menu does not identify which
source created it.

## 7. Discovery result

Eight repetitions across eight finite worlds produced:

| Diagnostic | Result |
| --- | ---: |
| mechanism-type macro F1 | 0.991 |
| active-edge support F1 | 0.996 |
| active-edge sign accuracy | 0.992 |
| feedback-loop truth geometry | 0.920 |
| return-time truth Spearman | 0.942 |
| recovery-time truth Spearman | 0.981 |
| matched selection/creation balanced accuracy | 0.998 |
| matched raw-menu half-\(L_1\) | 0.006 |
| matched choice TV | 0.013 |
| history-gate specificity margin | 0.936 |
| active same-author signature AUC | 0.830 |
| hidden-source alias refusal | 1.000 |
| null false-positive rate | 0.000 |

The matrix feedback loop \(L_u\) is therefore recoverable in this finite
family even when population menu and choice marginals are matched. The
spectral-radius rank correlation was only 0.473 against a registered target
of 0.80. This does not negate the recovered loop geometry. It shows that
compressing a noisy, possibly non-normal matrix \(J_u^O\) to
\(\rho(J_u^O)\) is too sensitive to preserve author ordering.

The validated core is consequently

\[
\widetilde{\Lambda}^{eco}_u
=
\left(
K_u^C,K_u^O,K_u^X,L_u,
\tau_u^{return},\tau_u^{recovery},
G_u^{history},R_u^{refusal}
\right).
\]

\(J_u^O\) remains a defined local operator, but its spectral radius is an
engineering diagnostic rather than a validated GRAIN measurement coordinate.
No claim about ecological criticality, instability, burst threshold, or
author rank in \(\rho(J_u^O)\) is licensed.

## 8. Claim boundary

If the finite battery passes, GRAIN may claim that:

- opportunity is an endogenous path object, not only a nuisance condition;
- selection, creation, return, feedback, and history gating are distinct
  mathematical operators;
- matched topic/situation proportions do not identify those operators; and
- an anonymous author ecology signature can exist in designed path data.

The battery cannot establish that these mechanisms exist in human text, that
they are personality traits, or that the simulator labels name psychological
constructs. Logistic coefficients, \(K=4\), \(d=2\), thresholds, and synthetic
AUC values remain engineering diagnostics rather than theory primitives.

## 9. Reproduction

```bash
python scripts/run_suica_m4_opportunity_ecology.py \
  --config configs/m4_opportunity_ecology.json
python -m pytest -q tests/test_m4_opportunity_ecology.py
```

Primary artifacts:

- `configs/m4_opportunity_ecology.json`
- `results/m4_opportunity_ecology/decision.json`
- `results/m4_opportunity_ecology/metrics.csv`
- `results/m4_opportunity_ecology/matched_marginal_metrics.csv`
- `reports/SUICA_M4_OPPORTUNITY_ECOLOGY_DISCOVERY.md`
