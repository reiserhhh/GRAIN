# HM153: Context diagnostic shape, not accuracy optimization

Prospective finite plan, 2026-10-11. NOT executed at creation. Seal this plan,
module, runner, verifier and tests before inspecting the new vocabulary fields.
HM152/151/150 and their failures remain immutable. Original18/100/44 stays OPEN.
Previous goal turn was read-only route review: NO_PROGRESS_EXECUTION, not a wait.

## Unknown and discriminating observation

HM152 K can change even when context applies exactly the same multiplicative
lexical tilt to high/low persona laws. Different base laws then have different
normalizers. This does not establish a change in relative diagnostic preference.
Competing accounts: (A) a common tilt, up to scalar gauges; (B) different lexical
tilts under H/L. Compare candidate words, not another prediction score.

For each aligned second-slice position let

    delta = (h_Hcarry-h_Lcarry) - (h_Hreset-h_Lreset)
    U(v) = W[v] . delta
    DeltaA = A_Hcarry-A_Lcarry-A_Hreset+A_Lreset
    k(v) = U(v)-DeltaA
    Gamma(v,w) = k(v)-k(w) = (W[v]-W[w]) . delta.

Gamma=0 for all v,w iff the normalized H/L diagnostic contrast changes by a
vocabulary-constant offset. This is equivalent to the two context tilts being
proportional. Nonzero Gamma rejects strict common lexical tilt, NOT every
generic grammar/prompt explanation and NOT establishes human personality.

The full-vocabulary oscillation D=max U-min U is the Hilbert projective distance
between normalized positive tilt measures T_H proportional to p_Hcarry/p_Hreset
and T_L proportional to p_Lcarry/p_Lreset. These auxiliary measures are not
trait posteriors or an author's choice distribution. Extreme D can be rare-token
driven: retain argmin/argmax native IDs and their reference probability.

Declare q(v) proportional to sqrt(p_Hreset(v)p_Lreset(v)). The tied linear head
gives q=softmax(W[(h_Hreset+h_Lreset)/2]); this is a geometric reference law,
NOT a causal claim that the midpoint is an actual decoder state.

    m=E_q U; S(v)=U(v)-m; a=m-DeltaA; k(v)=S(v)+a.

Keep shape AND calibration, signed terms, and cancellations. Weighted SD_q(U)
measures variation among plausible baseline alternatives; not trait strength.
Both terms are logit-gauge invariant under independent constant shifts. Dual
invertible coordinate changes preserve the field. No automatic noise deletion.

## Fixed support, resource and output contract

All972 frozen HM152 cases from all702 HM150 articles; first32 native tokens,
boundary16. All15552 second-token units, all108 persona condition cells. The
two targets/three families/six tasks/three styles/three generation seeds remain.
Seeds NOT people;14 refusals/519 generator truncations remain. No new articles,
decoder calls, prompts, API, training, private/MEPS data, Claude, Git, lockbox,
automation or old queues. Model-dependent states/weights are still required.

Additionally compare E/N on all54 neutral articles x3 families =162 pairs,2592
aligned units. Common q0 is geometric mean of the four reset laws. Report
weighted field cosine, covariance, (U_E+U_N)/2 and (U_E-U_N)/2 variances with
their exact sum identity. Negative/zero/undefined retained; no alignment fit,
latent psychological correlation or information-share claim.

One FP32 MPS output-head projection per case/pair; weights are exact archive
BF16 values converted to FP32. No transformer decoder instantiated. Main has
31104 rows; paired comparison7776 rows; total38880,1134 projection batches.
CPU64 reductions after projection, full vocabulary151936.600seconds INCLUDING
dependency checks/loading/IO. Fail and preserve partial receipts if exceeded;
no automatic rerun/fallback/budget renewal. Audit independently capped120seconds.
No selecting subset or growing runtime from observed outcomes.

Save manifest/pairs,status,per-case and cross-pair compact npz,token_accounts.csv,
case_accounts.csv,condition_accounts.csv,cross_target_accounts.csv,summary and
output seal. Selected native accounts must match at abs1e-4 PLUS rel1e-5; exact
stored algebra abs1e-9 PLUS rel1e-9. No score-based promotion threshold.

CPU64 alternative checks all selected-vector accounts/CSV/sums108conditions/
2592cross accounts. Full-vocabulary recomputation only fixed cases0,6,485,971
positions0,15; cross pairs0,161 position15. Same-agent alternate arithmetic,
NOT blind independent replication and NOT all-vocabulary rerun.

## What changes the mainline

Constant field: retain calibration transport; stop claiming lexical diagnostic
shape from K alone. Nonconstant plausibly weighted field: context changes
relative diagnostic preferences, so evidence compilation needs that conditional
field, not only an additive scalar. E/N alignment/anti-alignment describes shared
and distinct action on the same evidence; it is not the trait correlation.
Next investigate source-supported reusable target fields/conditions, not AUC,
rank, head, kernel, prompt or likelihood-classification rescue.

## Prior art and exact limits

Existing HM10B1 already uses CLR/softmax geometry; HM88/102 study attention-event
normalizers; HM108 output-digit four-corner action. None are claimed absent.
This packet separates alternative-token diagnostic shape from constant
calibration in actual HM152 writer-prefix observations. PICLe persona likelihood
ratios and Persona Vectors steering are predecessors, not our inventions.

[Cohen and Fausti, Section4, Eq24](https://arxiv.org/html/2309.02413v2) supplies
finite-positive-cone Hilbert distance. Its positive-LINEAR-operator contraction
theorems do NOT apply to nonlinear transformer transitions here.
[Compositional geometry Section2.1](https://link.springer.com/article/10.1007/s11004-020-09895-w)
defines perturbation as normalized componentwise multiplication and CLR.
Their operations justify the common-tilt equivalence, not psychological labels.
OpenAI/math pinned adc7f124, Universal-Tensor-Squares preliminaries motivates
operation-compatible dual tests; text does not satisfy its Specht/GL hypotheses.
No new group-theoretic theorem is transferred or claimed.
