# Complete-vector target relations and their observation support

2026-10-13 JST. HM242 finite construction/verification complete; no live job.
Canonical: `results/grain_math_response_v1/target_relation_operator_r1`.

## 1. The additional measurement object

An author contributes a pair of1024D window means (x_A,x_B), not two independent
people. A target-dependent relationship between those vectors need not be a
target-dependent mean vector. A cold source function can therefore be bilinear:

    phi_bq(xA,xB) = (xA-refA_b)^T T_bq (xB-refB_b).

T_bq is a SOURCE-only difference of sample cross-covariances between the high
and low inherited q groups. Fixed source grand means provide the reference;
no batch of application authors is used to standardize a new author. Output is
a signed, unbounded covariance witness, not a posterior probability, trait
level, diagnosis or calibrated psychometric scale. All source targets and the
unlabelled common covariance remain available.

Prior interfaces already established this general kind of relationship:
HM175 within/between target-profile dependence in8cells; HM176/177 paired
author and target assignment in4states; HM239 identity/moment separation in
full vector means. HM242's addition is the complete1024D target covariance
operator, its exact dual cold evaluation, and the empirical source/support
comparison. It is not invention of covariance or proof of an attention circuit.

## 2. Within and between profiles remain distinct, not deleted

For binary target group a containing n_a authors, m_va is its view mean and
m_vg is a complete-profile mean. The sample cross-covariance decomposes exactly:

    C_a = W_a+B_a
    W_a = sum_g sum_(i:g) (x_iA-m_Ag)(x_iB-m_Bg)^T/(n_a-1)
    B_a = sum_g n_g (m_Ag-m_Aa)(m_Bg-m_Ba)^T/(n_a-1).

Then T_q=C_1-C_0=W_q+B_q. Common C uses all authors and the same profile
decomposition. The n_a-1 denominator is kept in BOTH components. This exact
finite decomposition is not a population causal partition or two independent
psychological mechanisms. A singleton profile has an empirical within sum0
by definition; its population covariance is not thereby identified as0.
One of the old183-author bank's profiles is a singleton and is reported intact.

The target reference is empirical authors inside each binary group. It is NOT
HM209's uniform32-profile reference. Between-profile structure is preserved;
no topic, situation, common component or residual is removed from a predictor.

The dual representation uses only source-author matrices:

    T_bq = Z_bA^T H_bq Z_bB
    <T_0q,T_1r>_HS = sum(H_0q * (K_A01 H_1r K_B01^T)).

All signed cross-component/cross-target products are retained. A large norm
does not establish target evidence; a negative product is not negative trait
variance. Linear kernels can miss non-linear dependence, so zero covariance
does not mean independence or absence of author/target information.

## 3. Actual finite observation

SAME386 historical PANDORA SOURCE authors, disjoint203/183 banks, both fixed
twelve-comment views, inherited full five-bit profiles; all five targets.
No new raw text, labels, neural forwards, API, private data, training or fits.

| Target | Within-profile cross-source operator cosine | Total | Between-profile |
| --- | ---: | ---: | ---: |
| E | -.058981 | -.061889 | -.004925 |
| N | +.001049 | +.000080 | -.032865 |
| A | +.030280 | +.023624 | +.008457 |
| C | -.027471 | -.020052 | +.023839 |
| O | +.053146 | +.047574 | -.010383 |

The common within-profile operator cosine is+.614158. The actual5target
contrasts do not show comparably large ambient agreement; positive3/5 is not
a significance test. This diagnoses a difference between shared paired-vector
structure and its target-indexed covariance contrasts, not general absence of
personality signal. Sampling variability, reference composition, embedding
geometry and higher/conditional relationships remain distinct explanations.

One fixed cyclic B relink inside complete profiles keeps every profile mean
and B unchanged (literal max3.52e-20) while changing W (squared operator
differences3.47e-5..8.00e-5). This is a source identity audit, not new texts,
new human behavior, or an optimization. The old mean atlas cannot observe that
specific pairing change; the new bilinear interface can.

## 4. Ambient similarity is not functional agreement on observed support

Both frozen banks scored all24old author pairs/all9old evidence frames. On the
full histories the within-profile witness score-vector cosines are:

    common .985311; E -.703786; N -.073997; A -.212846;
    C +.116045; O +.617590.

The O witness thus agrees locally on this SAME24-author support while its
ambient operator cosine is only+.053146. Keep that positive result AND the
E/A reversals. Neither metric is a questionnaire correlation, personality
accuracy, independent test-retest coefficient or automatic transfer claim.

Mathematically, for fixed augmented pair coordinates v(x) (the vectorized
outer product of the centered windows), a finite support defines a Gram

    Q_X = average_x v(x)v(x)^T,
    average_x phi_0(x)phi_1(x) = vec(T_0)^T Q_X vec(T_1).

Different source references can be incorporated into augmented affine pair
features [x_A,1] tensor [x_B,1]. Q_X is used here only to describe observed
scores, not fit or rotate an operator. Euclidean/Frobenius geometry corresponds
to a different reference over the ambient pair space. This formalizes why
weak global operator alignment can coexist with local functional agreement.
HM157/158/233 already distinguished geometry from effective target action;
this is its concrete second-order source-witness instance, not a new general
geometry theorem. We do not choose O or alter the estimator based on this table.

## 5. Scientific decision and remaining mainline

Close this unconditional covariance-contrast family after its finite result.
Do not select dimensions, weights, ranks, thresholds, source partitions or
regularizers to rescue its five targets. Retain the common relation as a
usable GRAIN observation and all target-specific witnesses as research outputs.

The next missing connection is a condition-qualified target observation:
which same-event relation is relevant to which queried/observable target,
and how its source-supported meaning relates to the existing vector action.
Inventory HM177's assignment, HM208/209's addressed windows and HM233's
same-event meaning bridge before a new finite plan. The next packet must add
a target-use/measurement capability, not repeat generic author recurrence or
globally align operators to LLM answers. No new factored personality axes are
named. Choice, condition response, state, path and residuals remain in scope.
Original18 protocols/100cards/44questions are still OPEN.

## 6. Mathematical provenance and reproduction

Cross-covariance operators, tensor products and Hilbert-Schmidt contractions
are mature tools: [Gretton et al.2005, Sections2-3](https://www.gatsby.ucl.ac.uk/~gretton/papers/GreBouSmoSch05.pdf).
Their characteristic/universal-kernel independence theorems are not applied
to our linear covariance witness. OpenAI/math pinned adc7f124's signed-tensor
03b-isotypic1-120 was reread for retaining specified signed components; its
symmetric-group/Specht hypotheses do not hold for these author vectors, and
no new theorem was transferred. Newness here is a project-specific source
construction and empirical separation, not a claimed original HSIC method.

Prepare/compute/verify/check ran once each, all exit0. Compute.603991s and
alternate verification1.386966s CPU. Seven new/27focused tests pass;3newPython
Ruff lint/format pass. Blockwise literal ambient covariances verify all1296
bank/target/component Grams and7776cold values; max4.25e-19. Separate cold
process exact readback. Same agent alternate arithmetic, not blind replication.
These tests authenticate the calculation, not a human psychological construct.
The final ancestor integration set passed55tests. One earlier integration
invocation used a nonexistent test filename and ran no tests; the corrected
existing-file command passed without any scientific-code/result change.

```sh
cd '/Volumes/mobile3/projects/Sliced Utterance In-Context Assessment'
'/Volumes/mobile3/projects/project persona/.venv/bin/python' scripts/run_grain_target_relation_operator_hm242.py check
'/Volumes/mobile3/projects/project persona/.venv/bin/python' scripts/run_grain_target_relation_operator_hm242.py cold
```

Cold accepts only frame identity and two windows of distinct event_id/vector/
condition records; source banks are fixed, application labels are rejected.
The downstream scoring no longer invokes a decoder; the cached E5 encoder
remains neural. Old sealed failures/results and scores are unchanged.
