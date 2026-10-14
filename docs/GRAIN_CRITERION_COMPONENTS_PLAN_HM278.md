# HM278: shared criterion configuration versus within-configuration meaning

Finite plan written before computation, 2026-10-14 JST.

## Scientific decision, not a precision branch

HM277 observed a small positive N author link for F-J, while the complete F
reading was negative. The unknown is whether this relation is associated
with N conditional on the other four binary criteria, or with their shared
configuration. These are both potentially meaningful author properties.
Do NOT delete the latter as noise or fit a weight to improve prediction.

Reuse HM182's other-four profile interface and HM212's conditional assignment
orbit. Those previous studies addressed source vector contrasts/native actions,
not these new all386-author chronological-pair criterion readings. The new
ability is a cold source-qualified paired reading that retains composition
and within-configuration criterion meaning separately, without needing a
new author's other traits. This is not a new conditional-expectation theorem.

## Fixed construction

Use only HM277 public numerical readings/label-independent author weights and
HM273's frozen SOURCE bundle. All386 historical authors, both banks203/183,
all five binary criteria, two spaces and both old windows remain unchanged.
No new vectorization, geometry, kernels, sign choices or scoring parameters.
This is historical exploratory reuse, not fresh lockbox/paper reproduction.

For target q, let B_q be its other-four binary profile. On each SOURCE bank:

    m_jq = mean(Y_q | B_q=B_jq, SOURCE bank of j)
    e_jq = Y_jq - m_jq.

Every source profile side is required; absent support is an error, not zero.
Let D_uj be the unchanged opposite-bank F-J source-author weight difference:

    d_total_uq = sum_j D_uj Y_jq
    d_composition_uq = sum_j D_uj m_jq
    d_within_uq = sum_j D_uj e_jq
    d_total = d_composition + d_within.

No text, coordinate, context or criterion component is discarded. The word
"within" specifies a SOURCE criterion comparison, not an assertion about
the application author's B profile. All score inputs are opposite-bank only.
The held author's criterion departure is used for evaluation only, never as
an application scoring input. This is conditional on coarse BINARY criteria,
not continuous traits, community, topic, identity background or causal N.

## Competing predictions and primary family

If the relation is only a configuration resemblance, within-configuration
target assignment should not orient the paired score unusually. If it has
criterion-conditional meaning, the actual target assignment should yield a
signed covariance beyond its conditional assignment reference. Signs are not
flipped and no target is chosen after observing this result.

Primary for all22 pairs is

    T_sq = mean_u d_within_suq e_uq.

The held residual e_uq is centered within its OWN bank and B group solely in
evaluation. Report all10 primary T, conditional r, orbit SD, two-sided rank
and Holm rank. Also retain raw bank-centered correlations and criterion
covariance accounts for total/composition/within, by bank and each window.
Do not treat windows/banks as four independent replications.

199 fixed target assignments per q permute ONLY Y_q within bank x B_q.
Rebuild SOURCE score AND held criterion from each same assignment. Preserve
all other-four rows and all32 counts. The two bank permutations are independent;
the two spaces and three windows share their respective assignments.
These ten conditional nulls differ, so use Holm, not a fictitious joint
max-statistic from incompatible target orbits. Historical reuse limits the
interpretation to an exploratory conditional assignment reference.

There is an exact finite-group variance, requiring no Gaussian assumption.
Let V_0,V_1 be within-stratum permutation covariances of e in the two banks,
H=D_01+D_10^T, and n=386. Then

    E_orbit T = 0
    Var_orbit T = tr(H^T V_0 H V_1)/n^2.

This includes BOTH directions' dependence; neither bank is an independent
replicate of the other. An orbit SD ratio is not a Gaussian z significance.
Conditional covariance zero does not establish all conditional independence;
we are not implementing or claiming a full KPC/KCM test or latent cause.

## Execution and verification

One120s CPU cap, immutable stages prepare/compute/verify/check. Focused tests
first, scientific code frozen before prepare. No neural/API/private/MEPS/
training/Git/Claude/automation/lockbox/old queue. Cold numeric exemplar must
reproduce all three channels without an evaluated label. Verify direct
source-author sums, bank exclusion, all old totals, conditional assignment
counts and exact variance by a different stratum-block formula; tiny exhaustive
orbit is a unit test only, not a replacement for the natural-data question.

## Literature and source reuse

- Berrett et al.2019, https://arxiv.org/abs/1807.05405 : categorical-stratum
  permutation is an established special case. No continuous estimated CPT
  density/MCMC or its approximation-error guarantee is implemented here.
- Huang/Deb/Sen2022, https://www.jmlr.org/papers/v23/21-493.html : conditional
  dependence is more general than a scalar covariance. Its KPC estimator and
  conditional-independence equivalence are NOT transferred to our statistic.
- Muandet/Jitkrittum/Kuebler2020,
  https://proceedings.mlr.press/v124/muandet20a.html : conditional moment
  questions have mature kernel methods; this finite bank is not their MMR test.
- OpenAI/math pinned adc7f124 foundations98-116/166-232: explicitly qualify
  likelihood/reference and conditional products. No tree threshold or spin
  theorem is applied to humans; this remains design inspiration.

After this result, consolidate target-functional meaning rather than enlarge
assignment counts or tune r. Original18/100/44 remains OPEN.
