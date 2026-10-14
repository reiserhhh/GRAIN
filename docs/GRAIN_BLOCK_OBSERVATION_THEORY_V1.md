# Observation-design uncertainty is not author-generating uncertainty

HM254,2026-10-13. An executable finite-bank law and cold criterion interface.
This uses mature finite-sampling, representation and errors-in-variables
algebra. It is not a new theorem or a personality-accuracy promotion. All
HM253 science is unchanged. Original18/100/44 remain OPEN.

## 1. The condition is part of the measurement object

For each fixed author bank F, retain every event's feature
q=[1,c,(1-c)z,cz]. c is the inherited coarse venue condition, not a complete
topic/situation descriptor. Neither condition choice nor either coordinate
mass is removed. Consecutive chronological triples define B=12 blocks.
Uniform independent S3 assignments within blocks produce three observation
views, each with one event from every block. This group acts on the observation
assignment, NOT on a native text's order or a person's behavior.

Define nu_b=mean_j q_bj, Sigma_b=mean_j[(q_bj-nu_b)(q_bj-nu_b)^T],
mu_F=mean_b nu_b and Gamma_F=sum_b Sigma_b/(2B^2). Then exactly:

    E(phi_v|F)=mu_F
    Cov(phi_v,phi_w|F) = 2 Gamma_F if v=w; -Gamma_F otherwise.

Proof: within a triple a uniformly selected event has covariance Sigma_b.
For distinct views the ordered pair is uniform over six distinct event pairs.
Since sum_j(q_bj-nu_b)=0, its covariance is -Sigma_b/2. Independent BLOCK
assignments add covariances with weight1/B^2; no independence of human
texts or latent personality is required.

## 2. Which subspace contains the design fluctuation?

For stacked views Phi=(phi_0,phi_1,phi_2),

    Cov(Phi|F)=(3 I_3-1_3 1_3^T) tensor Gamma_F.

The common-view direction has eigenvalue0 and the two-dimensional contrast
space has eigenvalue3. This is the familiar trivial/standard S3 representation
split, not a new group-theoretic finding. It identifies the protocol-induced
fluctuation: views always sum to3mu_F, so their common direction cannot change
under this assignment. Contrast variation is not another stable author factor.

This DOES NOT license deleting genuine text dynamics: HM245's native sequence
actions and all GRAIN state/choice/relational channels remain separate. The
group here only changes who gets an already encoded event as an observation.
Treating every context/order difference as such a design fluctuation would
change the measured object and discard evidence.

For a frozen source score beta and a new author's bank,

    Var(phi_v^T beta|F)=2 beta^T Gamma_F beta,
    Var((phi_v-phi_w)^T beta|F)=6 beta^T Gamma_F beta.

These now have a cold executable interface. They are conditional finite-bank
sampling variances, not psychological variance fractions, single-conversation
confidence intervals, SOURCE coefficient uncertainty or behavioral prediction
error. No Gaussian model was assumed.

## 3. A corrected finite-bank bridge collapses to mean regression

SOURCE uniform-author second moments are M=E_u[mu_F mu_F^T], G=E_u Gamma_F,
m=E_u[mu_F Y^T]. Averaging over observation assignments gives

    K_cross=M-G; A_single=M+2G.

Thus correcting the conditional complementary-view covariance yields

    (K_cross+G)beta=m  <=>  M beta=m.

Under the same solution contract this is ordinary whole-bank regression. It
does not add observations or target information. In this packet the relative
moment tolerance is1e-12; direct OLS uses1e-6 on mu because its singular values
are squared in M. No threshold, rank or regularization was selected from data.

The uncorrected cross-moment matrix can be indefinite. One natural SOURCE
fold has minimum eigenvalue-4.7541e-7; it is retained, not projected to PSD.
Full sample rank is not a certificate of an independent-proxy latent model.

## 4. Why the correction is not universally correct

The condition F (collected bank) differs from U (author-generating law).
By total covariance under any generating-bank law,

    Cov(phi_0,phi_1|U)
      = Cov(mu_F|U)-E[Gamma_F|U].

Those two terms can cancel. Therefore negative covariance conditional on F
does not establish correlated errors conditional on U. Arbitrarily adding
Gamma can alter the intended latent propensity rather than recover it.

The exhaustive128-row example has theta in{-1,+1}, six iid errors in{-1,+1},
q=theta+error, two triples and Y=theta. It gives M_11=7/6, G_11=1/6 and:

    orbit proxy slope=1,
    finite-bank regression slope=6/7,
    single-view regression slope=2/3.

The first recovers the stipulated generating propensity; the second is the
regression on the noisy bank mean. This is standard errors-in-variables
behavior, not evidence that either law describes natural PANDORA comments.
The required law must come from qualified collection/design knowledge, not
from choosing whichever r is largest.

## 5. Exact accounting for the old deterministic allocation

Let K_old beta_old=m_old be an old HM253 equation, delta_K=K_old-(M-G),
delta_m=m_old-m. The exact signed identity is

    M beta_old-m
      = G beta_old + delta_m-delta_K beta_old
        +(K_old beta_old-m_old).

All four terms and all Gram cross terms are saved. No term is declared a
personality fraction or automatically removed. The old fixed interleave was
not a randomized draw from the new orbit. This decomposition cannot identify
the cause of all weak criterion correlations; it just states precisely what
changing observation contracts changes in the old SOURCE equation.

## 6. Actual result and next mainline decision

386 SAME public authors/13,896 SAME events;16 fixed observation assignments.
All coefficients calibrated on the other author fold; independent cold CLI
reads no labels, LLM answers or private text. Bank r values ENACO:

| Method | E | N | A | C | O |
| --- | ---: | ---: | ---: | ---: | ---: |
| orbit proxy | .02415 | .02040 | -.02548 | -.01923 | .03610 |
| finite bank | .04404 | -.00645 | .07958 | -.01057 | .07731 |
| single view | .02988 | -.03271 | .08819 | -.00608 | .10467 |

These are descriptive historical binary-criterion links. No method/partition
promotion, significance search, paper-benchmark comparison or psychological
factor naming. A/O positive associations are retained. Different methods
have different statistical risks; closing this branch does not claim they
must have equal finite-sample accuracy.

Exact mean partition variances ENACO for bank scoring are
.02912/.02915/.02372/.01661/.02242. For orbit-proxy scoring they are
9.9614/.9360/1.8800/.13646/.57479. These are score squared units, not probabilities
or human variance proportions. With only16 draws, empirical/exact average
variance ratios across all15 method-target cells are.9519-1.0217; no precision
extension was needed or run.

DECISION: close the fixed-bank mean-correction branch. Preserve its useful
uncertainty interface. Do not continue invertibility/rank/partition/precision
rescue as a mechanism program. Next construction must add target-qualified
joint evidence/conditional response or a justified generating observation
unit, rather than merely reweight the same means. FIRST inventory HM242/247
and HM250/251: existing relation witnesses and behavior anchors must not be
rediscovered or silently redefined as a new target law.

## 7. Prior art, scope and reproduction

[Niepert/Domingos2014 Sections2.1-2.2](https://homes.cs.washington.edu/~pedrod/papers/mlc14.pdf)
already distinguishes finite exchangeability and infinitely extendible iid
mixtures. The elementary covariance and regression identities are mature.
The pinned [OpenAI/math observation-law section166-210](https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/The-exact-reconstruction-threshold-for-the-three-state-symmetric-channel-September-25-2026/build/sections/02-foundations.tex)
inspired explicitly declaring independence before evidence composition. Its
tree/spin threshold and symmetry hypotheses are NOT transferred to people.

HM196 finite inventory, HM216 catalogue orbits and HM176 author covariance
are prior project results. The new system ability is the paired criterion
compiler plus cold conditional variance on the SAME actual public event bank,
and the explicit branch-closing comparison against the generating estimand.

Canonical: results/grain_math_response_v1/block_observation_law_r1.
prepare/compile/cold/analyze/verify/check each once,all exit0,no background job.
Compile.00861s,analysis.09091s excluding process/hash/write overhead. Zero new
neural/API/private/training calls; historical vector encoding remains neural.
35focused tests pass(8new),5 Python lint/format PASS. Enumerating27,792 natural
block assignments and289,500 cold scalar readings gives covariance error
1.39e-17, projected variance error1.07e-13, direct-OLS relative error2.59e-12.
Same-agent alternate arithmetic, not independent blind replication.

One malformed apply_patch was rejected atomically before creating the runner;
one prefreeze C417 style warning was fixed before registering science. No
scientific failure, rerun, postseal edit or old-result rewrite occurred.

Read-only check:

```sh
"/Volumes/mobile3/projects/project persona/.venv/bin/python" \
  scripts/run_grain_block_observation_hm254.py check
```

Goal ACTIVE; original18/100/44 OPEN. This packet is not full theory closure.
