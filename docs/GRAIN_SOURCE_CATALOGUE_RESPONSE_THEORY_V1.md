# Source catalogue sampling is not population response change

HM216,2026-10-12. The new construction and finite experiment are complete.
Goal ACTIVE/nojob; original18/100/44 still OPEN. This is a mechanism decision,
not prediction optimization, a new permutation theorem or psychological DIF.

## 1. What changed in the interpretation

HM215 showed that matching all32 current profile readings does not match the
SAME next-evidence response of two empirical instruments. That result remains
correct. HM216 identifies the instruments' provenance and tests the missing
competitor: their203/183 authors are hash-interleaved within each full target
profile of one386-author source catalogue. They are not two independently
sampled natural populations. The original response discrepancy is ordinary
relative to other allocations of that same catalogue.

| Allocation family | Observed cross-ratio RMS | Reference median | Reference2.5%-97.5% | Upper rank |
| --- | ---: | ---: | --- | ---: |
| Profile only | .045337949 | .046504567 | .042269905-.049599336 | .700 |
| Profile and exact two-window condition counts | .045337949 | .045163072 | .043461452-.046402809 | .400 |

These are log-response units and conditional allocation ranks, not accuracy,
Pearson r, psychological significance or proof of equal populations. The
second family moves only77 of386 authors;309 remain frozen. A large retained
conditional discrepancy cannot be assigned to condition effects just because
the restricted orbit cannot move most of the catalogue.

## 2. Measurement reference, catalogue and natural law are distinct

A useful contract is

```text
reading_q = M_q(evidence, target_reference, source_catalogue, observation_law)
response_q(x0,x1) = reading_q(x1)-reading_q(x0).
```

The target reference fixes profile/author weighting. The source catalogue is
the finite material used to construct an empirical law. The observation law
specifies how text vectors become observations. A natural population law is
not established simply by fixing those software ingredients.

For the inherited single-window Gaussian mixture, with prototypes z_i and
fixed s2, omit the query-only normalizer as in HM209:

```text
A_bg(x) = (1/n_bg) sum_(i in bank b,profile g) exp(x.z_i/s2-|z_i|^2/(2s2))
log[A_bg(x1)/A_bg(x0)]
  = log E_(omega_bg(x0)) exp((x1-x0).z_i/s2).
```

The observed response depends on the available prototypes and their weights.
Different finite libraries can give different responses even under a common
stipulated Gaussian rule and a common target reference. This identity does
not recover neural attention or demonstrate a natural psychological cause.

## 3. Whole-author group action and its limits

Hold every author's two windows, full profile and condition counts together.
Act on bank memberships with the product of symmetric groups within profiles.
The second family uses profile plus exact condition-count pair strata. These
are full groups; the set of their resulting count-preserving allocations is
the orbit. It is not the invalid restricted "balanced-permutation" set.

The observed allocation plus199 fixed draws in each family supplies the
declared rank(1+#draw>=observed)/200. It is a finite conditional diagnostic.
Random exchangeability of the hash-derived population and the historical
N-guided panel selection have not become a new randomized human study.

Support is part of the result:32 full-profile strata move386 authors, whereas
311 exact-condition strata contain only28 movable strata and77 authors.
All target labels, vectors, pairings and bank/profile counts remain unchanged.
No condition or author choice is deleted or relabeled as psychological noise.

## 4. Exact orbit-density expectation

For stratum h of N_h authors, n_bh fixed allocations to bank b, total profile
count n_bg, and two kernel values k_i0,k_i1:

```text
E[A_bv] = sum_h n_bh mean_h(k_v)/n_bg
E[J_b] = sum_h n_bh mean_h(k_0 k_1)/n_bg
E[I_b] = E[A_b0] E[A_b1]
         + sum_h n_bh(1-n_bh/N_h) S_h(k_0,k_1)/n_bg^2.
```

S_h uses divisor N_h-1. Singleton/full-allocation terms contribute zero
sampling covariance. Keep its sign. The identity also follows by pair
inclusion probabilities: n/N for i=j, n(n-1)/(N(N-1)) for distinct authors
in one stratum, and product inclusion probabilities across strata.

Under profile-only exchange, single and same-author-pair density expectations
are the pooled catalogue in both banks. Their response drift is zero by this
construction, not by proving a universal personality law. The product reader
has unequal expected densities when bank counts differ: both estimated
marginals depend on the same sampled catalogue. Its observed orbit-mean
cross-ratio RMS is .000563593. This is not a new personality component.

Expectation of densities, log densities, target scores and responses are
different operations. Do not use this compiler to assert that averaging
individual posterior scores produces the pooled catalogue posterior.

## 5. What remains measured after orbit construction

Single-window source actions under the profile-only mean density remain
nonzero: E/N/A/C/O RMS .002978854/.003673905/.003605962/.003279970/.005888620,
first averaging both text windows within each of24 authors. The mean-density
bank discrepancy vanishes by the group construction while target-dependent
response remains. This post-outcome description was read from sealed
orbit_mean_densities, without a new fit or parameter choice. It is neither an
accuracy gain nor proof that catalogue-sensitive components are worthless.

The exact-condition mean-density discrepancy remains .041469056. Because
309 authors are frozen, this cannot tell whether condition-linked expression,
within-condition fingerprints or unsupported strata generated that remainder.
Do not enlarge the orbit by binning conditions until a scientific construction
requires it; no support/strata search was performed here.

## 6. Prior art and next decision

[Hemerik and Goeman2018 sections2.1/3](https://arxiv.org/html/1411.7565)
supplies group-invariance/identity requirements. [Gretton et al.2012 pp724-726](https://jmlr.org/papers/volume13/gretton12a/gretton12a.pdf)
distinguishes empirical sample functions from population distribution tests.
No MMD or characteristic-kernel result was implemented. Finite-population
inclusion covariance is mature mathematics. OpenAI/math adc7f124 signed
03b-isotypic70-120 inspired explicitly retaining the stipulated action and
multiplicity, not a transferred Specht-module bound. Its Gaussian streaming
08 hypotheses were checked and do not describe reused prototype libraries.

Stop bank/seed/bootstrap precision rescue. A useful next scientific task must
add or identify a target-linked evidence relationship, not make two arbitrary
source halves agree. First inventory the common-source action/anchor/native
interfaces and distinguish software catalogue robustness from actual target
meaning. Preserve HM212's actual-label contact and all earlier choice/state/
dynamic findings at their own support; this experiment does not negate them.

## 7. Reproduction

Canonical results/grain_math_response_v1/source_catalogue_response_r1.
prepare/compute/verify each ran once;0 new neural/API/private/train/fit calls.
398 group draws,400 total allocation readings reuse the SAME24 panel authors,
not400 independent replications. All6000 allocation/target/law accounts and
30 summaries survive. Inputs, scientific code and completed outputs are sealed.

9 new tests plus7 HM215 and10 HM214 tests passed(26);3 new Python files passed
Ruff/format before sealing. Different-arithmetic same-agent verification:
197376 literal values,68600 stratum checks,6 selected reference allocations,
maximum8.479328e-14. It is not a blind independent audit. Old HM215 replay
error9.547918e-15,control error0. Compute4.535962s and verify.981898s exclude
prepare/hash/write/processstartup;not total walltime. No scientific retries.

```sh
# Read-only verification; do not rerun science over the sealed r1.
"/Volumes/mobile3/projects/project persona/.venv/bin/python" \
  scripts/run_grain_source_catalogue_response_hm216.py check
```
