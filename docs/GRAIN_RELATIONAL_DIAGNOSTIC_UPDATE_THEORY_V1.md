# Full-vector relational diagnostic updates, HM209

## Object and construction delta

HM209 turns paired full-vector observations into a positive diagnostic update
on the exact HM207/208 evidence. It does not optimize accuracy. HM176/177
already supplied categorical paired observation laws; dependence as evidence
is not a new principle. The added interface keeps the full 1024-dimensional
vectors and reads the same addressed 12-event windows through two source laws
with identical single-window marginals but different prototype coupling.

All 386 source authors, two old 203/183 author folds, 32 inherited five-bit
target profiles, five targets E/N/A/C/O, and 24 source-disjoint historical
application authors remain. No application labels, native answers, raw text,
new neural calls, API or training enter the constructor. The upstream cached
E5 representation remains neural. Cold inference after vectorization is
numerical, not LLM-free raw-text measurement or recovered neural reasoning.

## 1. Positive source laws

Let g be one of 32 profiles, n_g the source count, and z_iA,z_iB the paired
source window means. Use the previously frozen HM204 Gaussian scale
s2 = 0.05525111588037168, without fitting or searching it:

```text
K_s(x,z) = normalized Gaussian density with mean z and covariance s2 I
J_g(a,b) = (1/n_g) sum_(i:g) K_s(a,z_iA) K_s(b,z_iB)
I_g(a,b) = [(1/n_g) sum_(i:g) K_s(a,z_iA)]
           * [(1/n_g) sum_(j:g) K_s(b,z_jB)]
d_g(a,b) = log J_g(a,b) - log I_g(a,b).
```

J retains source-author pairing; I independently selects the two prototypes
within the SAME profile. Integrating either observation window gives the same
conditional marginal in both laws. This is a declared finite mixture channel,
not an identified natural text-noise distribution or a causal intervention
on an author. Missing source profiles are rejected, not assigned zero evidence.
For a singleton profile J_g=I_g exactly; there is no estimable within-profile
prototype coupling in that cell.

The prior is fixed uniform over profiles, with authors uniform within each
profile. It is HM207's reference, not HM204's uniform-author prior and not the
empirical population prevalence. Implemented log potentials omit the Gaussian
query-only term and normalization constants, which cancel in J/I and profile
posteriors. Individual omitted-factor potentials are not normalized densities.

## 2. Diagnostic update, not merely recurrence

Bayes gives an exact finite operation:

```text
P_J(g|a,b) = P_I(g|a,b) exp(d_g) / E_(P_I)[exp(d_G)]
delta_q = log odds_J(T_q=1|a,b) - log odds_I(T_q=1|a,b)
        = log E_(P_I | T_q=1)[exp(d_G)]
          - log E_(P_I | T_q=0)[exp(d_G)].
```

A coupling factor common to all profiles cancels from every target posterior,
even if it changes the observation density. Decompose the log factor into
all 32 characters of the five-bit profile:

```text
d_g = sum_m d_hat_m chi_m(g)
d_hat_m = (1/32) sum_g d_g chi_m(g).
```

The common coefficient d_hat_0 cancels from delta_q. The other coefficients
describe profile dependence, but a nonconstant coefficient alone does not
guarantee an improvement for a particular target. They are not new measured
factors, independent information fractions, or psychological trait axes.

The separate observation log-evidence ratio is

```text
log p_J(a,b)/p_I(a,b) = log E_(P_I)[exp(d_G)].
```

It is not the target update or just d_hat_0. This distinction separates
explaining a text pairing from explaining its target assignment.

## 3. Target-indexed local fields

For view v and either source law L, differentiate the target log odds:

```text
F_L,q,v = grad_(x_v) log odds_L(T_q=1|a,b)
        = [E_L(z_v | a,b,T_q=1) - E_L(z_v | a,b,T_q=0)] / s2
F_rel,q,v = F_J,q,v - F_I,q,v = grad_(x_v) delta_q.
```

Under J both prototypes share one author index; under I their within-profile
weights factorize. These fields show where a small vector change modifies
the relational update. They do not identify a causal attention head, a human
response operator or a psychological direction. Tests compare the gradients
with finite changes; this is interface verification, not a precision campaign.

## 4. Frozen evaluation and results

Each bank is compiled from its own author fold only and evaluated on the
opposite SOURCE fold afterward. Both evaluation references are retained:
observed opposite authors and an equally weighted average over 32 profiles.
The fixed scale had previously used all 386 source vectors without labels;
the representation and cohorts were used historically. These results are not
a fresh nested-CV benchmark, significance test or untouched lockbox.

Paired minus independent binary log scores, in nats per author:

| Target | Bank 0, observed | Bank 1, observed | Bank 0, profile-balanced | Bank 1, profile-balanced |
| --- | ---: | ---: | ---: | ---: |
| E | -0.000121652 | +0.000049778 | -0.000115555 | +0.000033208 |
| N | +0.000524795 | +0.000047676 | +0.000414302 | -0.000051695 |
| A | -0.000090803 | +0.000094176 | -0.000099725 | +0.000058053 |
| C | -0.000070912 | +0.000006555 | -0.000072402 | +0.000011102 |
| O | +0.000112063 | +0.000037552 | +0.000077089 | +0.000103269 |
| Joint 32-profile | +0.000459616 | +0.001613190 | +0.000244449 | +0.000872717 |

O is positive in both directions and both references, a small local diagnostic
relation at this frozen channel. N's observed-author result is positive both
ways, but the second profile-balanced result reverses. E/A/C mix signs.
Joint-profile gains are positive but tiny. None are a large predictive gain,
population significance claim or proof of universal target transport.

Held-author marginal probability-change RMS is 0.000538 to 0.001945 across
targets/banks. Cross-source signed local-field products on all 24 full frames
are E -0.000640282, N +0.000462617, A -0.000294497, C +0.000033134,
O +0.000096308. These products are not correlations or information fractions;
a positive field product does not require a positive log-score gain.

All 216 application frames per bank are retained: full/full and four fixed
subset edits in either window, with the companion full. Conditions, choice,
state-containing vectors and all components remain; no C/E component is
removed. Conditions are input provenance, not a separately identified
conditional likelihood. Native responses are not used or compared in HM209.

## 5. Cold interface and reproducibility

Cold input has exactly frame_key and two nonempty windows of distinct
event_id/vector/condition records. Labels and answers are rejected; missing
windows are not zero vectors. Repeated IDs are rejected. Distinct IDs do not
by themselves establish biological or statistical independence.

The input-bound source banks return both normalized 32-profile laws, five
readings, five log-odds updates, the common log factor and observation evidence
ratio. Actual separate cold CLI execution on the first full frame produced
updates [-0.002270937,-0.009333049,+0.002654358,-0.003385362,-0.004910278];
common factor +0.010148786 and evidence ratio +0.009435803 are distinct.

Canonical artifacts: results/grain_math_response_v1/relational_diagnostic_update_r1.
The prepared manifest binds configuration, scientific code, source inputs and
stage seals. Each stage ran once. 11 new / 43 focused tests and three Python
lint/format checks passed before sealing. One direct-distance/explicit-product
audit checked 3392 values plus all profile updates/character inversions;
maximum error 5.113097e-15. Same agent, different arithmetic, not blind audit.
Build/analyze/verify recorded 0.149061/0.215571/0.055566 seconds, excluding
preparation/hash/process setup. No scientific failure, retry or score search.

```sh
# Completed r1 is sealed; do not overwrite or rerun its stages.
# From the implementation repository, this read-only cold command is usable:
"/Volumes/mobile3/projects/project persona/.venv/bin/python" \
  scripts/run_grain_relational_diagnostic_update_hm209.py cold
```

## 6. Mathematical ancestry and next scientific decision

Conditional Bayes, finite mixtures and dependence factorization are mature.
[Kernel Bayes' Rule, 2013](https://jmlr.org/papers/v14/fukumizu13a.html) is
related inference literature; this construction does not use its RKHS inverse
estimator or claim its convergence result. The verified abstract of
[Approximate Bayesian Conditional Copulas, 2021](https://arxiv.org/abs/2103.02974)
addresses conditional dependence, not this particular mixture implementation.
No full-text replication is claimed.

[OpenAI/math pinned foundations, lines 98-116 and 166-184](https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/The-exact-reconstruction-threshold-for-the-three-state-symmetric-channel-September-25-2026/build/sections/02-foundations.tex)
inspired the distinction between observation likelihood and posterior evidence
and the explicit branch-independence assumption. Its tree/spin symmetry and
reconstruction theorem do not transfer to text or personality. New value is
the same-evidence executable connection and its empirical answer, not a new
Bayes theorem or discovery of coupling for the first time.

Next inventory the existing same-event target-assignment/observable-anchor
interfaces to determine which source-supported evidence relation should enter
a measurement definition. Do not tighten the Gaussian scale, tune profile
weights, optimize r, rename character coefficients as factors or launch a new
packet automatically at this closeout. Goal ACTIVE/nojob; original18/100/44OPEN.
