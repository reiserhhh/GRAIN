# HM57: event distributions, nonlinear readout and context collisions

2026-10-09. CPU-only follow-up to the frozen HM56 results. This is a declared
result-after reference, not an independent preregistered replication. Preserve
HM46-HM56 code, materials and results. No model loading, forwards, API, fitting,
search, generation, private data, training, publication or old queue.

## Scientific decision

HM56 excludes common linear mixtures of isolated event scores. This can arise
from known decoder nonlinearity. Do not mistake that exclusion for a new
composition mechanism. Reuse every HM56 candidate logit; compare two fixed
standard geometries and test collisions in full distributions, not just means.

If fixed logit pooling explains an effect, adopt the known explanation. If two
accounts have identical qualified event distributions but different native
readouts, no deterministic function of that identical summary can distinguish
them. That identifies an observation-interface omission rather than a demand
for a more accurate regressor. It does not identify which contextual circuit
causes the difference or make the readout a human personality parameter.

## Fixed mathematical references

Let p_e = softmax(l_e) on the ten declared digit answers, v_j = j/9, and use
weights obtained only from explicit design histories. Event weights count
current own occasions; mention weights count own new/alias reports. Neither
counts another person's actions as the focal person's events. No weights are
selected using the whole-account readout.

    m = sum_e w_e p_e
    g = softmax(sum_e w_e l_e) = normalize(product_e p_e**w_e)
    B = dot(v,m), G = dot(v,g), N = dot(v,p_native)
    N-B = (G-B) + (N-G).

m and g are linear/logarithmic opinion pools. On the categorical simplex they
are mixture/exponential paths, respectively. Their KL barycenter identities are:

    sum w KL(p_e || r) - sum w KL(p_e || m) = KL(m || r)
    sum w KL(r || p_e) - sum w KL(g || p_e) = KL(r || g).

These finite-support identities are proved by expansion, not by transferring a
network-consensus or human-posterior theorem. The ten-answer law is conditional
on the candidate alphabet; it is not the whole language-model distribution.

For a two-event logit path p_t = softmax(t*l_x+(1-t)*l_y), d=l_x-l_y:

    S'(t) = Cov_p(v,d)
    S''(t) = E_p[(v-E_p v)*(d-E_p d)**2].

Ordinary multiclass pooling can leave the interval of isolated scalar scores.
Compare against a synthetic positive pooling world, an explicit scalar-range
counterexample and a same-summary/context-shift world; acceptance cannot require
all worlds to fail.

The squared discrepancy keeps its signed cross term:

    ||N-B||^2 = ||G-B||^2 + ||N-G||^2 + 2*(G-B)'*(N-G).

The first difference is a reference pooling effect; the second is a discrepancy
to this fixed reference. They are not causal percentages and not orthogonal.

## Complete finite roster and collision tests

- All 1296 raw readouts: 144 isolated-event readings and 1152 account readings.
- Both fixed event and mention bases, all four regimes, all six queries.
- 144 regime pairs plus 96 order-reversal pairs, both bases, retaining every sign.
- Compare entire ten-answer m/g arrays. For an identical summary, the best common
  prediction of two native profiles is their midpoint, with paired profile RMS
  floor ||N_hi-N_lo||/2. This standard bound cannot be lowered by richer fitting
  of that same summary. It concerns only the specified summary/native readout.
- Alias/balanced and other/balanced preserve own event distributions; order
  reversal preserves both unordered bases. Distinct/alias changes event weights.
  The observed collision status must be checked, not hardcoded as a result.

## Outputs and acceptance

Archive plan/spec/code before analysis, a full raw-logit snapshot and hash links
to original metadata/arrays and HM56 verification. Write raw-binding, scalar
decomposition, probability decomposition, paired changes, collision certificates,
regime/query summaries and zero-call run status in event_probability_pooling_r1.

The verifier must read original raw JSON independently, reconstruct weights
without the runner's ledger helper, use scipy logsumexp/product formulas, check
all rows, cross terms and KL identities, and rebuild reported summaries. Focused
tests include gauge/permutation invariance, derivatives, invalid data, true
pooling and context-collision worlds. Freeze failures; do not change inputs to
improve a score. Acceptance requires arithmetic/provenance/roster checks, not a
better RMSE, not a new personality correlation and not global completion.

## Primary mathematical ancestry actually read

- Nielsen (2020), *An Elementary Introduction to Information Geometry*,
  [categorical/exponential and mixture families, sections 3.9 and 3.13-3.14](https://pmc.ncbi.nlm.nih.gov/articles/PMC7650632/).
- Bandyopadhyay & Chung (2016 revision), *Distributed Estimation using Bayesian
  Consensus Filtering*, [III-B eq.8 and Theorem6 proof](https://arxiv.org/html/1403.3117v3).
  Only the normalized geometric pool is used; graph convergence is not applied.
- OpenAI/math fixed commit adc7f1241b42e322a6451854ab7e4b4c146bf78a:
  Conditional-information-under-deterministic-coordinate-sweeps, setup section;
  Posterior-replicas-and-conditional-information-in-Gaussian-regression, model
  section through the first conditional-information bound. The design lesson
  is to preserve the correct conditioning object; Thorp/spherical-Gaussian
  assumptions are not transferred to LLMs. No new theorem is claimed here.

Next action is selected by the mechanism result: ordinary pooling law versus
an ordered/contextual evidence interface. No teacher-fidelity optimization.
