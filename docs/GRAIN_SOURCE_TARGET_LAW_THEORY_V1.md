# SOURCE-supported joint target law and observation functionals

2026-10-13 JST. HM229 completed finite numerical construction.
This document describes a reusable interface and a scalar-compression finding,
not a new probability theorem or an accuracy promotion.

## 1. Build a law before choosing a score

For fixed SOURCE authors i=1,...,386, preserve their full five-column anchor
rows y_i, their two existing E5 prototype means z_i0,z_i1, and the inherited
Gaussian variance s^2=0.05525111588037168. A supplied addressed history gives
the same pooled vector x as the earlier full SOURCE scorer.

    k_i(x) = (1/2) sum_v exp(-||x-z_iv||^2 / (2 s^2))
    w_i(x) = k_i(x) / sum_j k_j(x)
    F_x(dy) = sum_i w_i(x) delta_y_i(dy)

F_x preserves joint anchor configurations; separate per-column resampling would
not preserve that joint law. It is a declared empirical SOURCE law conditional
on similarity, not an identified natural posterior of the author's personality.
It is a Gaussian kernel conditional-distribution construction, not an invention
of kernel regression or distributional prediction.

Different observation functionals now share ONE underlying law:

    level_q(x) = integral y_q dF_x
    CDF_q(t|x) = integral 1(y_q<=t) dF_x
    Cov_qr(x) = integral (y_q-level_q)(y_r-level_r) dF_x
    old_high_q(x) = integral bit_q(code_i) w_i(x)

The final expression refers to each SOURCE row's inherited code. Its log odds
replay the old full SOURCE score within 5.33e-15 on all 72 laws. No new fitting
changes those scores. The equal-author SOURCE law is the fixed reference;
adding another application batch cannot change an already-scored author's law.

## 2. A comparison is a declared observation, not a coordinate difference

For independent draws Y~F_x and Y'~F_x', define

    Pi_q(x,x') = P(Y_q>Y'_q) + (1/2) P(Y_q=Y'_q)
    d_q(x,x') = 2 Pi_q(x,x') - 1.

This specifies a probability-of-superiority functional, not independence of
biological authors or repeated comments. Self Pi=.5 and swapped Pi sums to 1.
For binary laws, Pi=.5+.5(p_x-p_x'), so ordinal comparisons reduce exactly to
the single high-probability score. Continuous marginal shapes need not do so.
CDF crossing prevents first-order stochastic dominance; it does not by itself
require a directed comparison cycle or a new latent dimension.

The cited one-dimensional dice [1,5,9], [3,4,8], [2,6,7] all have mean 5 but
win cyclically with probability 5/9. Hence HM228's strict A cycle rules out
deterministic scalar ordering on those readings, but does not alone rule out
all one-dimensional distributional comparison models. This is mature
nontransitive-dice mathematics, not a newly discovered GRAIN theorem.

## 3. Actual same-evidence findings

386 SOURCE authors and the SAME 24 application authors /288 events /3 arms /
5 targets were retained. All computations use existing numeric vectors;
0 new neural/API/private/training calls and no parameter search.

On the inherited 32-edge full-history graph, mean-vs-Pi directions differ in
8/160 target-edge cells (E/N/A/C/O:3/1/1/1/2). Binary-vs-continuous comparisons
differ in 31/160 cells (8/9/5/3/6). Thus compressing the law to a mean or a
high/low score can discard observable relationships even on the same evidence.

Full-history CDF crossings occur on 31/31/29/32/32 of 32 edges. Nevertheless,
none of the five SOURCE-law comparison graphs has a strict cycle, either on
the inherited graph or on all mathematical author pairs. The same holds for
both inherited removal arms. This does not erase the neural observer's A cycle:
the two interfaces implement different comparison laws.

Native HM228 A cycle 15->16->17->15 corresponds here to Pi=.502649/.494124/
.503226; its second edge has the opposite direction. This frozen SOURCE law
does not explain the observed neural cycle. Do not tune it until the cycle fits.

Raw SOURCE-Pi/native-odd edge cosine E/N/A/C/O is +.493343/+.105997/-.080302/
+.407569/+.103157. This is finite action geometry, not Pearson personality
accuracy. One fixed joint SOURCE-row permutation gives +.212827/-.494805/
+.402831/+.190914/+.434257. It is only a descriptive assignment control:
A/O contacts are larger in that control, and no p-value/selection claim follows.
In the all-arm CSV that control column intentionally repeats the full-history
permutation reference, not an arm-specific shuffled-law computation.

## 4. Why this is not an instrument-precision bottleneck

The full-history SOURCE masses have effective support 375.17-381.19 of 386
authors and TV distance .0441-.0688 from uniform. All target pair superiority
departures from .5 are at most .002789/.004828/.005876/.003356/.003561.
The author-specific tilt exists but is small under this inherited law. Improving
floating-point reconstruction does not create target meaning or stronger
individual evidence. Bandwidth sharpening is not the next accepted mechanism.

Distribution variance .075-.101 is variation under the SOURCE law, not a
single-person confidence interval, error variance or state/trait fraction.
The application panel is historically reused, not untouched CV/lockbox evidence.
Only SOURCE y values enter the construction. Reading the existing authorized
public NPZ decompresses the entire y block before SOURCE row selection;
application y is not used and is excluded from the cold process.

## 5. Cold interface and reproducibility

Source bundle: source_means/codes/variance/anchors only. Application records:
opaque author integer, 12 unique event IDs, vectors, and old support/counter
indices. Application labels, LLM responses, raw text and weights are rejected.
The numeric cold backend is independent of an LLM decoder; its existing E5
input representation remains neural. No fully model-free text encoder is claimed.

    PY='/Volumes/mobile3/projects/project persona/.venv/bin/python'
    "$PY" scripts/run_grain_source_target_law_hm229.py prepare
    "$PY" scripts/run_grain_source_target_law_hm229.py compute
    "$PY" scripts/run_grain_source_target_law_hm229.py analyze
    "$PY" scripts/run_grain_source_target_law_hm229.py verify

Run from this implementation repository with the parent project's Python.
The canonical source_target_law_r1 is already sealed: do not rerun those stages
there. Use check for read-only verification or --out a new authorized packet.
Cold scoring uses score_grain_source_target_law_cold.py with --bundle/--inputs/
--output. The output path must not exist; no silent overwrite or retry.

All 8,640 mathematical pair accounts and 31,752 CDF entries were recomputed
with sorted-support/literal arithmetic. Binary replay max 5.33e-15; cold parent
replay 0; self/swap/common-law identities pass. 35 focused tests and four
Python lint/format checks pass. Same-agent alternative calculation, not blind
independent replication. Cold computation .704117 s; analysis .119668 s.

## 6. Theory position and next decision

New project interface: evidence vector -> ONE source joint law -> several
explicit target observation functionals. New finite finding: scalar compression
changes some author comparisons, while the current law does not explain neural
cycle/target aliasing. This is a construction/representation increment, not a
claim of new mathematics, a completed personality scale, or a theory breakthrough.

Next inventory source-qualified conditional response/target-meaning interfaces
on the SAME evidence units. Before another experiment, specify which competing
observation law it distinguishes and what new measurement construction it enables.
No source bandwidth, head, rank, graph-size, label-flip or observer-fidelity grid.
Preserve GRAIN choice, conditions, state, dynamics and joint configuration.
Original 18 protocols/100 cards/44 computational questions remain OPEN.

## References and mathematical inspiration

- [Thas et al. (2012), Probabilistic index models](https://rss.onlinelibrary.wiley.com/doi/full/10.1111/j.1467-9868.2011.01020.x): comparison functional and ties; no semiparametric fit imported.
- [Angel & Davis (2016), A Direct Construction of Non-Transitive Dice Sets](https://arxiv.org/html/1610.08595): cited equal-mean counterexample; not new natural/model observations.
- [Tamás & Csáji (2024), Recursive Estimation of Conditional Kernel Mean Embeddings](https://www.jmlr.org/papers/v25/23-0168.html): conditional-law viewpoint (abstract read); its estimator/consistency theorem is not used here.

OpenAI/math pinned adc7f124, three-state-channel foundations lines 150-235:
the law recursion comes before functionals; combining experiments requires
conditional independence and a probability-law tilt. We borrow that design
discipline, NOT its tree reconstruction theorem or independence assumptions.
No mathematical proof guarantees our arbitrary text similarity kernel is the
natural author/personality conditional law.
