# SOURCE-to-reader order compatibility

2026-10-13, HM249. Explicitly post-HM248 diagnostic, not prospective
confirmation, a fitted predictor, or a new mathematical theorem. Canonical
observations: `results/grain_math_response_v1/source_reader_order_r1`.

## 1. The object being tested

HM248 supplies ten actual eight-comment inputs for each of 24 historical
authors, from five frozen SOURCE selectors and two removal arms. Each input
has a five-coordinate SOURCE profile under each old F/B/D account, and five
native reader digit log odds. Exact token aliases are one observation:
240 roles reduce to 228 distinct within-author inputs, with 12 aliases.

For a fixed author, target q and SOURCE account a, write s_j for SOURCE_a(x_j)[q]
and y_j for the native reader's high/low digit log odds. The tested hypothesis
is y_j=g(s_j), allowing any nondecreasing g. Test nonincreasing g separately.
Each author and target may have a different g: this is more generous than a
single population calibrator. It is still a one-coordinate, ordered reader
hypothesis, not all possible vector-to-reader functions.

The inputs have equal comment count, not identical content, length, community,
or semantic opportunity. SOURCE-personality association and model judgment
are distinct observation laws, even when their target names coincide.

## 2. Finite order bound, without fitting g

For the scalar order define

```text
B_plus  = (1/2) max_(s_i <= s_j) (y_i - y_j)_+
B_minus = B_plus(-s, y)
B_either = min(B_plus, B_minus)
```

Any nondecreasing g whose maximum error is e obeys

```text
y_i - e <= g(s_i) <= g(s_j) <= y_j + e
```

for s_i<=s_j, hence e >= (y_i-y_j)/2. Taking all pairs yields B_plus.
Conversely, for e=B_plus the finite envelope

```text
g(s_j) = max_(s_i <= s_j) (y_i-e)
```

is nondecreasing, is at least y_j-e, and is at most y_j+e. Thus the bound
is exact on the observed finite scalar order. Equal SOURCE values impose
both inequalities. The code computes bounds and witnesses, not this g.
B_either is the best achievable maximum error over the union of the two
orientations; choosing an orientation does not erase a positive union bound.

For computational guarding, require SOURCE separation >2*delta and subtract
eta from the half-gap. Delta is inherited 1e-11; eta is twice HM248's measured
head reconstruction discrepancy, 8.338373035599034e-6. This is an empirical
computational adjustment, not a confidence interval or an assurance against
all model/runtime/scientific error. Near ties are unobserved, not rescued.

## 3. Fixed shared-profile competitor

Retain the old v=(1,-1,1,1,1), with no sign search:

```text
x_j = v * SOURCE_a(x_j)
z_j = dot(v, native_reader(x_j))/5
```

If every coordinate of x_i is <= the corresponding coordinate of x_j,
any coordinatewise nondecreasing multivariate reader must satisfy z_i<=z_j.
The same pair half-gap bounds its maximum discrepancy, including nonlinear
interactions. Guarded comparisons require strict separation in all coordinates.
No comparable pairs means UNOBSERVED, not evidence for a successful reader.
This tests the specified shared evaluation order, not every possible shared
factor, arbitrary rotation, or nonmonotone multivariate mapping.

## 4. Actual finite result

All 24 authors conflict with both scalar orientations, for every one of the
five targets and all three SOURCE accounts. The mean guarded union bounds
under D, in E/N/A/C/O order, are

```text
0.321536 / 0.174711 / 0.534723 / 0.260772 / 0.427422 nats
```

These are finite maximum-error lower bounds, not Pearson r, AUC, effect-size
correlations, human-trait errors, or population significance. The 360 cells
are repeated author/account/target combinations, not 360 independent people.

The full-profile shared competitor has only 103/104/46 guarded constraints
out of 977 possible distinct within-author pairs for F/B/D. It is observable
for 19/22/16 authors and conflicts for 7/8/4. The remaining 5/2/8 authors are
unobserved. Preserve this limited coverage rather than reject every shared
valuation mechanism or accept the nonconflicting cases as validation.

## 5. Separate post-outcome coarse-condition description

After freezing and verifying the primary result, a read-only grouping used
the existing `retained_conditions` count. Group separately by author AND exact
old binary-venue count BEFORE taking both orientation bounds. This allows a
different calibration and even a different orientation within each stratum.

There are 36 observable author/count strata and 729 unique within-stratum
pairs. Every author has at least one stratum conflicting in both orientations
for each target and account. Counts of conflicting strata, in E/N/A/C/O order:

```text
F: 30 / 32 / 31 / 30 / 32 of36
B: 29 / 31 / 31 / 31 / 31 of36
D: 31 / 32 / 32 / 31 / 32 of36
```

This description does not alter primary CSVs, seals, acceptance, or their
pre-analysis contract. Binary-venue counts are not full scenario, topic,
subreddit identity, length, or opportunity controls. No psychological causal
claim follows from this count matching.

## 6. Mainline consequence and prior art

HM248's positive result remains: source-only rules locate actual evidence
whose removal changes model judgments. HM249 adds a specific constraint:
their named scalar values cannot reproduce all those judgments by merely
changing an ordered scale, even author-by-author. Do not fit isotonic models,
reverse signs to rescue results, or launch rank/kernel/head precision grids.

This changes the construction problem from scalar calibration to a qualified
evidence valuation law. Possible explanations include other SOURCE coordinates,
nonmonotone context interactions, the SOURCE instrument's omissions, or a
different reader target contract. The result does not choose among them.
Future work must produce a positive, prospectively testable action rule that
distinguishes these explanations, not another generic incompatibility table.

The order bounds and feasibility ancestry are mature:
[Stout, 2015/2017, Section2](https://arxiv.org/html/1507.02226v2).
The unweighted proof above is derived directly; inconsistent symbols in the
extracted weighted formula are not copied or claimed to be a paper error.
[Heimersheim/Nanda, 2024, Section2.2](https://arxiv.org/html/2404.15255v1)
provides an existing distinction between mediation and amplification/calibration;
HM249 does not itself localize a circuit.

[OpenAI/math pinned foundations98-116](https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/The-exact-reconstruction-threshold-for-the-three-state-symmetric-channel-September-25-2026/build/sections/02-foundations.tex)
motivates asking what an observation preserves. Its posterior sufficiency
is within one specified experiment; a SOURCE evidence score is not the LLM
reader posterior. No tree-reconstruction or new psychological theorem is
transferred. Current novelty is the addressed empirical interface test, not
the mathematics itself or a completed theory of personality measurement.

## 7. Execution boundary

One prepared/analyzed/verified finite packet; 0 new model/API/private/training
calls and 0 fitted predictions. All 1,200 old endpoints retained, including
aliases. Analysis compute0.1252415s. 19 focused tests and three Python files'
lint/format passed. Alternate checks: 4,752 order accounts, 14,655 pair rows,
all source endpoints and native odds, and addressed witnesses. Maximum pair
error6.883e-15, source replay0, native odds5.107e-15. Same-agent alternate
arithmetic is not independent blind replication. All seals pass; no job.
Goal active; original18/100/44 open. Historical failures and HM243's partial
collection remain unchanged. Historical E5/Qwen remain neural instruments.
