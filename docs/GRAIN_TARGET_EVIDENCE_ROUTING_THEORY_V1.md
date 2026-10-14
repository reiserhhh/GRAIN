# Conditional target evidence and profile routing, HM210

## Construction delta

HM210 explains the SAME HM209 marginal target update, rather than improving
its accuracy. The new callable interface returns all 80 hypothetical
conditional target readings and three signed accounts for each target. It
distinguishes evidence within an unchanged other-target profile from a change
in the weights of those profiles. Neither account nor their interaction is
discarded. HM184/185 conditional mean edges and HM176/177 coupled categorical
laws already exist; conditioning, Bayes and finite differences are not new
mathematical discoveries.

The evidence, old source folds and channel are unchanged: 386 SOURCE authors,
203/183 folds, two 12-event windows, full 1024D vectors, all five inherited
binary targets E/N/A/C/O and all 32 profiles. Both source banks are retained.
The same 24 source-disjoint historical application authors supply 216 frames
per bank. No new neural calls, native answers, application labels, raw text,
API, training, bandwidth or reader search occur. E5 remains the upstream
neural encoder. The inherited all-386 label-free scale and historical cohorts
do not constitute a fresh nested-CV benchmark.

## 1. Target-conditioned observation

Let P_J and P_I be the positive HM209 paired and within-profile marginal-product
posteriors. For target q, c denotes the other FOUR TARGET BITS, not a topic,
situation or observed questionnaire profile of the current author. Define:

```text
w_L(c|x) = P_L(T_-q=c|x)
r_L(c,x) = P_L(T_q=1|T_-q=c,x)
m_L(x) = sum_c w_L(c|x) r_L(c,x),             L in {I,J}.
```

All 16 hypothetical c are returned. Cold scoring does not require knowing or
guessing the author's actual other-target labels. It marginalizes over them
only when producing m_L.

For the relation factor d_g=log(J_g/I_g), the exact conditional log-odds change
is a finite target difference:

```text
D_q d(c,x) = d_(1,c)(x) - d_(0,c)(x)
          = log odds_J(T_q=1|c,x) - log odds_I(T_q=1|c,x).
```

Proof: the posterior normalizer and fixed prior cancel in each within-c odds;
subtract the two laws. Thus an other-target-only factor d_g=u(c) has D_q d=0.
Nevertheless, it can change w(c|x), hence change m_q whenever r_I depends on c.
This is a route through the DECLARED joint observation law, not a newly
localized neural or psychological pathway.

With the existing signs s_j=2T_j-1 and full finite character expansion,

```text
d_g(x) = sum_S d_hat_S(x) product_(j in S) s_j
D_q d(c,x) = 2 sum_(S contains q) d_hat_S(x) product_(j in S except q) s_j.
```

Characters containing q govern its conditional update; characters not
containing q may still move its marginal reading through profile routing.
Mixed characters cannot be declared independent personality information.
This is standard finite log-linear/character algebra, not a new group theorem.

## 2. Exact three-account explanation

Construct four diagnostic cross-readings, ordered II,IJ,JI,JJ:

```text
M_ab = sum_c w_a(c|x) r_b(c,x)
C = M_IJ - M_II                         conditional change at old weights
R = M_JI - M_II                         profile routing at old response
X = M_JJ - M_IJ - M_JI + M_II           signed interaction
C + R + X = m_J - m_I.
```

The four cross-readings need not be four naturally realized populations.
They are fixed diagnostic readouts of the same posterior tables; this is not
causal mediation. X is retained even when negative.

Evaluate the binary log score l(y,M)=y log M+(1-y)log(1-M) at each corner and
apply the SAME differences. The three score accounts sum to the original
HM209 log-score change. Score interaction also includes the nonlinear score
transformation; probability and log-score accounts are different quantities.
Their ratios are not psychological information fractions.

Three useful consequences, exercised in synthetic unit tests:

- An other-target-only factor can move q when the old joint law associates it
  with those targets, even though every D_q d is zero.
- If r_I is constant across other profiles, that indirect route vanishes.
- A mixed factor can have opposite nonzero conditional changes while their
  marginal effect cancels exactly. A zero total need not mean no conditional
  action; conversely, a positive total need not imply target-local evidence.

## 3. Executed findings

All observed-author mean score accounts below are in **10^-6 nats/author**.
Bank 0 is 203 SOURCE -> 183 held; bank 1 is 183 SOURCE -> 203 held.

| Bank | Target | Conditional C | Routing R | Interaction X | Original total |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | E | -116.579 | -4.632 | -0.441 | -121.652 |
| 0 | N | +525.347 | -4.031 | +3.479 | +524.795 |
| 0 | A | -85.119 | -5.834 | +0.150 | -90.803 |
| 0 | C | -89.251 | +19.235 | -0.896 | -70.912 |
| 0 | O | +106.228 | +3.546 | +2.289 | +112.063 |
| 1 | E | +47.696 | +0.127 | +1.954 | +49.778 |
| 1 | N | +39.348 | +8.567 | -0.239 | +47.676 |
| 1 | A | +73.384 | +18.499 | +2.292 | +94.176 |
| 1 | C | +24.456 | -18.692 | +0.791 | +6.555 |
| 1 | O | +29.725 | +6.901 | +0.926 | +37.552 |

O's small positive update is chiefly within-profile evidence at this channel,
not merely other-profile routing. The profile-balanced O conditional accounts
are +72.506/+94.783 in the same units, versus routing +3.962/+7.135 and
interaction +0.622/+1.351. Both evaluation references are preserved, not
chosen after inspecting which produces gains. Bank 1 balanced N stays negative:
conditional -53.496, routing +1.823, interaction -0.022, total -51.695.
Bank 1 C demonstrates cancellation: +24.456 conditional, -18.692 routing,
with +0.791 interaction, rather than an absent positive conditional account.
All complete per-author accounts and balanced summaries remain in artifacts.

A descriptive readback of the already computed arrays found mixed positive
and negative D_q d across the 16 hypothetical profiles for **all 183 and all
203 held authors, for every target**. Conditional log-odds RMS ranges from
0.013520 to 0.020577 across banks/targets. These signs depend on the declared
source law and sparse profile prototype counts; they are not proof that every
human has a universal conditional personality structure. Cross-bank sign
agreement on the 24 full cold frames is E/N/A/C/O = 191/217/210/178/201 of
384 hypothetical cells per target, not independent people or a significance
test. Do not promote the atlas as uniformly transported axes.

## 4. Cold capability and accounting

Cold input remains exactly frame_key and two nonempty windows of distinct
event_id/vector/condition records. No actual other-target labels are supplied.
The actual separate cold CLI emitted all 80 hypothetical cells. Its original
probability updates, E/N/A/C/O, are:

```text
[-0.000567708, -0.002332459, +0.000663389, -0.000846116, -0.001226652]
```

These differ in units from HM209's log-odds changes; do not mix the two.
Its O conditional log-odds changes span -0.059868 to +0.030708 across profiles.
The same event can therefore have opposing conditional directions without
providing its actual author's other labels.

Canonical output: results/grain_math_response_v1/target_evidence_routing_r1.
prepare/compute/verify/cold each ran once, with immutable input/science/stage
seals. Ten new tests plus eleven HM209 tests passed. One same-agent literal
alternate audit checked 3,200 corner/conditional-odds values and all old
probability and mean log-score updates: maximum error 3.3306690738754696e-16.
This is implementation verification, not accuracy optimization or independent
replication. Compute/verify internal timings were 0.016309/0.042088 seconds;
these exclude preparation, hashing and process setup.

## 5. Prior art and next scientific decision

[Amari 2001, Sections II-IV](https://people.csail.mit.edu/jrennie/trg/papers/amari-ig-hierarchy-01.pdf)
distinguishes log-linear interaction and mixture/marginal coordinates.
HM210 uses exact finite conditioning and diagnostic corners, not that paper's
KL-optimal hierarchy estimator. [Oizumi, Tsuchiya and Amari 2015](https://arxiv.org/abs/1510.04455)
motivates specifying which disconnected constraints define an information
quantity; no integrated-information or human causal claim is imported.
[OpenAI/math, pinned posterior-replicas section 02](https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Posterior-replicas-and-conditional-information-in-Gaussian-regression-September-27-2026/build/sections/02-replicas.tex)
and its section 06 preserve conditioning and pair-law assumptions. That
thinking inspired the inventory; Gaussian replica/projection bounds do not
apply automatically to these text profiles. The missing application interface
is new to this code line, not a new mathematical theorem.

The next mainline question is what qualifies a conditional route as evidence
for an author target rather than source-profile association alone. Inventory
the SAME event units, source-supported anchors and query-specific actions in
HM160/184/185/193/199-203 before designing a new finite discriminator. Keep
evidence changes, reader changes and reference changes separate. Do not adjust
scale, profile weights, derivative precision or atlas signs to make gains
positive. No next packet or job is started here. Goal ACTIVE/nojob; original
18 protocols/100 cards/44 computational questions remain OPEN.
