# Target-specific evidence acquisition and its transport boundary

HM206, 2026-10-12. The finite archive experiment is complete. This is a
source-supported construction, not a new information-theoretic theorem.

## 1. What was constructed

An acquisition rule can ask which available observation to read BEFORE its
fine content is revealed. The source law is indexed by availability M,
an observed early cell A, an available late condition C, and late state B.
It retains condition selection rather than removing it as irrelevant noise.
SOURCE386 has two disjoint 17-event windows and inherited 203/183 author folds.
All five old binary targets and their complete 32 profiles remain present.

For a fixed source law, write mu=E_P[Y|M,A] and
mu_C(B)=E_P[Y|M,A,C,B]. The exact squared-error information value is

    V_P(C)=E_P[(mu_C(B)-mu)^2|M,A,C].

This value equals source Bayes risk reduction. Target mutual information
I(Y;B|M,A,C) and source-address information I(U;B|M,A,C) are distinct
objectives. U is the empirical source author address, not a psychological
class. The same entropy criterion is used for the two MI objectives.

The cold numeric interface accepts availability, an addressed early cell,
and a target index. It returns support and probabilities for choosing a
condition. It neither generates a prompt nor authenticates an author.
The experiment is randomized retrospective archive access, not a human
do(topic) intervention or a natural conditional-independence claim.

## 2. Why positive source value does not imply held-author benefit

For any frozen update delta=mu_C(B)-mu and any evaluation law Q, algebra gives

    Gain_Q(C)=2 E_Q[(Y-mu) delta]-E_Q[delta^2].

The source tower property makes E_P[(Y-mu)delta]=E_P[delta^2], hence V_P>=0.
It does not impose that equality under Q. An update with the wrong relation
to the held outcome can increase risk. A larger entropy reduction or update
norm therefore does not automatically mean better target evidence.

Actual source target/identity selections disagree in 7.9%-49.4% of weighted
states where both conditions are available. Nevertheless, all five pooled
target-risk gains are negative on the opposite author fold, and all ten
target-by-direction gains are negative. The frozen target rule is also worse
than identity-MI and uniform acquisition for all five pooled targets.
This result identifies a transport boundary, not its unique cause. Finite
calibration error, conditional distribution change and representation loss
remain competing explanations; they are not resolved by this experiment.

## 3. What this changes in GRAIN

Keep separate:

    source target link -> observation value -> acquisition choice
    observed new evidence -> frozen update -> realized target benefit.

Author distinguishability is not interchangeable with target information.
The selector is usable as an explicitly source-qualified interface, but it
is not promoted as a transferable adaptive personality instrument. Do not
rescue these results with cell, smoothing, temperature or policy grids.
Next work must add a missing evidence/meaning mechanism on genuinely shared
units rather than optimize this empirical table.

## 4. Provenance and mathematical ancestry

Canonical results: `results/grain_math_response_v1/target_observation_design_entropy_recovery_r1`.
The primary `target_observation_design_r1` build failed when six complements
rounded to -2.220446049250313e-16. That failure and code seals are retained.
One separately sealed companion maps only entropy inputs within eight
float64 epsilons below zero to zero; scores, gains and positive masses are
not clipped or renormalized. All four companion stages ran once.

25 focused tests pass. A same-agent alternate audit has maximum discrepancy
8.88e-16; this is arithmetic verification, not independent replication.
All authors have partial support, 380/386 complete support; missing support
is explicitly reported, never imputed as a neutral result. The historical
encoder and global cell thresholds predate this split, so this is not
untouched whole-pipeline CV. No new neural/API/private/training operation.

[Deep CAT](https://arxiv.org/abs/2502.19275) and
[Bayesian experimental design](https://arxiv.org/html/2310.10783v2) already
study target-directed acquisition. The pinned OpenAI/math three-state
foundations217-232 inspire declaring the observation experiment; their tree
reconstruction theorem is not transferred. The risk identity above is
ordinary squared-error algebra and conditional expectation.
Original18/100/44 remain OPEN; the full goal is not complete.
