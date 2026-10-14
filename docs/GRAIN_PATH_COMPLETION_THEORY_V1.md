# Local preference and whole-path completion are different observation laws

HM264, 2026-10-13. Sections below precede primary computation. Standard finite
BEST/matrix-tree enumeration plus conditional probability, not novel mathematics.

## 1. What is measured

Keep an ordered state path and a SOURCE catalogue j=(author,window) with counts
N, start s and end t. The catalogue is a finite declared inventory, not a claim
that the author always uses these counts. Distinct events are depleted once;
overlapping triples are not independent evidence. Each source author carries
a complete five-bit anchor profile, never an application questionnaire value.

The ordered-window law is a hypothesis about observation, not a personality
law. Completion is an explicit condition. It is not called noise or subtracted.

## 2. Exact graph formulas

On the incident graph support, require out-in=1_s-1_t. Let tau_t be the minor
determinant of diag(out)-N rooted at t. The number of distinct state words is

    C(N;s,t) = tau_t * product_i(out_i-1+1_(i=t))! / product_ij N_ij!.

Parallel edge labels are divided out; loops and s=t are retained. No-edge C=1
only when s=t. Impossible/disconnected balances give C=0, not a fitted repair.

A locally uniform outgoing-edge depletion process has success probability

    h(N;s,t) = C(N;s,t) * product_ij N_ij! / product_i out_i!.

For a remaining graph with current state i and fixed terminal t,

    p_local(k) = N_ik/out_i,
    p_complete(k) = C(N-E_ik;k,t)/C(N;i,t)
                  = p_local(k)*h(N-E_ik;k,t)/h(N;i,t).

This is finite completion conditioning/Doob h-transform. It can set a locally
available move to zero because that move strands unused edges, without deleting
the original edge or calling it irrelevant to the author.

## 3. Source reference and target meaning

Let pi_j be uniform over SOURCE author-window catalogues. For prefix E, write
l_j(E) for its local depletion likelihood and h_j(E) for remaining success.
The three distinct joint weights, before normalization, are

    LOCAL: pi_j*l_j(E)
    PC:    pi_j*l_j(E)*h_j(E)/h_j(empty)
    GC:    pi_j*l_j(E)*h_j(E).

PC keeps the original catalogue prior by conditioning within each catalogue;
GC selects more completable catalogues even before observing any state. Given
weight w_j, the SOURCE anchor readout is sum_j w_j Y_j / sum_j w_j, with the
entire 32-profile posterior kept. Departure is from that law's OWN reference
prior, not a universal .5. A missing denominator is unobserved, never neutral.

These are posterior SOURCE anchor assignments under a declared finite law,
not human trait posteriors or endorsement probabilities. SOURCE assignments
are calibration anchors; no human causal action is inferred from them.

## 4. Positive mechanism and hard limitation

Designed A:0,1,0,2,3; B:0,2,0,1,3. Same bag/endpoints/outdegrees, but after
observing 0->1 the local rule cannot distinguish equal-prior A/B (posterior1/2),
whereas completion distinguishes them (A posterior1). No precision tuning.
The target assignment is designed, not a psychological finding.

Within one inventory, complete words with the same start/counts have identical
likelihood. Their exact order remains unidentifiable under this exchangeability
assumption. Thus this layer does not capture every HM263 order interaction.
Fixed finite inventories can lose support for new natural paths; no infinite
process theorem guarantees generalization. Missing source law, mathematical
counting correctness and psychological interpretation are different questions.

## 5. Mainline consequence

Author measurement must specify reference population AND observation/selection
law. Whole-history feasibility can change an anchor score without changing
initial local preferences. Future construction may need a source-supported
law over inventories, rather than treating an observed author window as a
literal reusable future budget. This packet diagnoses that need, not a solution
by smoothing, more ranks, more states, or closer reproduction of LLM answers.

## Sources

- Halberstam/Hutchcroft2024, finite BEST background in Section2.1:
  https://arxiv.org/html/2405.20276v1
- Fortini/Petrone, predictive mixtures of Markov chains (recurrence qualified):
  https://arxiv.org/html/1406.5421v3
- OpenAI/math pinned foundations217-232, observation coarsening inspiration:
  https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/The-exact-reconstruction-threshold-for-the-three-state-symmetric-channel-September-25-2026/build/sections/02-foundations.tex

No tree threshold, transient-extreme-point theorem or psychological factor
claim is transferred. The formulas above are finite derivations, not new prior art.

## 6. Actual HM264 outcome (post-computation)

All772 SOURCE inventories and240 historical held whole paths were processed
once, retaining all19 prefixes,3references,3laws,2anchor assignments,5marginals
and32-profile posteriors. Primary elapsed4.646778s;0 new neural, raw private,
application questionnaire,API,training or Git calls.

SOURCE completion success ranges1/48..1, median.318813;48/772 inventories have
success1. Global-completion selection alters the SOURCE prior before text:
pooled actual E,N,A,C,O difference(GC-PC) is
(-.000043914,-.022528925,-.018879375,+.013291438,-.022218302).
This is an observation-selection effect, not a change in a human trait.

All240 two-event prefixes have pooled support. PC-versus-local marginal RMS
differences are(.017057843,.016435715,.019592607,.020748078,.020491117).
The one old-permuted anchor gives(.016865153,.019037879,.017124669,.021701326,
.016767613), so response amplitude alone does not establish target specificity.
No new permutation grid or significance claim was used.

Pooled support LOCAL/PC at prefix6 is156/154, prefix12 is21/18, prefix16 is1/1,
and prefix17 and18 are0/0. Both separate banks also have0 complete support.
Every complete held record remains with null target scores. Fixed SOURCE
inventories therefore do NOT furnish a usable natural whole-window measurement
law on this cohort. Prefix summaries condition on different support sets;
larger late RMS is not improvement. No best shorter prefix is selected.

The exact toy gives posterior1/2->1 without changing local first-step preference.
772 independent rational-Gaussian determinant accounts and32968 exact prefix
identities were checked;profile normalization maximum1.55e-15, score/departure,
source factorial and fixed cold CLI errors0.29 focused tests and4 new Python
lint/format checks pass. Preflight corrected a mis-specified toy fixture
(a loop before the sink also had success1/2) and missing shebangs before freeze.
Same-agent audit, not blind replication; old HM235 failed primary is unchanged.

The observed-history-as-future-budget shortcut fails. Next inventory existing
source-supported observation-law/unknown-kernel and writer/readout opportunity
interfaces before a new finite plan. Require a construction of variable paths
that can transport to new evidence, not smoothing/shortening/more states/rank
or re-proving generic nonadditivity. Original18/100/44OPEN;goal active,no job,
no next experiment queued.
