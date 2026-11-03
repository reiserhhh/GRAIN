# HM290: unknown shared kernels under partial author observations

2026-10-14 JST. Finite pre-computation plan. Not accuracy/precision tuning.

## Inventory and additional capability

HM171 already supplies positive external SOURCE orientation and cold scores;
HM176/177 already supply paired laws; HM245/V6 already use path signatures.
HM281 ALREADY integrates Dirichlet kernels on fully observed paths. None of
these generic constructions is repeated as a new finding. HM282 only sums
masked paths AFTER replacing random SOURCE kernels by their posterior means.
Its explicit integrated-kernel coarsening interface remains open.

This packet makes that missing interface executable: only observed venue or
anonymous-expression bits enter early inference, while unknown transition rows
are shared across the path and into the next episode. The response criterion is
the actual next frozen vector-state of an existing human text,not an LLM answer.
These are coarse text-behavior proxies,not independently named psychological acts.

## Unchanged support and law

Reuse SAME386 SOURCE/120 held authors/two18-state episodes,all source row priors,
three banks/references,all32 profiles/two old anchor assignments and donor map.
No recompile/refit of SOURCE kernels,hidden-bit leakage or boundary transition.
No application labels,raw/private text,API,new neural calls/training or old queue.
Full observations reproduce HM281 kernel_shared. Fixed-mean controls are exactly
the saved HM282 programs,not a reoptimized baseline.

Exact target per SOURCE i,for observation masks D_t:

    L_i(O) = integral sum_(compatible hidden paths x)
             mu_i(x0) product_t K_i(x_(t-1),x_t) dDir_i(mu_i,K_i).

Integrate one shared kernel,not draw a new kernel per event. It generally differs
from the HMM using E[K_i] at every edge. Full paths reduce to old beta ratios.
Partial-state inference additionally needs a distribution over hidden starts
and transition counts; a single current-state marginal is insufficient.

Use a collapsed fully adapted particle algorithm on (last_state,n0,N) PER
source i. Given a particle,current origin a and mask M, compute

    q_i,r(b)=(alpha_i,ab+N_r,ab)/(alpha_i,a+ + N_r,a+)
    m_i,r=sum_(b allowed) q_i,r(b).

Weight/resample predecessor sufficient statistics by m_i,r,then sample b from
q restricted to M,then increment the counts once. SOURCE normalizer accumulates
the mean incremental masses. Label codes affect profile readout only,not filtering.

## Approximation and fixed numerical contract

64 particles; TWO fixed seed realizations. Realization0 is primary; realization1
only diagnoses integration variability. Seeds depend on fixed reference/view
and realization,NEVER author ID,unobserved bits or future observations. Identical
observed prefixes therefore replay identically. No particle/seed/precision grid
or increase after outcomes. Coarse likelihoods and posteriors are approximations,
not exact integrated values or sampling confidence intervals. Full-view replay
is exact to ordinary arithmetic because every hidden path is specified.

Retain six contracts from HM282. Full/C/S reference evidence uses integrated
likelihoods; conditional contracts use L(full)/L(C) or L(full)/L(S),with the
SOURCE reference held as in the old explicit conditional design. Within-source
kernel counts for those two conditional contracts use the now fully observed
early path. This is NOT causal do(C) or an average of per-kernel ratios.
Coarse likelihood estimates may violate exact monotonicity relative to the
known full likelihood;report EVERY violation,never clip or silently certify
exact conditional probabilities. Corresponding output beliefs are finite
particle approximations under the declared model,not human trait probabilities.

Late paths restart without an early-last to late-first edge. Unknown-kernel
posterior counts and SOURCE belief carry; every next vector-state is predicted
before incorporating its outcome. Self and old start-matched stranger memory
are both retained. Strangers are not full topic/time/demographic controls.

## Fixed scientific readouts

For every bank,realization and contract report paired per-author late transition
log-score differences: own-fresh,own-stranger,and integrated-partial minus OLD
fixed-mean same-contract. No winner/best-accuracy selection. Report all old-anchor
profiles and five marginal trajectories,reference-count uncertainty and numerical
realization differences. A shared posterior's movement is epistemic,not automatic
true personality/emotion change. Preserve condition choice and expression together.

One360-second compute cap; no scientific rerun/cap renewal. Source-cohort and
numerical conditional exploratory ranges use the old999 bootstrap/seed282,not
independent-donor population coverage. For a sign change across realizations,
mark that contrast unresolved rather than increasing the numerical budget.

## Verification and scope closure

Tests: full exact beta/old observer replay; fully hidden normalized mass; hidden
bit/author-ID/label invariance; no artificial episode edge; positive support;
partial tiny-world exhaustive likelihood with fixed finite MC diagnostics.
Verify actual full/fresh/carry against immutable HM281 arrays and fixed controls
against HM282. Cold process takes source bundle and identified episodes only;
no neural runtime or application-label input. Same-agent audit,not blind review.

This completes an APPROXIMATE usable masked-kernel adapter if run succeeds;it
does NOT prove exact18-event coarsened integration,clinical meaning,all GRAIN
mechanisms or original18/100/44 complete. The generative assumptions remain
declared models,not claims every natural text is stationary Markov.

## Primary mathematics and borrowing boundary

[Carvalho etal2010](https://arxiv.org/pdf/1011.1098), sections2-3, supplies existing
resample-propagate and parameter sufficient-statistic particle learning. Our
Dirichlet/count specialization is not their Gaussian model or a new SMC theorem.
HM281/Fortini-Petrone already supplies the full-path Markov-mixture law.
OpenAI/math pinned adc7f124 Posterior-replicas.../08-streaming and reconstruction
02-foundations inspires conditioning on what was observed before prediction.
Its Gaussian/independent-side/tree bounds are not applied. No new geometry,
group,information or personality theorem is claimed.

Canonical results/grain_math_response_v1/partial_kernel_r1; new core/runner/cold
CLI/tests,full numerical outcomes and parent Chinese mechanism report. STOP at
the fixed result;do not use forecast metrics to start another parameter search.
