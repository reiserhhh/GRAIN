# Source-observed author functions: HM170

## Object and contribution boundary

This is an executable connection between an observed source endpoint law and
an author-relative GRAIN score. It is not a new conditional-expectation theorem,
histogram estimator, venue predictor, personality factor, or universal latent
decoder. HM160 already had observable conditional functions for four declared
simulator prototypes; C3/C4 already provided anonymous coordinate cold scoring;
F14/R08 already modeled recorded venue histories. HM170 combines a continuous
natural-source observation family, a frozen author-balanced reference, and an
explicit choice/within decomposition in one independently runnable numeric rule.

An anchor determines what a score means. Here the three anchors are recorded
publication continuity and anonymous-coordinate continuity. A personality or
behavioral construct requires an appropriate additional anchor, not renaming
these coordinates or improving their correlations.

## Observation law

For author u, event t has anonymous15 vector h_ut and recorded condition c_ut.
Chronological adjacency refers to sampled cached records, not every publication.
Two source-cut coordinate bits define s_ut in {0,1,2,3}. For each strictly later
within-window record, y_utq is one of three binary observations on that SAME
later record: same venue, coordinate0 high, or coordinate5 high. The outcomes
are not three independently sampled events. Adjacent pairs also share records.

With U source authors and n_u eligible pairs per author, define:

    w_ut = 1/(U*n_u)
    R(c,s) = sum_ut w_ut 1{c_ut=c,s_ut=s}
    psi_q(c,s) = sum_ut w_ut y_utq 1{c_ut=c,s_ut=s} / R(c,s)
    mu_q(c) = sum_s R(s|c) psi_q(c,s)
    ref_q = sum_cs R(c,s) psi_q(c,s).

The coordinate thresholds are source-event medians, explicitly a different
weighting from the author-balanced probabilities/reference. No pseudocounts,
application normalization, extrapolation, latent truth or LLM answer fitting
is inserted. A missing cell is unsupported, not a probability of zero.

These equations are ordinary finite conditional probabilities. They define a
source-observation reading; they are not the individual author's identified
conditional response law and are not a do-intervention on venue.

## Author composition and retained conditions

For cold window empirical mass P_u(c,s), return:

    m_uq = sum_cs P_u(c,s) psi_q(c,s)
    theta_uq = m_uq - ref_q
    choice_uq = sum_c [P_u(c)-R(c)] mu_q(c)
    within_uq = sum_cs P_u(c,s) [psi_q(c,s)-mu_q(c)]
    theta_uq = choice_uq + within_uq.

All are oriented in the chosen endpoint's probability units, not standard
deviations, personality percentiles or posterior psychological truth. The
identity follows by adding/subtracting sum_c P_u(c)mu_q(c). Changing the author's
identifier does not change a reading. Changing the current scoring batch does
not change the reference. Both choice and conditional composition are retained.
Within is not noise-free trait or causal reaction; choice is not discarded.

The cold CLI receives only an opaque author key and distinct current event
identities/conditions/coordinates. It cannot access future endpoints or fit new
parameters. Its arithmetic is neural-free, but upstream E5 vectors and the
historical learned directions remain neural-derived.

## Actual evidence and what it enables

Fixed624 natural authors x36 records, source403/validation101/held120; all21216
within-window adjacent pairs are strictly ordered, no ties. Source13702 pairs
cover all8 cells (minimum753);1248 cold author windows are supported.

Early score versus later observed endpoint, validation/held respectively:

| Endpoint | Validation101 r | Held120 r | Held early/later score r |
| --- | ---: | ---: | ---: |
| Same recorded venue | .426210 | .413571 | .701861 |
| Anonymous coordinate0 high | .759214 | .809693 | .814313 |
| Anonymous coordinate5 high | .469510 | .582142 | .647271 |

All199 matched whole-author later-profile permutations gave two-sided tail
rank.005 for each role/endpoint. This is a finite exploratory conditional tail
rank, not a precise population p-value or confirmatory multiplicity result.
Historical application cohorts have been reused; they are not untouched.

The construction now has a concrete author-to-observable connection. It does
not solve the unresolved semantic target/action transport of HM167-169: those
are generated-model interventions, while this is a natural publication law.
Do not join different observation worlds and call that a single mechanism.

The later-coordinate endpoints are endogenous to the same representation.
Their strong correlations support coordinate/publication persistence, not
independent psychological validity. Source fit is construction, not validation.
Window spans have median429.703/448.380 days, so this is not short-term state,
controlled opportunity or single-conversation dynamics evidence.

## Prior art and next scientific decision

[Park and Muandet2020](https://proceedings.neurips.cc/paper_files/paper/2020/file/f340f1b1f65b6df5b5e3f94d95b11daf-Paper.pdf),
sections2.2/3, distinguish conditional expectations from raw cross-covariances.
We use a finite empirical partition, not their RKHS regression, universal
consistency proof, or a newly invented conditional operator.

OpenAI/math's pinned adc7f124 four-state Potts reconstruction manuscript
(2026-10-05; introduction/proof-strategy reading) inspires explicit observation
families and branch-dependence checks. Its bounded-tree broadcast assumptions,
infinite depth and L3 capacity theorem are NOT transferred to natural authors.
The repository also warns that unformalized manuscripts may contain errors:
https://github.com/openai/math.

Next inventory existing SAME-event numerical endpoint/action interfaces. The
next finite construction must add meaningful target qualification to a source
anchor or author response family, not optimize these r values, cutoffs, venue
bins, coordinate indices, kernels or dimensions. Keep the entire three-target
family and all negative comparisons. Original18/100/44 remain OPEN.
