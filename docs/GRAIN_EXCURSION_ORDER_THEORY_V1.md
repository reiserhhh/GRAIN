# Conditional complete-path order measurement

HM265,2026-10-13. Constructed and tested on frozen numeric public derivatives.
Goal ACTIVE/nojob; original18/100/44 OPEN. Not a new probability theorem.

## 1. The observation family

A path X=(x0,...,xT) has directed counts N_ij. Select its most frequent state
(ties lowest state). Partition at its visits into a prefix, return excursions
and suffix. Fix the first/last excursion as well as prefix/suffix; permute all
interior excursions. This finite orbit O(X) preserves occupancies, directed
counts, endpoints, first/last edges and length. Within-excursion sequences are
also fixed. The orbit is generally a STRICT subset of the same-count Euler
family. Numeric reordered paths are not attested natural texts or new subjects.

Any unknown homogeneous first-order Markov kernel P gives

    Pr_P(X) = initial(x0) * product_ij P_ij^N_ij.

This likelihood is constant inside O(X). Integrating over an unknown mixture
of kernels preserves the equality, including author-specific mixtures. Thus,
conditional on the declared orbit statistic, the null order law is uniform
without fitting P or adding pseudocounts. Natural texts need not satisfy this
null. Informative observation, higher history, task opportunity and changing
states may break it. This is not a causal intervention removing those causes.

## 2. The measurement, not a whole-person posterior

For an adjacent triple a, compile the SOURCE field

    F_g(a) = mean_SOURCE_author mean_window 1_(Y_u=g) M_uw(a),
    psi_g(a) = F_g(a) / sum_h F_h(a),

using all32 joint binary-anchor profiles g. Unsupported cells are missing.
Let B_X=E_uniform_O(X) M_(Pi X). Keep all three quantities:

    S_g(X) = <M_X,psi_g>,
    A_g(X) = <B_X,psi_g>,
    D_g(X) = S_g(X)-A_g(X).

Contract each profile vector with the five profile bits for target readings.
S/A are averages of local SOURCE fields, not a likelihood product or full-path
personality posterior. D is a signed conditional order contrast, not a new
psychological dimension. A retains choices, conditions, within-excursion
history and boundary material; it is NOT simply a topic/first-order nuisance.
No component is deleted or declared noise. Raw S=A+D remains recoverable.

The conditional operator is idempotent on this declared orbit; E_orbit D=0.
That orthogonality is under the uniform orbit law, NOT automatically under
the actual author population. For natural covariance always keep cross terms:

    Var(S) = Var(A)+Var(D)+2 Cov(A,D).

Because first/last edges are fixed, both cropped adjacent triple pair margins
are invariant. A field of the form psi(a,b,c)=f(a,b)+g(b,c) has D=0 exactly.
An ordered triple field can have nonzero D while every pair count is unchanged.
The designed two-order toy gives +1/16 and -1/16. This identifies the specific
measurement distinction; it is not evidence of a human personality cause.

## 3. Exact calculation without orbit enumeration

Each nonempty segment contributes fixed internal triples. A width3 observation
crosses at most three segments. For k movable labeled blocks, an oriented pair
appears adjacent with probability1/k; an oriented triple with probability
1/(k(k-1)). Prefix/first and last/suffix probabilities are accounted separately.
Repeated blocks induce equal multiplicity for every distinct word, so labeled
permutation expectation equals uniform distinct-word expectation. Integer
numerators with denominator k(k-1) give the exact triple expectation. Empty
or singleton orbits are retained; D=0 there by construction, not absent traits.

## 4. Actual complete-path construction

1012 windows:386 SOURCE authors/two18-state windows and120 held authors/two
complete windows. Three references;actual and one old-permuted SOURCE profile.
No held labels, raw/private text, API, neural calls or training. Historical
state coding comes from a neural encoder/coarse GRAIN partition; cold numerical
scoring is not a claim that raw-text encoding is wholly LLM-free.

One fixed pass2.589160s CPU. Pooled SOURCE fields support240/240 complete held
orbit scores; individual banks214/240 and197/240. This is a DIFFERENT conditional
functional from HM264's failed inventory posterior: it does not make that
generative law succeed.217/240 held and714/772 SOURCE windows have >1 word;
maximum orbit10080. All unsupported entries remain null. No state/prefix fit.

Each SOURCE author is scored using the opposite bank.295 authors have both
windows fully supported. Same-target finite Pearson contacts and window r:

| Target | S vs SOURCE anchor | A vs SOURCE anchor | D vs SOURCE anchor | S window r | A window r | D window r |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E | .10599 | .11667 | -.08520 | .52818 | .54122 | -.01261 |
| N | .02676 | .03557 | -.07060 | .53137 | .54825 | -.06128 |
| A | .06618 | .07658 | -.06497 | .42885 | .44307 | .01299 |
| C | .09689 | .10861 | -.08630 | .51473 | .51910 | -.03519 |
| O | .02037 | .02700 | -.04696 | .56431 | .55998 | .11992 |

These are SOURCE-only binary-anchor exploratory descriptions, not continuous
Big5 prediction, held performance, significance or independent replications.
All5x5 contacts remain. Old-permutation A window r is .70432/.24533/.52364/
.64961/.43646. Thus strong window consistency alone is not target meaning.

A result-afterward descriptive check splits the two banks:151/144 supported
authors. Actual A window r ranges .18726.. .64916; D ranges -.15068.. .12180.
Positive A consistency is not solely a pooled bank-offset effect, but remains
a property of this observation, not proof of a trait. No threshold promotion.

Pooled held D RMS actual ENACO=.008307/.008162/.007606/.009627/.008970;
old permutation=.007853/.007948/.008976/.007226/.009585. Similar amplitudes
and weak D recurrence do not support treating D as the hidden stable personality
layer in this state family. Preserve O's small positive recurrence, not a new
named factor. The result points toward understanding A, not amplifying D.

## 5. Verification and next decision

All1012 exact cropped pair-margin checks;974 orbits exhaustively enumerated
(58747 numeric words, not additional subjects); direct profile contraction
max1.11e-16. All240 cold CLI values/missing masks match exactly.37 focused tests
(16new), four Python Ruff lint PASS. One prefreeze formatting check differed
in core block-comprehension wrapping; preparation followed and sealed that code,
so the style-only discrepancy was retained rather than rewriting frozen science.
A failed atomic prefreeze documentation patch changed no files. No primary
scientific failure, retry or postfreeze scientific edit. Same-agent not blind.

Next:inventory which author/condition/within-block relations the stable A keeps
on SAME evidence and how they act on an observable target. Do not maximize D,
flip its signs, refine states until r improves, refit a transition kernel or
repeat generic order/nonadditivity. Full author-generative/psychological law
remains open; this packet supplies a useful conditional measurement component.

## Primary sources

- Fortini/Petrone Section3: https://arxiv.org/html/1406.5421v3 . Transition-count
  predictive dependence alone does not establish Markov exchangeability.
- Halberstam/Hutchcroft2024 Section2.1: https://arxiv.org/html/2405.20276v1 .
  Finite Euler counting is mature; no infinite-process theorem transferred.
- OpenAI/math adc7f1241b42e322a6451854ab7e4b4c146bf78a, foundations217-232:
  https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/The-exact-reconstruction-threshold-for-the-three-state-symmetric-channel-September-25-2026/build/sections/02-foundations.tex
  Observation/conditional-expectation inspiration only, not tree thresholds.

The prior probability/group tools are not the novelty claim. The project-specific
construction is an exact, complete-path, SOURCE-qualified conditional score with
an explicit unknown-kernel competitor and retained reference/order accounts.
