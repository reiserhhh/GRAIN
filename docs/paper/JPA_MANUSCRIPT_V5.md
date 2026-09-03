# [BLINDED MANUSCRIPT — Journal of Personality Assessment — v5]

<!-- v5 (2026-07-07): full restructure to the JPA/APA-7 empirical format
     per author instruction — Introduction + theory, Study 1 (Method/
     Results: development), Study 2 (Method/Results: sealed test),
     exploratory Benchmark section, General Discussion. Method now
     carries full reproducibility detail (tokenizer, slicer, leak mask,
     class map, eligibility, power, software versions). Tables 1-4
     realized. v4.2 content and corrections carried forward; every
     number ledger-traceable. Final format details (word limits,
     portal) to be verified against the T&F author portal by the
     author before submission.
     v5.1 (2026-07-07): guidelines VERIFIED in browser (page updated
     2026-06-22): <=40 pages ALL-INCLUSIVE (abstract, tables, refs,
     captions, notes) -> Appendix A moves to supplemental material at
     submission; abstract 200 words (done, 198); NO keywords (removed);
     double-anonymous review, 2 referees; American spelling; double
     quotation marks; T&F standard APA refs; Word template via
     ScholarOne; no submission/publication/page charges; generative-AI
     declaration + data availability statement REQUIRED (added);
     CRediT supported; Open Science Badges available (apply:
     Preregistered + Open Materials).
     v5.6 (2026-07-11): factor-fluidity conjecture (F10) + three
     label-free instruments (time / text position / personal axes);
     exploratory subsection extended and retitled, A.7(iv) added.
     Directional predictions registered and committed BEFORE the runs
     (parent repo commit 60ce99c).
     v5.8 (2026-07-12): round-12 mechanical number sweep of the v5.5-v5.7
     additions (36 items, Sonnet-executed, read-only): one summary range
     fixed (P4 motion statistics -> 'smallest p = .096'; the old .24-.66
     mixed statistics and dropped the smallest p), realized-phi clarified,
     m=3 harmonized into EN; trilingual.
     v5.7 (2026-07-11): continuization of position (F10.6, cut -> flow;
     predictions pre-registered at e839ed9): construct-specific shapes —
     first-person true flow (monotone), tension/directive boundary
     micro-registers, two style clusters mid-dip. -->

# Text as Behavior in Chosen Situations: Theory, Development, and a Preregistered First Test of Group-Referenced Analysis of Individual Natural-Language Expression (GRAIN)

## Abstract

People reveal themselves in what they write, and much of the literature
that scores personality from everyday language treats what a text is
about, and where it was written, as noise. We argue that in
self-selected writing this removes signal, because writers choose
their contexts, and the choices are themselves consistent,
personality-expressive behavior. We develop a framework that reads
accumulated text as behavior distributed over chosen situations,
measured through three channels — context choice, a context-invariant
style base, and person-by-context signatures — and operationalize it as
Group-Referenced Analysis of Individual Natural-Language Expression (GRAIN). In Study 1 (PANDORA;
17.6 million Reddit comments; n = 3,183), removing context means
lowered disjoint-occasion stability for every style measure
(Δ = −.09), and a purity criterion admitted four of nineteen
constructs as candidate style traits, all form habits. Study 2, a
sealed preregistered test (n = 1,058), confirmed two of seven
directional hypotheses — first-person rate predicted Neuroticism
(r = .111), political-venue choice predicted Openness (r = .096) —
while the omnibus criterion failed; no significant effect ran opposite
its prediction. Exploratory benchmarks locate the unfitted instrument
against supervised models (mean r = .27). We discuss what a measure,
unlike a predictor, may claim.

<!-- JPA checklist (2026-06-22): unstructured abstract of 200 words;
     do NOT include keywords. -->

---

## Introduction

Over the past two decades, the language people produce in ordinary life
has become a serious source of personality data. Pennebaker and King
(1999) showed that stylistic word use is reliable across time and genre;
Yarkoni (2010) mapped word-level correlates of the Big Five across nearly
700 bloggers; and Schwartz et al. (2013), analyzing roughly 700 million
words from 75,000 social media users, found that open-vocabulary language
features recover personality, gender, and age with striking specificity.
Language-based assessments correlate with self-reports, agree with
informant reports, and are stable over months (Park et al., 2015). For a
field that has long wanted behavioral alternatives to the questionnaire
(Baumeister et al., 2007), these are encouraging results, and they have encouraged applications
ranging from large-scale epidemiology to clinical screening.


The newest wave of this work runs through large language models, and
its shape is instructive. Much of it retrofits the new machinery onto
instruments designed around human response processes: the models are
administered psychometric inventories themselves (Pellert et al.,
2024), whose validity arguments were built for human respondents and
do not automatically carry over; or they are deployed as zero-shot
raters that infer a person's questionnaire score from text (Peters &
Matz, 2024; Wright et al., 2026), which leaves the questionnaire — not
the verbal behavior — as the criterion and the construct list
untouched; or they generate new items for the old self-report format
(Hommel et al., 2022). Calls for psychology to engage seriously with
these systems are now common (Demszky et al., 2023), but the
engagement has mostly swapped the machinery while keeping the
measurement framework fixed. We start from the other end. If
instruments are increasingly executed, and in part discovered, by
machines, the method itself should be designed around what machines
natively observe — accumulated verbal behavior in chosen contexts, at
volumes no human coder can read — with the human-auditable layer built
on top as a matter of governance rather than serving as the measuring
device. That design brief, and not any particular technology, is what
the framework below tries to satisfy; the General Discussion returns
to it.

Less attention has been paid to a methodological habit that runs through
much of this work. Because personality is understood as something that
should hold across situations, and because text varies enormously with
what it happens to be about, the field's instruments have mostly
been built to keep topical and situational variance out of the
measurement. The closed-vocabulary tradition does this by design:
LIWC-style dictionaries fix their word categories in advance, and
their style core — function words — was chosen in part because it is
nearly content-free (Tausczik & Pennebaker, 2010; Boyd & Schwartz,
2021). Stylometry made the same choice a century earlier and for the
same reason, favoring function words and character patterns precisely
because they are robust to topic (Mosteller & Wallace, 1963;
Stamatatos, 2009). And where context cannot be excluded by feature
choice, the natural next step is to hold it constant or remove it
statistically — the operation we put to a direct test below. The intuition is
reasonable on its face. It is also, we will suggest, in quiet conflict
with the field's own findings, since the open-vocabulary features that
predict best are largely topical (Schwartz et al., 2013; Eichstaedt et
al., 2021). If what people write about were mostly noise around the
person, removing it should have helped.

We recently put the nuisance assumption to a direct test in PANDORA
(Gjurković et al., 2021), a corpus of 17.6 million Reddit
comments from roughly ten thousand users that its authors compiled for
personality research and kindly shared with us, applying context
centering in the strongest
forms we could construct, including two-way fixed-effects estimates fitted
out of sample with cross-fitting. Every variant lowered what we will
call disjoint-occasion stability, that is, the correlation between scores
computed from a person's earlier writing and from their later writing,
with no text shared between the two; the drop held for every style
measure we examined, and pooled across measures the loss was Δ = −.09 (95% CI [−.10, −.08]). The result
seemed paradoxical only until we considered where the "nuisance" variance
comes from. In observational corpora nobody assigns topics. Writers
choose their venues and subjects, the choices repeat, and repeated choice
is exactly the kind of consistent, psychologically expressive behavior
that assessment is supposed to capture. Statistically controlling for
context in such data amounts to adjusting away a pathway through which
personality reaches the page.

There is a precedent for the reorientation we propose. When behavioral
consistency across situations proved elusive, one response was to defend
traits and another was to abandon them; the resolution that endured did
neither. Mischel and Shoda (1995) argued that the situational variability
itself is structured, that stable if-then profiles are signatures of
personality rather than error around it. Our proposal applies the same
lesson to verbal behavior. The context-linked variance that language
researchers remove is not error around the person. In self-selected
corpora, a good portion of it appears to be the person, expressed through
a channel that the removal destroys.

### The Present Studies

This article therefore proceeds in three steps. The theoretical
sections assemble the case that context choice is personal, drawing on
situation-selection research, the behavioral-signature literature,
Brunswik's representative design, stylometry, and work on
within-person distributions, and they state a framework with formal
consequences, including an account of when the standard practices
(centering, reliability interpretation) mislead and when they are
safe. Study 1 develops the instrument on a large self-selected corpus
and tests the framework's two development-stage predictions: that
context centering must reduce person-level stability in such a corpus
(Proposition 2), and that a purity criterion will sort lexical
constructs into person-borne and venue-borne families along the
form–content line (Proposition 3). Study 2 subjects the framework to a
sealed, one-shot preregistered test of seven directional hypotheses on
quarantined criterion labels, reported in full whatever the outcome. A
final section adds exploratory benchmark analyses — a fit gradient
against supervised models on the same users, transport to an
assigned-context corpus, and register robustness — that locate the
instrument's performance and its costs. Throughout, we try to claim no
more than the evidence licenses; the framework forbids several common
practices, and its own record under those prohibitions is part of the
argument. All scoring resources, code, the complete claims ledger
including corrections and retractions, the sealed preregistration, and
the one-shot confirmatory report are public.

## Why Context Choice Is Personal

Four lines of research, none of them ours, converge on the premise that
doing something *somewhere* is informative about the doer.

The first is the situation literature. Buss (1987) distinguished
selection, evocation, and manipulation as mechanisms by which persons and
environments come to correspond, and placed nonrandom selection of
settings first among them. Contemporary situation research has made the
point quantitative: in experience-sampling work, the situations people
report themselves in, and how they construe them, track their traits with
regularity (Rauthmann et al., 2015; Rauthmann & Sherman, 2023). Extraverts
are found in more social situations partly because they arrange to be.
In a text corpus this mechanism is not inferred but logged: the venue and
subject of every post is a recorded act of selection. Treating that
record as noise discards behavior of precisely the sort personality
psychology spent decades arguing matters.

The second line concerns what happens within situations. Shoda et al. (1994) observed children across repeated camp situations and
found that profiles of situation-behavior contingencies, the if-then
signatures, were stable and discriminative even when average behavior was
not. The textual analogue is straightforward to state: how a writer's
style shifts when they enter a particular kind of venue may itself be a
stable personal pattern, distinct from their average style. We treat such
signatures as a third measurement channel, and note that they require
their own null model, since some style shift in a venue is normative for
everyone who posts there.

Third, there is an old argument that assessment should sample the
ecology. Brunswik's representative design holds that generalization
requires sampling stimuli and situations from the organism's actual
environment rather than constructing systematic item sets (Dhami et al., 2004). Viewed this way, the accumulated writing of a
person's daily life is not a poor cousin of the standardized test but the
more representative sample, closer in spirit to Mehl's naturalistic
sound-sampling of daily speech (Mehl et al., 2006) or to
the thin slices of expressive behavior from which observers judge
strangers with surprising accuracy (Ambady & Rosenthal, 1992). What the
ecological tradition has lacked in text is standardized scoring, and that
lack, rather than any shortage of behavior, is what a text instrument has
to supply.

Fourth, a century of stylometry indicates where in language the personal
layer resides. Attribution work from Mosteller and Wallace (1963) onward
has repeatedly found that individuals are identified less by what they
discuss than by small habits of form: function words, punctuation and
orthographic habits, favored constructions. Van Halteren et al. (2005)
argued from such regularities for a "human stylome," and forensic work on
idiolect points the same way (Wright, 2017). Psycholinguistics adds that
stable individual differences extend even to native grammatical knowledge
(Dąbrowska, 2018). If context-invariant personal style exists in text, it
should live disproportionately in form; content vocabulary, by contrast,
is the natural carrier of situation. We will see that this division is
almost exactly what channel-separated measurement recovers.

A fifth consideration frames the whole. Fleeson (2001) showed that a
person's states are best described as a density distribution, with the
trait as parameters of that distribution; a corpus of text slices is such
a distribution, sampled from life rather than elicited by items. And the
idiographic tradition, from Allport (1937) through its modern
formalization in non-ergodicity arguments (Molenaar, 2004; Fisher et al., 2018), warns that between-person structure does
not exhaust the individual. Furr's (2008) decomposition of profiles into
normative and distinctive components gives that warning a measurable
form, which we adopt for the residual layer of the framework.

Taken together, these literatures suggest a specific reading of a text
corpus: it is behavior, distributed over situations the person chose,
sampled from the person's ecology, in which the personal appears through
selection, through form-borne habits, through patterned situational
shifts, and in a residual that factor lists do not capture. The framework
below is an attempt to make that reading precise enough to be wrong.

## A Framework for Text as Situated Behavior

The framework needs three assumptions, none exotic. First, a corpus can be
read as a sample of behavioral observations. We partition each person's
text into slices of fixed token length and treat each slice as one
observation of verbal behavior in a context, so that a person is
represented by a distribution of observations rather than by a single
document. This is the density-distribution view of traits (Fleeson, 2001)
transposed to text, and it matters for the same reason it mattered there:
means, spreads, and contingencies of the distribution are different
quantities, and collapsing them loses information. Second, every slice
carries a context label, and in free ecologies the context was chosen.
The choice distribution appears to be quite stable in its own right; in
our development corpus, the similarity of a user's venue distribution
across time separates same-user from different-user pairs with an area
under the curve near .84, where .50 is chance and 1.00 is perfect
separation, before any style measure enters. Third, we
assume slice scores decompose additively into a person base, a context
effect, a person-by-context interaction, and noise. We treat this as a
working approximation rather than a metaphysical commitment; several of
its testable consequences are examined below, and one of our own
estimators exists mainly to sound an alarm when the approximation fails.

In symbols: for person u and slice i written in venue r(i),

    y_ui = f_u + b_r(i) + γ_u,r(i) + ε_ui,        r(i) ~ π_u,

where f_u is the person's context-invariant base, b_r the style offset
of venue r, γ_u,r the person-by-venue signature, ε_ui noise, and π_u the
person's venue-choice distribution. A person's raw construct score is
the slice mean ȳ_u = f_u + m_u + γ̄_u + ε̄_u, whose mixture term

    m_u = Σ_r π̂_ur b_r

is a function of the choice profile alone; π̂_ur is the realized share
of u's slices falling in venue r.

Under these assumptions, personality can reach a text score through
exactly three routes: through the choice distribution itself, through the
context-invariant base, and through the structure of the interaction
term. We refer to these as the choice, base, and signature channels. The
point of naming them is not taxonomy but bookkeeping: every claim about
"style" is implicitly a claim about which channel carries the observed
covariance, and the channels behave differently under the operations
researchers routinely apply.

Three propositions make this concrete. Proofs, together with estimator definitions sufficient to
recompute every quantity, are given in Appendix A; here we state the
content in words and key identities.

*Proposition 1 (stability is a composite).* The mean of a person's slice
scores estimates the sum of their base and a choice-weighted mixture of
context effects. Because the choice distribution is stable, the mixture
term is stable too, and retest correlations of raw scores therefore
reflect both channels at once. Formally, with stable shares the
disjoint-occasion covariance of raw scores is Var(f) + Var(m) +
2 Cov(f, m), up to signature and noise terms (Appendix A.2). The practical consequence is easy to
underestimate. In a simulated corpus built to contain no personal style
whatsoever, only styled venues and person-stable venue preferences,
ordinary disjoint-occasion retest correlations still reach .64. High
"reliability" of a text measure, taken alone, does not indicate a style
trait; it indicates only that something person-linked is stable, and
that something may be the person's taste in venues.

*Proposition 2 (centering is mediator adjustment).* Subtracting estimated
context means from scores removes the mixture term, and with it every
part of the person signal that travels through choice. Under independence
of choice and context effects, the removed retest covariance is exactly
Var(m) + 2 Cov(f, m) (Appendix A.3), and simulation with known ground truth reproduces
it almost exactly (predicted −1.335, observed −1.332 in covariance
units). When choice and context effects are coupled, as they are whenever
people seek out venues that suit their style, the estimated context
effects additionally absorb part of the base itself, and centering
removes more than the closed form predicts; in our simulations the
regression of estimated on true context effects reaches a slope of 1.40
under strong coupling. The boundary condition is the useful part: when
contexts are assigned rather than chosen, the mixture term is null and
centering is harmless. This is, we believe, why the same operation that
behaves sensibly in prompted-essay research quietly damages measurement
in social media corpora.

*Proposition 3 (identification needs disjointness).* If stability is a
composite, separating the channels requires comparing scores computed on
disjoint sets of contexts, so that shared context effects cannot carry
the correlation. Splitting at the level of individual venues removes most
of the mixture; splitting at the level of venue categories removes
mediation that operates through correlated context effects within a
category. From this we derive a purity criterion for trait language: a
construct is described as a style trait only when it stays stable across
entirely different categories of venue (category-disjoint stability) and
only a small fraction of that stability can be attributed to stable
venue choice rather than to style (a low choice-mediated share). The share is estimated by
covariance accounting, mediated share = 1 − Cov(f̂_e, f̂_l)/Cov_raw,
where f̂ = ȳ − m̂ and the mixture estimate m̂ is cross-fitted, with
venue effects estimated on other persons (Appendix A.4). The thresholds we use (disjoint stability of at least .15 with
a mediated share below .30) are operational conventions, not derived
constants, and we flag them as such.

Two further elements complete the framework. The first is a residual
layer. Following Furr's (2008) decomposition, we subtract the normative
profile from each person's vector of construct scores and treat what
remains, the distinctive profile, as a measurable quantity in its own
right. The theoretical motivation comes from the idiographic tradition
and from non-ergodicity arguments (Molenaar, 2004; Fisher et al., 2018):
between-person structure should not be expected to exhaust the
individual. Empirically — in exploratory analyses on a single development cohort —
it does not. Distinctive profiles correlate with
themselves at a median of .50 across occasions months apart, against a
stranger-pairing baseline of essentially zero, and the correlation rises
to .68 when computed on form-based constructs alone. A further sign
points the same way: roughly a third of the person-identifying signal
recoverable by a modern text embedding lies outside our construct list
altogether, yet is itself stable across occasions. Individuals appear to
be codable well beyond what any fixed factor list captures, which is
what the idiographic tradition has said all along, now with a number
attached.

The second element is a set of comparison licenses. Because channels
respond differently to changes of platform, prompt, and language, a score
earns comparability only within a stated radius, and the radius differs
by construct family. Within-person monitoring in a fixed context is the
least demanding case; comparisons across themes require purity;
comparisons across registers require register-specific norms; and
comparisons of score levels across regimes (free versus assigned versus
domain-locked writing) are not licensed at all, because Proposition 2
implies that no statistical adjustment can equate them. We found it
useful to encode these licenses in the released software, which declines
out-of-license comparisons rather than leaving the matter to a footnote.

## Study 1: Developing the Instrument on a Self-Selected Corpus

Study 1 had two goals: to fix the instrument — the slicing rules, the
construct battery, the choice profile, and the estimators — and to test
the framework's development-stage predictions on data with no access to
criterion labels. Everything in this study was conducted on development
users only, with the confirmation labels of Study 2 still sealed.

### Method

#### Transparency and Openness

We report how eligibility was determined, all exclusions, all
constructs in the battery, and all estimators; there are no unreported
measures. Every analysis, correction, and retraction in the program is
recorded in a public, time-stamped claims ledger, and the scoring
resources are version-pinned by cryptographic hash. Analyses used
Python 3.14 with pandas 3.0, SciPy 1.17, and scikit-learn 1.8;
exact versions and seeds are pinned in the repository. Large parts of
the analysis pipeline were implemented by AI systems under human
direction; we treated that as a reason for stricter governance, and an
adversarial audit with full recomputation rights, documented in the
ledger, caught thirteen substantive errors across eleven rounds, all
corrected before this report. Simulation licenses for every estimator
(generative worlds that deliberately violate the model's assumptions)
are described with the formal results in Appendix A. No new data were
collected; both corpora are secondary, public, and pseudonymous, and
neither is redistributed.

#### Corpora

The main corpus is PANDORA (Gjurković et al., 2021), 17.6 million
Reddit comments from roughly ten thousand users, compiled by its
authors for personality research and provided to us directly. A subset
of users carries Big Five percentile scores (n = 1,401 after the
authors' preparation), and a further subset carries MBTI-type labels;
these labeled users were quarantined from the start (see Study 2), and
Study 1 used only unlabeled development users. The development pool
comprised users with sufficient text in two non-overlapping time
windows; the primary stability analyses used 3,183 such users. The
corpus carries no demographic variables, so invariance across age,
gender, or device could not be examined, and we flag this as a
limitation rather than silently ignoring it. The assigned-context
contrast corpus is the stream-of-consciousness essay collection of
Pennebaker and King (1999), in which students wrote continuously under
a fixed instruction, so that topic and occasion were set by the
researchers rather than chosen by the writers. Both corpora have long
publication histories and are used under their original names and
access terms; the only unpublished data we ourselves collected (a small
market-forum corpus) plays no part in any analysis reported here.

#### Slicing and Leak Masking

Scoring begins by grouping a person's comments by venue (subreddit).
Comments are filtered to English, deleted and removed bodies are
dropped, each body is truncated at 1,500 characters, and each user
contributes at most 400 comments. A user's venues are the subreddits
in which they wrote at least 4 comments, capped at the 8 largest by
volume; each such venue is one *condition*. Within a condition,
comments are concatenated in time order and tokenized with the pattern
`\w+(?:['’-]\w+)?|[^\w\s]`, which treats a word (including internal
apostrophes and hyphens, so that "don't" is one token) or a single
punctuation mark as one token. The token stream is cut into
consecutive, non-overlapping 128-token windows; windows shorter than
24 tokens are dropped, and at most 10 slices are kept per
user-condition cell (per temporal half, where halves are used).
Curly apostrophes are normalized to ASCII before slicing. Finally, any
slice matching a fixed leak pattern is discarded: the pattern covers
Big Five trait names, all sixteen MBTI type codes, "Myers-Briggs",
"Enneagram", and generic assessment vocabulary ("personality",
"trait", "percentile", "score", "introvert(ed)", "extravert(ed)" and
variants), so that no score can be driven by a writer discussing
personality typology in so many words. A measure has no business
reading the answer, or the answer's vocabulary, off the page.

#### Construct Scores

A construct's slice score is the percentage of tokens matching its
fixed word list (matched tokens / tokens in slice × 100); a person's
construct score is the mean over their slices; composite constructs
are fixed weighted sums of such rates (e.g., the affect–tension
composite is 0.40 × negative-affect + 0.65 × conflict-threat + 0.75 ×
uncertainty). The battery contains nineteen lexical constructs
(Table 1): four rationally constructed lexicons (first-person usage,
directive–action language, an affect–tension composite, and
novelty–play vocabulary) and fifteen word-class rates discovered
bottom-up, as clusters of words with correlated within-person usage,
on one random half of the development users and retained only if
their disjoint-occasion stability replicated on the untouched other
half. Word lists are short, public, and version-pinned; revising any
list requires demonstrating, on frozen data, exactly what the revision
changes. Nothing is fitted per person.

#### The Choice Profile

The choice profile is computed from where slices come from rather than
what they say. Each venue belongs to one of twelve content classes
(politics and news, gaming, sports, narrative media, technology, and
so on) under a frozen venue-to-class map built once on development
users and pinned by hash; the map assigns 5,435 venues. One class,
which contains personality-typology venues, is excluded from every
label-facing analysis because its very venue names leak the criterion.
With s_uk person u's share of slices in class k and s̄_k the cohort
mean share, the choice axes are log-ratios a_uk = log((s_uk + ε) /
(s̄_k + ε)) with ε = 10⁻⁴, giving eleven usable axes plus a choice
entropy H_u = −Σ_k s_uk log s_uk. Someone who places 40% of their
comments in political venues, against a population average near 10%,
has a high politics-choice score regardless of anything stylistic in
their writing.

#### Stability and Purity Estimators

All person-level stability quantities are disjoint-occasion
correlations: each person's comment stream is split into an early and
a late portion separated by at least 90 days, each portion is scored
separately, and the two scores are correlated across persons. Because
no text is shared between portions, item-overlap inflation is
impossible by construction. Two constrained variants implement the
purity contrast of Proposition 3: a same-category split, in which both
portions are restricted to venues from the same content classes, and a
different-category split, in which the two portions draw on entirely
different classes. A construct whose stability survives the
different-category split is being carried by the person; one whose
stability collapses there was being carried by the venues. The
choice-mediated share is estimated by the covariance accounting of
Appendix A.4, with venue effects cross-fitted on other users — a
requirement adopted as standing policy after a self-estimation leak
was caught in an early version. The operational purity criterion —
category-disjoint stability of at least .15 with a mediated share
below .30 — uses thresholds that are conventions, and simulation
shows the share estimator is upward-biased under strong style–choice
coupling, so shares are read as intervals rather than points
(Appendix A.4). Estimators were licensed against nine generative
worlds that violate the model's assumptions deliberately; seven
licensed cleanly and two recorded failures, which is what forced the
interval reading.

### Results and Interim Discussion

#### The Centering Test

Proposition 2 predicts that removing context means
from self-selected text should reduce, not improve, the stability of
person scores. It does. Across the five style measures of our initial
battery, condition-mean centering lowered disjoint-occasion stability in
every case, with a pooled change of −.09 (95% CI [−.10, −.08]); partial
centering at any shrinkage level behaved the same way; and two-way
fixed-effects centering estimated out of sample, the strongest version we
could construct, still cost between .05 and .07 per measure once a
self-estimation leak in our first attempt was found and corrected. The boundary condition has, so far, only a simulated instantiation:
with venue assignment made exogenous, the same operation removes
essentially nothing (+.01 in covariance units). The essay corpus
cannot test it directly, because a single fixed prompt leaves no venue
variation to center — the boundary case in degenerate form.


#### Where the Person Signal Lives

Applying the purity criterion of
Proposition 3 to our full inventory of nineteen lexical constructs sorted
them cleanly (Table 1). Four passed as candidate style traits, and all four are
form habits: first-person rate, apostrophe omission, negation, and
profanity. Every construct defined by content vocabulary failed, with
estimated choice-mediated shares between roughly .33 and .61; six were
mixed; one (an affect composite) had too little disjoint stability to
classify at all. The apostrophe case is worth a sentence, because it is
the strongest style trait in the inventory (category-disjoint stability
.37) while being invisible to people: in blind coding, human raters
identified it at chance (0 of 8), whereas the construct held a
cross-validated stability of .65 on users never seen during its
construction. Individuality in written language seems to reside
disproportionately in small mechanical habits, which is precisely what
the stylometric tradition would have predicted, and content vocabulary
behaves like what it is, a record of where the writer goes. Consistent
with this division, character n-gram co-selection axes, fitted on one
half of the users and transported to the other, added twenty-one
dimensions of stable signal not redundant with the word-level battery,
about fifteen of them form-like.


**Table 1.** *The nineteen-construct battery with its purity
decomposition (development users, label-free).*

| Construct | Example vocabulary | Raw | Cat.-disjoint | Share | Family |
|---|---|---|---|---|---|
| first_person | I, me, my, mine | .68 | .27 | .28 | Form habit |
| wcl_60 | its, dont, im, thats, cant | .71 | .37 | .05 | Form habit |
| wcl_13 | not, just, don, because, why | .46 | .17 | .01 | Form habit |
| wcl_23 | pretty, man, damn, profanity | .54 | .17 | .17 | Form habit |
| wcl_03 | so, really, love, great, definitely | .57 | .14 | .51 | Mixed |
| wcl_36 | my, he, she, family, mom, dad | .57 | .16 | .39 | Mixed |
| wcl_11 | use, used, version, fix, screen | .54 | .10 | .49 | Mixed |
| wcl_45 | that, believe, true, fact, however | .52 | .16 | .32 | Mixed |
| wcl_07 | state, country, party, government | .60 | .11 | .34 | Mixed |
| wcl_15 | yeah, oh, lol, haha, gonna | .34 | .14 | .33 | Mixed |
| directive_action | imperative and advice verbs | .41 | .10 | .47 | Venue signature |
| novelty_play | novelty and creativity terms | .38 | .03 | .59 | Venue signature |
| wcl_25 | show, story, character, movie | .52 | .08 | .54 | Venue signature |
| wcl_02 | team, win, season, fan, vs | .55 | .04 | .59 | Venue signature |
| wcl_54 | is, as, which, case, likely | .46 | .10 | .33 | Venue signature |
| wcl_22 | the, there, one, always, called | .47 | .10 | .38 | Venue signature |
| wcl_35 | watch, stuff, youtube, video, music | .45 | .07 | .61 | Venue signature |
| wcl_20 | run, level, hit, kill, build | .50 | .02 | .60 | Venue signature |
| tension_core | 0.40 neg. affect + 0.65 conflict + 0.75 uncertainty | .32 | .05 | — | Undetermined |

*Note.* Raw = disjoint-occasion stability of raw scores; Cat.-disjoint
= stability across venues from entirely different content classes;
Share = estimated choice-mediated share of raw stability (bootstrap
95% CIs and the full decomposition in the Supplement; shares are upper
bounds under coupling, Appendix A.4). Family gate: category-disjoint
≥ .15 and share < .30 (conventions). The four form habits are the only
constructs the criterion certifies as candidate style traits;
tension_core lacked the disjoint stability needed to classify at all.


Study 1's summary is uncomfortable for our own instrument and exactly
what the framework predicts. Removing context means damaged every
measure, as Proposition 2 requires of self-selected text. And of
nineteen constructs built or adopted in good faith as "style"
measures, the purity criterion certified four — all form habits, none
of them content vocabularies — while classifying every content-defined
construct as venue-borne or mixed. Individuality in written language
appears to reside disproportionately in small mechanical habits, which
is what a century of stylometry would have predicted; content
vocabulary behaves like what it is, a record of where the writer goes.
The battery that enters Study 2 therefore carries its own admission
audit: most of it, by its own criterion, measures venues.

## Study 2: A Sealed Confirmatory Test

Study 1 fixed the instrument and showed, on development users, where
person signal should and should not live. Study 2 asked the only
question that settles anything: fixed in advance, do the channels
predict quarantined criterion labels in the preregistered directions?

### Method

#### Preregistration and Custody

The confirmation labels (Big Five percentile scores for the held-out
PANDORA users) were quarantined before development began and were
never read by any development analysis. The confirmatory protocol —
hypotheses, directions, eligibility rules, correction procedure, and
success rule — was committed to a public repository whose initial
commit hash serves as the seal. The analysis script was then audited
by an independent adversarial process, revised, pinned by its own
hash, and run exactly once; a sentinel file makes a second run
detectable. The realized eligible sample was required to fall within
15% of the preregistered estimate for the run to proceed (it did:
1,058 realized against 1,108 estimated). The output was published as
computed, favorable or not.

#### Hypotheses

Seven confirmatory directional hypotheses and one secondary bridge
hypothesis were fixed (Table 2). Four address style-base constructs
(H1–H4) and three address the choice channel (H6–H8); groundings were
recorded in the preregistration, which also flagged H8 as the weakest
entry. The success rule was at least 4 of 7 confirmed at
Benjamini-Hochberg q < .05 within the set, with every significant
effect in the preregistered direction. H5 (a bridge hypothesis onto
MBTI Thinking, n = 303) was declared secondary because it is
underpowered, and is excluded from the denominator.

#### Participants and Power

Eligibility — at least 4 venue conditions and at least 12 post-mask
slices — was estimated from dump metadata only, without scoring text
or reading labels: 1,108 of the 1,401 labeled users were expected to
qualify, and 1,058 did. A priori power at n = 1,108 (two-tailed
α = .05) was .76 for an observable r of .08, .92 at .10, and > .99 at
.15. The preregistration also recorded attenuation ceilings: with
questionnaire reliability near .85 and disjoint-occasion reliabilities
from the development registry (.60–.82 by construct), a true validity
of .20 would surface as an observable r of roughly .14, so
single-construct correlations in the .10–.25 band were declared the
realistic success zone in advance.

#### Analysis

One Pearson correlation per hypothesis on the eligible users, Fisher
95% confidence intervals, Benjamini-Hochberg correction within the
seven-hypothesis set, and the success rule above. The preregistration
additionally declared one non-confirmatory report: the incremental
ΔR² of the battery over a 194-category closed-dictionary ridge
baseline for Neuroticism and Openness.

### Results


**Table 2.** *Preregistered hypotheses and outcomes (sealed one-shot
run; confirmatory n = 1,058, bridge n = 303).*

| H | Predictor (channel) | Criterion | Dir. | r | 95% CI | q | Outcome |
|---|---|---|---|---|---|---|---|
| H1 | affect–tension composite (base) | Neuroticism | + | .052 | [−.009, .112] | .218 | ns |
| H2 | first-person rate (base) | Neuroticism | + | .111 | [.051, .170] | .002 | confirmed |
| H3 | novelty–play (base) | Openness | + | −.046 | [−.106, .014] | .230 | ns |
| H4 | directive–action (base) | Agreeableness | − | .029 | [−.031, .089] | .397 | ns |
| H6 | politics/news choice axis (choice) | Openness | + | .096 | [.036, .155] | .006 | confirmed |
| H7 | choice entropy (choice) | Openness | + | −.009 | [−.069, .052] | .776 | ns |
| H8 | gaming choice axis (choice) | Extraversion | − | −.041 | [−.101, .020] | .260 | ns |
| H5 | directive–action (bridge, secondary) | MBTI Thinking | + | .062 | [−.051, .174] | — | ns (excluded from set) |

*Note.* Dir. = preregistered direction; q = Benjamini-Hochberg within
the seven confirmatory hypotheses; grounding notes for each hypothesis
are in the preregistration (H2: first-person–Neuroticism/depression
meta-analysis, Edwards & Holtzman, 2017; H8 was flagged a priori as
the weakest entry). No significant effect ran opposite its
preregistered direction.

The omnibus rule failed: two of seven hypotheses survived correction,
against the preregistered bar of four. The two survivors are one from
each of the framework's primary channels, and the style-side survivor
is the one construct in the battery that the purity criterion
certifies as person-borne: first-person rate predicted Neuroticism
(r = .111, 95% CI [.051, .170], q = .002), and choice of political
venues predicted Openness (r = .096, [.036, .155], q = .006). The
declared non-confirmatory check showed no incremental ΔR² of the
battery over the closed-dictionary baseline for either trait (both
cross-validated R² values were negative at this n, reflecting the
weakness of both feature sets for absolute prediction).

The five failures divide into two kinds, and only one of them is
explained by the framework. Three involve style constructs — the
affect–tension composite, the novelty measure, and the
directive-language measure — that the purity criterion of Study 1,
derived after these hypotheses were fixed, classifies as venue-borne
or unclassifiable (Table 1); had the criterion existed at
preregistration time, it would have barred all three from
confirmatory testing. The other two were choice-channel hypotheses
that simply failed: a venue-breadth conjecture (H7), and a
gaming–extraversion prediction (H8) that the preregistration itself
had flagged as its weakest entry. For these two we have no
after-the-fact account, and we decline to construct one. We report
the failure as a failure: what the opening licenses is deliberately
narrow — two channel-level directional claims, at small effect sizes,
in one population — and what it illustrates more broadly is that the
framework's admission rules and its confirmatory record point the
same way.

## Benchmark and Generalization Analyses

The analyses in this section are exploratory and clearly labeled as
such in the ledger: the first reuses the confirmation labels after
their one-shot opening, and the others use development data. Their
purpose is to locate the instrument — against fitted prediction
models, across writing regimes, and across registers — rather than to
confirm anything.

### Measurement Versus Prediction on the Same Corpus

A fair question about any new instrument is how it compares with what
already exists, and for PANDORA the comparison is unusually direct,
because the corpus's authors published supervised baselines — ridge and
lasso models over tf-idf-weighted n-grams, fitted to the Big Five
labels — and shipped their code and predictions with the data. We
reproduced their official Big Five baselines from that package to
within Δr ≤ .0014 of the shipped values: their n-gram model averages
r = .246 across the five traits (n = 1,402), and their augmented model,
which adds predicted MBTI and Enneagram features, averages r = .293
(n = 1,386).

The comparison has to be read carefully, because the two exercises
answer different questions (Yarkoni & Westfall, 2017). A prediction
model may fit anything that helps; it is judged on held-out accuracy,
and its coefficients carry no commitment about what is being measured.
A measure fixes its constructs and directions before seeing criterion
data and is judged on what its scores mean. To put both on one footing
we ran an exploratory fit gradient on exactly the users of our sealed
test (n = 1,058, identical folds), moving from full fitting to none
(Table 3). A ridge model over tf-idf unigrams, the model family of the
official baselines, reaches a mean r of .272 on these
users, sitting between the two official baselines, which suggests our
eligibility rules did not select an unusually easy or hard cohort.
Restricting the same fitting to the sixteen frozen, preregistered
features of our instrument (four style constructs, eleven venue-choice
axes, choice entropy) drops the mean to .090, with the gains
concentrated where the framework says person signal lives, in
Neuroticism (r = .166) and Openness (r = .105). Removing fitting
entirely leaves the preregistered single constructs of the sealed test
(r = .111 and .096). These analyses reuse labels that were already
unsealed at the confirmatory opening; they are exploratory by
construction and are recorded as such in the ledger.

**Table 3.** *An exploratory fit gradient on the sealed-test cohort (n =
1,058, identical five-fold splits), with the corpus authors' official
baselines as an external anchor.*

| Model | Fitted to labels | E | N | A | C | O | Mean |
|---|---|---|---|---|---|---|---|
| Ridge over tf-idf unigrams | features and weights | .31 | .30 | .29 | .18 | .28 | .27 |
| — content vocabulary only (stop words removed) | features and weights | .32 | .30 | .29 | .18 | .26 | .27 |
| — function words only (~318, fitted) | features and weights | .15 | .15 | .13 | .05 | .19 | .13 |
| Ridge over the 16 frozen instrument features | weights only | .09 | .17 | −.01 | .10 | .10 | .09 |
| Preregistered constructs (sealed test) | nothing | — | .11 | — | — | .10 | — |
| Official baseline, n-grams (n = 1,402) | features and weights | .33 | .24 | .23 | .16 | .26 | .25 |
| Official baseline, + MBTI/Enneagram (n = 1,386) | features and weights | .39 | .28 | .27 | .27 | .25 | .29 |

*Note.* Official rows are on the authors' own user set and folds,
reproduced from their released package to Δr ≤ .0014; they are context,
not a same-cohort contrast. The sealed-test row reports the two preregistered survivors; the
remaining cells were not hypothesized. The two indented rows split the
full-fit vocabulary into its content and function-word parts.

Read as a gradient, the three rungs price the ingredients separately.
Fitting combination weights over the committed features buys little
beyond the best single construct, so the instrument's parsimony costs
almost nothing. Opening the vocabulary roughly triples the mean correlation, and the
concentration of that advantage in content is measured rather than
presumed: restricting the fitted vocabulary to roughly three hundred
function words drops the mean to r = .134, while removing exactly
those words and keeping the content vocabulary costs almost nothing
(r = .268 against .272). On the framework's reading the fitted models
are harvesting the
choice channel wholesale — topics, venues, and all — without asking
which part of it is the person and which the place. For prediction
that harvesting is legitimate and effective. A measure must refuse it
until the purity question is answered, and on this corpus the price of
the refusal is the difference between r ≈ .27 and r ≈ .10. We take the
gradient as making the cost of interpretability explicit, rather than
leaving it to be discovered.

### Transport to an Assigned-Context Corpus

The framework expects direction to travel better than
level. A battery fitted entirely on Reddit and applied without
refitting to the student essays retained directional validity (mean
cross-validated correlation with Big Five scores of about .14,
comparable to a transported text-embedding baseline of about .15), while
score levels and covariance structure did not transport; within Reddit,
all nineteen constructs survived a register shift from top-level posts
to replies once volume was matched, with level shifts confined to two
venue-linked constructs.

### Register Robustness Within the Platform

Within Reddit, the dialogue-register boundary between top-level posts
and replies offers a cheap transport test: the same person, the same
venues, a different communicative footing. On 1,976 deep development
users with at least 6 slices in every register-by-half cell, all
nineteen constructs met the robustness rule (cross-register,
opposite-half correlation at least .70 of the within-register value);
material level shifts appeared only for a family-narrative word class
(d = 0.86, replies richer in personal narrative) and a sports-linked
class (d = 0.56), for which per-register norms are recommended.
Same-occasion cross-register correlations ran about .08 above
opposite-half ones, a reminder that adjacency inflates and that
disjoint-occasion designs remain the honest default.

### Instruments From the Aggregation Geometry

The instrument set gained a further, more mathematical layer late in the
program, developed in a parallel deployment of the framework and imported
here; all three analyses are exploratory and label-free. The first reads
aggregation through a projective coordinate. With φ(r) = r/(1−r), the
Spearman–Brown lengthening family becomes exactly linear, φ(r_W) =
W·φ(r₁), so plotting φ against a designed number of slices W tests the
measurement model itself rather than any single reliability figure. On
2,767 development users (nested designed subsets, W = 1, 2, 4, 8), the
affect–tension composite followed the saturating alternative with an
asymptote matching its realized full-half stability (fitted φ_max = 0.49
against a realized φ of 0.46): its low reliability is a ceiling, not a
sample-size accident. The apostrophe construct violated the saturation
model in the opposite direction — its fitted asymptote falls far below
its realized large-W value — which is the signature of zero-inflation at
the slice level: many 128-token slices offer no apostrophe opportunity
at all, so at small W the construct behaves as a hurdle process rather
than a rate. The second instrument is a time-reversal null: under
stationarity, each person's occasion sequence is exchangeable with its
reversal, so per-person sign flips give an exact null for mean drift.
First-person rate drifts downward at −0.22 to −0.25 rate-units per year
(p < .0005; n = 4,454 pooled, 3,278 venue-constant), and the drift
survives holding each writer's single largest venue fixed — a
within-person change, not a venue-mix change — while the construct
remains rank-order stable: the classic maturation pattern of a trait
measure, and a boundary condition for within-person monitoring. The
third instrument tests factor claims against a random-matrix benchmark:
exactly four components of the nineteen-construct battery exceed the
Marchenko–Pastur edge (empirical shuffle edge 1.169), and all four
replicate across disjoint halves (loading congruences .93–.99).

The same geometric register then posed a further question, arriving as
a conjecture: is the factor frame itself fixed, or does it flow with
conditions — and if it flows, do the axes stop being nameable and
become carriers of individuality instead? Directional predictions were
registered before the runs. Across time the population frame is rigid:
the top-four eigenframe, recomputed over four within-person time
blocks, moves no more than within-person block-permutation nulls allow (all
four motion statistics nonsignificant across two cohort definitions,
smallest p = .096). Across
position in a text it flows: comparing the opening against the closing
128 tokens of 21,496 long comments, the first two axes hold
(congruences .97 and .94) while the third and fourth reorient (.73 and
.78; subspace distance p < .002 against per-comment swap nulls), and
mean levels shift — first-person rate runs markedly higher in openings
(user-level difference 0.69 rate units, p < .0005) and affect–tension
slightly higher in closings, while a registered hunch that directive
language rises toward closings failed (p = .10). A continuous
refinement — windows indexed by relative position within a comment,
with composition-matched within-comment permutation nulls and a
composition-free paired stratum (m = 3) — resolves the shape per
construct:
first-person rate declines gradually along the text (monotone,
curvature not significant — a true flow), affect–tension jumps in the
closing zone, directive language turns out to be elevated at openings
(a boundary structure the two-point contrast could not see), and two
style clusters dip mid-text; some constructs flow, others live in
boundary micro-registers. Across persons the
frame is a signature: each writer's leading within-person eigenvector
matches itself across time-disjoint halves better than stranger
pairings (mean |cos| = .257 vs. .224, p < .0033, n = 2,430), sits at
the odd–even estimation ceiling (≈ .25), so any within-person rotation
is at most mild, and — the registered kill condition — the
self–stranger gap widens rather than closes (.277 vs. .216) once the
shared four-factor frame is projected out: personal orientation is not
a noisy copy of the population factors. Individual axes also disperse
widely against those factors (median best alignment .49, against a .33
random-direction median). The joint reading is specific: the population
factors are stable coordinates, but at the person level the orientation
itself is trait-like, individually informative, and not usefully
nameable — one more measurement object that is natural for machine
consumption and awkward for lexical interpretation. Formal statements
and the associated falsifiers are collected in Appendix A.7.

## General Discussion

The framework began as an attempt to explain one uncomfortable result,
and its value should be judged by how much else it organizes. Several
standing features of the literature follow naturally once text is read
as behavior in chosen situations. The predictive advantage of
open-vocabulary methods over closed dictionaries (Schwartz et al., 2013;
Kern et al., 2014) is expected, because topical features carry the
choice channel that style dictionaries exclude by construction; on this
reading the advantage reflects channel coverage more than lexical resolution; our own fit gradient reproduces the pattern within a single corpus, where opening the vocabulary roughly triples mean correlation over the committed feature set (Table 3). The decades-long robustness of function words as individual
markers, from authorship attribution to Pennebaker's program
(Mosteller & Wallace, 1963; Tausczik & Pennebaker, 2010), is equally
expected, because form habits are where the context-invariant base
lives. The recurrent finding that language-based models lose accuracy when
moved across platforms (Jaidka et al., 2018), while rank-order
relations survive in attenuated form — as in our own transport to the
essay corpus — matches the framework's boundary
structure: levels are regime-bound, direction travels. And the repeated
disappointment of affect word-rates as measures of stable individual
differences — dictionary affect scores can even run in the wrong
direction against the well-being of the populations producing them
(Jaidka et al., 2020) — has a simple description here: they are heavily context-loaded and behave as states
sampled at the wrong timescale, which is what the density-distribution
view would suggest.

The account also forbids things, and the prohibitions are its most
practical output. It forbids topic and venue centering as a default in
self-selected corpora, and specifies the design alternative (assigned
contexts) under which the same operation becomes safe. It forbids trait
interpretation from reliability alone, supplying instead an operational
purity check that our own inventory largely failed. And it forbids
comparing score levels across free, assigned, and domain-locked writing,
a comparison that current practice performs implicitly whenever
platform-trained models are applied elsewhere. We would regard the
framework as damaged if a well-designed study showed context centering
improving disjoint-occasion person stability in a clearly self-selected
corpus, or showed level comparability across regimes achieved by
statistical adjustment alone. Table 4 collects the framework's
commitments, the tests each has faced, and their current status.

**Table 4.** *The framework's commitments, the tests they have faced,
and their current status.*

| Commitment | Test | Status |
|---|---|---|
| Centering reduces person stability in self-selected text (P2) | 5 measures, pooled Δ = −.09 [−.10, −.08]; strongest cross-fitted variant −.05 to −.07 | Supported (Study 1) |
| Raw stability is a base + choice composite (P1) | Style-null simulation: raw retest .64, disjoint −.07 | Supported (simulation) |
| Centering removes exactly Var(m) + 2Cov(f, m) (P2) | Oracle simulation: −1.332 observed vs −1.335 predicted | Supported (simulation) |
| Purity criterion sorts constructs along the form–content line (P3) | 4 form / 8 venue / 6 mixed / 1 undetermined | Supported (Study 1, Table 1) |
| Channel predictions hold on quarantined labels | 2 of 7 confirmed; omnibus rule (≥ 4/7) failed | Partially supported (Study 2, Table 2) |
| No significant wrong-direction effects | 0 of 7 observed | Supported (Study 2) |
| Direction transports across regimes; level does not | Battery .14 ≈ embedding .15 on essays; levels and covariance fail | Supported (exploratory) |
| Constructs survive register shifts | 19 of 19 robust; 2 level shifts flagged | Supported (exploratory) |
| Estimators remain valid under assumption violations | 7 of 9 worlds licensed; 2 recorded failures adopted as interval readings | Mixed, incorporated (Appendix A) |
| The supervised advantage is content-borne | Content-only r = .268 ≈ full .272; function-words-only .134 | Supported (exploratory, Table 3) |

*Note.* Sources for every number are in the public claims ledger;
"simulation" rows are licensing results for the estimators, not
evidence about people.



Several limitations bound what we claim. All trait-level evidence comes
from one platform plus one prompted-essay corpus, in English, from
people who write a great deal; eligibility rules discard light writers,
and nothing here speaks to them. The corpora carry no demographics, so
measurement invariance across age, gender, or device could not be
examined. The confirmed effects are small (r ≈ .10-.11), far below what
any individual-level decision would require, and we make no applied
claims. The omnibus preregistered criterion failed, and although the
pattern of failures is informative, it was diagnosed with a criterion
built afterward; the diagnosis is a hypothesis about the next study, not
a rescue of this one. Finally, the program was largely executed by AI
systems under human direction. We treated that as a reason for stricter
governance rather than lighter: estimators were licensed against
generative worlds violating the framework's own assumptions,
confirmatory material sat behind a one-shot public seal, and an
adversarial audit with full recomputation rights caught thirteen real
errors across eleven rounds, all documented. The audit trail is part of
the evidence, in the same sense that a telescope's calibration record is
part of an observation.

Two studies would move the account decisively. The first is a
within-person, multi-regime design: the same participants writing in
free and assigned contexts, twice, in two languages, with questionnaire
and informant anchors. That single design tests level equating across
regimes (the one comparison we currently forbid), state structure below
the month timescale, dialogue registers, and the cross-language
generality of the form-based channel, and its full specification is
public. The second is already running in a weak form: structural
predictions about text not yet written, sealed by hash in advance. Both
follow the same policy that produced the present results, which is to
make the framework easy to kill and then report whether it died.

A further direction follows from what the program kept finding at its
edges. The strongest style trait in our inventory was not proposed by
a person: it was discovered by machine clustering, is invisible to
human raters (0 of 8 in blind coding), and yet replicates on users
never seen during its construction (r = .65). And when a modern text
embedding is allowed to identify writers, roughly a third of its
stable person-signal lies outside our entire construct list. Both
facts point the same way: if individuality in language is
fingerprint-like — codable in full only by a high-dimensional
representation, never exhausted by a human-readable factor list — then
the natural instrument is partly a machine one, and a battery like
ours is best understood as the licensed, auditable window onto that
representation rather than as the representation itself. We expect
scale development and interpretation methods to move in this
direction, with learned representations proposing candidates and
frameworks like the present one deciding what they may claim:
transported fits, purity gates, convergence licenses, and sealed tests
are exactly the governance such instruments need, because our own
development also showed how easily vector scores mislead without
them — random directions in embedding space are person-stable too, so
stability alone licenses nothing there. On this view the assessment of
the future is neither hand-built nor black-box, but machine-proposed
and theory-audited — the design brief with which we began.

Personality assessment has long treated the standardized test and the
spontaneous sample as belonging to different traditions, one scorable
and one rich (cf. McClelland et al., 1989). A person's accumulated writing suggests the distinction is
not fundamental. What people write about, where they choose to write it,
and the small mechanical habits that persist regardless are all behavior;
read together, under rules that respect which channel carries which
signal, they yield an assessment that is standardized in scoring and
ecological in origin. The present framework is one way to make that
reading precise, and, so far, it has survived its own tests only in the
places it said it would.

## Open Practices Statement

All formal derivations, simulation code, frozen scoring resources, the
complete claims ledger including corrections and retractions, the sealed
preregistration, and the one-shot confirmatory report are available at
[ANONYMIZED REPOSITORY LINK]. No raw user text or identifiers are
redistributed; the source corpora are available from their original
publishers. The fit-gradient analyses of Table 3 reuse the confirmation labels
after their one-shot opening; they are labeled exploratory in the
ledger and carry no confirmatory weight.

## Declarations

**Disclosure statement.** The authors report there are no competing
interests to declare.

**Funding.** [Completed on the unblinded title page.]

**Declaration of generative AI use.** Generative AI systems were used,
under human direction, to implement analysis code, run the
preregistered analyses, and draft manuscript text; the authors directed
the program, verified the results against the public claims ledger, and
take full responsibility for the content. The
role of AI here is constitutive rather than incidental: the
instrument is designed to be machine-executed at scale, part of its
battery was machine-discovered (including a construct invisible to
human raters), and the framework's residual layer concerns signal
that only high-dimensional representations carry (see General
Discussion). An adversarial audit protocol
with full recomputation rights, documented in that ledger, governed
this use (see Transparency and Openness).

**Data availability statement.** The PANDORA corpus was created by
Gjurković et al. (2021) and was provided to us by its authors; the
essay corpus was created by Pennebaker and King (1999). Both are used
under their original names and access terms, and neither is
redistributed. All scoring resources, analysis code, the complete
claims ledger, the sealed preregistration, and the one-shot
confirmatory report are available at [ANONYMIZED REPOSITORY LINK]; no
raw user text or identifiers are redistributed.
[Acknowledgment of the PANDORA team moves to the unblinded title page.]

## References

<!-- Entries compiled v4; bibliographic details re-verified at the
     round-12 pre-submission audit. -->

Allport, G. W. (1937). *Personality: A psychological interpretation*. Henry Holt.

Ambady, N., & Rosenthal, R. (1992). Thin slices of expressive behavior as predictors of interpersonal consequences: A meta-analysis. *Psychological Bulletin, 111*(2), 256–274.

Baumeister, R. F., Vohs, K. D., & Funder, D. C. (2007). Psychology as the science of self-reports and finger movements: Whatever happened to actual behavior? *Perspectives on Psychological Science, 2*(4), 396–403.

Boyd, R. L., & Schwartz, H. A. (2021). Natural language analysis and the psychology of verbal behavior: The past, present, and future states of the field. *Journal of Language and Social Psychology, 40*(1), 21–41.

Brunswik, E. (1956). *Perception and the representative design of psychological experiments* (2nd ed.). University of California Press.

Buss, D. M. (1987). Selection, evocation, and manipulation. *Journal of Personality and Social Psychology, 53*(6), 1214–1221.

Dąbrowska, E. (2018). Experience, aptitude and individual differences in native language ultimate attainment. *Cognition, 178*, 222–235.

Edwards, T., & Holtzman, N. S. (2017). A meta-analysis of correlations between depression and first person singular pronoun use. *Journal of Research in Personality, 68*, 63–68.

Demszky, D., Yang, D., Yeager, D. S., Bryan, C. J., Clapper, M., Chandhok, S., Eichstaedt, J. C., Hecht, C., Jamieson, J., Johnson, M., Jones, M., Krettek-Cobb, D., Lai, L., JonesMitchell, N., Ong, D. C., Dweck, C. S., Gross, J. J., & Pennebaker, J. W. (2023). Using large language models in psychology. *Nature Reviews Psychology, 2*(11), 688–701.

Dhami, M. K., Hertwig, R., & Hoffrage, U. (2004). The role of representative design in an ecological approach to cognition. *Psychological Bulletin, 130*(6), 959–988.

Eichstaedt, J. C., Hommel, B. E., Wollang, F.-J. M., Kotova, V., Zacher, H., & Schmukle, S. C. (2022). Transformer-based deep neural language modeling for construct-specific automatic item generation. *Psychometrika, 87*(2), 749–772.

Jaidka, K., Giorgi, S., Schwartz, H. A., Kern, M. L., Ungar, L. H., & Eichstaedt, J. C. (2020). Estimating geographic subjective well-being from Twitter: A comparison of dictionary and data-driven language methods. *Proceedings of the National Academy of Sciences, 117*(19), 10165–10171.

Jaidka, K., Guntuku, S. C., & Ungar, L. H. (2018). Facebook versus Twitter: Cross-platform differences in self-disclosure and trait prediction. In *Proceedings of the Twelfth International AAAI Conference on Web and Social Media*. AAAI Press.

Kern, M. L., Yaden, D. B., Schwartz, H. A., Giorgi, S., Park, G., Hagan, C. A., Tobolsky, V. A., Smith, L. K., Buffone, A., Iwry, J., Seligman, M. E. P., & Ungar, L. H. (2021). Closed- and open-vocabulary approaches to text analysis: A review, quantitative comparison, and recommendations. *Psychological Methods, 26*(4), 398–427.

Fisher, A. J., Medaglia, J. D., & Jeronimus, B. F. (2018). Lack of group-to-individual generalizability is a threat to human subjects research. *Proceedings of the National Academy of Sciences, 115*(27), E6106–E6115.

Fleeson, W. (2001). Toward a structure- and process-integrated view of personality: Traits as density distributions of states. *Journal of Personality and Social Psychology, 80*(6), 1011–1027.

Furr, R. M. (2008). A framework for profile similarity: Integrating similarity, normativeness, and distinctiveness. *Journal of Personality, 76*(5), 1267–1316.

Gjurković, M., Karan, M., Vukojević, I., Bošnjak, M., & Šnajder, J. (2021). PANDORA talks: Personality and demographics on Reddit. In *Proceedings of the Ninth International Workshop on Natural Language Processing for Social Media* (pp. 138–152). Association for Computational Linguistics.

Kern, M. L., Eichstaedt, J. C., Schwartz, H. A., Dziurzynski, L., Ungar, L. H., Stillwell, D. J., Kosinski, M., Ramones, S. M., & Seligman, M. E. P. (2014). The online social self: An open vocabulary approach to personality. *Assessment, 21*(2), 158–169.

McClelland, D. C., Koestner, R., & Weinberger, J. (1989). How do self-attributed and implicit motives differ? *Psychological Review, 96*(4), 690–702.

Mehl, M. R., Gosling, S. D., & Pennebaker, J. W. (2006). Personality in its natural habitat: Manifestations and implicit folk theories of personality in daily life. *Journal of Personality and Social Psychology, 90*(5), 862–877.

Mischel, W., & Shoda, Y. (1995). A cognitive-affective system theory of personality: Reconceptualizing situations, dispositions, dynamics, and invariance in personality structure. *Psychological Review, 102*(2), 246–268.

Molenaar, P. C. M. (2004). A manifesto on psychology as idiographic science: Bringing the person back into scientific psychology, this time forever. *Measurement: Interdisciplinary Research and Perspectives, 2*(4), 201–218.

Mosteller, F., & Wallace, D. L. (1963). Inference in an authorship problem. *Journal of the American Statistical Association, 58*(302), 275–309.

Park, G., Schwartz, H. A., Eichstaedt, J. C., Kern, M. L., Kosinski, M., Stillwell, D. J., Ungar, L. H., & Seligman, M. E. P. (2015). Automatic personality assessment through social media language. *Journal of Personality and Social Psychology, 108*(6), 934–952.

Pellert, M., Lechner, C. M., Wagner, C., Rammstedt, B., & Strohmaier, M. (2024). AI psychometrics: Assessing the psychological profiles of large language models through psychometric inventories. *Perspectives on Psychological Science, 19*(5), 808–826.

Pennebaker, J. W., & King, L. A. (1999). Linguistic styles: Language use as an individual difference. *Journal of Personality and Social Psychology, 77*(6), 1296–1312.

Peters, H., & Matz, S. C. (2024). Large language models can infer psychological dispositions of social media users. *PNAS Nexus, 3*(6), Article pgae231.

Rauthmann, J. F., & Sherman, R. A. (2023). Patterned person–situation fit in daily life: Examining magnitudes, stabilities, and correlates of trait–situation and state–situation fit. *European Journal of Personality*. Advance online publication. https://doi.org/10.1177/08902070221104636

Rauthmann, J. F., Sherman, R. A., & Funder, D. C. (2015). Principles of situation research: Towards a better understanding of psychological situations. *European Journal of Personality, 29*(3), 363–381.

Schwartz, H. A., Eichstaedt, J. C., Kern, M. L., Dziurzynski, L., Ramones, S. M., Agrawal, M., Shah, A., Kosinski, M., Stillwell, D., Seligman, M. E. P., & Ungar, L. H. (2013). Personality, gender, and age in the language of social media: The open-vocabulary approach. *PLoS ONE, 8*(9), Article e73791.

Shoda, Y., Mischel, W., & Wright, J. C. (1994). Intraindividual stability in the organization and patterning of behavior: Incorporating psychological situations into the idiographic analysis of personality. *Journal of Personality and Social Psychology, 67*(4), 674–687.

Stamatatos, E. (2009). A survey of modern authorship attribution methods. *Journal of the American Society for Information Science and Technology, 60*(3), 538–556.

Tausczik, Y. R., & Pennebaker, J. W. (2010). The psychological meaning of words: LIWC and computerized text analysis methods. *Journal of Language and Social Psychology, 29*(1), 24–54.

van Halteren, H., Baayen, H., Tweedie, F., Haverkort, M., & Neijt, A. (2005). New machine learning methods demonstrate the existence of a human stylome. *Journal of Quantitative Linguistics, 12*(1), 65–77.

Wright, A. G. C., Ringwald, W. R., Vize, C. E., Eichstaedt, J. C., Angstadt, M., Taxali, A., & Sripada, C. (2026). Assessing personality using zero-shot generative AI scoring of brief open-ended text. *Nature Human Behaviour, 10*(3), 541–555.

Wright, D. (2017). Using word n-grams to identify authors and idiolects: A corpus approach to a forensic linguistic problem. *International Journal of Corpus Linguistics, 22*(2), 212–241.

Yarkoni, T. (2010). Personality in 100,000 words: A large-scale analysis of personality and word use among bloggers. *Journal of Research in Personality, 44*(3), 363–373.

Yarkoni, T., & Westfall, J. (2017). Choosing prediction over explanation in psychology: Lessons from machine learning. *Perspectives on Psychological Science, 12*(6), 1100–1122.


## Appendix A: Model, Proofs, and Estimator Definitions

**A.1 Model and notation.** For person u and slice i written in venue
r(i),

    y_ui = f_u + b_r(i) + γ_u,r(i) + ε_ui,        r(i) ~ π_u,

where f_u is the person's context-invariant base, b_r the style offset
of venue r, γ_u,r the person-by-venue signature, ε_ui noise, and π_u
the venue-choice distribution. Venue effects are mean-zero and
independent across venues unless stated; γ and ε are mean-zero,
independent of the rest, and independent across occasions. The raw
score is the mean over a person's slices,

    ȳ_u = f_u + m_u + γ̄_u + ε̄_u,        m_u := Σ_r π̂_ur b_r,

with π̂_ur the realized share of u's slices in venue r; m_u depends on
the choice profile alone. Stability quantities are covariances (or
correlations) across persons between scores computed on an early and a
late occasion with no shared text.

**A.2 Proposition 1 (stability is a composite).** When the venue sets
of the two occasions overlap and shares are stable, m_u^(e) ≈ m_u^(l),
and expanding ȳ under A.1 with γ, ε independent across occasions gives

    Cov(ȳ^e, ȳ^l) = Var(f) + Var(m) + 2 Cov(f, m) + [signature terms].

Raw and shared-venue retest correlations therefore carry the full
mixture variance. The simulated flesh-null world (f ≡ 0, styled venues,
person-stable preferences) makes the size concrete: raw retest r = .64
and shared-venue retest r = .43, while the disjoint estimand of A.4
reads −.07 on the same world.

**A.3 Proposition 2 (centering is mediator adjustment).** Choice
mediates person → text (f_u → π_u → venue → y). Venue centering
replaces y_ui by y_ui − b̂_r(i), which removes m̂_u from ȳ_u. When
choice is independent of venue effects (π ⊥ b) and b̂ is estimated out
of sample (cross-fitted), the removal is exact in expectation, and the
change in disjoint-occasion covariance is

    Δ Cov(retest) = − [ Var(m) + 2 Cov(f, m) ]  ≤ 0

whenever the mixture is person-stable and non-negatively coupled. With
oracle (true) venue effects the identity reproduces in simulation to
−1.332 observed against −1.335 predicted, in covariance units. Under
coupling — people seeking venues that suit their style — b̂ absorbs
part of f itself (the regression of b̂ on b reaches slope 1.40 under
strong coupling), so centering removes more than the first-order
identity states. Boundary condition: under assigned contexts π is
identical across persons, hence Var(m) = Cov(f, m) = 0 and centering
removes only estimation noise.

**A.4 Proposition 3 (identification needs disjointness) and the purity
criterion.** Let the venue sets A_u (early) and B_u (late) be disjoint,
venue effects independent mean-zero across venues, and π ⊥ b. Then
Cov(m^A, m^B) = 0, and

    Cov(ȳ^A_e, ȳ^B_l) = Var(f) + Cov(f, m^A) + Cov(f, m^B),

so the disjoint-set estimand isolates Var(f) up to the base–mixture
cross-terms: exact when Cov(f, m) = 0, upper-bound-flavored under
positive coupling. Three channels survive coupling and are handled
explicitly: (i) the two cross-terms above; (ii) venue effects
correlated within a content class, which re-open mediation across
"disjoint" venues and motivate the category-disjoint variant; and
(iii) under strong coupling a residual mixture–mixture term, because
the weights themselves depend on b. The covariance-accounting
estimator of the total mediated share is

    mediated_total = [Var(m) + 2 Cov(f, m)] / Cov_raw
                   = 1 − Cov(f̂_e, f̂_l) / Cov_raw,        f̂ := ȳ − m̂,

with m̂ cross-fitted (venue effects estimated on other persons; a
self-estimation leak in an early version was caught and corrected, and
the cross-fitting requirement is now standing policy). In generative
worlds that violate the assumptions deliberately, mediated_total is
upward-biased under strong coupling (.541 against a target of .426 in
the recorded failing world), so on real data it is read as an upper
bound on mediation; equivalently, 1 − mediated_total is a lower bound
on the base share, the category-disjoint stability supplies the
matching upper-bound-flavored arm, and the two bracket the truth. The
operational purity criterion of the main text — category-disjoint
stability of at least .15 with a mediated share below .30 — applies
these estimands with thresholds that are conventions, flagged as such.

**A.5 Scoring definitions (as frozen in code).** A construct's slice
score is the percentage of tokens matching its fixed public list
(tokens matched / tokens in slice × 100); a person's construct score is
the mean over their slices; a composite construct is a fixed weighted
sum of construct scores. Venues map to twelve content classes by a
frozen map built once on development users. With s_uk person u's share
of slices in class k and s̄_k the eligible-cohort mean share,

    choice axis      a_uk = log( (s_uk + ε) / (s̄_k + ε) ),   ε = 10⁻⁴,
    choice entropy   H_u  = − Σ_k s_uk log s_uk,

with one class excluded as reference, leaving eleven axes. Disjoint-
occasion stability is the Pearson correlation, across persons, of
scores from the earliest and latest portions of each person's writing,
separated by at least 90 days with no shared text; the same-category
and different-category variants constrain the venues of the two
portions as described in the main text. Sealed-test eligibility
required at least 4 venue classes and at least 12 slices per person.

**A.6 Overidentifying checks.** Three constraints hold under the model
and are monitored as misspecification alarms: (i) Spearman–Brown
consistency along the volume curve; (ii) the single-factor transport
bound Cross(e, l) ≤ √(w_e · w_l) for any register or occasion pair —
instantiated in the register study as .739 ≤ √(.782 × .857) = .819;
and (iii) agreement, within sampling error, between mediated shares
estimated by the shared-versus-disjoint contrast and by covariance
accounting. Disagreement flags class-correlated venue effects or a
failure of additivity, and one observed disagreement of exactly this
kind is what forced the interval reading of A.4 rather than a point
estimate.


**A.7 Invariance, aggregation geometry, and spectral checks.** Four
further formal elements, added late in the program. (i) Every
permutation null in this article is a choice of group G acting on
labels; a construct carries G-signal iff its statistic's distribution is
not G-invariant. The stranger-pairing null uses the derangement class of
S_N; sign-flip nulls use (Z/2)^N; the refusal to shuffle time (the
contiguous-halves rule) is the trivial-group row of the same table. An
artifact, in this language, is a statistic that is constant on a larger
orbit than the analyst believed. Stationarity itself becomes a testable
invariance: reversing each person's occasion sequence maps drift slopes
to their negatives, so per-person sign flips give an exact null for mean
drift (the time-reversal instrument of the exploratory section). (ii)
With φ(r) = r/(1−r), Spearman–Brown lengthening is exactly φ(r_W) =
W·φ(ρ) (proof: φ(r_W) = Wρ/(1+(W−1)ρ−Wρ) = Wρ/(1−ρ)); reliability
growth is linear flow in a projective coordinate. Under slice-clustered
noise the curve bends to φ(W) = aW/(1+bW), concave with asymptote a/b,
and the fitted asymptote must not fall below the realized large-W value
— a consistency falsifier that the apostrophe construct fails, exposing
slice-level zero-inflation. Because stable venue choice acts as signal
in a retest, φ-curvature reads noise geometry only: it cannot detect the
choice channel, and it complements rather than replaces the purity
decomposition. One consequence is adopted as policy: composite
constructs should be blended in φ, where independent-noise composition
is additive, not in r. (iii) Factor claims about the battery are tested
against the Marchenko–Pastur edge (1 + √(p/n))², with an empirical
within-column-shuffle edge and a cross-half eigenvector-replication
requirement. (iv) Whether the frame itself is fixed is one more
invariance question: for a condition variable t, factor fixity asserts
that the top-k eigenframe — a point on the Grassmann manifold Gr(k, p)
— is constant in t up to sampling noise, testable by subspace distance
against within-person condition-label permutations; a frame that
replicates within persons while failing to be shared across them stops
being a nameable population object and becomes a personal quantity.
Predictions for the three fluidity instruments were registered before
the runs, and the registered kill condition (residualized
self-congruence at or below stranger congruence) was not triggered.
