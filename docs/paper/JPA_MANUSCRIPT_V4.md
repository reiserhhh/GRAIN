# [BLINDED MANUSCRIPT — Journal of Personality Assessment — v4]

<!-- v4 (2026-07-07): reframed per author decision — theory argument +
     instrument development + sealed validation with benchmark context.
     New: "Developing the Instrument" part; "A Sealed Confirmatory Test";
     "Measurement Versus Prediction on the Same Corpus" (EXPL-1 fit
     gradient, Table 1; official baselines reproduced to delta<=.0014);
     full reference list (entries re-verified at round-12 audit).
     v3 register and v3.1 fixes carried forward. Blinding unchanged. -->

# Text as Behavior in Chosen Situations: Theory, Development, and a Preregistered First Test of Group-Referenced Analysis of Individual Natural-Language Expression (GRAIN)

## Abstract

People reveal themselves in what they write, and a sizable literature now
uses everyday language to assess personality. Much of that work treats
what a text is about, and where it was written, as noise: features are
residualized on topics, corpora are stratified by venue, and measurement
is often confined to closed style dictionaries. We argue that in
self-selected writing this habit removes signal rather than noise,
because writers choose their contexts and the choices are themselves
consistent, personality-expressive behavior. We develop this observation
into a framework in which accumulated text is behavior distributed over
chosen situations, reaching measurement through three channels: context
choice, a context-invariant style base, and person-by-context
signatures. The framework's commitments were built and tested on
PANDORA, a corpus of 17.6 million Reddit comments shared with us by its
authors (Gjurković et al., 2021). As predicted, removing context means
lowered person-level stability for every style measure examined (pooled
Δ = −.09); a purity criterion separating person-borne from venue-borne
stability admitted four of nineteen lexical constructs as candidate
traits, all of them form habits. A sealed, one-shot preregistered test
then confirmed two of seven directional hypotheses — first-person rate
predicted Neuroticism (r = .111) and political-venue choice predicted
Openness (r = .096) — while the omnibus criterion failed; the survivors
are the framework's two channel-level predictions. An exploratory fit
gradient on the same users locates these unfitted scores against
supervised models (mean r = .27). We discuss what a measure, unlike a
predictor, is entitled to claim.

**Keywords:** personality assessment; verbal behavior; situation
selection; behavioral signatures; idiolect; language analysis

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
field that has long wanted behavioral alternatives to the questionnaire,
these are encouraging results, and they have encouraged applications
ranging from large-scale epidemiology to clinical screening.

Less attention has been paid to a methodological habit that runs through
much of this work. Because personality is understood as something that
should hold across situations, and because text varies enormously with
what it happens to be about, researchers routinely treat topical and
situational variance as a nuisance. The moves are familiar: residualize
features on topic models, stratify corpora by forum or genre before
analysis, or restrict measurement to closed style dictionaries precisely
because content words seem contaminated by circumstance. The intuition is
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

This article therefore makes three connected contributions. The first
is theoretical: we assemble the case that context choice is personal,
drawing on situation-selection research, the behavioral-signature
literature, Brunswik's representative design, stylometry, and work on
within-person distributions, and we state a framework with formal
consequences, including an account of when the standard practices
(centering, reliability interpretation) mislead and when they are safe.
The second is an instrument: a scoring procedure that reads a person's
accumulated text as behavior in chosen situations, built from fixed
public word lists and a venue-choice profile, with every construct
forced through a purity check before it may be called a trait. The
third is evidence under custody: a preregistered, sealed test of seven
directional hypotheses on quarantined labels, which the framework
passed only in part, followed by benchmark analyses that locate the
instrument against prediction models fitted to the same corpus.
Throughout, we try to claim no more than the evidence licenses; the
framework forbids several common practices, and its own record under
those prohibitions is part of the argument.

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

## Developing the Instrument

The instrument was developed and stress-tested on public corpora, with
every analysis, correction, and retraction recorded in a public claims
ledger; we summarize here and keep procedural detail in the Supplement.
The main corpus is PANDORA (Gjurković et al., 2021), which we obtained
directly from its authors; development analyses used 3,183 of its users
with sufficient text in non-overlapping time windows. The
assigned-context contrast is the stream-of-consciousness essay corpus of
Pennebaker and King (1999), in which students wrote continuously on a
fixed instruction, so that topic and venue were set by the researchers
rather than chosen by the writers. Both corpora have long publication
histories, and we use them under their original names and terms; the
only unpublished data we collected ourselves (a small market-forum
corpus) plays no part in the analyses reported here. Confirmation
labels were quarantined from the start behind a preregistration sealed
by the initial commit hash of a public repository, and were read
exactly once, in the sealed test reported below.

### From Text to Scores: The Procedure in Brief

Because the argument so far has been abstract, it may help to walk
through what is actually computed; exact parameters and code are in the
Supplement. Scoring begins by concatenating a person's comments within
each venue and cutting the result into consecutive 128-token windows,
where a token is, roughly, a word. Windows shorter than 24 tokens are
dropped, and so is any window in which the writer explicitly states a
personality label ("as an INTJ..."), since a measure has no business
reading the answer off the page. Each surviving slice is scored against
fixed word lists. The lists are short, public, and version-pinned. A
first-person score, for instance, is simply the percentage of tokens
that are first-person forms: in the sentence "I think my plan might
work, but you should try it first," two of twelve tokens (I, my) are
first-person, so the slice scores 16.7 per 100 tokens. A person's score
on a construct is the average over their slices; composite constructs
are fixed weighted sums of such rates. Nothing is fitted per person,
and revising any list requires demonstrating, on frozen data, exactly
what the revision changes.

The choice profile is computed from where the slices come from rather
than from what they say. Each venue belongs to one of twelve content
categories under a fixed venue-to-category map built once on development
users (politics and news, gaming, sports, and so on), and a person's
profile is their distribution of comments over the categories, expressed
relative to the population. Someone who places 40% of their comments in
political venues, against a population average near 10%, has a high
politics-choice score regardless of anything stylistic in their writing.

Stability and purity are computed by splitting each person's writing in
time. We take the earliest and the latest portions, separated by at
least 90 days, score each portion separately, and correlate the two
scores across people; that is disjoint-occasion stability. For purity,
the same split is computed twice more: once using only venues from the
same categories in both portions, and once using venues from entirely
different categories. A construct whose stability survives the
different-category split is being carried by the person; one whose
stability collapses there was being carried by the venues.

### Two Development Results That Shaped the Battery

Two development results, both obtained while the confirmation labels
remained sealed, determined the battery's final form.

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

Applying the purity criterion of
Proposition 3 to our full inventory of nineteen lexical constructs sorted
them cleanly. Four passed as candidate style traits, and all four are
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

## A Sealed Confirmatory Test

The strongest evidence, in both directions, comes
from the preregistered opening. Seven directional hypotheses over
quarantined Big Five labels (n = 1,058 eligible users) and one secondary
bridge hypothesis were fixed in public, with eligibility rules,
Benjamini-Hochberg correction, a priori power calculations, and a
success rule of at least four of seven confirmed, before any label was
read; the analysis script itself was hash-pinned after an adversarial
audit, and ran once. The omnibus rule failed: two of seven hypotheses
survived correction. The two survivors are one from each of the framework's primary
channels, and the style-side survivor is the one construct in the
battery that the purity criterion certifies as person-borne:
first-person rate predicted
Neuroticism (r = .111, 95% CI [.051, .170], q = .002), and choice of
political venues predicted Openness (r = .096, [.036, .155], q = .006).
No significant effect ran opposite to its preregistered direction. The five failures divide into two kinds, and only one of them is
explained by the framework. Three involve style constructs — an
affect–tension composite, a novelty measure, and a directive-language
measure — that the purity criterion, derived after the fact,
classifies as venue-borne or unclassifiable; had the criterion existed
at preregistration time, it would have barred all three from
confirmatory testing. The other two were choice-channel hypotheses
that simply failed: a venue-breadth conjecture, and a
gaming–extraversion prediction that the preregistration itself had
flagged as its weakest entry. For these two we have no after-the-fact
account, and we decline to construct one. We report the
failure as a failure. What the opening licenses is deliberately narrow:
the two channel-level directional claims, at small effect sizes, in one
population. What it illustrates more broadly is that the framework's
admission rules and its confirmatory record point the same way.

## Measurement Versus Prediction on the Same Corpus

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
(Table 1). A ridge model over tf-idf unigrams, the model family of the
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

**Table 1.** *An exploratory fit gradient on the sealed-test cohort (n =
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

## Transport to Assigned Contexts

The framework expects direction to travel better than
level. A battery fitted entirely on Reddit and applied without
refitting to the student essays retained directional validity (mean
cross-validated correlation with Big Five scores of about .14,
comparable to a transported text-embedding baseline of about .15), while
score levels and covariance structure did not transport; within Reddit,
all nineteen constructs survived a register shift from top-level posts
to replies once volume was matched, with level shifts confined to two
venue-linked constructs.

Table 2 summarizes each commitment of the framework, the test it has
faced, and its current status, with sources for every number.

## Discussion

The framework began as an attempt to explain one uncomfortable result,
and its value should be judged by how much else it organizes. Several
standing features of the literature follow naturally once text is read
as behavior in chosen situations. The predictive advantage of
open-vocabulary methods over closed dictionaries (Schwartz et al., 2013;
Kern et al., 2014) is expected, because topical features carry the
choice channel that style dictionaries exclude by construction; on this
reading the advantage reflects channel coverage more than lexical resolution; our own fit gradient reproduces the pattern within a single corpus, where opening the vocabulary roughly triples mean correlation over the committed feature set (Table 1). The decades-long robustness of function words as individual
markers, from authorship attribution to Pennebaker's program, is equally
expected, because form habits are where the context-invariant base
lives. The common observation that language models and dictionaries
alike transfer poorly across platforms in level, while rank-order
relations survive in attenuated form, matches the framework's boundary
structure: levels are regime-bound, direction travels. And the repeated
disappointment of affect word-rates as trait measures has a simple
description here: they are heavily context-loaded and behave as states
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
statistical adjustment alone.

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

Personality assessment has long treated the standardized test and the
spontaneous sample as belonging to different traditions, one scorable
and one rich. A person's accumulated writing suggests the distinction is
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
publishers. The fit-gradient analyses of Table 1 reuse the confirmation labels
after their one-shot opening; they are labeled exploratory in the
ledger and carry no confirmatory weight.

## Declarations

<!-- Funding: [tbd]. COI: none. Ethics: secondary analysis of public
     pseudonymous corpora; no new data collection. Consent: n/a. -->

Data and materials. The PANDORA corpus was created by Gjurković et al.
(2021) and was provided to us by its authors; the essay corpus was
created by Pennebaker and King (1999). Both are used under their original
names and access terms, and neither is redistributed here.
[Acknowledgment of the PANDORA team moves to the unblinded title page.]

## References

<!-- Entries compiled v4; bibliographic details re-verified at the
     round-12 pre-submission audit. -->

Allport, G. W. (1937). *Personality: A psychological interpretation*. Henry Holt.

Ambady, N., & Rosenthal, R. (1992). Thin slices of expressive behavior as predictors of interpersonal consequences: A meta-analysis. *Psychological Bulletin, 111*(2), 256–274.

Brunswik, E. (1956). *Perception and the representative design of psychological experiments* (2nd ed.). University of California Press.

Buss, D. M. (1987). Selection, evocation, and manipulation. *Journal of Personality and Social Psychology, 53*(6), 1214–1221.

Dąbrowska, E. (2018). Experience, aptitude and individual differences in native language ultimate attainment. *Cognition, 178*, 222–235.

Dhami, M. K., Hertwig, R., & Hoffrage, U. (2004). The role of representative design in an ecological approach to cognition. *Psychological Bulletin, 130*(6), 959–988.

Eichstaedt, J. C., Kern, M. L., Yaden, D. B., Schwartz, H. A., Giorgi, S., Park, G., Hagan, C. A., Tobolsky, V. A., Smith, L. K., Buffone, A., Iwry, J., Seligman, M. E. P., & Ungar, L. H. (2021). Closed- and open-vocabulary approaches to text analysis: A review, quantitative comparison, and recommendations. *Psychological Methods, 26*(4), 398–427.

Fisher, A. J., Medaglia, J. D., & Jeronimus, B. F. (2018). Lack of group-to-individual generalizability is a threat to human subjects research. *Proceedings of the National Academy of Sciences, 115*(27), E6106–E6115.

Fleeson, W. (2001). Toward a structure- and process-integrated view of personality: Traits as density distributions of states. *Journal of Personality and Social Psychology, 80*(6), 1011–1027.

Furr, R. M. (2008). A framework for profile similarity: Integrating similarity, normativeness, and distinctiveness. *Journal of Personality, 76*(5), 1267–1316.

Gjurković, M., Karan, M., Vukojević, I., Bošnjak, M., & Šnajder, J. (2021). PANDORA talks: Personality and demographics on Reddit. In *Proceedings of the Ninth International Workshop on Natural Language Processing for Social Media* (pp. 138–152). Association for Computational Linguistics.

Kern, M. L., Eichstaedt, J. C., Schwartz, H. A., Dziurzynski, L., Ungar, L. H., Stillwell, D. J., Kosinski, M., Ramones, S. M., & Seligman, M. E. P. (2014). The online social self: An open vocabulary approach to personality. *Assessment, 21*(2), 158–169.

Mehl, M. R., Gosling, S. D., & Pennebaker, J. W. (2006). Personality in its natural habitat: Manifestations and implicit folk theories of personality in daily life. *Journal of Personality and Social Psychology, 90*(5), 862–877.

Mischel, W., & Shoda, Y. (1995). A cognitive-affective system theory of personality: Reconceptualizing situations, dispositions, dynamics, and invariance in personality structure. *Psychological Review, 102*(2), 246–268.

Molenaar, P. C. M. (2004). A manifesto on psychology as idiographic science: Bringing the person back into scientific psychology, this time forever. *Measurement: Interdisciplinary Research and Perspectives, 2*(4), 201–218.

Mosteller, F., & Wallace, D. L. (1963). Inference in an authorship problem. *Journal of the American Statistical Association, 58*(302), 275–309.

Park, G., Schwartz, H. A., Eichstaedt, J. C., Kern, M. L., Kosinski, M., Stillwell, D. J., Ungar, L. H., & Seligman, M. E. P. (2015). Automatic personality assessment through social media language. *Journal of Personality and Social Psychology, 108*(6), 934–952.

Pennebaker, J. W., & King, L. A. (1999). Linguistic styles: Language use as an individual difference. *Journal of Personality and Social Psychology, 77*(6), 1296–1312.

Rauthmann, J. F., & Sherman, R. A. (2023). Patterned person–situation fit in daily life: Examining magnitudes, stabilities, and correlates of trait–situation and state–situation fit. *European Journal of Personality*. Advance online publication. https://doi.org/10.1177/08902070221104636

Rauthmann, J. F., Sherman, R. A., & Funder, D. C. (2015). Principles of situation research: Towards a better understanding of psychological situations. *European Journal of Personality, 29*(3), 363–381.

Schwartz, H. A., Eichstaedt, J. C., Kern, M. L., Dziurzynski, L., Ramones, S. M., Agrawal, M., Shah, A., Kosinski, M., Stillwell, D., Seligman, M. E. P., & Ungar, L. H. (2013). Personality, gender, and age in the language of social media: The open-vocabulary approach. *PLoS ONE, 8*(9), Article e73791.

Shoda, Y., Mischel, W., & Wright, J. C. (1994). Intraindividual stability in the organization and patterning of behavior: Incorporating psychological situations into the idiographic analysis of personality. *Journal of Personality and Social Psychology, 67*(4), 674–687.

van Halteren, H., Baayen, H., Tweedie, F., Haverkort, M., & Neijt, A. (2005). New machine learning methods demonstrate the existence of a human stylome. *Journal of Quantitative Linguistics, 12*(1), 65–77.

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
