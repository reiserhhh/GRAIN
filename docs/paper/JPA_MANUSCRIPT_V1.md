# [BLINDED MANUSCRIPT — Journal of Personality Assessment]

<!-- Double-blind: no author names, no institution, no identifiable repo URL.
     The public repository is cited as [ANONYMIZED REPOSITORY LINK — view-only
     copy provided for review]; the real URL + seal hash replace it on
     acceptance. Separate title page holds author/affiliation/waseda.jp email.
     Every number in this manuscript must trace to a CLAIMS_LEDGER row; the
     pre-submission round-12 audit re-verifies the full sweep. -->

# Group-Referenced Analysis of Individual Natural-Language Expression (GRAIN): A falsification-first method for personality assessment from spontaneous language

## Abstract

Personality assessment has long wanted one instrument to combine the
standardization of self-report with the spontaneous behavioral richness of
performance-based measures. We present GRAIN, an analysis method that treats a
person's everyday writing as a performance-based test with frozen,
mechanical scoring: spontaneous text is sliced into many small utterances
inside the contexts the person chose, and read through three channels —
context choice, a context-invariant style base, and person-by-context
signatures. The method rests on a falsification: the field's default of
statistically removing topic/venue variance is shown, empirically and then
algebraically, to DELETE person signal whenever contexts are self-selected
(the removed retest covariance equals the choice-mediated share); context
must be handled by design, not subtraction. Validity governance is built
in: version-pinned scorers, data tiers with a hash-sealed one-shot
preregistration, estimators licensed against generative worlds that violate
the model's own assumptions, graded comparison licenses, and an adversarial
builder/auditor protocol that caught and corrected 13 builder errors across
11 audit rounds. In a complete worked construction on public Reddit and
essay corpora (N = 3,183 development users), trait-grade signal
concentrated in form habits (pronouns, orthography, negation) while content
vocabulary rode on context choice; the sealed confirmatory opening verified
two channel predictions (first-person rate to Neuroticism, r = .111;
political-venue choice to Openness, r = .096) while the preregistered
omnibus rule failed (2/7) — reported in full. Code, manifests, and the
sealed preregistration are public.

**Keywords:** personality assessment; natural language; psychometrics;
preregistration; open-vocabulary analysis; idiolect

---

## 1. Introduction

<!-- ~1,600 words. Hook: the topic-as-nuisance default and its quiet failure. -->

A decade of language-based assessment has established that personality
leaves usable traces in everyday writing (Pennebaker & King, 1999; Schwartz
et al., 2013; Park et al., 2015; Eichstaedt et al., 2021; Boyd & Schwartz,
2021). The field's best predictive models, however, purchase their accuracy
with two debts. The first is interpretability: open-vocabulary and
embedding models yield thousands of opaque features whose scores cannot be
audited item-by-item, revised without refitting, or explained to the person
measured. The second debt is quieter and, we will argue, conceptual: the
near-universal treatment of topic and venue as nuisance variance. Whether by
topic modeling, corpus stratification, or residualizing on content, the
default pipeline seeks the person "controlling for" what they chose to
write about.

This paper begins from a falsification. In a corpus of 17.6 million Reddit
comments, we attempted the nuisance treatment in its strongest forms —
naive condition-mean centering, shrunken partial centering, and two-way
fixed-effects centering estimated out-of-sample with cross-fitting — and
every variant REDUCED the disjoint-occasion stability of every style
measure (pooled Δ = −.091, 95% CI [−.102, −.080]; cross-fit fixed-effects
variants −.05 to −.07). The reason is not statistical misfortune but
structure: when contexts are self-selected, context is a mediator of
person→text effects. Conditioning on a mediator removes the mediated
signal; we derive the identity (the removed retest covariance equals
Var(m) + 2Cov(f, m), where m is the person's context-mix term and f the
context-invariant base), its boundary condition (centering is harmless
exactly when contexts are exogenously assigned, as in prompted essays), and
its second-order aggravation under strong person-context coupling. What
the field has treated as cleaning is, under self-selection, deletion.

The constructive consequence is a reframing we call the rind model, after
the watermelon: topic and venue are not packaging to discard but the rind
that carries person signal, because people pick their rinds. A text-based
instrument should therefore read three channels: (C1) context CHOICE
itself — which venues, which topics, in what proportions; (C2) a style
BASE — habits that survive crossing from rind to rind; and (C3)
person-by-context SIGNATURES — the if-then patterning of behavior across
situation classes, the textual descendant of behavioral signatures
(Mischel & Shoda, 1995). Context variance is controlled by DESIGN
(fixing or sampling rinds) and never by statistical subtraction.

This ambition has an assessment-tradition genealogy that predates the
computational turn. Self-report inventories bought standardization at the
price of item transparency, response styles, and the limits of
introspective access; the performance-based tradition (from projective
techniques onward) reached for spontaneous behavior instead, and paid in
scoring subjectivity and contested reliability (Meyer et al., 2001; Mihura
et al., 2013). A person's accumulated everyday writing is, we argue, the
performance sample the tradition was missing: abundant, ecologically
produced, impossible to rehearse item-by-item — and, unlike a projective
protocol, scorable by FROZEN mechanical rules whose every count can be
audited. GRAIN's founding ambition is exactly this synthesis: the
standardization of the questionnaire applied to the spontaneous behavior of
the performance tradition. Frameworks are cheap, however; what we
principally offer the assessment community is the validation machinery that
made this one falsifiable at every joint:

1. **Frozen, interpretable scoring.** All scores are counts from fixed
   lexica and fixed context maps, version-pinned with equivalence gates
   (a scorer revision that changes any frozen score by any amount restarts
   the validation chain — measured, not asserted).
2. **Wrong-world licensing of estimators.** Every estimand used for a claim
   must first demonstrate, in generative worlds that VIOLATE the model's
   assumptions one at a time, that it distinguishes the theory being true
   from being false. Two licensing failures were caught this way and are
   reported as recorded FAILs with their mechanism.
3. **Tiered data governance with a sealed lockbox.** Development labels are
   quarantined from confirmation labels; the confirmatory analysis was
   preregistered, hash-sealed in a public repository, and executed exactly
   once by a script whose own hash was pinned by an adversarial pre-audit.
   We report the opening verbatim: the omnibus success rule FAILED (2 of 7
   hypotheses at BH-FDR q < .05; the rule required 4), while the two
   confirmations are precisely the framework's two core channels.
4. **An adversarial builder/auditor protocol.** Every substantive result was
   attacked by an independent auditor with full recomputation rights before
   entering the claims ledger. Across 11 rounds the audits caught 13 real
   builder errors — self-estimation leakage, optimizer boundary collapse,
   estimand asymmetries, a construct-identity swap caused by silently
   refitting a clustering, and interpretive overstatements — each recorded
   with its correction. We propose the protocol, and its error taxonomy, as
   a reusable reproducibility method for computational measurement work.

Two further contributions follow from the framework rather than preceding
it. Channel decomposition of every construct in the inventory shows that
trait-grade, context-invariant signal concentrates in FORM habits —
pronoun use, orthographic habit (apostrophe omission), negation, profanity
— while content vocabulary owes its apparent stability to context choice;
this aligns the instrument with the forensic-stylometry finding that
individuality lives in co-selection of form (van Halteren et al., 2005;
Wright, 2017), and yields an admission gate (a "flesh-purity" criterion)
that would have prevented, in advance, the specific hypothesis that failed
most clearly at confirmation. And a residual-individuality layer
operationalizes the idiographic intuition that no factor list exhausts a
person (Allport, 1937; Molenaar, 2004): after subtracting the normative
profile, the remaining deviation pattern is stable across months (median
profile r = .505 against a stranger null of −.02; .684 on form constructs
alone) — measurable individuality beyond the factor structure, quantified
with the profile-decomposition discipline of Furr (2008).

We present the framework and its formal results (Section 2), the
validation architecture (Section 3), the worked construction with its
falsification series and confirmations (Sections 4-5), and the licensing
limits — what comparisons the instrument does and does not permit —
together with the designed next study (Section 6).

## 2. The Rind Framework and Formal Results

### 2.1 Model and channels

For author *u*, text slice *i*, and context ("rind") *r*(*i*) — the venue or
topic shell inside which the slice was produced — we write

> *y*_ui = *f*_u + *b*_r(i) + *γ*_u,r(i) + *e*_ui,  with *r*(*i*) ~ *π*_u,

where *f*_u is the author's context-invariant style base ("flesh"), *b*_r a
context style offset, *γ*_u,r a person-by-context interaction, *e*_ui slice
noise, and *π*_u the author's distribution over contexts. The single
substantive commitment is the last clause: **contexts are chosen**, and the
choice distribution *π*_u is itself a stable property of the person. Three
measurement channels follow. C1 (choice): *π*_u relative to the population —
which venues, in what proportions. C2 (style base): *f*_u — habits that
survive crossing contexts. C3 (signatures): the structure of *γ*_u,r — the
person's if–then patterning across situation classes, the textual descendant
of behavioral signatures (Mischel & Shoda, 1995). Contexts come in regimes:
free (the writer picks venue and topic; social platforms), assigned (an
experimenter fixes the prompt; classic essay corpora), and domain-locked (an
external domain fixes the macro-topic; special-interest communities). Which
channels are measurable depends on the regime: fixing the rind by design
silences C1 and purifies C2; free collection yields C1 at the cost of mixing
rinds into every raw score.

### 2.2 Six formal results

Proofs and their simulation verifications are in Supplement A; here we state
what changes practice. (F1) The raw user mean is a composite:
E[ȳ_u] = *f*_u + *m*_u, with *m*_u = Σ_r *π*_ur·*b*_r. Because *π*_u is
person-stable, the context-mix term *m*_u is person-stable too — a raw
"style" score measures flesh PLUS frozen context choice. (F2) Retesting on a
shared context set cannot separate them: shared-set retest covariance equals
Var(*f*) + Var(*m*) + 2Cov(*f*, *m*). In a generative world with *f* ≡ 0 —
no personal style whatsoever — the raw disjoint-occasion retest still reads
*r* = .64 from choice alone. Stability, by itself, is not evidence of a
style trait; this single fact reorganizes how text-based "reliability" must
be read. (F3) Splitting contexts into disjoint sets kills the mix–mix term
when context effects are independent and choice is uncoupled from them;
person–choice coupling survives as cross terms, so disjoint-set estimates
upper-bound flesh. A class-disjoint variant removes category-level mediation
as well (in its licensing world, condition-level splits over-read flesh by
+.106 when context effects correlate within class, while the class-disjoint
arm reads the planted flesh variance within tolerance). (F4) **The centering
theorem.** Statistical removal of context means — the field's default
"topic control" — is mediator adjustment. Under choice–effect independence
it removes retest covariance by exactly Var(*m*) + 2Cov(*f*, *m*) (oracle
simulation: −1.332 observed against −1.335 predicted), and under strong
person–context coupling it over-removes even that, because estimated
context effects absorb the flesh of the people who chose them (regression
of estimated on true effects has slope 1.40 at coupling κ = 0.7; observed
removal −1.583 against a first-order −1.294). The boundary condition is
exact: with exogenously assigned contexts the mix term vanishes and
centering is harmless. Removing topic variance is cleaning only in
assigned-prompt designs; under self-selection it is deletion. (F5) The
precision cost of context heterogeneity is weighted by a construct's
context loading — venue-mix entropy costs reliability in proportion to how
much of the construct's variance rides on context. (F6) The model
over-identifies: transport across occasions or registers is bounded by
√(rel₁·rel₂); independent mediation estimators must agree; their
disagreement is a built-in misspecification alarm. (The alarm fired twice
in our own data; the affected shares are reported as bands, not points.)

### 2.3 Consequences for instrument design

Three rules follow. Context is controlled by design, never by subtraction
(F4). A construct may be called a style trait only after a purity check —
substantial class-disjoint stability with a small choice-mediated share —
because F1 guarantees that raw stability conflates the two channels. And
every score carries a comparison license: within-context monitoring,
cross-theme, cross-register, cross-regime, and cross-language comparisons
are separately licensed by evidence, with cross-regime LEVEL comparison
forbidden outright (Section 6); the prohibition is enforced in the released
software, which raises an exception rather than returning a number.

## 3. Validation Architecture

The framework is only as credible as the machinery that could have killed
it. Five components — each of which caught at least one real error during
development — constitute what we propose as a reusable governance standard
for computational assessment.

**Frozen scoring with equivalence gates.** All scores are counts from
version-pinned lexica and a fixed context map; no model weights, no
fine-tuning. A scorer revision must prove equivalence or restart the
validation chain: the v3.1 revision (apostrophe normalization) changed no
headline value by more than .002 and was adopted as the ingestion default;
the v4 revision (disjointifying the 20 words shared across lexica, each
reassignment logged with its rationale) left every frozen construct score
bit-identical while removing manufactured between-category covariance from
the wider feature space.

**Tiered data with a sealed lockbox.** Development users (*N* = 8,678) are
permanently separated from confirmation labels. Development-tier anchor
labels are used for orientation only, and every contact is disclosed in the
preregistration itself. The confirmation tier — 1,401 Big-Five-labeled
users, 375 dually labeled users, and the reserved half of an essay corpus —
was sealed behind a preregistration whose seal is the initial commit hash
of a public repository: hypotheses, directions, eligibility rules, the
multiplicity correction, and the success rule were fixed in public before
any confirmatory label was read, with a budget of two openings for the
project's lifetime. Opening #1 (Section 5) was executed once, by a script
whose own hash was pinned by an adversarial pre-audit, and is reported in
full regardless of outcome.

**Evidence tiers with ledger-first reporting.** Every claim lives in a
public claims ledger at one of four tiers — exploratory; same-data
pre-committed; held-out; lockbox-confirmatory — and prose may not use
stronger language than the ledger row supports. The ledger records
retractions and corrections in place. Every number in this manuscript
traces to a ledger row and its source artifact.

**Wrong-world licensing of estimators.** Before an estimator's output may
support a claim, the estimator must demonstrate — in generative worlds that
violate the model's assumptions one at a time — that it can tell the
framework being true from being false. Nine worlds were run (choice-null,
flesh-null, recovery grids, class-correlated contexts, coupled choice, a
centering phase diagram, timescale aliasing, signature-null, leakage-only).
Seven passed. Two produced recorded FAILs sharing a single mechanism — the
composition bias of estimated context effects under strong person–context
coupling — which converted the affected estimators into declared bounds
rather than point estimators. The suite also caught a defect in itself (a
tie-unsafe test statistic that fabricated signal in a heavily tied null
world), yielding the standing rule that harnesses must replicate frozen
estimator conventions exactly.

**An adversarial builder/auditor protocol.** Every substantive result was
attacked by an independent auditor with full recomputation rights before
entering the ledger, under criteria pre-committed before execution. Across
eleven rounds the audits caught thirteen real builder errors: nine
mechanical — among them a self-estimation leak in a centering "rescue," an
optimizer boundary collapse that included one falsely converged fit, an
estimand asymmetry that had reversed a verdict, and a construct-identity
swap caused by silently refitting a clustering instead of transporting it —
and four interpretive over-claims, among them reading a disattenuation
blow-up as evidence and over-labeling topical axes as form. Each error, its
catch, and the standing rule extracted from it are part of the public
record. We regard this audit trail not as a confession appendix but as the
method's central reliability evidence: an assessment framework built by a
fallible process that demonstrably catches its own failures.

All development and auditing were AI-assisted under this protocol, with the
human investigator fixing criteria, adjudicating scope, and holding final
responsibility; builder and auditor roles were always executed by
independent instances. We disclose this both as method transparency and
because the governance pattern — freeze, seal, license, audit — is
precisely what makes AI-assisted measurement research checkable.

## 4. Worked construction I: falsification and channels (development tier)

<!-- TO WRITE as Studies:
  4.1 Data and tiers (PANDORA 17.6M comments; N=3,183 half-eligible; Essays
      dev half 1,255; funnels table). Ethics: secondary public data.
  4.2 Study 1 — centering falsified 3 ways + F4 empirics (P2 -.091; E1
      cross-fit; phase diagram; Essays boundary).
  4.3 Study 2 — choice channel: axes held out (5/5, r_B .48-.68, shrinkage
      .027); purity decomposition (flesh/rind shares per construct, with
      bands per the estimator-agreement alarm); form/content division;
      register transport 19/19 (opposite-half, volume-matched).
  4.4 Study 3 — inventory growth under gates: open-vocabulary A/B (15/15),
      co-selection axes (21/48, ~15 form), blind coding with legibility
      classes (8/15; machine-only construct human 0/8), flesh-purity gate
      families F4/C8/comp6/und1.
-->

## 5. Worked construction II: anchors and the sealed confirmation

<!-- TO WRITE:
  5.1 Development-tier anchor performance (TF CV r=.346 from 31 scores; JP
      .182; EI .139; SN .120; Essays transport mean .144; expectation frame
      vs full-text baselines .25-.32; parse/base-rate disclosures).
  5.2 The lockbox opening (preregistered, one-shot): full 8-row table
      (H1-H8), BH-FDR, SUCCESS RULE FAILED 2/7 — reported verbatim; H2
      first_person->N r=.111 [.051,.170] q=.002; H6 politics->O r=.096
      [.036,.155] q=.006; power table a priori; eligibility funnel
      1,412->1,058/303; ΔR² add-on uninformative (frozen convention
      overfits, disclosed); no re-analysis; remaining budget.
  5.3 Residual individuality: seed-pattern stability .505/.684 vs stranger
      null; identification AUC .887/.891; the 32% unfactored share (M3).
-->

## 6. General discussion

<!-- TO WRITE: what is licensed (L0-L4 table as the honest deliverable);
     what failed and why that is the method working; limitations (one
     platform + students, English, heavy writers, no demographics);
     the native-corpus design (2 regimes x 2 sessions x 2 languages) as the
     specified next study; AI-assisted research disclosure. -->

## Open Practices Statement

<!-- All code, frozen lexica, class maps, manifests (SHA-256), the sealed
     preregistration (tag + hash), all audit rounds, and every reported
     number's source artifact are available at [ANONYMIZED REPOSITORY LINK].
     No raw user text or user identifiers are redistributed; dataset access
     follows the original publishers' terms (PANDORA; Essays). The
     confirmatory analysis was preregistered and executed once; the
     preregistration seal is the repository's initial commit hash. -->

## Declarations

<!-- Funding: [tbd]. COI: none. Ethics: secondary analysis of public
     pseudonymous datasets; no new data collection; exemption statement.
     Consent: not applicable (secondary public data). Data availability:
     per original dataset publishers; manifests provided. Code availability:
     public repository (anonymized for review). AI use: manuscript and
     analyses developed with AI assistance under the builder/auditor
     protocol documented in Section 3; final responsibility rests with the
     author. -->

## References

<!-- APA 7. Core: Pennebaker & King 1999; Schwartz et al. 2013; Park et al.
     2015; Eichstaedt et al. 2021; Boyd & Schwartz 2021; Mischel & Shoda
     1995; Fleeson 2001; Molenaar 2004; Fisher et al. 2018; Furr 2008;
     van Halteren et al. 2005; Wright 2017; Dąbrowska 2018; Biber 1988;
     Mosteller & Wallace 1963; Gjurković et al. 2021 (PANDORA); Yarkoni
     2010; Kern et al.; Benjamini & Hochberg 1995. -->
