# GRAIN: Joint Content and Ownership in Evidence Observation

2026-10-09. HM68 finite mechanism update, not an accuracy optimization.
Parent goal active; original18 protocols/100 cards/44 questions remain open.

## What Changed

Cross-target composition depends on both co-occurring content and its assigned
person. The effect is present before answer decoding, not only in the final
score. However, content assigned to another person still changes the queried
person's observation substantially. A pure same-person redundancy account is
therefore inadequate for this reader/material battery.

This is a mixed observation law, not a claim that the reader binds every fact
correctly, or that human personality traits interact in the same way. Keep the
common reader Tq, author evidence distribution Pu and human response Bu distinct.

## Exact Finite Law

Let x/y denote two behavioral alternatives and g their companion ownership:
g=+1 assigns the companion to A, g=-1 to B. The focal event belongs to A.
The observed value F may be a six-query score vector, centered candidate logits,
or the question-free block16 body endpoint. Each stage retains its own units.

The exact expansion on this declared eight-corner support is

    F = c0 + cx*x + cy*y + cg*g + cxy*x*y
        + cxg*x*g + cyg*y*g + cxyg*x*y*g.

For Ks = F(+,+,s)-F(+,-,s)-F(-,+,s)+F(-,-,s),

    Kown   = 4*(cxy+cxyg),
    Kother = 4*(cxy-cxyg),
    D      = Kown-Kother = 8*cxyg.

The even/odd group projections for g -> -g separate content interaction
invariant under this ownership exchange from ownership-dependent interaction.
They do not label the components psychologically or certify correct binding.
Interactions can change with the query and lexical family; no universal
eight-coefficient formula for arbitrary language is claimed.

## Actual Observations

Complete six-target design: all30 ordered distinct focal/companion pairs in
two lexical families, plus6 neutral companions per family. All eight corners
and all six queries were read. There are576 new public authored bodies and
3456 actual passive local forwards, not576 human participants.

| Componentwise RMS over30 pairs | Family0 | Family1 |
| --- | ---: | ---: |
| Native score D | .091115 | .083045 |
| Native score neutral D0 | .003325 | .004813 |
| Native score D-D0 | .092414 | .082801 |
| Centered-logit D | .361893 | .339444 |
| Centered-logit neutral D0 | .016253 | .018537 |
| Body-endpoint D | .036732 | .035805 |
| Body-endpoint neutral D0 | .004859 | .005267 |
| Score D after deleting logit xy/xyg | .052519 | .042313 |

Neutral behavior alternatives keep actor counts, event count and format. They
are nominally irrelevant, not guaranteed psychologically meaningless. D-D0 is
descriptive: behavioral polarity coordinates do not share a physical unit.

The answer-link control removes only logit cxy/cxyg, keeps all six other terms,
and uses the unchanged decoder. It shows that role-specific single-evidence
slopes followed by nonlinear scoring also produce ownership interaction.
Actual nonzero centered-logit D rules out an exclusively answer-link explanation.
Do not subtract RMS values or their squares as causal mediation percentages.

Generic composition remains large. Native-score Kown/Kother RMS are
.101918/.090310 in family0 and .073157/.078372 in family1. The historical
agreeableness-to-openness illustration retains similar suppression when the
openness companion belongs to B. Hence no pure qualified-only explanation.

Companion-query D averaged over all five other focal alternatives:

| Query | Family0 | Family1 |
| --- | ---: | ---: |
| Extraversion | +.090660 | +.056266 |
| Neuroticism | -.030555 | -.024521 |
| Agreeableness | +.147124 | +.184892 |
| Conscientiousness | +.068982 | +.037148 |
| Openness | +.014557 | +.040643 |
| Anthropomorphism | -.011303 | -.023479 |

The effect is not universally positive or a new personality factor. Both
families have18/30 positive companion-query D values. Result-after full-table
family cosine is .676986 for native score, .556965 for log-odds, .591846 for
centered logits and .315861 for body endpoints. These describe finite action
patterns, not human validity, transport accuracy, or independent populations.

## Author-Level Consequence: Joint Relationships Matter

An author observation profile is a distribution over qualified event/context
records, not simply an independent average of isolated behavior scores:

    Theta(u,q) = integral Tq(E,R,C) dPu(E,R,C).

Here E is evidence content, R ownership/event relations, and C context/history.
This is an observation functional, not the author's true psychological trait.
Its finite expansion is

    Theta(u,q) = sum_S c(q,S) * m(u,S),
    m(u,S) = E_Pu[product of coordinates indexed by S].

Thus a cxyg component requires the joint x*y*g moment. Univariate means and
pairwise correlations of these three coordinates cannot recover it in general.
This concerns the declared evidence coordinates, not all possible embeddings:
a richer full-vector mean may encode such interactions upstream (HM62).

Result-after analytic witness, using every actual stage table:

    Pplus(x,y,g)  = (1 + delta*x*y*g)/8,
    Pminus(x,y,g) = (1 - delta*x*y*g)/8, delta=.5.

Both distributions are positive and normalized. All order0/1/2 character
moments agree; all univariate and pairwise marginals therefore agree. Yet

    E_Pplus[F]-E_Pminus[F] = 2*delta*cxyg = delta*D/4 = D/8.

This was numerically checked for all432 group/stage tables. Native-score profile
difference RMS is .011389/.010381, centered-logit .045237/.042430. These are
reweightings of captured designed accounts, not two real author populations.
No actual author distribution or new personality score was fitted. The witness
locates the information lost by a particular lower-order aggregation, rather
than declaring that every mean embedding or factor model fails.

## How This Updates the Measurement Interface

1. Preserve qualified events and their relations before author aggregation.
2. Keep generic context composition and ownership-dependent composition apart.
3. Retain both the common evidence skeleton and compatible contextual directions.
4. Integrate their joint distribution, not independent event scores by default.
5. Keep the decoder separate from pre-answer observation; no accuracy rescue.

The next mechanism question is which joint content/ownership actions can be
expressed as a portable evidence law for author scoring, while preserving the
generic context channel. Do not replace that question with fitting Tq more
closely, fixing a prompt, adding Fourier order, or naming a reader effect Bu.
No new experiment is launched by this document; a bounded discriminating plan
is still required. Psychological connections and standalone unseen-text scoring
remain separately open.

## Ancestry and Verification

[de Wolf2008](https://homepages.cwi.nl/~rdewolf/publ/other/fourier.pdf), sections
2.1--2.2 (PDF pp1--3), provides the established character basis, orthogonality,
inversion and Parseval. The parity witness is a standard consequence of that
basis. [Shapley-Taylor](https://arxiv.org/abs/1902.05622) was consulted at abstract
level; no claim to implementing its attribution index or discovering interaction
analysis. OpenAI/math scalar/matrix transport distinctions were design inspiration,
not an applied PDE/manifold/uniqueness theorem or a proof that accuracy tuning
creates a psychological breakthrough.

Canonical: results/grain_math_response_v1/ownership_composition_r1.
Plan/code/material bound before acquisition; actual observer matched HM61.
1728 complete token-bag proofs exchange exactly two actor-token positions;
body-prefix capture differences across future queries are0. Weights were not
individually hashed; config/tokenizer/observer dependencies were bound.
Collector completed3456/3456 attempts in849.354923 seconds without retries.
No fitting, training, generation, API, labels, private data, patching or gradients.

51 focused tests (10 new ownership tests), Ruff lint/format pass. Primary
verifier checks1054224 values with maximum8.882e-16. Result-after alternate
target/summary/toy/spectrum/profile readback checks1244721 values with
maximum9.095e-13. All four numeric worlds at three seeds retained actual arrays.
Same-agent alternate arithmetic is not independent scientific replication.

The initial analyzer saved stages but omitted the promised named spectra;
the separate result-after companion exported all432 spectra without changing
frozen primary code/results or rerunning the model. Its profile witness is a
declared result-after extension, not a preregistered outcome. Preserve all
HM46--68 material, including negative explanations; no Git/publication/lockbox.
