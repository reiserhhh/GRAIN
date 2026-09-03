# V7 Psychometric Projection: Construction Charter

## Purpose

V7 starts a new construction line for GRAIN.  Its object is not a generic text
classifier, a replacement Big Five inventory, or an author-identification
system.  Its object is a **frozen, inspectable mapping from repeated natural
language observations to relative positions in a reference population**.

The construction problem is:

\[
  D_u = \{(x_{u i}, c_{u i}, o_{u i})\}_{i=1}^{n_u}
  \quad\longmapsto\quad
  (\widehat{\mathcal G}_{u;R,O,E,A}, \mathcal S_u),
\]

where \(x\) is text, \(c\) is observed context/provenance, \(o\) is the
observation operator, \(\widehat{\mathcal G}\) is a reference-relative
landmark-distance geometry, and \(\mathcal S_u\) states whether the available
observations support that profile. This is a measurement candidate only. It
becomes a psychological construct only after a separate construct-validation
programme.

## V7 Primary Object

The V7 primary object is a versioned **relative geometry bundle**, not a
post-hoc factor label:

\[
\mathcal G =
  (R, O, E, A, M, L, Q, V, \nu),
\]

with reference population \(R\), observation operator \(O\), representation
extractor \(E\), author aggregation \(A\), a regularized metric \(M\),
identifier-free landmarks \(L\), reference support distribution \(Q\),
validity/support domain \(V\), and version \(\nu\). It must retain all of
these to produce a reproducible profile for an unseen author.

For a frozen bundle, a new author is scored without refitting:

\[
  \widehat{\mathcal G}_{u^*} =
  \left(d_M\!\left(A(E(O(D_{u^*}))),\ell_j\right)\right)_{j=1}^{m}.
\]

If the author has insufficient observation support, text falls outside the
declared operator domain, or a required runtime artifact is absent, GRAIN must
return `SCORE_REFUSE` rather than fabricate precision.

W4/W4b's high effective-rank result means that a small, rotated latent-factor
coordinate system is not identified by current data. The older `FactorBundle`
remains a historical technical probe only. It is not the V7 primary
measurement object and cannot supply V7 factor names or personality scores.

## Construction Principles

1. **No label enters discovery.** Big Five, MBTI, symptom scales and behaviour
   may only be attached in a later external-anchor stage.
2. **Context is a channel, not blanket-removable noise.** Topic, setting,
   expression opportunity and a person's selection of them remain observable
   provenance.  V7 compares observation designs rather than subtracting them
   by default.
3. **Multiple observation operators are competing measurement designs.** Whole
   text, native comments/turns, fixed windows, semantic blocks and nested
   aggregation are all legitimate arms.  Fixed 128-token slices have no
   privileged theoretical status.
4. **Discovery is permissive; promotion is evidence-labelled.** A candidate can
   remain `TEXT_STRUCTURE_DIAGNOSTIC` or `CONDITIONAL_BRANCH` without being
   discarded.  It cannot be called a personality dimension because it helps
   author matching.
5. **The old V3/V4 lockbox is historical evidence only.** It neither selects
   V7 candidates nor closes V7 questions.

## Initial V7 Status Vocabulary

| Status | Meaning |
| --- | --- |
| `L0_INGREDIENT` | Reproducible text feature, not yet an author score. |
| `L1_POPULATION_STRUCTURE` | Cross-person structure found in a discovery population. |
| `L2_SCOREABLE_CANDIDATE` | Frozen geometry can profile unseen authors relative to a reference population. |
| `L3_INTERNALLY_REPLICATED` | The same frozen object survives author-disjoint technical replication. |
| `L4_EXTERNALLY_ANCHORED_CANDIDATE` | A declared projection or geometry relation has separate external criterion evidence. |
| `L5_DOMAIN_SCALE_PROTOTYPE` | A future domain-specific construction programme has fixed an operational protocol. |
| `TEXT_STRUCTURE_DIAGNOSTIC` | Useful descriptive structure that is not scoreable as an author object. |
| `SCORE_REFUSE` | Input does not meet the bundle's declared support conditions. |

## First Experiment

`scripts/run_suica_v7_operator_smoke.py` is intentionally label-free.  It
uses PANDORA Tier-U only as a local reference population and compares
observation operators on:

- author-disjoint frozen-score coverage;
- within-author technical resampling error;
- disjoint-observation score consistency;
- factor-space transport from discovery to confirmation; and
- required support / refusal rates.

It does **not** test a psychological hypothesis, name a factor, or evaluate
Big Five/MBTI.  Its role is to make the V7 measurement object executable
before any new construct claims are attempted.
