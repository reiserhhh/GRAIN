# GRAIN Linguistic & Psycholinguistic Foundations v1 — literature map

Created: 2026-07-07, on the user's two theses: (1) constructs should have a
linguistic-theoretical base — the current inventory may be under-connected
to psycholinguistics; (2) personality is fingerprint-like: no factor set
exhausts a person, but the person is CODABLE, and what remains after
subtracting the group is the individuality (residual = 個性).

Verdict up front: thesis (1) is partially right — GRAIN's constructs were
data-discovered (OP-5) and validation-first, but several sit on
long-standing literatures the docs never cited; connecting them costs
little and strengthens the paper. Thesis (2) has an exact formal home
(Furr's distinctiveness; Molenaar's non-ergodicity; the forensic stylome)
and GRAIN already possesses the machinery and two prior lessons (E6/E9b
stranger nulls) needed to measure it — see OP-32.

## 1. What each GRAIN component already stands on (and should cite)

| GRAIN component | Existing literature base | Key sources |
|---|---|---|
| Function-word / style-base channel (C2; first_person) | Function words as unconscious, stable style markers; "linguistic style" as an individual difference; I-usage <-> negative emotionality meta-analytically | Pennebaker & King 1999; Mosteller & Wallace (function-word attribution); Edwards & Holtzman meta (already cited in prereg H2) |
| Open-vocabulary discovery (OP-5 wcl clusters) | Open-vocabulary beats closed dictionaries for personality prediction (r .31-.41 vs .21-.29); differential language analysis | Schwartz et al. 2013 PLoS ONE; Park et al. 2015 JPSP (LBA validity: self/informant agreement, 6-month stability); Eichstaedt et al. 2021 closed-vs-open review |
| Rind/register model (venues as registers) | Register variation is DIMENSIONAL and systematic; co-occurrence of lexico-grammatical features defines registers | Biber 1988 (MD analysis, 67 features, 6 dimensions); Biber cross-linguistic MD work — the rind model is a self-selection-aware cousin of MD register theory, and the 12 content classes are empirical registers |
| Choice channel (C1) | Sociolinguistic style-as-choice; audience design; register selection as identity work | Labov (stylistic variation); Bell (audience design); Johnstone 1996 The Linguistic Individual |
| React channel (C3, if-then signatures) | If-then situation-behavior signatures as the stable unit of personality | Mischel & Shoda 1995 Psych Review (CAPS) — C3 is its text-behavioral descendant; cite it |
| Slice-distribution view (states, E5/E7) | Personality as density distributions of states | Fleeson 2001; Fleeson & Jayawickreme (whole trait theory) |
| Trait-space != state-space (E7d) | Non-ergodicity: inter-individual structure need not equal intra-individual structure | Molenaar 2004 manifesto; Fisher, Medaglia & Jeronimus 2018 PNAS; Adolf & Fried 2019 (ergodicity sufficient-not-necessary) — E7d is a measured instance |
| Individual grammars (why machine-only constructs like wcl_60 exist) | Native speakers differ substantially in grammar/collocation; usage-based entrenchment is speaker-specific | Dąbrowska 2012/2018/2019 (individual differences in native attainment); Barlow 2013 (individual usage-based grammar); Schmid (entrenchment) |
| Fingerprint thesis / codability | The stylome: measurable, largely unconscious, constant author-unique trait set; idiolect as co-selection | van Halteren et al. 2005 (human stylome); Wright 2017/2018 (idiolectal co-selection, n-gram textbites); Grant & MacLeod (resource-constrained identity); Coulthard 2004 (idiolect & uniqueness) |
| Residual = individuality | Profile = normative + distinctive; raw similarity is normativeness-inflated; distinctiveness is the person-specific part | **Furr 2008 J Pers (the exact formalization of the user's thesis)**; Wood & Furr 2016 (normativeness critique); Biesanz SAM (normative vs distinctive accuracy) |
| Idiographic ambition with nomothetic tools | Idiographic/nomothetic integration | Allport 1937 (the original frame); Beck & Jackson 2020-2021 (idiographic personality networks) |

## 2. Where the psycholinguistics is genuinely thin (gaps worth owning)

1. **Production mechanisms.** GRAIN counts surface rates; it has no model of
   WHY a trait raises a rate (lexical access, planning, affect-priming).
   Psycholinguistic production models (Levelt-tradition) are not engaged.
   Cost of the gap: construct interpretations stay correlational. Cheap fix:
   frame constructs as "behavioral residues" (Boyd/Pennebaker tradition),
   not mechanisms; flag mechanism work as future.
2. **Register theory formalization.** The rind model reinvented pieces of
   Biber's MD framework with self-selection added. A mapping section
   (rind regimes <-> MD dimensions; class map <-> register taxonomy) would
   let reviewers locate GRAIN in a 40-year literature instead of treating
   the rind model as ad hoc.
3. **Lexicon construction.** The 7-11-word lexicons are ad hoc where LIWC
   categories went through psychometric development (OP-19 already tracks
   the LIWC-license question). Dictionary-construction standards
   (Boyd's LIWC-22 papers) are the reference point.
4. **Idiolect-grade features.** The stylome literature says the strongest
   individual signal sits in co-selection patterns (n-grams, collocations),
   not single-word rates — consistent with our M3 finding that ~32% of
   embedding-identifiable person signal escapes the 19-score battery.
   A v4 "co-selection construct" family (character/word n-gram profiles,
   audited like wcl clusters) is the literature-indicated growth direction.

## 3. The residual-individuality program (OP-32; user's thesis 2)

Formal chain: profile p_u = normative p-bar + distinctive d_u (Furr).
- **Codability** (fingerprint half of the thesis): identification metrics —
  SUICA-19 AUC 0.887 / embedding 0.891 / choice profile 0.839-0.909
  (stylome-consistent).
- **Factor incompleteness** (no factor set exhausts a person): OP-9 M3 —
  ~32% of embedding-space person-stable signal is NOT subsumed by the
  battery, yet remains person-stable (top-PC retest 0.66-0.72). That
  number IS the measured "factors can't cover everything" share at
  current battery size.
- **Residual stability** (the 個性 claim proper): distinctive-signature
  stability = same-user early/late correlation of z-scored deviation
  profiles vs stranger null — pilot: run_suica_op32_distinctive_residual
  _pilot_v1.py (T1; ledger row OP-32-P).
- Governance note: raw profile similarity WITHOUT the normative
  subtraction would repeat the mistake the project already made twice
  (E6 pre-audit, E9b labeling) — Furr's warning and our stranger-null
  lesson are the same lesson.

## 4. Immediate paper consequences

- Add a "theoretical location" subsection: C2 <- function-word stylistics;
  C1 <- register self-selection (Biber + audience design); C3 <- CAPS
  if-then signatures; E5/E7 <- whole-trait density distributions + 
  non-ergodicity; residual track <- Furr + stylome.
- The falsified centering result gains a literature frame too: topic
  models/LDA-residualization as the default "topic-as-nuisance" practice —
  our F4 theorem says exactly when that default destroys signal.
- OP-19 (LIWC), OP-28 (LLM worlds), OP-33 (co-selection construct family)
  become the literature-driven v4 items.
