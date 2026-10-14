# Sources for a constructive theory of text-based psychological measurement

Subsequent reading and bibliographic corrections are maintained in
[the unified literature register](GRAIN_LITERATURE_REGISTER.md) and
[the bidirectional audit](GRAIN_BIDIRECTIONAL_THEORY_AUDIT_20260912.md).
The register adds MEM/BLT competitors, pins the SPAR source review, and corrects
the Blackwell follow-up publication year (2025 issue; DOI contains 2024).
This dossier preserves the earlier seminar's reading record.

Research scope: scientific goal selection, not validation of an already-fixed
GRAIN architecture. Sources inspected on 2026-09-12. Reading depth is explicit;
linked full texts were inspected at the relevant sections, not read exhaustively.
Repository descriptions establish available operations, not psychological validity.
No external repository was installed or executed during this review.

## Closest measurement work and constructive implications

| ID | Primary source and reading depth | Established contribution | Implication for the proposed goal |
|---|---|---|---|
| P1 | Brickman, Gupta & Oltmanns (2025), [LLMs for Psychological Assessment](https://journals.sagepub.com/doi/pdf/10.1177/25152459251343582), introduction and experimental-design section, pp.1-6 | A psychological-assessment workflow already connects language collection, modeling and interpretation; its example uses narrative interviews and questionnaire criteria. | A complete workflow alone is not our breakthrough. Need an additional constructive mechanism or prediction. |
| P2 | [Evaluating construct validity of text embeddings](https://link.springer.com/article/10.1140/epjds/s13688-022-00353-7), publisher abstract/introduction | Embedding quality is assessed relative to intended constructs rather than generic semantic quality. | A better representation is target-dependent; embedding geometry does not itself define a psychological scale. |
| P3 | Fleeson & Jayawickreme (2015), [Whole Trait Theory](https://pmc.ncbi.nlm.nih.gov/articles/PMC4472377/), indexed author full-text descriptive/explanatory sections; direct reopen hit CAPTCHA | Traits described through state distributions and explained by social-cognitive processes. | Distributional and state/trait thinking is inherited, not a GRAIN invention. A new goal must add an estimable construction law. |
| P4 | Rauthmann et al. (2014), [DIAMONDS](https://rap.ucr.edu/pub3.pdf), author-hosted abstract and [PubMed record](https://pubmed.ncbi.nlm.nih.gov/25133715/) | Situation characteristics are empirically organized and linked to cues, affordances and behavior. | Distinguish recorded opportunity, chosen venue and subjective appraisal. Do not replace all three by a topic label. |
| P5 | Karelaia & Hogarth (2008), [lens-model meta-analysis](https://www.insead.edu/sites/insead/files/assets/faculty-personal-site/natalia-karelaia/documents/3KarelaiaDeterminantsoflinearjudgment.pdf), source located; abstract verified through author/institution records | Cue validity, utilization and judgment achievement have an established statistical tradition. | Several LLM readers are useful evidence-use probes; they do not automatically identify an author's true construct. Full equation checks remain required. |
| P6 | Rasch (1977), [specific-objectivity archive](https://www.rasch.org/rasch.htm), archive listing; [derivation note](https://www.rasch.org/rmt/rmt11a.htm), mathematical summary | Non-identical items and invariant person comparisons are old measurement questions. | Do not claim first measurement without identical utterances. Compare the additional challenge of learned evidence units and endogenous observation. Original memoir not yet fully checked. |
| P7 | Egami et al. (2022), [How to make causal inferences using texts](https://naokiegami.com/paper/CausalText.pdf), abstract and workflow | Discovery of a text representation changes the inferential problem; discovery/estimation separation addresses some risks. | New evidence discovery is legitimate. Separate frozen evaluations from representation development, rather than banning exploratory discovery. |

Previously inspected Wilson/ECD, semantic measures and LST-R remain relevant in
V2-V4. This dossier does not treat those frameworks as limited to fixed binary
items or claim that context-conditioned measurement is unoccupied territory.

## Mathematics with a direct job to do

| ID | Source and reading depth | Mathematical object | Useful question, and boundary |
|---|---|---|---|
| M1 | Hyvarinen, Sasaki & Turner (2019), [nonlinear ICA with auxiliary variables](https://proceedings.mlr.press/v89/hyvarinen19a/hyvarinen19a.pdf), sections 3-4, equations 4-9 and Theorem 1 | Contrastive estimation under conditionally independent sources, invertible mixing and sufficiently varied auxiliary modulation | Can recorded conditions reveal shared latent structure? Recovery is up to componentwise transformations under assumptions; an arbitrary text encoder or arbitrary topic variable does not satisfy them automatically. |
| M2 | Gresele et al. (2020; UAI2019), [Incomplete Rosetta Stone](https://proceedings.mlr.press/v115/gresele20a/gresele20a.pdf), sections 1-3 | Shared latent sources across sufficiently different noisy views | Linking views could identify more than isolated embeddings. Two deterministic readers of the same text are not automatically the independent noisy views in this model. |
| M3 | Park & Muandet (2020), [measure-theoretic conditional mean embeddings](https://papers.neurips.cc/paper_files/paper/2020/file/f340f1b1f65b6df5b5e3f94d95b11daf-Paper.pdf), section 4, estimator and Theorems 4.2-4.4 | A conditional distribution represented as an RKHS-valued conditional expectation, estimated by vector-valued regression | Gives a rigorous candidate for condition-to-response objects without requiring a bounded inverse covariance operator. Consistency needs stated kernel, sampling and regularization conditions; it is not a psychological-identification theorem. |
| M4 | Szabo et al. (2015), [two-stage distribution regression](https://proceedings.mlr.press/v38/szabo15.pdf), setup and excess-risk section | Distributions are sampled, then observations are sampled within each distribution | Authors and texts per author provide two different sample sizes. Replacing each author by a bag is established; dependent/selected texts and learned measurement coordinates need additional treatment. |
| M5 | Park et al. (2021), [conditional distributional effects](https://proceedings.mlr.press/v139/park21c/park21c.pdf), section 5.1 | Conditional witness functions localize distribution differences | A numerical witness could point to evidence regions rather than only output a factor name. Applying it to authors is not automatically causal treatment-effect analysis. Kernel witnesses are smoothed contrasts, not raw density-ratio or semantic truth. |
| M6 | Blackwell (1951/1953), [collected source entries](https://celebratio.org/Blackwell_DH/article/255/) and [2024 general-state proof](https://doi.org/10.1016/j.econlet.2024.112146), abstracts; original Berkeley PDF returned403 | Compare statistical experiments by attainable decision performance, with common parameter space | Ask whether a representation/design preserves the desired decisions, rather than whether it reconstructs every hidden state. Exact Blackwell/Le Cam claims need the original hypotheses; no new proof is claimed here. |
| M7 | Driver, Oud & Voelkle (2017), [ctsem](https://www.jstatsoft.org/article/view/v077i05), original abstract and author repository | Latent dynamic processes, multiple noisy indicators, irregular observation intervals | Dynamic response and measurement need not be separate ad-hoc pipelines. Token order is not elapsed psychological time; choose the index explicitly. |
| M8 | Salvi et al. (2021), [signature kernel/Goursat PDE](https://arxiv.org/abs/2006.14794), original abstract; Cass, Piatti & Pei (2025), [numerical schemes](https://arxiv.org/html/2502.08470v1), introduction/algorithm discussion | Kernels of ordered paths through PDE computation; more recent numerical approximations | Ordered multi-factor interactions have mature mathematical tools. A signature is a representation candidate, not a personality model; truncation, time augmentation and sampling choices matter. |
| M9 | Di et al. (2009), [multilevel FPCA](https://arxiv.org/abs/0906.1457), author abstract | Separate between-person and within-person functional variation | A functional author representation need not be forced into the current static/dynamic feature list. Variance components alone do not name psychological constructs. |

These branches serve different purposes: identification (M1-2), conditional
representation/estimation (M3-5), comparison of information (M6), and dynamics
(M7-9). They are not ingredients that must all be placed in one formula.

## Existing implementations and direct competition

| Project | What was inspected | Operation worth comparing or reusing | Boundary |
|---|---|---|---|
| [OscarKjell/text](https://github.com/OscarKjell/text) | README and [author manual](https://oscarkjell.r-universe.dev/text/doc/manual.html) | Embeddings, training/prediction, semantic projection, token/layer extraction | Token access and psychological text scoring already exist. |
| [koheiw/LSX](https://github.com/koheiw/LSX) | README and publication links | Seed-guided latent semantic document scaling, including cross-language applications | Seed-to-text scaling is not a novel GRAIN goal by itself. |
| [maifeng/SPAR_measure](https://github.com/maifeng/SPAR_measure) | README, `src/spar_measure/api.py` and `core.py` | Bipolar semantic axes, active retrieval, reusable/precomputed embeddings | Direct near-neighbor to query-first evidence construction. Source is GPL-3.0-or-later; inspect before incorporating code. Whitening decorrelates, not generally makes coordinates statistically independent. |
| [devon7y/semanticfa](https://github.com/devon7y/semanticfa) | README | Semantic analysis of scale items without human response data | Item meaning structure is not author response structure; do not use it as evidence that personality factors were recovered. |
| [rprimi/embeddcv](https://github.com/rprimi/embeddcv) | README | Embedding-based content-validity analysis | Candidate tool for construct content, not person scoring validation. |
| [cdriveraus/ctsem](https://github.com/cdriveraus/ctsem) | README, original package paper abstract | Hierarchical continuous-time state-space models | Compare before inventing another bespoke AR decomposition. |
| [ilkhem/iVAE](https://github.com/ilkhem/iVAE) | README/configuration inventory | Identifiable-VAE reference implementation | Identification assumptions and experimental modality matter more than adding a VAE. |
| [crispitagorico/sigkernel](https://github.com/crispitagorico/sigkernel) | README | Differentiable path kernel on CPU/GPU | Reference for a justified ordered-path question, not an automatic upgrade. |
| [stanfordnlp/pyvene](https://github.com/stanfordnlp/pyvene) | README, intervention API documentation and source excerpt | Serializable activation interventions | Traces model cue use, not human causal mechanisms. |

The `text` paper, SPAR's linked firm-communication paper, and the package papers
are sources for distinct contributions, not evidence that every README claim
has been experimentally validated. Current repository branches were read;
reproducible implementation must pin versions/commits and review licenses first.

## Questions changed by these sources

1. A context-response distribution is a defensible *candidate measurement
   object*, but simply applying a kernel or writing an operator is not novel.
2. Existing IRT already compares persons given different calibrated items.
   The proposed additional task is to discover valid evidence and its comparison
   rules from non-identical, selected, context-dependent expression.
3. Require neither exact exchangeability within every context cell nor a unique
   low-dimensional personality basis as a precondition for exploration. Declare
   the smaller target relation/function that a local method can estimate.
4. Recorded-before-response is not the same as randomized/exogenous. Conversely,
   response-derived context is allowed for descriptive conditional analysis; it
   cannot silently be interpreted as an independently manipulated opportunity.
5. Multiple LLM readers plus one noisy criterion do not by themselves identify a
   uniquely true cue-validity/utilization decomposition. Reader interventions
   contribute one kind of evidence, not the full psychological identification.

The desired new result must be constructive: a rule/estimator that handles a
specified difficult case, predicts when a design or representation should
change, and produces useful psychological measurement beyond the closest
established alternative. A more ornate overview is not that result.

## Additional competitors found between seminar rounds 2 and 3

These sources materially narrow the novelty claim; they are not decorative
references added after choosing the architecture.

| ID | Primary source and reading depth | What changes |
|---|---|---|
| P8 | Liu & Wang (2017), [examinee-selected items and nonignorable missingness](https://pubmed.ncbi.nlm.nih.gov/28390145/), complete primary abstract | Choice patterns and responses already have a joint IRT model with correlated latent variables. Modeling selection separately is not new. |
| P9 | Liu, Qiu & Wang (2019), [examinee-selected items with rater effects](https://pubmed.ncbi.nlm.nih.gov/31452553/), primary abstract and indexed methods text; direct PMC reopen hit CAPTCHA | Jointly treating selected constructed responses and rater severity already exists. Several LLM raters plus a selection model alone would be an adaptation, not the core discovery. |
| P10 | Zoanetti, Zhou & Tan (2020), [NLP features in explanatory item response models](https://people.acer.org/en/publications/integrating-natural-language-processing-features-within-explanato/), institutional presentation abstract | NLP-derived item properties already enter explanatory measurement models. This is a presentation, not a verified full peer-reviewed methods paper. |
| P11 | Jeon et al. (2021), [latent-space item-response interaction maps](https://arxiv.org/abs/2007.08719), primary abstract | Person-item interaction geometry is already modeled instead of discarded. We should not demand additive separability as the definition of valid measurement. |
| P12 | Speer et al. (2026), [validity of open-ended personality assessment](https://journals.sagepub.com/doi/10.1177/10944281251413746), publisher abstract and reference list only; full paper restricted | Prompt design changes the supply of trait-relevant narrative evidence. Convergence to direct narrative ratings is a different target from convergence to self-report. A useful new mechanism must go beyond showing that targeted prompts or fine-tuning help. |
| P13 | [Natural Language Processing Psychometrics](https://arxiv.org/abs/2608.07316), August 2026 preprint abstract | Interpretable linguistic evidence, synthetic-to-human transfer and psychological prediction are active adjacent work. Full methods and claims still require comparison before a first-of-kind statement. |
| M10 | Kiraly & Tomioka (2012), [algebraic/combinatorial matrix completion](https://icml.cc/2012/papers/510.pdf), introduction, observation-mask example, Proposition2.13 and Example2.16 | Overlap/connectivity alone cannot justify general interaction recovery. Rank and observation pattern matter; necessary conditions need not give unique completion. This supplies a route to linking design, not a proof that one common prompt identifies all person responses. |

Additional references located, but not yet relied on for a methods claim:
[Zayas, Lee & Shoda's if-then assessment chapter](https://bpb-us-e1.wpmucdn.com/blogs.cornell.edu/dist/5/11179/files/2024/09/2021Zayas_Lee_Shoda_modelingmind_draft_17August2020.pdf),
[the 2026 interviewer-evaluator-judger framework](https://www.nature.com/articles/s41598-026-62333-5)
(publisher open failed; indexed abstract accessible), and the primary papers in
P12's reference list. Finding them is not equivalent to reading their full methods.

## Final-round novelty check

| ID | Primary source and reading depth | Consequence |
|---|---|---|
| M11 | Wang & Singh (2018), [selectively sampled column subset selection](https://www.jmlr.org/papers/volume18/15-233/15-233.pdf), introduction, objective and assumptions discussion | Feedback-driven observation selection and representative-column recovery already have theory. A new prompt-selection rule must add something beyond relabeling active matrix sampling. |
| P14 | [Enhancing IRT for Cognitive Diagnosis](https://arxiv.org/abs/1905.10957), 2019 primary abstract | Joint use of question semantics and measurement parameters already exists. Learning coordinates is not by itself enough to establish novelty. |
| P15 | Lee, Son & Jia (2025 online; 2026 issue), [multi-agent automatic item generation](https://link.springer.com/article/10.1007/s10869-025-10067-y), publisher abstract/introduction | An LLM system proposing context-specific, construct-relevant items with human review is also an existing pattern. Our object is open expression and its measurement evidence, not just new questionnaire wording. |

P11 was subsequently checked in the original [model paper](https://arxiv.org/pdf/2007.08719),
section2.2: its latent geometry represents interactions, not automatically an
ability space. The [2025 R Journal package paper](https://journal.r-project.org/articles/RJ-2025-021/)
provides continuous-response extensions as well as binary models; abstract and
package overview inspected. These are substantive comparison candidates.
[mirt](https://github.com/philchalmers/mirt) and the
[CRAN lsirm12pl mirror](https://github.com/cran/lsirm12pl) were located as
implementation candidates; no source execution or version compatibility check.

Final reading boundary: this is a targeted cross-disciplinary scan with explicit
full-text-section versus abstract-only records, not an exhaustive systematic
review or a verified first-of-kind search. P12's underlying data/protocol and
active-design/learned-measurement competitors require full methods comparison
before a novelty claim is finalized. None of the listed theorems is claimed to
have been newly proved or empirically verified in our system during this seminar.
