# HM159: source design, not reader accuracy

2026-10-11. Prospective finite CPU packet. Original 18/100/44 remains open.

## New construction ability and the competing explanations

Can GRAIN construct a new author's conditional expression propensity without
fixing a finite latent-class count or fitting a reader's answers? HM140 supplied
a choice law; HM142 imposed k=3; HM143 coupled local finite classes. F04 already
compared response means and slopes. None establishes that an unknown population
law is uniquely fixed by arbitrarily many short author histories.

Explanation A: sufficiently many source authors determine the population law,
so a unique conditional author score follows from that short source design.
Explanation B: different population laws can produce the exact same source
histories but different conditional author scores. A different observation
design, not a more accurate reader, is needed to resolve the functional.

Freeze four-history application profiles, both matched source populations,
all five possible histories, and a continuous Beta(2,3) population. No winning
axis/world, accuracy target, head/layer/rank search, or hidden application fitting.

## Declared observation world

p is a propensity for expression alternative R=1. Under fixed condition C,
Y=C xor R and R|p ~ Bernoulli(p). In the free phase, C|p ~ Bernoulli(p) and
R|p ~ Bernoulli(p) are conditionally independent. Therefore retaining (C,Y)
gives two conditional Bernoulli observations per event. This is a stated model
assumption, NOT a universal property of choice and speech.

Four distinct frozen GRAIN15 vectors encode the four (C,Y) addresses exactly.
They are public-source dictionary prototypes, not newly generated natural text.
The simulator emits these exact vectors, so decoding is a finite interface check.
Sources and application histories are separate. Oracle p/masses never enter the
cold constructor. Target is conditional expression propensity, not named Big5.

## Mathematical discriminator and positive construction

Use p_i=i/5 and the positive/negative parts of (-1)^i binom(5,i)/16. Finite
differences annihilate polynomials of degree <=4, hence the two populations
produce identical four-event histories for every source-author count, but can
differ on higher-degree or threshold targets.

The cold mean after history k of 4 is

    mu_k = integral p B_(4,k)(p) dG / integral B_(4,k)(p) dG.

Its numerator has degree5. Variance requires degree6. A fixed-four source
cannot identify the mean in general; fixed-five can identify the mean, and
three joint choice/response events can identify mean and variance in this
declared world. Compile these polynomial functionals into Bernstein weights
and apply directly to observed source frequencies, without estimating G.

Erasing C gives Y probability 2p(1-p), which is invariant under p -> 1-p.
The reflected matched populations remain indistinguishable at any event count.
Do not delete selection as nuisance. Repeated descriptions of the same event
do not create extra independent trials; include a deduplication negative control.

Exact finite-grid LP bounds are sharp only on that witness grid. Polynomial
identification holds throughout [0,1]. Continuous Beta verifies the latter.
Population-law ambiguity, uncertainty about this author, and finite-source
sampling error must remain separately reported. Do not clip failed empirical
moment variances or silently fit a mixture to repair them.

## Runs and acceptance

Prepare freezes dictionary provenance, plan, config, implementation and tests.
Run exact rational aliases and all cold profiles; new vector-event sources at
N=2000/20000, three seeds, three worlds. These two N values separate sampling
from structural ambiguity; they are not a performance optimization grid.
Save frequencies, counts, oracle-only evaluation arrays, all statuses, bounds,
profiles and hashes. Run a distinct rational/enumeration audit and focused tests.

Accept finite construction only if alias equality is exact, the joint law
distinguishes them, Bernstein identities hold, all source-only cold profiles
replay, and duplicate handling does not promote fake independent evidence.
Failure of sampled moments remains a failure. No automatic budget renewal.

## Prior art and exact delta

Tian/Kong/Valiant (NIPS2017), Introduction/Proposition1/Sections2.1-2.2,
already establish limited per-entity observations, moment matching and population
recovery. Vinayak/Kong/Valiant/Kakade (ICML2019), Sections1-3/Remark3.1, explicitly
separate a unique observed fingerprint from nonunique mixing laws. We do not
claim their theorem, finite-difference cancellation, Bernstein basis or LP as new.

OpenAI/math adc7f124, joint metric/connection Introduction lines1-130, motivates
specifying which observation family separates internal objects. Its smooth
manifold/unitary/elliptic assumptions are not imposed here or transferred.

The project delta is a cold functional constructor connecting qualified GRAIN
events, preserved choice/response, source design degree, and author uncertainty,
without a forced finite-class prior. Existing published mathematics is used
instead of another reader-precision experiment.

- https://papers.nips.cc/paper_files/paper/2017/file/bc4e356fee1972242c8f7eabf4dff517-Paper.pdf
- https://proceedings.mlr.press/v97/vinayak19a/vinayak19a.pdf
- https://github.com/openai/math/tree/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Determination-of-a-metric-and-a-unitary-connection-from-one-boundary-patch-October-5-2026

No private/API/neural training/Git/Claude/automation/lockbox/old queue. Goal active.
