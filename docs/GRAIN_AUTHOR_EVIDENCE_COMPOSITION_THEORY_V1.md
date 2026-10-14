# Local observation equivalence and author evidence composition

2026-10-10. HM102 executed on sealed public numeric HM101 carriers, no fitting
or new neural calls. This is a measurement-construction rule, not a new general
mathematical theorem or a completed personality scale.

## 1. A common observer is not yet an author measurement

For evidence e, target t and head h, register a source query q_e,t,h and value
covector d_t,h in the carrier's actual coordinates. Define

    l_e,a = q_e . k_e,a, c_e,a = d . v_e,a,
    Z_e = sum_a exp(l_e,a), N_e = sum_a exp(l_e,a)c_e,a,
    f_e = N_e/Z_e.

Token-sum Z counts atoms. Unit-mean Z divides both N and Z by the record's
token count. Neither changes f. Neither is automatically the author's event
frequency or evidential reliability. HEADS ARE COMPOSED SEPARATELY, then summed.

Given a declared author composition P, two possible objects are

    S_declared = sum_h E_P f_h,
    S_partition = sum_h E_P(Z_h f_h)/E_P Z_h.

The second observes a tilted author law dP_tilde,h = Z_h dP/E_P Z_h. It is
not intrinsically wrong, but is a DIFFERENT measurement. Their exact gap is

    S_partition - S_declared = sum_h Cov_P(Z_h,f_h)/E_P Z_h.

This covariance is a construction account, not a fraction of psychological
information. It can be positive, negative, or cancel between heads/targets.

## 2. The local normalized observer forgets a scale needed by one merge

Translate every key in one evidence unit by kappa_e. For every query, all
logits receive the same q . kappa_e and local normalized attention/readouts
remain invariant. The positive lift (N_e,Z_e), however, scales by
exp(q . kappa_e). This is observational equivalence for the LOCAL reader,
not a symmetry of the original neural encoder or a permissible natural-text
intervention. If a calibrated partition is retained as data, the full reader
does not treat those carriers as equivalent.

In projective language, (N,Z) and (aN,aZ), a>0, have the same normalized
readout. Adding positive lifts is well defined, but addition does not descend
to independently rescaled projective classes. For two distinct local readings
f1,f2, the merge

    (a1 Z1 f1+a2 Z2 f2)/(a1 Z1+a2 Z2)

changes as a1/a2 changes, even though both local functions are unchanged.
With positive P on the support, invariance under EVERY independent positive
rescaling holds iff each local head reading is constant on that support.
Proof: vary one positive scale to dominate; the merge tends to its local
reading. Conversely equal readings trivially remain equal. A sum of heads may
accidentally cancel, so the universal condition is stated headwise.

In contrast, E_P f is determined by P and the local normalized observations.
This identifies a missing assembly contract, not an impossibility of author
measurement and not a reason to discard contextual mass/choice information.

## 3. Compatible assembly of retained positive lifts

Represent a positive lift as (ell=logZ,f). The stable binary operation is

    ell_new = logaddexp(ell1,ell2),
    f_new = exp(ell1-ell_new)f1+exp(ell2-ell_new)f2.

Insert declared author weights by ell_e <- ell_e+log P_e. For a fixed common
scale, this operation is associative and commutative and preserves split/rejoin.
These are standard normalizer properties, NOT a new streaming algorithm.
They ensure partition merge consistency after its SCALE has been declared;
they do not make that scale identifiable from f alone.

Splitting a fixed unit's frozen atoms and rejoining preserves the token measure.
Re-encoding split texts changes contextual K/V and is a different operation.
Likewise duplicating token atoms changes token mass; unit-mean mass treats a
duplicated atom sequence differently. Same-event duplicates still require the
existing event ledger: this packet does not infer event identity from row IDs.

## 4. Executed finite witness

Use all864 current K/V captures from HM101, four physical route arms, two
wording banks, two bridges, six action families and six frozen target queries.
Queries have12 heads and keys2 GQA heads,128 dimensions. Across seven endpoint
registrations, each36x128 query system has rank36. A query-only minimum-norm
key offset satisfies all target/head logit shifts simultaneously, max residual
3.109e-15, norm .856--1.185, condition243--438. It is not an outcome-fitted
predictor, nor a claimed on-manifold intervention in natural language.

Apply fixed theta_e=+/- .5*(previous_level-1), base0 and global+.5 controls.
Vector-translation local reading error2.109e-15; log partition shift error
7.994e-15. The two synthetic author laws have equal current marginals.

In the physically neither arm, current K/V ignores history and S_declared
contrast is zero (max8.882e-16). After a representation-only local gauge, the
partition merge contrast reaches .455655, with new-word token-sum RMS .175763
for the +.5 gauge. This is a counterexample relative to the declared author
measurement, not discovery of psychological information in disconnected paths.
Global gauge keeps both estimands invariant (merge max change2.665e-15).

The actual both-arm declared contrast remains new-word RMS .176971 for every
gauge. Its base unit-mean partition alternative is .213931 RMS; these magnitudes
do not rank predictive quality and are not Pearson correlations. No successful
history signal or condition-selection account was removed.

## 5. Reusable interface and next scientific consequence

`math_author_evidence_merge.observe` exports local f and logZ. `compose` returns
the declared expectation, partition alternative and covariance gap separately.
The cold CLI `score_grain_author_evidence_merge_cold.py` opens only compiled
query JSON and numeric K/V/record-position NPZ, no text/model/native answer.

`compose_grain_author_evidence.py` requires an explicit author-law declaration
and `--mode`. Partition mode additionally requires a scale contract; records
must be unique within each declared profile.192 positive synthetic profiles
were actually scored by both modes;1152 interface contrast checks exactly
match the primary table. Both standalone CLIs also produced byte-content-equivalent
JSON objects to the demonstrated interface (different file names).

This closes the finite observation-to-author assembly ambiguity. It does NOT
close natural evidence eligibility, unseen-condition transport, shared/target-
specific response construction or psychological interpretation. Current f still
depends on frozen neural K/V and source queries; it is not raw-text LLM-free
personality prediction. Next develop compatible target response laws on common
evidence, using declared author integration rather than letting partition
calibration silently redefine the author. Review earlier target-response
experiments before adding another packet; no precision/head/rank grid.

## 6. Mathematical ancestry and reproducibility

Vuckovic et al2020, section3.1 supplies the Boltzmann-Gibbs measure viewpoint:
https://arxiv.org/pdf/2007.02876 . Milakov/Gimelshein2018, sections3/3.1 supplies
online/parallel stable normalizers: https://arxiv.org/pdf/1805.02867 .
Katharopoulos et al2020, section3.2 supplies kernel numerator/denominator
assembly: https://proceedings.mlr.press/v119/katharopoulos20a/katharopoulos20a.pdf .
We do not claim their finite feature map is exact for arbitrary softmax queries.

OpenAI/math pinned algebra.tex, lines1-160, motivates checking whether joining
representations respects their declared meaning. Its finite monoid/F2/star-
height theorem is not transferred to this continuous numerical problem:
https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Finite-Monoid-Computations-and-a-Uniform-Generalized-Star-Height-Bound-September-25-2026/build/sections/algebra.tex

Ancestry in this project: NEXT70 normalized content+mass; S15--17 within-body
mass/covariance; HM88 hierarchical token laws versus author composition;
HM101 physically controlled history input. These are retained, not renamed
as new discoveries. HM102 adds the cross-context equivalence/merge witness
and an explicit reusable author-construction contract.21 focused tests pass
(14 new across mechanism/interface), seven Python files Ruff/format pass;
331776 stream checks and direct-token verification max2.220e-15, same agent,
not third-party reconstruction. No neural/API/private/training/Git/lockbox.

Canonical: `results/grain_math_response_v1/author_evidence_merge_r1`.
Primary stages prepare -> cold CLI -> analyze -> verify, followed by separately
sealed interface demonstration. Existing paths are exclusive; never overwrite
sealed results to rerun. Original18/100/44 and the main goal remain OPEN/ACTIVE.
