# Condition-relative relational fields, HM208

## Scientific object and result

The object is an addressed two-window text relation with source target
links, not a purified personality vector. HM208 qualifies HM207's SAME
relation into condition composition, relative expression and oriented mixed
accounts without changing the full observation or selecting a better score.

386 source authors, 203/183 author folds, two disjoint12-event windows,
full1024 dimensions and all32 inherited binary target profiles are retained.
The24 old native-contact authors supply identical event edits and cached
responses. No new raw text, application questionnaire, neural forward, API
or training was used. The condition is the inherited TOP20 venue bit, NOT a
complete topic, situation, opportunity or psychological-state description.

The O relative-expression relation is positive under both frozen template
references: source cross-fold inner products3.037841e-7 and3.245610e-7.
The composition-only O account is3.883537e-10 in both comparisons. This
coarse composition-only account does not exhaust the O candidate. N also
has a positive expression-expression account, but other accounts reduce
the full comparison. E/C negative findings remain. These inner products
are not Pearson correlations, accuracy or independent information shares.

## 1. Source-only conditional reference

Let x[u,v] be an author's mean vector in window v, p[u,v] the observed
condition-one fraction, and tau[b,v,z] the source-fold event-weighted mean
for condition z. The template fit does not use target labels. Define

```text
C[u,v] = tau[v,0] + p[u,v] (tau[v,1]-tau[v,0])
E[u,v] = x[u,v]-C[u,v]
x[u,v] = C[u,v]+E[u,v].
```

C is composition under this declared template; E is expression relative to
that template, not discarded error. Two reference conventions were fixed
before analysis: each fold's own templates, and the opposite SOURCE fold's
templates. Both are retained. They are not independent replications.

272 source author-windows with just one observed condition remain in the
data. No unobserved individual response is imputed from them. A globally
missing source condition is rejected rather than replaced by zero.

## 2. Addressed conditional relation

G is the inherited five-bit target profile, chi_m(G) its finite character.
The operator reference is uniform over32 profiles and uniform over authors
within each profile. It is an explicitly constructed reference, not the
natural population distribution. Center each component within the SAME
profile, r_a,v = a_v-E_ref[a_v|G], a in {C,E}. Then

```text
r_C,v+r_E,v = r_HM207,v
T_ab,m = E_ref[chi_m(G) r_a,0 r_b,1^T]
T_pair,m = T_CC,m+T_CE,m+T_EC,m+T_EE,m.
```

Across two source author folds, the Hilbert-Schmidt comparison is

```text
<T_pair,m^0,T_pair,m^1>
  = sum_ab sum_cd <T_ab,m^0,T_cd,m^1>.
```

All16 SIGNED accounts are needed. Keeping only the four diagonal terms
changes the estimand; negative mixed terms are not automatically noise.
All31 nonconstant characters, including all5 single-target characters,
are retained. The199 EXACT HM207 within-profile second-window permutations
move C and E together, so they do not introduce a new reference search.

The binary composition-only matrix has the exact factorization

```text
T_CC,m = E_ref[chi_m(G) ptilde_0 ptilde_1] delta_tau_0 delta_tau_1^T.
```

Here ptilde_v=p_v-E_ref[p_v|G]. Rank at most one follows from the construction,
not from discovering an intrinsic one-dimensional psychological trait.

## 3. Frozen cold observation and action

Cold input has only frame_key and two windows of distinct event_id/vector/
condition records. It has no target, native answer or application profile.
Frozen source templates produce cold C/E and fixed profile-averaged centers
mu_c,v. Four source operators are evaluated on four observation leg pairs:

```text
s_ab,cd,m = (x_c,0-mu_c,0)^T T_ab,m (x_d,1-mu_d,1)
sum_ab,cd s_ab,cd,m = old HM207 pair reading.
```

For a one-window edit with the other window fixed, this is an exact linear
field on the changed component. BOTH p and E are recomputed after actual
selected/control comment edits. Holding p fixed when condition counts
change would measure the wrong action.

The actual cold CLI returns the same five total values for both references,
up to roundoff. The inference arithmetic uses no LLM, but the historical
E5 representation remains neural and no LLM-free raw-text claim is made.

## 4. Reference change is redistribution, not new evidence

If tau changes to tau', define D=C'-C. Then C'=C+D and E'=E-D. After profile
centering, d=(p-E_ref[p|G]) (delta_tau'-delta_tau). Accordingly,

```text
T'_CC = T_CC+T_CD+T_DC+T_DD
T'_CE = T_CE+T_DE-T_CD-T_DD
T'_EC = T_EC+T_ED-T_DC-T_DD
T'_EE = T_EE-T_ED-T_DE+T_DD.
```

Window subscripts are implicit on the left/right factors. The complete
operator and complete cold score are invariant; the component attribution
can change. This is bilinear algebra, not a new gauge theorem and not proof
that natural topic effects vanish. A component must carry its reference
qualification instead of being treated as an intrinsic purified trait.

## 5. What the executed data decide

For the prespecified five-target sum, source comparisons are

| Account | Own template | Opposite template |
| --- | ---: | ---: |
| EE versus EE | +2.226486e-7 | +2.282917e-7 |
| Remaining15 signed accounts together | -1.508497e-7 | -1.564928e-7 |
| Full pair relation | +7.179892e-8 | +7.179892e-8 |

The full reference upper rank remains .340. No improvement has been made
to the original relation. O's EE ranks .025/.010 and N's .030/.060 are
descriptive exploratory ranks, not multiplicity-corrected psychological p.
The source O result survives these two coarse reference choices, while its
model-action contacts remain unequal across source folds: EE own contacts
.082/.335 and .084/.331; cyclic-stranger contacts are negative in all four
cells. The native model saw only the edited own window, not the companion;
these contacts do not establish that it uses this two-window operator.

Useful construction delta: choice and relative expression can be kept in
ONE addressed relation, with all16 cold readings summing to the original
measurement. This permits competing evidence organizations to be studied
without erasing choice, state or context. It does not establish author
fingerprint exclusion, psychological validity, or a universal predictor.

## 6. Mathematical ancestry and next question

- Lancaster/finite-character accounts are already in HM144/HM207. This
  packet does not rediscover them as a new theorem.
- [Quintero et al.2026, sections2.1 and4.1](https://arxiv.org/html/2603.29972v2)
  studies reference-dependent composition attribution. Our tensor
  construction is not its Oaxaca-Blinder estimator; the useful lesson is
  to declare and retain the reference, not choose a flattering one.
- [Yu and Elwert, sections2.1-2.3](https://arxiv.org/html/2306.16591v4)
  separates prevalence, response heterogeneity and selection. The causal
  treatment assumptions are not established for these natural venue bits;
  no sorting-on-gains or human do-condition identification is borrowed.
- [OpenAI/math signed-isotypic, pinned adc7f124, lines72-112](https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Signed-tensor-densities-and-diagram-budgets-for-coordinate-sweeps-September-26-2026/build/sections/03b-isotypic.tex)
  inspires retaining oriented sectors and multiplicity. Its symmetric-group
  density theorem is not transferred to these Z2^5 text observations.

Next inventory existing SAME-event source-qualified conditional-response
and diagnostic-action interfaces before any new finite experiment. The
question is how a source-supported relation becomes a reusable target
evidence operation, not how to maximize EE, its r, or a condition taxonomy.
Do not repeat HM186/189 displacement statistics, tune templates/venue
granularity, delete mixed terms, or assume the native model read a companion.
Original18 protocols/100 cards/44 questions remain OPEN; goal ACTIVE/nojob.

## 7. Reproduction and audit

Canonical: results/grain_math_response_v1/condition_relational_operator_r1/.
prepare/build/analyze/verify each executed once, followed by a separate cold
CLI. The inputs and scientific code are hash-bound; do not rerun over r1.
10 new tests plus8 HM207 tests passed before sealing;52 focused neighboring
interface tests passed after execution. Three Python files passed
Ruff check/format. One same-agent alternate audit covered2116 literal
accounts and7936 finite-field accounts, max7.517417e-21. Parent source and
all199 reference totals differ by at most8.999725e-22; cold/action totals
by at most6.098637e-20. These are implementation checks, not science gains.

build .693584s, analyze1.115211s, verify .309181s; summed2.117977s excludes
prepare/input hashing and process overhead. No scientific failure or rerun.
The preseal summary-index axis fix and its regression test are recorded.
All experiment handles are terminal; no new background job is running.
