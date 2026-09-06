# VM05 raw same-ready source comparison

2026-09-07, round33. **Complete raw/reference-free stage, not full-program
acceptance.** The34 new conditions finished normally in2,517s of fitting-stage
time; all170 saved-fold replays had zero numerical error. The unchanged144-person
Big5 profile was reused and independently replayed, maximum8.33e-17, not refitted.
There are35 source heads /175 accepted saved folds in this report.

## Matched estimators and results

Every row uses the same36 comments per author, same eligible authors, source-only
labels, task-specific official5 outer folds and fixed3-fold inner search. Big5
uses144 Big5-only authors; each MBTI axis uses1,539 MBTI-only authors, excluding
historical reference authors. These are GRAIN-support-selected cohorts, not the
full624/6,294 source comparison, not all PANDORA and not the bridge population.

Here **g** means the16 sorted distances to frozen landmarks. This is the declared
formal profile comparator, **not the entire GRAIN4W framework or every historical
GRAIN feature family**. BGE/E5 are cached user-level embeddings; Qwen is block10
token-weighted body mean/std. Concatenated features use the same joint fold-fit
scaler/PCA16 and statistical head. No outer-result-selected weights or layer.

| Task / pooled metric | g | TF-IDF | BGE | E5 | Qwen | g+Qwen | g+BGE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Big5 mean Pearson |-.067957|.077950|.158473|.170271|.057208|.057573|.166768|
| EI AUC |.472598|.565262|.628519|.620472|.604889|.605358|.629422|
| SN AUC |.488620|.584707|.641738|.641101|.604863|.604772|.641619|
| TF AUC |.542748|.726055|.735849|.734191|.716673|.716701|.736014|
| JP AUC |.517255|.559249|.624937|.634304|.624719|.624727|.626177|

No axis has been flipped because its profile AUC was below.5. Mean Pearson and
AUC are different metrics; do not average these five rows into a single score.

## The relevant GRAIN increment

Adding a neural representation to a weak profile is not evidence that the profile
contributes unique signal. The important paired contrasts are therefore
`(g+Qwen)-Qwen` and `(g+BGE)-BGE`, not only the combinations minus g.

| Task | g+Qwen minus Qwen, pooled95% CI | g+BGE minus BGE, pooled95% CI |
|---|---|---|
| Big5 |+.000365 [-.002406,.002937]|+.008295 [.000022,.016776]|
| EI |+.000469 [-.000116,.001088]|+.000903 [-.002221,.004516]|
| SN |-.000092 [-.000717,.000515]|-.000119 [-.004735,.004432]|
| TF |+.000028 [-.000412,.000489]|+.000165 [-.001851,.002258]|
| JP |+.000009 [-.000478,.000508]|+.001241 [-.001625,.003928]|

Of the20 pooled/mean-official-fold GRAIN-increment intervals,19 include zero.
The sole exception is Big5 g+BGE pooled, whose lower endpoint is only.000022.
Its mean-fold delta is+.008001[-.000291,.016802], positive in4/5 folds. This is
a small, estimand-sensitive, unadjusted lead, **not a robust family-wide increment**.
The near-zero lower endpoint must not be presented as decisive significance.

The bootstrap uses1,000 shared within-official-fold author draws, seed731, for
all seven observers and nine fixed contrasts. Intervals are pointwise and
conditional on the already fitted OOF heads. They do not include retraining,
hyperparameter-selection uncertainty or familywise correction. Analysis code was
implemented after some point estimates were visible; comparator families were
already declared. All90 paired rows, both estimands and all negative results
are retained. Independent real-data scalar replay uses SciPy/sklearn directly:
35 heads /70 metrics agree within9.71e-17. Bootstrap endpoint independence was
tested on synthetic examples, not rerun independently for all real-data draws.

## Mechanism interpretation

1. Under this supported-cohort/readout protocol, sorted landmark distances alone
   carry much less externally readable source-label information than ordinary
   embeddings. Formal geometry and personality-label prediction are not the same
   endpoint. This does not negate previously validated GRAIN measurement objects.
2. The combinations mostly reproduce their neural component's score. Large
   improvements over g alone are primarily compatible with the neural component
   doing the work. They cannot be credited to a new GRAIN-personality mechanism.
3. Joint PCA is itself a restrictive readout:16 profile columns compete with
   thousands of neural columns. The current comparison neither proves that g
   contains zero unique information nor identifies projection as the cause.
   Do not silently replace the frozen experiment with an outcome-tuned blend.
4. Full-source versus same-ready scores change both training population and
   evaluation population. The changing E5/BGE ranking cannot be attributed to
   model size or eligibility alone. Cross-cohort differences are not paired effects.
5. This remains source-trained regression/logistic readability, not the native
   LLM's trait judgment, internal causal use, or independent bridge validity.

### Saved-head algebra audit

All60 hybrid saved heads (10 conditions x5 outer heads plus full head) were
checked separately using480 synthetic feature rectangles, not real author rows.
The mixed GRAIN/neural finite difference is zero within4.44e-16 on the regression
or log-odds scale. Effective-coefficient pullback agrees within4.43e-16. Three
synthetic unit tests include the counterexample that sigmoid curvature can
create probability-scale mixed differences despite additive log odds.

This confirms that the executed head tests additive, projection-restricted
usefulness. It does not test an arbitrary interaction such as `g^T U V^T h`,
and does not demonstrate that such an interaction would improve prediction.
See [the algebra and next-step boundary](../../docs/GRAIN_VECTOR_READABILITY_THEORY_NOTE_V1.md).
No nonlinear source model or source-result-guided rescue was added.

## Artifacts and next action

Private root: `results/grain_vector_mechanisms/`.
- `vm05_raw_ready_primary_r1`:34 fits and explicit reused-profile evidence;
  batch77d1d31511f51c54ffe98163835726d7ae9d68c71e5aff0ef8a4bdf676422f7c.
- `vm05_raw_ready_paired_comparison_r1`:35 observer rows, trait metrics,90 paired
  rows; identityd5e55b439997d64bb2e8dc88dbe4ba22d3fb56781ff958c7f3197c0adb292cb8.
  Paired CSV SHA256d44f1a4e52549a13fc3fae051f02a116018dfc3c28727c7a5f6dd2b5cb46a5bc.
- `vm05_raw_ready_scalar_audit_r1`: independent35-head/70-score audit,
  identityafae3c1e1a12763cbbbfdaa1547e671807cf489d85c166d4b9c2ac17343cad8a.
- `vm05_raw_ready_additivity_audit_r1`:60 saved heads/480 artificial rectangles,
  identityd156388e913a7c5b1327f726135fd69186bfe4b3db575876c902db7dd10a184b.

Run the existing declared secondary readout family, then history/direct and
source controls/nulls; keep full same-ready arm/reference/readout scope open.
The new315-case cache-only supervisor started only after this complete paired
packet and actual predecessor exit. It performs no new neural forwards. Strong
controls stay paused while it owns the CPU leases. Bridge/MEPS psychological
values remain closed; no public release, push or scorer modification occurred.
