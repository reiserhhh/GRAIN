# P1L: cached entity differences to conditional response coefficients

2026-09-07. Frozen ine300353 before fitting; now complete and independently audited.
Results in ENTITY_OPERATOR_REPORT.md; prospective design retained below.
No new neural forwards. Decision: do entity-state vector differences predict
held-setting response components beyond template/person-name means and vector norm?

## Frozen input and split

Use only the completed P1K cache. For each of4 settings and2 queried entities,
X=n(role+)-n(role-) at the one name token, block16. Read only the canonical event/
positive-criterion/A=Yes state; common-prefix equality was already verified across
all later queries. These8 rows are event/entity contrasts, not8 independent humans.

Y contains24 coefficients: native versus N intervention, each3 criterion families
x4 exact polarity/coding coefficients. Exclude EQD because it exactly duplicates
native. No human labels or future data. The descriptive full-matrix singular
spectrum is not validation. All prediction splits leave one setting out, keeping
its two entities together;4 folds. No seed/layer/alpha selection.

Four frozen models: training mean; per-name training mean; Ridge on full X; Ridge
on ||X|| alone. Center X/Y only on training, scale X by the training RMS row norm,
then alpha=1 (dual system K+I). Return training mean if centered norm is zero.
This kernel normalization makes regularization comparable across1536D versus1D,
without feature selection. Full-vector fit still has only6 training cases/fold.

Save every OOF prediction and coefficient-wise/criterion-wise/full SSE ratio
against training-mean and zero predictors. Keep negative and near-zero components.
Do not promote a predictor on an aggregate dominated by code-only effects. Report
whether semantic components transfer on event/trait AND how venue compares.

## Tiny-sample null and resource bound

Enumerate all24 permutations of the4 setting labels, carrying both names and all
Y components together, and rerun the same4-fold full-vector fit. Compare the full
OOF SSE with each jointly permuted data set's SSE; identity included. This is a
small exchangeability diagnostic (minimum tail fraction1/24), not population
significance: settings were authored/nonrandom and names/actions have dependencies.
No hyperparameter search, bootstrapped pseudo-people, external data or new model.
Cap: zero neural calls, at most24 permutation fits x4 folds plus main models,
60seconds intended CPU budget; abort/preserve if exceeded. No automatic expansion.

## Interpretation

A reusable shared prefix does not imply a single readout. For a frozen finite
intervention, with other states held at the recipient block,

    d_q = R_q(n + delta, h_other) - R_q(n, h_other)
        = integral_0^1 <grad_n R_q(n+t*delta,h_other),delta> dt.

This is the fundamental theorem of calculus along a path (where differentiable),
not a new neural or psychological law. P1K's parity decomposition acts on R_q;
P1L asks whether local vector differences predict that finite conditional response
on held settings. No neural gradients are estimated in P1L. Successful prediction
would still use LLM-created states and would not establish an LLM-free personality
model, a binding-ID subspace, or a clinical measurement instrument. Failure sets
limits on this small fixed linear operator, not on all vector mechanisms.
