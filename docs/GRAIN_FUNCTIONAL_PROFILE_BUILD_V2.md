# Build the fixed-reference functional profile

2026-09-09. Research extension; released deterministic GRAIN APIs are unchanged.
Use the existing Project Persona environment from this independent repository.
Rebuilding requires the private S1/M8 caches and frozen local observer, not API
keys, downloads or new labels.

## Commands

```bash
PY="/Volumes/mobile3/projects/project persona/.venv/bin/python"
B="results/grain_vector_mechanisms"
"$PY" scripts/run_grain_profile_states.py prepare --output-dir "$B/shared_author_profiles_m25_v1"
"$PY" scripts/run_grain_profile_states.py collect --output-dir "$B/shared_author_profiles_m25_v1"
"$PY" scripts/analyze_grain_functional_profiles.py --state-dir "$B/shared_author_profiles_m25_v1" --output-dir "$B/functional_author_construction_m25_m26_v1"
"$PY" scripts/analyze_grain_functional_covariance.py --analysis-dir "$B/functional_author_construction_m25_m26_v1" --output-dir "$B/functional_covariance_m25b_v1"
"$PY" scripts/package_grain_functional_profile.py --analysis-dir "$B/functional_author_construction_m25_m26_v1" --state-dir "$B/shared_author_profiles_m25_v1" --output-dir "$B/functional_profile_bundle_m26_v1"
```

These directories already exist and are complete. Do not overwrite or restart.
`collect` resumes or returns if complete; fresh reproduction uses new output
names. Analysis/export refuse existing destinations. Private text, token records,
author outputs and arrays stay ignored/local. A private bundle is not cleared for
public distribution merely because it contains coefficients rather than texts.

The collector verifies model/template/query tokens and old endpoint vectors. It
captures layer16 boundary states, stopping before suffix attention, without
gradients or generated answers. Native comparison targets come from old caches.

## Score encoded material

```python
from grain_research.vector_mechanisms.functional_bundle import FrozenFunctionalBundle

bundle = FrozenFunctionalBundle.load(
    "results/grain_vector_mechanisms/functional_profile_bundle_m26_v1"
)
parts = bundle.score("neuroticism", P, Q, question_token_ids, observer_configuration)
```

P is ordered query-token by hidden state; Q is the final-token state, both from
the declared observer. `parts` contains `mean`, `configuration`, `final`, `score`.
No clipping or questionnaire percentile conversion occurs. BGE/E5 vectors cannot
be substituted for these Qwen-specific coordinates. New text still needs the
frozen neural prefix; scoring cached arrays needs only NumPy.

For an author-view, `states` maps each query to `full`, `e1`, `e2` pairs `(P,Q)`:

```python
profile = bundle.profile(states, query_token_ids, observer_configuration)
positions = profile["positions"]
decomposition = bundle.shared_private(
    three_query_configuration_responses, view=0, edit="e1"
)
```

The nine coordinates are three levels, three E1 C responses and three E2 C
responses. This API requires S1-compatible material/edit selection. For new
protocols, retain full values but declare/rebuild the profile reference. Arbitrary
edits are not normative measurements. Saved OLS shared/private components are
inspectable; their shared term did not transport and is not the recommended
cross-query estimator. Use the signed cross-path covariance ledger instead.

## Outputs and tests

- `reference_rank_checks.csv`: every SVD candidate; full-rank fallback retained.
- `native_readout_metrics.csv`: model levels and natural edit agreement.
- `functional_links.csv`, `shared_private_metrics.csv`: group/query relationships.
- `signed_cross_path_covariance.csv`: signed pairs, including native remainder.
- `author_profile_metrics.csv`: separate level/response/combined results.
- `replay_audit.json`: portable function and batch-invariance verification.

```bash
"$PY" -m pytest tests/test_grain_functional_profiles.py tests/test_grain_functional_bundle.py tests/test_grain_functional_covariance.py -q
"$PY" -m pytest tests -q -k 'grain and not nonlinear_readout'
"$PY" -m pytest tests/test_grain_nonlinear_readout.py -q
```

Keep LightGBM tests separate from Torch on this Mac (OpenMP runtime conflict).
No old long-running queue needs to restart to use this construction.
