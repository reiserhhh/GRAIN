# GRAIN open-stream operators: choosing the right comparison

Use this extension when event times, pool size, membership or register change.
The original scoring path remains available; do not switch all six options on
by default. Each option answers a different question. All imports below use
`grain_core`. Lower losses are better.

| Question | Operator | What to supply and what to read |
| --- | --- | --- |
| G-1: Is apparent disagreement just sampling variation? | `stream_precision`, `liquid_reference_error`, `liquid_common_range`, `dual_view_increment.assess_precision` | Supply positive counts and **calibrated** event variances. For nested pools supply error cross-covariance as well. Precision-weighting the source population and shrinking each unit are different estimands. In a common-range chart `center` is the Euclidean feasible-ball center; `precision_weighted_point` is a separate summary. `event_sigma` is an isotropic event SD in each canonical block, not raw-text SD. A fixed multiplier is a sensitivity check; use `path_quantile` on exchangeable disjoint calibration paths for a pathwise coverage claim. |
| G-2: Did anything arrive since the cutoff? | `ConstrainedRiccatiCell.event_path`, `window_delta_at_cutoff`, `counterfactual_window_delta`, `liquid_axis_series.encode_bipolar_origin` | Use zero-order hold for a sustained exposure, idle-then-impulse for discrete arrivals. Pick the time unit, start, cutoff and pulse convention in advance. An empty arrival window can have zero event contribution while its prior state continues evolving. The first window helper includes its left boundary; the second uses an open left endpoint. |
| G-3: Who stopped contributing here? | `selective_silence.assess`, `compute_selective_silence_panel` | Fix an observed feed and a past roster. Retained here, active elsewhere, and nowhere observed partition exposure mass. Do not feed outages as empty windows. The signed flux weights prior readings, not imagined current ones. `online_avoid_rate` and `offline_absent_rate` are legacy aliases; prefer `active_elsewhere_rate` and `not_observed_rate`. Neither a motive nor a MAR/MNAR classification is observed. |
| G-4: Did people change, or did the mix change? | `open_cohort.decompose`, `decompose_open_cohort_step`, `PrefixHygiene` | Supply event counts and mean coordinates by person in two observed bins. Separate within-person movement, stayer weights, entry and exit. Report the convention (symmetric vs sequential) and overlap. A raw aggregate change still exists with no stayers, but a within-person comparison then has no reference. Prefix filters see no future records; choose lifetime vs rolling duplicate rules intentionally. |
| G-5: Does a feature change meaning with register? | `pragmatic_register.CrossFittedRegisterCalibrator`, `mtmm_validity_audit`, `stream_readout.crossfit_register` | Compare additive and register-interaction readouts with disjoint units. Penalties for reported held-author predictions are selected inside each training fold. Use an annotated endpoint and report each behavior separately. Fixed bilingual act markers provide a reproducible linguistic baseline, not personality labels. |
| G-6: Should an individual's history travel to this condition? | `transport_gate.ConditionScopedRiccatiStream`, `evaluate_transportability_gate`, `stream_readout.ScopedInnovation` | Emit a pre-arrival prediction, then innovation, then update. Compare paired carry vs fresh losses on calibration units; apply that decision without using evaluation labels. Report both raw alternatives and the gated loss. Different endpoints may warrant different gates. |

## A minimal downstream check

```python
from grain_core.open_cohort import decompose
from grain_core.selective_silence import assess

before = {"A": (9, 0.0), "B": (1, 1.0)}
after  = {"A": (1, 0.0), "B": (9, 1.0)}
d = decompose(before, after)
assert abs(d["change"] - 0.8) < 1e-12
assert d["within"] == d["entry"] == d["exit"] == 0

s = assess([[("A", "board", 2.0), ("B", "board", -1.0)]],
           [("A", "another_board", 1.0)], "board")
assert s["active_elsewhere"] == s["not_observed"] == 0.5
```

For the full pool and its screened subset, record sampling units and shared
events. Do not plug independent-reader errors into nested-reader estimates.
Give simple and complex comparisons the same observations, history and uncertainty
budgets. If both permit the same decision, report that before interpreting
the more complex trajectory.

## Evidence and reproducibility

In the source-only PANDORA panel, total mean log-length change went in the
opposite direction to within-person change in 14 of 66 comparable transitions.
Another 23 had no stayer reference. This is a practical reason to show both
lines rather than interpreting the pool mean as change in a typical speaker.
In 80 held R23 venue-window pairs, adding activity elsewhere and bridge-share
change to current retention and log incumbent count reduced next-window
retention MSE from 0.006897 to 0.006481. Signed activity elsewhere did **not**
improve next-window length change on the separate source-only panel.

- [Independent replay and final time split](../reports/open_stream_replay.json): public source observations and synthetic calibration; [command-line harness](../examples/open_stream_replay.py) accepts local paths and emits aggregates only. The program's `--help` documents the inputs.
- [Integrated R23 aggregate results](../reports/open_stream_r23_integrated.json): 400,000 PANDORA comments and 121 RealPersonaChat speakers. The public text itself is not redistributed. The original R23 experiment is maintained in Project Persona.
- The R23 retention test uses a retrospective choice of venues and calendar; its held windows are a time split on that panel, not a live market backtest. The bridge measure concerns changes in where authors participate. A mean ratio of component norms is not a percentage of variance explained.
- Tests: `python -m pytest -q tests/test_open_stream_contracts.py tests/test_open_stream_upgrades_g1_g6.py`.
