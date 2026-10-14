# GRAIN Mechanism Supplement: Evidence Provenance Map

**Status:** internal reproducibility companion to the trilingual mechanism supplement
(`GRAIN_MECHANISM_SUPPLEMENT_{EN,JA,ZH}_V1.md`) and the package index
(`GRAIN_MECHANISM_THEORY_PACKAGE_V1.md`). Every number below was located in the named
artifact by the 2026-09-13 independent audit (`CLAUDE_TODO_CLOSURE_AUDIT.md`); rounding
follows the primary report. Simulated authors are declared-world parameter groups, not people.

| Supplement statement | Primary artifact (under `results/grain_vector_mechanisms/mechanism_todo_v1/`) | Value used | Boundary |
|---|---|---|---|
| Three-layer reading contrast (held wording) | `proposition_response_r1/completion/summary.json` | content r=.403131/MSE=.045369; binding r=.894985/MSE=.018183 (15646/18432); full-event r=.895985/MSE=.011838 | 192 simulated parameter groups, paired wordings |
| Selective misread shift | `proposition_response_r1/audit/selection_bias.csv`, `audit/summary.json` | mean |shift| .035705, max .152778; envelope widths .302300/.523245 | finite material ranges, not CIs |
| Version ledger vs. mean baselines | `dialogue_claim_ledger_r1/summary.json` | 72/72 operations and scores; MAE .958333/.875/0 | authored public dialogues |
| Content collision | `evidence_identity_r1/content_collision.json` | .993314 vs. targets +/-.165707 | source-calibrated readings |
| Abstract identity query failure | `evidence_identity_r1/observer_sanity.csv`, `composition_metrics.csv` | .000406 vs. .998065; held TF-IDF r .619 vs. typed NLI .166 | fixed NLI reader |
| Evidence-history comparison | `evidence_history_r1/evaluate/comparison.csv`, `update_witnesses.json`, `square_root_repair_freeze.json` | level MSE .026027/.035029/.044062; coverage .9688/.8750/.7383; witness lower bound .0762939; repair 7.88e-7 -> 5.86e-12 | 3 seeds, 768 application groups |
| Choice/opportunity confound | `choice_design_r1/choice_metrics.csv` | r .917/.930 vs .771/.796; RMSE 1.47/1.53 vs .633/.640 | simulated selection worlds |
| Response sharing | `response_sharing_r1/evaluate/comparison.csv` | crossed r .8414/MSE .044659 vs .156086; smooth .014255 vs .014327 | three declared worlds |
| Finite query-moment action | `query_moment_finite_r1/contract.json`, `evaluation/` | -33.3%/-25.1% (.02), -46.7%/-22.2% (.10); reversal 16/16 | frozen Qwen logits, 8 historical authors |
| Joint conditional observation | `joint_conditional_observation_r1/evaluate/metrics_by_seed.csv` | r .3578 / .0408 / .3594 | post-result product channel |
| Shared/specific query actions | `shared_query_actions_r1/component_actions.csv` | P share 73.5%-80.5%; 94.7% retained | 560 model calls |
| Independent program routes | `query_program_actions_r1/independent_audit.json`, `summary.json` | 1.55e-15; 98.6%; .004379 vs .288880 | 84 inputs |
| Temporal targets | `temporal_measurement_r1/metrics.csv` | .30905 -> .14319; coverage .69271 / .96875 | known dynamics |
| New-condition design | `new_condition_design_r1/evaluate/final_comparison.csv`, `identifiability_identity.json`, `next_observation_loss_surface.csv` | -25.9%/-30.0%; Var(d)=.034133; .036596 -> .010940 | greedy oracle reference |
| Pilot transport | `pilot_transport_r2/gaussian/metrics.csv`, `text/`, `gaussian/fits.json` | .333/.557 -> .969/.964; .6748 -> .2960 | 33 main fits; 3 bootstrap limits retained |
| Explanation chain | `explanation_chain_r1/summary.json` | +.013705 (8/12); +.076500 (12/12) | authored stimuli; frozen F03 |
| Two instruments | `local_instrument_r1/evaluation/metrics.csv`, `audit.json` | .2446 -> .1189; .5575 -> .6145; .4898 -> .2052 (coverage .7656); 3.77e-11 | 192 new simulated authors per instance |
| Handbook rebuild | `claude_todo_closure_20260913/d02_handbook_rebuild/rebuild_vs_r1_numeric_comparison.json` | all differences 0; audit PASS | same machine/dependencies |
| Cross-agent construction | `claude_todo_closure_20260913/d02_independent_construction/attempt1_comparison.json`, `attempt2_cold_vs_cached.json` | source moments diff 0; cold vs cached 1.44e-12; 22 NLI pairs | builder had development context |
| Natural coverage | `claude_todo_closure_20260913/m01_m09_dialogue_input_entry/natural_coverage/coverage.json` | 0/48 targets computable; 11 past-report, 4 intention, 1 request frames; 0 target-relevant | 12 licensed sentences, counts only |

Provenance rules follow `GRAIN_THEORY_EVIDENCE_PROVENANCE.md`: intervals and decision rules travel
with the point estimate; a repository artifact is an internal source; technical structure,
comparability, construct validity and use remain separate claim levels.
