# S20: a shared functional direction transfers and connects to source N

2026-09-08. COMPLETE. Frozen reference from24 authors, cached scoring for213
authors, actual intervention on24 different authors,48 forwards,0 backwards,
67.24seconds after loading. No human-label fitting, calibration or sign selection.

## Mainline result

The same mean integrated gradient predicts new authors' actual pairing-removal
effect at r=.9431, bootstrap95%CI[.9009,.9809], MSE/effect-energy=.1653 and
sign agreement22/24. Native score and feature replay errors are0.

On the189 source authors excluded from gradient estimation, the equal-view
mathematical readout correlates with the existing N questionnaire at r=.2000,
p=.00580, Fisher95%CI[.0589,.3332], author-bootstrap CI[.0611,.3306].
This is an observed source personality association for the operational direction,
not merely reconstruction of an internal score.

| Cohort | Functional mean N r | Functional view0 N r | S9 C N r | S12 learned C-reader N r |
|---|---:|---:|---:|---:|
| Primary189, outside reference | .199990 | .179990 | .198936 | .157481 |
| Reference24, secondary | -.007903 | .008610 | .555617 | .150574 |
| All213, secondary | .174895 | .168963 | .211552 | .142869 |

All cohorts and both pooling choices were fixed before outcomes. The primary
cohort is new to gradient estimation, not a new untouched PANDORA benchmark.
The wider source labels have appeared in previous development analyses.

Primary mean-readout versus S12-reader delta r=.04251, paired95%CI[-.15215,.23773].
All213 delta=.03203[-.12862,.22838]. The useful result is the direct source
association and functional transport, not a demonstrated superiority claim.

## What the formula does

Compute g_ref=mean of the24 integrated layer17 suffix gradients. For each author's
three native pairing vectors d_uv from the frozen prefix, return
z_u=mean_v g_ref^T d_uv. No questionnaire, held author's gradient, final decoder
suffix or generated explanation is needed by this reader. S20 uses cached
prefix vectors; S13 separately verified that prefix-only extraction reproduces
these vectors. It does not remove pretrained semantic computation from the prefix.

The functional mean's correlation with C is.2212 on primary189 and.2093 on all213.
Thus it is not simply a high-fidelity substitute for C. Its all213 source
covariance is.000055, versus C's.001046, in their raw contrast units.
These contrasts come from different interventions; do not interpret the ratio
as a causal percentage of C explained or use it to claim that all useful paths
were captured. Moderate internal-C correlation can coexist with useful Y linkage.

This now links: mean-null source signal(S9), joint KV access(S10/S12),
operative gradients rather than regression weights(S18/S19), and a transported
mathematical reader with source association(S20). Keep the existing T/F positive
control; no model update or release-scorer replacement is implied.

## Mathematical interpretation

For a context-dependent suffix F_u,
E_u=F_u(h_u)-F_u(h_u-d_u)=g_bar_u^T d_u.
Transport uses g_ref and error (g_ref-g_bar_u)^T d_u.
This applies the internal-path version of
[integrated-gradient completeness](https://proceedings.mlr.press/v70/sundararajan17a.html),
with finite shared gradient geometry motivated by
[active subspaces](https://arxiv.org/abs/1408.0545).
The transported correlation is empirical; no global approximation theorem is
claimed for arbitrary contexts or languages.

The next mechanism question is the location of query specificity. Is the shared
direction generic to numeric judgments, while the query-conditioned KV pairing
does most psychological selection? Or do the final gradients have substantial
trait-specific components? A matched-format cross-query response matrix can
distinguish these. Do not use the present source Y to choose an ensemble first.

## Artifacts and reproduction

Code/protocol4bbbd7c. Output directory:
results/grain_vector_mechanisms/functional_transfer_s20_r1/

It contains immutable manifest, private_reference.npz,
private_scores_before_outcomes.csv, private_packets, run_status.json,
private_source_connections.csv, metrics.csv, source_paired_intervals.csv,
summary.json. All private files remain Git-ignored.

Use parent venv Python:
- scripts/run_grain_functional_transfer.py prepare
- scripts/run_grain_functional_transfer.py run
- scripts/run_grain_functional_transfer.py evaluate

Preparation freezes the reference and all predictions before new endpoints and
outcome joins. A rerun needs a new directory; do not overwrite old analysis.
The52 focused S6-S20 tests pass, including actual MPS gradient export and new
author-isolation tests. Two existing SWIG deprecation warnings remain.
No protected bridge/MEPS questionnaire, lockbox, cloud upload or remote push.
