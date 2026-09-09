#!/usr/bin/env python3
"""Aggregate replayable context, route-exchange and state-opportunity discoveries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import (
    BASE,
    digest,
    read_json,
    write_json,
)
from scripts.summarize_grain_navigation_round import markdown_table


def summarize(report):
    """Require real completion of each cited packet and publish only aggregate values."""
    nav = BASE / "navigation_v1"
    dirs = {
        "exchange": nav / "NAV-06/endpoint_exchange_r1",
        "program": nav / "NAV-03/context_pullback_r2",
        "worlds": nav / "NAV-10/context_worlds_r2",
        "real": nav / "NAV-10/context_external_r2",
        "state": nav / "NAV-07/state_opportunity_r1",
        "atlas": nav / "NAV-12/context_atlas_r1",
    }
    for k in ("exchange", "worlds", "real"):
        if read_json(dirs[k] / "run_status.json")["status"] != "COMPLETE":
            raise ValueError("Packet incomplete: " + k)
    if (
        read_json(dirs["state"] / "summary.json")["status"]
        != "COMPLETE_REAL_OBSERVATIONAL_PACKET"
    ):
        raise ValueError("State analysis incomplete")
    if (
        read_json(dirs["atlas"] / "summary.json")["status"]
        != "COMPLETE_FIRST_LOCAL_ATLAS"
    ):
        raise ValueError("Atlas analysis incomplete")
    world = pd.read_csv(dirs["worlds"] / "metrics.csv").query("split=='held_scene'")
    world = world[world.method.str.endswith("_mean")]
    real = pd.read_csv(dirs["real"] / "metrics.csv")
    real = real[real.method.str.endswith("_mean")]
    kernels = pd.read_csv(dirs["exchange"] / "response_kernel.csv")
    kernels = kernels[kernels.query_a < kernels.query_b]
    exchange = pd.read_csv(dirs["exchange"] / "cross_vs_own.csv").query(
        "group=='shared'"
    )
    state = pd.read_csv(dirs["state"] / "paired_material_comparison.csv").query(
        "reference=='R_synth' and readout=='score'"
    )
    state = state[state.criterion.str.endswith("_adapted_dejected5")]
    timing = pd.read_csv(dirs["state"] / "occasion_specificity.csv").query(
        "reference=='R_synth' and readout=='score'"
    )
    interactions = pd.read_csv(dirs["atlas"] / "response_transfer.csv").query(
        "split=='held_scene' and effect=='actor_x_opportunity'"
    )
    interactions = interactions[interactions.method.str.endswith("_mean")]
    overlap = pd.read_csv(dirs["atlas"] / "overlap_consistency.csv").query(
        "split=='held_scene'"
    )
    program = read_json(dirs["program"] / "program.json")
    state4 = real.query("dataset=='free_chat' and method=='context4_mean'").iloc[0]
    tf4 = real.query("query=='TF_cont' and method=='context4_mean'").iloc[0]
    evidence = {
        str(p.relative_to(ROOT)): digest(p)
        for folder in dirs.values()
        for p in folder.glob("*.csv")
        if not p.name.startswith("private_")
    }
    write_json(
        nav / "round2_evidence.json",
        {
            "goal_status": "ACTIVE",
            "source_sha256": evidence,
            "report_script_sha256": digest(Path(__file__)),
            "scope": "local mechanism development and existing exposed external cohorts, no human label fitting",
        },
    )
    text = f"""# GRAIN navigation round 2: context and functional exchange

2026-09-09. Overall NAV-01--12 goal remains **ACTIVE**. This is a mechanism
development result, not a model-selection leaderboard or completion of route A.

**Key finding:** the four-block contextual program connects to the 46-person
free-chat low-state criterion at r={state4.function_external_r:.4f}
[{state4.function_r_lower:.4f}, {state4.function_r_upper:.4f}], with original
function fidelity r={state4.full_function_r:.4f}. This recovers part of the state
signal missing from static input-token geometry. T/F behaves differently: its
four-block human AUC is {tf4.function_auc:.4f}, below the earlier token-only
semantic/order candidate's .7047. We preserve both paths rather than forcing all
targets into one representation recipe. These are associations of the frozen
candidates, not a significant paired improvement claim. No human labels were
used to construct or select coefficients.

## 1. What changed

The token-only first-order operator from round 1 linked to T/F but lost
extraversion and low-state behavior. This round retains exact **1, 4 or 8 early
contextual blocks**, pulls the two fixed projections back to those states, and
tests fixed local charts. Construction uses only six synthetic reference texts.
There is no fitting on human labels, no MEPS construction and no original lockbox
access. Every method, including both individual charts, was scored before the
external label join. Reported mean-chart results are the fixed common comparator,
not a method selected on real scores; all twelve candidates are in the CSVs.

The complete runtime comparison loads eight extracted blocks and records states
at depths 1/4/8 in a single forward. A depth-1 score depends only on the first
state, but total measured batch cost still includes eight blocks. The original
28-layer model, its language-model head and the full 17-layer P/Q input are not
loaded at scoring time. This is an **extracted-neural-program**, not a claim of
Transformer-free inference. Its retained weight file is
{program["weight_bytes"] / 1e9:.3f} GB; pretrained knowledge remains essential.

## 2. Explicit computation

For early cut k, let H_k(x,q) be the retained contextual residual sequence.
At synthetic chart j, the two fixed projection derivatives are

```
J[j,a,k] = derivative of z_a with respect to H_k at reference text x_j
z_hat[j,a,k](x) = z_a(x_j)
                 + <J_evidence, A_j(H_k(x)) - A_j(H_k(x_j))>
                 + <J_query, H_k(x)[P] - H_k(x_j)[P]>
                 + <J_endpoint, H_k(x)[Q] - H_k(x_j)[Q]>
f_hat[k](x,q) = b_q + sum_a sum_j omega_j(x) z_hat[j,a,k](x)
```

Evidence correspondence A uses cosine softmax with fixed tau=8 and relative
position penalty=4. P and Q are exact **within-query** token-slot matches.
Weights are either 1/2 or softmax(8 * evidence-mean cosine to each reference).
No manual scene truth is required by the scorer. Chain-rule derivatives and
reference identity are exact; transporting those derivatives to new contexts is
the tested local approximation. Full-state P/Q reconstruction remains open.

Mathematical provenance: gradient-based attribution motivates the derivative
experiment, but this point-Jacobian operator is **not Integrated Gradients** and
does not inherit its path-integral guarantees. See
[Sundararajan et al., 2017](https://proceedings.mlr.press/v70/sundararajan17a.html).
Construction-only gradient route ranking followed by actual interventions follows
the distinction used in
[Syed et al., 2024](https://aclanthology.org/2024.blackboxnlp-1.25/).
Cross-query endpoint-delta transport and the chart comparison are our tested
combinations, not claims that those papers proved this personality mechanism.

## 3. Held controlled scenes

Here r measures agreement with the full frozen function, **not human validity**.
Each row has 24 renderings of only three held scene skeletons. Magnitude errors
are reported alongside correlation; better ordering does not imply score identity.

{markdown_table(world, ["query", "method", "r", "mae", "native_r"])}

The relationship between retained depth and fidelity is not monotone. In
particular, four contextual blocks recover much more low-state and T/F ordering
than the one-block candidate, but the scalar levels still have substantial bias.
More context alone is not a substitute for an appropriate approximation to the
remaining nonlinear computation.

## 4. Immediate real-text connection

Same already-exposed cohorts as round 1: 213 Big5-source authors, 64 MBTI-source
authors and 46 counseling participants, with three source views averaged per
author. These are exploratory endpoint checks, not a new full PANDORA benchmark.
MEPS criteria are never used to update paths, gates, references or weights.

{markdown_table(real, ["dataset", "query", "method", "n", "full_function_r", "function_external_r", "function_auc"])}

Complete intervals and paired native comparisons are in `NAV-10/context_external_r2/metrics.csv`.
Model fidelity and human association remain separate outcomes. Retain useful
local links without presenting every approximation as the same inference program.

## 5. Shared route identity is not shared use

162 actual native-checked interventions apply the source query's **edit delta**
at matched attention-head/MLP coordinates, only at the last input position. This
avoids inventing alignment between query strings with different numbers of P rows.
Within-query deltas on exactly the same site group are the transport control.
There are six held event edits from three scenes, not 162 independent subjects.

{markdown_table(kernels, ["query_a", "query_b", "candidate_jaccard", "endpoint_delta_cka", "native_response_r"])}

{markdown_table(exchange, ["target_query", "source_query", "native_r", "mean_abs_own", "mean_abs_cross", "sign_agreement"])}

In the second table r compares cross-query versus own-query intervention effects;
it is neither personality correlation nor whole-model recovery. High CKA can
coexist with different signs and very small native effects. Shared sites are a
place to investigate, not a ready-made shared semantic axis. The remaining
NAV-06 real-author GRAIN-group analysis has not been replaced by these synthetic
conditions.

## 6. State opportunity, paired on the same criterion

Both material views are now compared with the **identical** questionnaire endpoint
and participant set. This removes the earlier apples-to-oranges endpoint timing
comparison. It does not randomize materials or turn observational differences into
causal effects. The three overlapping negative-state composites are not independent
replications of each other.

{markdown_table(state, ["criterion", "free_r", "direct_r", "delta_r_free_minus_direct", "delta_lower", "delta_upper"])}

{markdown_table(timing, ["view", "before_r", "after_r", "partial_r_after_given_before", "partial_p", "score_vs_criterion_change_r"])}

The free-chat score retains a relationship with the post-session state after
conditioning descriptively on the preceding state. The direct-answer material
does not show the same pattern. The joint view/time difference remains jointly
confounded; it is saved as such, not interpreted as isolated personality change.
The controlled time-marker/event-order arm of NAV-07 remains open.

## 7. Local charts and signed interactions

We retain the full rectangle rather than deleting its non-additive part:

```
I = f(other,assigned) - f(self,assigned)
    - f(other,voluntary) + f(self,voluntary)
```

{markdown_table(interactions, ["query", "method", "r", "mean_abs_target", "mean_abs_predicted", "sign_agreement"])}

{markdown_table(overlap, ["query", "cut", "chart_correlation", "mean_abs_chart_disagreement", "affinity_minus_mean_mae"])}

The local chart average and observable-affinity gate are both actually tested.
High scalar ordering does not ensure correct mixed interactions; those terms
remain the next operator-development target rather than being relabeled noise.
The first local-atlas packet is complete, not a universal smooth field claim.

## 8. Execution and next scientific decisions

- NAV-06 local exchange complete; GRAIN real-author grouping still open.
- NAV-03 three contextual depths implemented and connected to real text;
  route A effective P/Q state construction still open, not replaced by B.
- NAV-07 paired observational packet complete; controlled time/order arm open.
- NAV-12 first local chart/interaction comparison complete; retain failures.
- Next prioritize actual context-conditioned interaction construction, natural
  real-text edits and query-identity factorial. No real-label parameter grid.
- NAV-08 reference, NAV-09 exact-prefix reuse and NAV-11 raw-text export retain
  their original scope in the active goal register.

The first contextual export (`context_pullback_r1`) failed before any evaluation
because its layer_types metadata still described 28 layers after cropping to 8.
The config helper was corrected, regression-tested and a fresh `r2` was built.
Six failed-export construction calls remain accounted for; no result was erased.

## Artifacts and replay

All result paths below are relative to `results/grain_vector_mechanisms/navigation_v1/`.
The public report contains only aggregates. Private author tokens, questionnaire
joins and pretrained weights remain gitignored locally.

| Packet | Directory | Key output |
|---|---|---|
| Shared mechanism | NAV-06/endpoint_exchange_r1 | shared_specific_edges.csv, cross_vs_own.csv, response_kernel.csv |
| Shallow program | NAV-03/context_pullback_r2 | program.json, context.safetensors, charts/ |
| Controlled check | NAV-10/context_worlds_r2 | metrics.csv, prediction_freeze.json |
| Real check | NAV-10/context_external_r2 | metrics.csv, prediction_freeze.json |
| State opportunity | NAV-07/state_opportunity_r1 | paired_material_comparison.csv, occasion_specificity.csv |
| Local atlas | NAV-12/context_atlas_r1 | conditional_signed_terms.csv, response_transfer.csv, overlap_consistency.csv |

Use the existing parent-project Python environment. Constructors require new run
IDs; neural evaluations are resumable with unchanged source/code hashes.

```bash
python scripts/analyze_grain_navigation_sharing.py
python scripts/run_grain_context_program.py construct --program <fresh-program-dir>
python scripts/run_grain_context_program.py worlds --program <program-dir> --out <fresh-worlds-dir>
python scripts/run_grain_context_program.py external --program <program-dir> --out <fresh-external-dir>
python scripts/analyze_grain_state_opportunity.py --out <fresh-state-dir>
python scripts/analyze_grain_local_route_atlas.py --out <fresh-atlas-dir>
python scripts/summarize_grain_navigation_round2.py
```

Aggregate provenance is recorded in `round2_evidence.json`; code tests cover
gradient finite differences, exact shallow/full early-state replay, chart identity,
different-query-length endpoint exchange, signed factorial accounting, and paired
participant/criterion alignment. Full goal closure requires the remaining ledger
items; this report does not mark them complete.
"""
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text, encoding="utf-8")
    print(str(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports/grain_vector_mechanisms/NAVIGATION_ROUND2_REPORT.md",
    )
    a = p.parse_args()
    summarize(a.report)
