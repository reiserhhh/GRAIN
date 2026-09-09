#!/usr/bin/env python3
"""Publish aggregate-only results of the third navigation loop."""

from __future__ import annotations

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


def summarize():
    """Rebuild the report from completed, frozen intervention packets."""
    nav = BASE / "navigation_v1"
    folders = {
        "natural": nav / "NAV-04/natural_evidence_r1",
        "independent": nav / "NAV-04/natural_programs_r1",
        "identity": nav / "NAV-05/query_identity_r1",
        "time": nav / "NAV-07/temporal_order_r1",
    }
    for folder in folders.values():
        if read_json(folder / "run_status.json")["status"] != "COMPLETE":
            raise ValueError("Finish collection before reporting")
    natural = pd.read_csv(folders["natural"] / "analysis/response_metrics.csv")
    main = natural.query("readout=='R_synth'")
    shallow = natural.query("readout=='context4_mean'")
    time = pd.read_csv(folders["time"] / "analysis/response_metrics.csv").query(
        "split=='held_scene'"
    )
    identity = pd.read_csv(folders["identity"] / "analysis/query_identity_matrix.csv")
    coding = identity.query(
        "split=='held_scene' and readout=='native' and change=='coding'"
    )
    tables = {
        str(p.relative_to(ROOT)): digest(p)
        for folder in folders.values()
        for sub in ("analysis", "logodds_analysis_r2")
        for p in (folder / sub).glob("*.csv")
    }
    tables.update(
        {
            str((folder / "predictions.csv").relative_to(ROOT)): digest(
                folder / "predictions.csv"
            )
            for folder in folders.values()
        }
    )
    write_json(
        nav / "round3_evidence.json",
        {
            "status": "ROUND3_DELIVERED_GOAL_ACTIVE",
            "sources": tables,
            "script_sha256": digest(Path(__file__)),
            "human_label_fitting": False,
        },
    )
    report = f"""# GRAIN navigation round 3: evidence operation and query identity

2026-09-09. NAV goal remains **ACTIVE**. These packets locate which parts of the
text-to-judgment computation already transport and which the shallow programs lose.

**Finding:** a function constructed on synthetic events follows natural-author
evidence changes across twelve held/evaluation-role PANDORA authors. Extraversion
response r=.928-.971; low-state r=.860-.990; T/F r=.646-.912. These are correlations
of paired **model responses**, not human personality-prediction accuracy. The
function still receives actual P/Q from the full prefix. This is evidence for a
portable local reader and a concrete target for the independent upstream program.

## 1. Natural evidence response: NAV-04

The twelve authors came from an existing exposed evaluation-role source, not a
new lockbox or construction set. Each has twelve excerpts. A frozen lexical rule
selected the first first-person proposition, without model attribution or labels.
The original excerpt body is retained inside explicit wrappers: my experience vs
a friend's, endorse vs do not endorse, today vs several years ago. Removal drops
the same selected excerpt. Other excerpts remain unchanged. These are controlled
counterfactual contexts, not verified facts about the author; endorsement wrapping
is not identical to a linguistic negation transformation.

288 native/PQ collections took432.94seconds. Three targets, eight conditions and
twelve authors were retained without success-based filtering. Synthetic/real
functions were unchanged. Independent candidates made288 shallow calls in
133.95seconds. Bootstrap resamples the12 authors, not individual edits.

{
        markdown_table(
            main,
            [
                "query",
                "axis",
                "r",
                "r_lower",
                "r_upper",
                "mae",
                "mean_abs_native_delta",
                "mean_abs_readout_delta",
                "sign_agreement",
            ],
        )
    }

The strongest gain in knowledge is not just level agreement: changing evidence
changes the fixed function in the same direction/pattern as the native observer.
Keep magnitude and sign columns: low-state endorsement r=.860 has only8/12
matching signs and its signed mean differs; correlation alone would hide this.

## 2. What the independent program loses

The token-only and four-block contextual operators are frozen round1/2
candidates, not retrained on these edits. All1/4/8-depth results remain in the
artifacts; the four-block comparison below was fixed before this packet.

{
        markdown_table(
            shallow,
            [
                "query",
                "axis",
                "r",
                "mean_abs_native_delta",
                "mean_abs_readout_delta",
                "sign_agreement",
            ],
        )
    }

Most shallow effects have too little amplitude. For low-state endorsement/time,
four-block response r=.700/.633 is present, but absolute effects are about.0005
against native.0070/.0054. T/F's useful static human association from round1 does
not imply evidence reasoning equivalence. Next develop the early-to-late
conditional computation, rather than a better human-label regressor.

## 3. Query wording and response coding: NAV-05

A genuine2x2 instrument factorial used the same96 worlds: binary Big5-E vs binary
MBTI-E/I wording, each answered with0/1 or A/B (positive=1/B). Four question-token
regions each received **their own** two synthetic references.456 full-model calls
(including reference gradients) took107.92seconds. Native and own-reference
predictions were scored without reusing another question's G coordinates.
Original continuous Big5 and original MBTI instruments remain unchanged.

{markdown_table(coding, ["query_a", "query_b", "axis", "r", "mae"])}

Held MBTI candidate mass stays essentially1, yet letter scores are about.718-.919
and digit scores about.088-.280. This is an output-coding effect, not parsing loss.
Construction-only calibration in logit coordinates separates a prior from evidence:

~~~text
d[q,c](x) = logit_positive(x) - logit_negative(x)
p[q,c](x) = sigmoid(d[q,c](x))
offset[q] = mean_construction(d[q,letters] - d[q,digits])
p_cal[q,letters](x) = sigmoid(d[q,letters](x) - offset[q])
~~~

Canonical decomposition: logodds_analysis_r2. Fixed offsets are1.9828(Big5) and
3.1654(MBTI). Held probability MAE between codes falls .1874 -> .0509 and
.6650 -> .0530. This calibrates one instrument to another, not to human truth.
Big5 logit-level code agreement is r=.9702; MBTI is .3904.

The residual is informative: MBTI actor-edit logit-response agreement is r=-.171,
while opportunity-response agreement is .885. A constant output prior explains
much of the score shift but not all query-specific evidence computation.
Target/coding identity therefore belongs inside the mathematical function.

Prompt-format sensitivity motivates this separation; see
[Sclar et al.](https://arxiv.org/abs/2310.11324). The numeric decomposition is our
local observer result, not an implication of that paper.

## 4. Time labels versus presentation order: NAV-07

Six event pairs contain negative and positive evidence. Factorially vary which
event is today versus six months ago, and independently reverse display order.
Two scenes construct the new present-focused question's references; four scenes
are held.24 worlds x2 questions plus18 reference calls =66calls,19.00seconds.
Event time is an experimental fact, not a synthetic personality label.

{
        markdown_table(
            time,
            [
                "query",
                "readout",
                "axis",
                "r",
                "r_lower",
                "r_upper",
                "mean_native_delta",
                "mean_readout_delta",
                "mae",
            ],
        )
    }

The present-focused function follows time-assignment effects at r=.854
[.669,.984] and order effects at.751[.617,.957]. Its native mean absolute
time/order changes are.079/.057; order is not negligible. Native mean
time-by-order interaction is-.082 for the present-focused question and-.103 for
the general query. The input relation is **query x event time x presentation
order**, not one universal recency weight.

This is a four-held-scene local result. The earlier46-person paired post-state
material comparison remains a separate observational link.
[Prompt permutation sensitivity](https://aclanthology.org/2022.acl-long.556/)
motivates checking order, but concerns demonstration ordering, not our event-time
manipulation; the mechanisms must not be conflated.

## 5. Mathematical update and next decision

~~~text
text + target + actor/endorsement/opportunity/time/order
  -> contextual state H_k
  -> target-conditioned evidence operation T
  -> P_hat,Q_hat (route A) OR z_hat1,z_hat2 (route B)
  -> fixed affine function
  -> instrument output scale/prior
  -> separate, frozen human external connection
~~~

Next implement a construction-only early-state residual operator (constant,
affine and one fixed quadratic kernel), generate actual P/Q coordinates, and
immediately repeat held-world, natural-edit and real-author comparisons. This asks
whether missing conditional operations can be represented by a reduced state map.
No full-model training, human-label tuning or MEPS fitting is needed.

NAV-04,05,07 now have actual first-packet closure. Useful-state construction,
real GRAIN group responses, fixed author norms, exact-prefix cache and runtime
packaging remain active. Earlier positive findings remain in force.

## Artifacts and replay

Private artifacts under results/grain_vector_mechanisms/navigation_v1:
- NAV-04/natural_evidence_r1: frozen source/edit designs, native/PQ and references.
- NAV-04/natural_programs_r1: independent token and contextual predictions.
- NAV-04/natural_evidence_r1/analysis: paired effects/components, metrics and CIs.
- NAV-05/query_identity_r1:2x2 instruments, own references and comparison matrix.
- NAV-05/query_identity_r1/logodds_analysis_r2: construction offsets/held diagnostics.
- NAV-07/temporal_order_r1/analysis: time/order factorial and interaction terms.
- round3_evidence.json: aggregate source hashes and report-code provenance.

Replay: run_grain_evidence_packets.py prepare/collect/evaluate with
--packet identity|evidence|time; evaluate_grain_evidence_programs.py;
analyze_grain_evidence_packets.py; analyze_grain_query_logodds.py.
Analysis requires a fresh output directory rather than overwriting completed
evidence. No raw author text, token arrays, keys or labels are published.
"""
    path = ROOT / "reports/grain_vector_mechanisms/NAVIGATION_ROUND3_REPORT.md"
    path.write_text(report, encoding="utf-8")
    print(path)


if __name__ == "__main__":
    summarize()
