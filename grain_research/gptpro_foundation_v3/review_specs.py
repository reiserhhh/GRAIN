#!/usr/bin/env python3
"""Review card specifications (rules bound to evidence). One function per card."""

from __future__ import annotations

import json
import subprocess
import sys

from grain_research.gptpro_foundation_v3.review_cards import Card, Evidence, Rule
from grain_research.gptpro_foundation_v3.review_derived import (
    load,
    mean_ci,
    paired,
    wilson,
)

R = "results/grain_gptpro_foundation_v3"
F02 = f"{R}/F02/recovery_full_r2"
F02_AUTH = f"{F02}/author_metrics.csv"


def f02_derived(root):
    """Oracle gaps / equivalence for C1, C2; C3 estimation errors and bound coverage."""
    rows = []
    key = ("seed", "author_id")
    for world, cand in (("c1", "personal_condition"), ("c2_xor", "b1"), ("c2_iid", "b1")):
        res = paired(root, F02_AUTH, key, "candidate", cand, "oracle_scoring_only", margin=0.005, world=world, role="application", metric="brier")
        rows.append({"quantity": f"{world}:{cand}-oracle_scoring_only:brier", **res})
    for prefix in ("12", "48"):
        for metric in ("pi_squared_error", "q_squared_error", "publication_squared_error"):
            res = mean_ci(root, F02_AUTH, key, world="c3", role="application", candidate="opportunity_visible", metric=metric, prefix=prefix)
            rows.append({"quantity": f"c3:opportunity_visible:{metric}:prefix{prefix}", **res})
        res = paired(root, F02_AUTH, key, "candidate", "opportunity_masked", "opportunity_visible", world="c3", role="application", metric="publication_squared_error", prefix=prefix)
        rows.append({"quantity": f"c3:masked-visible:publication_squared_error:prefix{prefix}", **res})
        cover = load(root, F02_AUTH, world="c3", role="application", candidate="opportunity_masked", metric="q_plugin_bound_contains_truth", prefix=prefix)
        k = sum(float(r["value"]) for r in cover)
        lo, hi = wilson(k, len(cover))
        rows.append({"quantity": f"c3:opportunity_masked:q_plugin_bound_coverage:prefix{prefix}", "n_units": len(cover), "mean": k / len(cover), "lo95": lo, "hi95": hi, "decision": "WILSON95"})
        empty = load(root, F02_AUTH, world="c3", role="application", candidate="published_only", prefix=prefix)
        rows.append({"quantity": f"c3:published_only:estimates_available:prefix{prefix}", "n_units": len(empty), "mean": sum(r["value"] != "" for r in empty)})
    for prefix in ("pi_squared_error", "q_squared_error"):
        res = paired(root, F02_AUTH, key, "prefix", "48", "12", world="c3", role="application", candidate="opportunity_visible", metric=prefix)
        rows.append({"quantity": f"c3:opportunity_visible:{prefix}:prefix48-prefix12", **res})
    return rows, [F02_AUTH]


def pytest_check(*files):
    """Extra check: run the named test files in a subprocess; record counts."""

    def run(root):
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *files],
            cwd=root, capture_output=True, text=True, check=False,
        )
        tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr[-200:]
        return f"pytest {' '.join(files)}", proc.returncode == 0, tail

    run.__name__ = "pytest_" + "_".join(f.split("/")[-1] for f in files)
    return run


def f02_eps_check(root):
    """Recompute epsilon non-transitivity from stored points."""
    d = json.loads((root / F02 / "exact_identification.json").read_text())["epsilon_nontransitivity"]
    a, b, c = d["points"]
    eps = d["epsilon"]
    ok = abs(a - b) <= eps and abs(b - c) <= eps and abs(a - c) > eps
    return "epsilon_nontransitivity_recomputed", ok, f"|a-b|={abs(a - b)} |b-c|={abs(b - c)} |a-c|={abs(a - c)} eps={eps}"


def f02_06() -> Card:
    """F02-06 construction rules per world."""
    pc = f"{F02}/paired_comparisons.csv"
    return Card(
        card_id="F02-06",
        package="F02",
        artifact="construction_rules.json",
        requirements=["R08", "R30", "A01"],
        depends_on=["F02-05"],
        derived=[f02_derived],
        extra_checks=[f02_eps_check, pytest_check("tests/test_grain_gptpro_f02.py", "tests/test_grain_gptpro_f02_recovery.py", "tests/test_grain_gptpro_f02_adverse.py")],
        rules=[
            Rule("C1-condition", "C1 conditional-response world, prefix 12, application",
                 "Record the condition with every event and score per-person condition cells; the per-person conditional table beats the source condition table.",
                 "SUPPORTED_LOWER_LOSS",
                 [Evidence(pc, "delta", {"world": "c1", "candidate": "personal_condition", "reference": "source_condition", "metric": "brier"}, expect=-0.050816),
                  Evidence(pc, "ci_high", {"world": "c1", "candidate": "personal_condition", "reference": "source_condition", "metric": "brier"}, expect=-0.048087)],
                 needs=["condition"], standard_method="per-person conditional frequency table (Laplace) is the best arm; its oracle gap is in derived_metrics (not declared sufficient)"),
            Rule("C1-mean-loses", "C1, application",
                 "A condition-blind person mean loses the conditional information: it is worse than even the source condition table.",
                 "INFORMATION_LOSS",
                 [Evidence(pc, "delta", {"world": "c1", "candidate": "personal_mean", "reference": "source_condition", "metric": "brier"}, expect=0.008315),
                  Evidence(pc, "ci_low", {"world": "c1", "candidate": "personal_mean", "reference": "source_condition", "metric": "brier"}, expect=0.007656)],
                 needs=["condition"], information="mean-only summaries discard the condition contrast"),
            Rule("C2-future", "C2 exact law",
                 "Histories equal in one-step law can differ in future law: TV 0 at horizon 1 but .5 at horizons 2-6, so the scoring state must carry the last two outcomes, not the last-step prediction.",
                 "EXACT_CERTIFICATE",
                 [Evidence(f"{F02}/future_equivalence.csv", "total_variation_00_11", {"horizon": "1"}, expect=0.0, tol=1e-12),
                  Evidence(f"{F02}/future_equivalence.csv", "total_variation_00_11", {"horizon": "2"}, expect=0.5, tol=1e-12),
                  Evidence(f"{F02}/future_equivalence.csv", "total_variation_00_11", {"horizon": "6"}, expect=0.5, tol=1e-12)],
                 needs=["history (order 2)"]),
            Rule("C2-population-table", "C2 xor and iid, prefix 12, application",
                 "When the transition law is shared across persons, the fixed source second-order table (B1) is the rule; person-level updating (B2, M) adds loss in both the xor and iid worlds.",
                 "HIGHER_LOSS",
                 [Evidence(pc, "delta", {"world": "c2_xor", "candidate": "b2", "reference": "b1", "metric": "brier"}, expect=0.021405),
                  Evidence(pc, "delta", {"world": "c2_xor", "candidate": "m", "reference": "b1", "metric": "brier"}, expect=0.011718),
                  Evidence(pc, "delta", {"world": "c2_iid", "candidate": "b2", "reference": "b1", "metric": "brier"}, expect=0.028630),
                  Evidence(pc, "delta", {"world": "c2_iid", "candidate": "m", "reference": "b1", "metric": "brier"}, expect=0.010861)],
                 needs=["history (order 2)"], standard_method="population second-order Markov table; its equivalence to the oracle scorer is tested in derived_metrics (margin ±.005 Brier, 90% CI)"),
            Rule("C2-standard-sufficient", "C2 xor and iid, prefix 12, application",
                 "The standard population second-order table reaches the oracle scorer within the pre-declared Brier equivalence margin (+-.005, 90% CI) in both C2 worlds; GRAIN person-level machinery adds nothing here.",
                 "KNOWN_METHOD_SUFFICIENT",
                 [Evidence("derived:c2_xor:b1-oracle_scoring_only:brier", "equivalent_90", expect=True),
                  Evidence("derived:c2_xor:b1-oracle_scoring_only:brier", "delta", expect=0.0000423, tol=5e-6),
                  Evidence("derived:c2_iid:b1-oracle_scoring_only:brier", "equivalent_90", expect=True),
                  Evidence("derived:c2_iid:b1-oracle_scoring_only:brier", "delta", expect=0.0001485, tol=5e-6)],
                 needs=["history (order 2)"], standard_method="population second-order Markov chain (sufficient under shared law)"),
            Rule("C1-oracle-gap", "C1, prefix 12, application",
                 "The per-person conditional table does not reach the oracle scorer: a cost of about .012 Brier remains at prefix 12 (not equivalent at +-.005). Whether more condition-specific events close it is not tested here.",
                 "HIGHER_LOSS",
                 [Evidence("derived:c1:personal_condition-oracle_scoring_only:brier", "delta", expect=0.011733, tol=5e-6),
                  Evidence("derived:c1:personal_condition-oracle_scoring_only:brier", "equivalent_90", expect=False)],
                 needs=["condition", "enough events per condition cell"], information="no prefix sweep was run for C1"),
            Rule("C3-published-only", "C3 exact law",
                 "Publication counts alone cannot identify opportunity rate pi or expression rate q: (pi,q)=(.8,.5) and (.4,1) both publish at .4; q is only bounded to [.4,1].",
                 "NOT_IDENTIFIED",
                 [Evidence(f"{F02}/exact_identification.json", key=["c3", 0, "published_rate"], expect=0.4, tol=1e-12),
                  Evidence(f"{F02}/exact_identification.json", key=["c3", 1, "published_rate"], expect=0.4, tol=1e-12),
                  Evidence(f"{F02}/exact_identification.json", key=["q_population_bounds_given_published_rate", 0], expect=0.4, tol=1e-12),
                  Evidence(f"{F02}/exact_identification.json", key=["q_population_bounds_given_published_rate", 1], expect=1.0, tol=1e-12),
                  Evidence("derived:c3:published_only:estimates_available:prefix12", "mean", expect=0, tol=0)],
                 needs=["opportunity denominator"], information="published-only view: no pi/q/publication estimates",
            ),
            Rule("C3-visible", "C3, opportunity visible",
                 "With opportunities recorded, pi and q are estimated per person and errors shrink with more events (prefix 48 vs 12).",
                 "SUPPORTED_FINITE",
                 [Evidence(f"{F02}/metrics.csv", "mean", {"world": "c3", "seed": "20260915", "role": "application", "candidate": "opportunity_visible", "metric": "q_squared_error", "prefix": "12"}, expect=0.013850),
                  Evidence(f"{F02}/metrics.csv", "mean", {"world": "c3", "seed": "20260915", "role": "application", "candidate": "opportunity_visible", "metric": "q_squared_error", "prefix": "48"}, expect=0.003152)],
                 needs=["opportunity denominator"]),
            Rule("C3-visible-prefix", "C3, opportunity visible, all seeds",
                 "Pooled over seeds, pi and q squared errors fall from prefix 12 to prefix 48 (paired, seed-stratified).",
                 "SUPPORTED_LOWER_LOSS",
                 [Evidence("derived:c3:opportunity_visible:pi_squared_error:prefix48-prefix12", "delta", expect=-0.012661, tol=5e-6),
                  Evidence("derived:c3:opportunity_visible:q_squared_error:prefix48-prefix12", "hi95", expect=-0.008048, tol=5e-6)],
                 needs=["opportunity denominator"]),
            Rule("C3-masked-bounds", "C3, opportunity masked",
                 "With opportunities masked, only the publication rate is estimable (error identical to the visible view) and q is reported only as a sample plug-in bound; the bound is not a confidence interval, and its empirical coverage (.92-.985 by seed/prefix; Wilson intervals in derived_metrics) is reported without a nominal level.",
                 "BOUNDS_ONLY",
                 [Evidence(f"{F02}/metrics.csv", "mean", {"world": "c3", "seed": "20260915", "role": "application", "candidate": "opportunity_masked", "metric": "q_plugin_bound_contains_truth", "prefix": "12"}, expect=0.92),
                  Evidence("derived:c3:masked-visible:publication_squared_error:prefix12", "decision", expect="IDENTICAL"),
                  Evidence("derived:c3:opportunity_masked:q_plugin_bound_coverage:prefix12", "mean", expect=0.925, tol=1e-9),
                  Evidence("derived:c3:opportunity_masked:q_plugin_bound_coverage:prefix48", "mean", expect=0.975, tol=1e-9)],
                 needs=["opportunity denominator for point estimates"]),
            Rule("eps-chain", "approximate equivalence",
                 "Approximate observational equivalence within epsilon is not transitive (points 0, .75, 1.5 with epsilon 1); approximate equivalences cannot be chained into an identification claim.",
                 "EXACT_CERTIFICATE",
                 [Evidence(f"{F02}/exact_identification.json", key=["epsilon_nontransitivity", "epsilon"], expect=1.0, tol=1e-12),
                  Evidence(f"{F02}/exact_identification.json", key=["epsilon_nontransitivity", "points", 2], expect=1.5, tol=1e-12)],
                 information="non-transitivity recomputed at review from the stored points (extra check)"),
        ],
        statuses={"implementation": "DONE", "protocol": "PARTIAL", "science": "SUPPORTED_FINITE"},
        gaps=[
            {"gap": "legacy F02-01 collision worlds not re-hashed (epsilon chain now recomputed at review; F02-04 adverse tests added in tests/test_grain_gptpro_f02_adverse.py)", "next": "F17-01 source acceptance (bookkeeping)"},
            {"gap": "C1/C2 type/parameter recovery metrics absent (only proper losses)", "next": "not required for construction rules; recorded"},
            {"gap": "C3 plug-in bound is not a confidence interval", "next": "F13 uncertainty rules"},
        ],
        non_claims=[
            "worlds are constructed finite simulations; rules are production rules for these worlds, not psychological laws",
            "no PANDORA/MEPS projection of opportunity visibility",
            "no parent card promoted",
        ],
        handoffs=[{"to": "F03", "what": "opportunity denominator requirement and published-only non-identification"}, {"to": "F16", "what": "scoring state must carry order-2 history in C2-type targets"}],
    )


SPECS = {"F02-06": f02_06}


F03 = f"{R}/F03/recovery_full_r1"
F03_AUDIT = f"{R}/F03/audit_full_r1"


def f03_derived(root):
    """Design x view x method identification map, IPW bias by prefix, exact cells."""
    import collections

    rows = []
    metrics = load(root, f"{F03}/metrics.csv", role="application")
    agg = collections.defaultdict(lambda: {"point": 0, "bound": 0, "cov_k": 0.0, "cov_n": 0, "width": []})
    for r in metrics:
        a = agg[(r["design"], r["view"], r["method"])]
        a["point"] += int(r["squared_error_n"])
        a["bound"] += int(r["width_n"])
        if r["contains_target"]:
            a["cov_k"] += float(r["contains_target"]) * int(r["contains_target_n"])
            a["cov_n"] += int(r["contains_target_n"])
        if r["width"]:
            a["width"].append(float(r["width"]))
    for (design, view, method), a in sorted(agg.items()):
        status = "point" if a["point"] else ("bounds" if a["bound"] else "no_output")
        rows.append({
            "quantity": f"map:{design}:{view}:{method}",
            "status": status,
            "point_rows": a["point"],
            "bound_rows": a["bound"],
            "mean": a["cov_k"] / a["cov_n"] if a["cov_n"] else "",
            "width": sum(a["width"]) / len(a["width"]) if a["width"] else "",
        })
    bias = collections.defaultdict(list)
    for r in metrics:
        if r["method"] == "ipw" and r["view"] == "all_opportunities" and r["bias"]:
            bias[(r["world"], r["design"], r["prefix"])].append(float(r["bias"]))
    for (world, design, prefix), v in sorted(bias.items()):
        rows.append({"quantity": f"ipw_bias:{world}:{design}:prefix{prefix}", "mean": sum(v) / len(v), "n_units": len(v)})
    cells = __import__("json").loads((root / F03 / "exact_population_cells.json").read_text())
    for design in ("free", "common", "fixed"):
        cs = [c for c in cells if c["design"] == design]
        widths = [c["general_bound"][1] - c["general_bound"][0] for c in cs]
        rows.append({"quantity": f"exact:{design}:general_bound_width", "min": min(widths), "max": max(widths), "n_units": len(cs),
                     "mean": all(c["general_bound"][0] - 1e-12 <= c["target"] <= c["general_bound"][1] + 1e-12 for c in cs)})
    for gamma in (0, 1):
        cs = [c for c in cells if c["gamma"] == gamma]
        rows.append({"quantity": f"exact:gamma{gamma}:oracle_ipw_max_abs_error", "max": max(abs(c["oracle_observed_propensity_mean"] - c["target"]) for c in cs), "n_units": len(cs)})
    targets = collections.defaultdict(set)
    for c in cells:
        targets[(c["r"], c["kappa"], c["lambda"], c["gamma"], c["z"])].add(round(c["target"], 12))
    rows.append({"quantity": "exact:target_varies_with_design", "mean": sum(len(v) > 1 for v in targets.values()), "n_units": len(targets)})
    rows.append({"quantity": "exact:published_only_response_bounds", "mean": sorted({tuple(c["published_only"]["response_bound"]) for c in cells}).__repr__(), "n_units": len(cells)})
    pc = load(root, f"{F03}/paired_comparisons.csv")
    for comp in ("fixed_vs_free", "common_vs_free"):
        sel = [r for r in pc if r["comparison"] == comp]
        rows.append({"quantity": f"design:{comp}:supported_cells", "mean": sum(r["decision"] == "SUPPORTED_LOWER_LOSS" for r in sel), "n_units": len(sel)})
    return rows, [f"{F03}/metrics.csv", f"{F03}/exact_population_cells.json", f"{F03}/paired_comparisons.csv"]


def f03_legacy_check(root):
    """Legacy level/half-contrast replay: max transform error over all rows."""
    rows = load(root, f"{F03_AUDIT}/legacy_target_comparison.csv")
    defined = [abs(float(r["transform_error"])) for r in rows if r["transform_error"] != ""]
    err = max(defined)
    blank = len(rows) - len(defined)
    return (
        "legacy_readout_transform_error",
        err == 0.0 and len(rows) == 32400,
        f"rows={len(rows)} defined={len(defined)} undefined(empty condition cell)={blank} max_error={err}",
    )


def f03_06() -> Card:
    """F03-06 opportunity rules."""
    pc = f"{F03}/paired_comparisons.csv"
    return Card(
        card_id="F03-06",
        package="F03",
        artifact="opportunity_rules.json",
        requirements=["R04", "R05", "R06"],
        depends_on=["F03-05"],
        derived=[f03_derived],
        extra_checks=[f03_legacy_check, pytest_check("tests/test_grain_gptpro_f03_recovery.py", "tests/test_grain_gptpro_f03_identification.py")],
        rules=[
            Rule("published-only", "all F03 worlds and designs; published-only view",
                 "A corpus of published texts without opportunity denominators identifies only the selected-text mean, a different target; no bound, IPW, pi or q output is produced, and the response bound is [0,1] in every exact cell.",
                 "NOT_IDENTIFIED",
                 [Evidence("derived:exact:published_only_response_bounds", "mean", expect="[(0.0, 1.0)]"),
                  Evidence("derived:map:free:published_only:general_bound", "status", expect="no_output"),
                  Evidence("derived:map:free:published_only:ipw", "status", expect="no_output"),
                  Evidence("derived:map:free:published_only:naive", "status", expect="point", note="selected-text mean only")],
                 needs=["opportunity denominator"], information="adding more published text does not add a denominator"),
            Rule("free-bounds", "free design, opportunities visible",
                 "Without a selection model, opportunities give only partial identification: exact general-bound width .356-.922 over the 96 free cells; the independent-opportunity assumption narrows it (plug-in coverage .985 general vs .898 independent, widths .701 vs .503).",
                 "BOUNDS_ONLY",
                 [Evidence("derived:exact:free:general_bound_width", "min", expect=0.356, tol=5e-4),
                  Evidence("derived:exact:free:general_bound_width", "max", expect=0.922, tol=5e-4),
                  Evidence("derived:map:free:all_opportunities:general_bound", "mean", expect=0.985, tol=5e-4),
                  Evidence("derived:map:free:all_opportunities:independent_opportunity_bound", "mean", expect=0.898, tol=5e-4)],
                 needs=["opportunity denominator", "selection assumption for narrowing"]),
            Rule("fixed-point", "fixed design",
                 "A fixed-opportunity design point-identifies the conditional response (exact width 0 in all 96 cells) and the fixed-design Beta estimator applies; the zero-width plug-in set is not an interval estimate (it contains the target in only .03 of cells).",
                 "EXACT_CERTIFICATE",
                 [Evidence("derived:exact:fixed:general_bound_width", "max", expect=0.0, tol=1e-9),
                  Evidence("derived:map:fixed:all_opportunities:general_bound", "mean", expect=0.03, tol=5e-3),
                  Evidence("derived:map:fixed:all_opportunities:fixed_beta", "status", expect="point")],
                 needs=["designed opportunity"], standard_method="Beta-binomial per condition cell"),
            Rule("design-beats-adjustment", "W3-A..D, prefixes 32/128/512, application",
                 "Design beats statistical adjustment: fixed and common designs have lower response MSE than the free design in all 12 world x prefix cells each (e.g. W3-D prefix 32 fixed -.0943, common -.0309).",
                 "SUPPORTED_LOWER_LOSS",
                 [Evidence("derived:design:fixed_vs_free:supported_cells", "mean", expect=12, tol=0),
                  Evidence("derived:design:common_vs_free:supported_cells", "mean", expect=12, tol=0),
                  Evidence(pc, "delta", {"world": "W3-D", "prefix": "32", "comparison": "fixed_vs_free"}, expect=-0.094346),
                  Evidence(pc, "delta", {"world": "W3-D", "prefix": "32", "comparison": "common_vs_free"}, expect=-0.030925)],
                 needs=["common or fixed opportunity"]),
            Rule("mnar-text-useless", "MNAR worlds W3-C/W3-D",
                 "Under response-dependent selection, more text does not remove selection bias: oracle observed-propensity IPW errs by up to .310 exactly (0 when selection is MAR), and sample IPW bias stays about .17-.19 from prefix 32 to 512, even though IPW lowers MSE (W3-C prefix 512 -.0149). Loss improvement is not evidence that selection was removed.",
                 "COUNTEREXAMPLE",
                 [Evidence("derived:exact:gamma1:oracle_ipw_max_abs_error", "max", expect=0.3105, tol=5e-4),
                  Evidence("derived:exact:gamma0:oracle_ipw_max_abs_error", "max", expect=0.0, tol=1e-9),
                  Evidence("derived:ipw_bias:W3-C:free:prefix32", "mean", expect=0.1871, tol=5e-4),
                  Evidence("derived:ipw_bias:W3-C:free:prefix512", "mean", expect=0.1748, tol=5e-4),
                  Evidence(pc, "delta", {"world": "W3-C", "prefix": "512", "comparison": "ipw_vs_naive_free"}, expect=-0.014920)],
                 needs=["selection design or known selection law"], information="additional text adds precision, not identification"),
            Rule("short-prefix-ipw", "MAR worlds W3-A/W3-B, prefix 32",
                 "With 32 events, prefix-MAP IPW is worse than the naive selected mean in MAR worlds (W3-A free +.0059, W3-B free +.0036; also under common opportunity), inconclusive at 128 (free), and better at 512. Do not apply per-person IPW at short prefixes; the likely mechanism (noisy prefix intercepts) is an interpretation, not separately tested.",
                 "HIGHER_LOSS",
                 [Evidence(pc, "delta", {"world": "W3-A", "prefix": "32", "comparison": "ipw_vs_naive_free"}, expect=0.005934),
                  Evidence(pc, "ci_low", {"world": "W3-A", "prefix": "32", "comparison": "ipw_vs_naive_free"}, expect=0.004061),
                  Evidence(pc, "delta", {"world": "W3-B", "prefix": "32", "comparison": "ipw_vs_naive_free"}, expect=0.003565),
                  Evidence(pc, "decision", {"world": "W3-A", "prefix": "128", "comparison": "ipw_vs_naive_free"}, expect="INCONCLUSIVE"),
                  Evidence(pc, "decision", {"world": "W3-A", "prefix": "512", "comparison": "ipw_vs_naive_free"}, expect="SUPPORTED_LOWER_LOSS")],
                 needs=["enough events per person for selection intercepts"]),
            Rule("same-observation", "unrestricted response-dependent selection",
                 "Different response targets (.1 to .9) can produce the identical observed law [P(S0),P(S1,Y0),P(S1,Y1)] = [.8,.1,.1]; without a selection assumption the sharp bound is [.1,.9].",
                 "EXACT_CERTIFICATE",
                 [Evidence(f"{F03_AUDIT}/same_observation_counterexample.json", key=["sharp_population_bound", 0], expect=0.1, tol=1e-12),
                  Evidence(f"{F03_AUDIT}/same_observation_counterexample.json", key=["sharp_population_bound", 1], expect=0.9, tol=1e-12),
                  Evidence(f"{F03_AUDIT}/same_observation_counterexample.json", key=["examples", 0, "observed_law", 0], expect=0.8, tol=1e-12),
                  Evidence(f"{F03_AUDIT}/same_observation_counterexample.json", key=["examples", 4, "observed_law", 0], expect=0.8, tol=1e-12)],
                 needs=["selection assumption or design"]),
            Rule("design-target-invariant", "F03 worlds",
                 "In F03 the opportunity design changes identification, not the estimand: the target is identical across free/common/fixed designs in all 48 parameter cells. Designs that change responses (reactivity) are an F06/H04 question, not settled here.",
                 "SUPPORTED_FINITE",
                 [Evidence("derived:exact:target_varies_with_design", "mean", expect=0, tol=0),
                  Evidence("derived:exact:target_varies_with_design", "n_units", expect=48, tol=0)]),
        ],
        statuses={"implementation": "DONE", "protocol": "PARTIAL", "science": "SUPPORTED_FINITE"},
        gaps=[
            {"gap": "wording correction: the 2026-09-16 F03 report's '32,400 old readouts replayed with zero error' includes 250 undefined rows (one condition cell empty); 32,150 defined rows have exactly zero error", "next": "recorded here; report left unedited (historical)"},
            {"gap": "exact-vs-large-sample integration check for bounds (F03-04) not run", "next": "optional; bounds are exact enumerations"},
            {"gap": "fixed-design Beta interval coverage not reported (only point MSE)", "next": "F13 uncertainty rules"},
            {"gap": "card-level orchestration (run_stage/--stage) absent; review built from recovery runs", "next": "recorded deviation"},
        ],
        non_claims=[
            "no simulated opportunity knowledge is projected to PANDORA or other natural corpora",
            "plug-in bounds are identified-set estimates, not confidence regions",
            "no parent card promoted",
        ],
        handoffs=[
            {"to": "F06", "what": "design value: fixed/common opportunity beats adjustment in all 12 cells; reactive designs need H04"},
            {"to": "F16", "what": "published-only corpora support only the selected-text target; scoring bundles must carry opportunity status or declare RAW selected-mean targets"},
            {"to": "F13", "what": "plug-in identified sets need separate sampling intervals"},
        ],
    )


SPECS["F03-06"] = f03_06


F04 = f"{R}/F04"


def f04_derived(root):
    """Standard-vs-GRAIN accounting per world and per-world verdict counts."""
    rows = []
    path = f"{F04}/application_full_r1/response_metrics.csv"
    key = ("seed", "author_id")
    for world in ("0", "L", "H", "V", "S", "T", "Q"):
        for cand in ("legacy_halves", "four_cell"):
            res = paired(root, path, key, "method", cand, "REML", value_field="profile_MSE", world=world, role="application", K="4", n="24")
            rows.append({"quantity": f"{world}:{cand}-REML:profile_MSE", **res})
    pc = load(root, f"{F04}/application_full_r1/paired_comparisons.csv", metric="profile_MSE")
    for method in ("REML", "four_cell", "legacy_halves"):
        sel = [r for r in pc if r["method"] == method]
        rows.append({"quantity": f"count:{method}:profile_MSE_supported_worlds", "mean": sum(r["decision"] == "SUPPORTED_LOWER_LOSS" for r in sel), "n_units": len(sel),
                     "status": ",".join(r["world"] for r in sel if r["decision"] == "SUPPORTED_LOWER_LOSS")})
    return rows, [path, f"{F04}/application_full_r1/paired_comparisons.csv"]


def f04_06() -> Card:
    """F04-06 response rules."""
    pc = f"{F04}/application_full_r1/paired_comparisons.csv"
    info = f"{F04}/information_full_r1/information_vs_mc_by_seed.csv"
    s_auc = f"{F04}/information_full_r1/S_mean_vs_profile_by_seed.csv"
    tail = f"{F04}/tail_full_r1/variance_tail_summary.csv"
    pair = f"{F04}/s_pair_cluster_r1/summary.json"
    return Card(
        card_id="F04-06",
        package="F04",
        artifact="response_rules.json",
        requirements=["R09", "R12", "A02"],
        depends_on=["F04-05"],
        derived=[f04_derived],
        extra_checks=[pytest_check("tests/test_grain_gptpro_f04_recovery.py")],
        rules=[
            Rule("single-condition", "linear response worlds",
                 "One condition gives a rank-1 design at every budget: repeats sharpen that condition's mean but never identify a slope.",
                 "EXACT_CERTIFICATE",
                 [Evidence(info, "rank", {"world": "L", "seed": "20260915", "K": "1", "n": "24"}, expect=1, tol=0)],
                 needs=["at least two conditions per person for a contrast"]),
            Rule("endpoint-design", "linear slope target, n=24, sigma^2=.25",
                 "For a linear slope, two balanced endpoints give variance .0104167 versus .01875 for four balanced conditions (ratio 5/9); Monte Carlo agrees. This is optimal for the linear slope only, not for unknown curves.",
                 "EXACT_CERTIFICATE",
                 [Evidence(info, "b_expected_MSE", {"world": "L", "seed": "20260915", "K": "2", "n": "24"}, expect=0.0104167, tol=1e-6),
                  Evidence(info, "b_expected_MSE", {"world": "L", "seed": "20260915", "K": "4", "n": "24"}, expect=0.01875, tol=1e-6),
                  Evidence(info, "b_empirical_MSE", {"world": "L", "seed": "20260915", "K": "2", "n": "24"}, expect=0.010542, tol=5e-6)],
                 standard_method="OLS optimal design (textbook)"),
            Rule("same-mean-different-slope", "S world, prefix 24",
                 "A person mean can erase a response distinction that the same events identify: mean-score AUC about .50 while slope AUC is 1.0 (paired authors share intercepts, slopes +/-.5).",
                 "COUNTEREXAMPLE",
                 [Evidence(s_auc, "mean_auc", {"seed": "20260915", "n": "24"}, expect=0.5093, tol=5e-5),
                  Evidence(s_auc, "slope_auc", {"seed": "20260915", "n": "24"}, expect=1.0, tol=1e-9)],
                 needs=["condition contrast target"], information="synthetic well-separated slopes; not a prediction-accuracy claim"),
            Rule("sharing-standard", "worlds 0/L/H/V/S/T, K=4, n=24",
                 "Source-only hierarchical shrinkage (REML/BLUP) lowers profile MSE versus personal OLS in 6 of 7 worlds (world 0 -.0108 [-.0121,-.0096]); in Q it is inconclusive. This is a standard mechanism; it is credited to the standard method, not to GRAIN.",
                 "SUPPORTED_LOWER_LOSS",
                 [Evidence(pc, "delta", {"world": "0", "method": "REML", "metric": "profile_MSE"}, expect=-0.010843),
                  Evidence("derived:count:REML:profile_MSE_supported_worlds", "mean", expect=6, tol=0),
                  Evidence(pc, "decision", {"world": "Q", "method": "REML", "metric": "profile_MSE"}, expect="INCONCLUSIVE")],
                 standard_method="REML/BLUP (standard)"),
            Rule("grain-increment", "all 7 worlds, profile MSE, legacy two-half sharing and four-cell vs REML",
                 "Against the standard REML reference, the legacy GRAIN two-half estimator and the four-cell reference win only where the linear model is wrong (T -.0180, Q -.0236); the legacy estimator is worse in 0/H/V/S and inconclusive in L. The GRAIN increment here is a nonlinearity effect, not a general sharing advantage.",
                 "SUPPORTED_FINITE",
                 [Evidence("derived:T:legacy_halves-REML:profile_MSE", "decision", expect="SUPPORTED_LOWER_LOSS"),
                  Evidence("derived:Q:legacy_halves-REML:profile_MSE", "decision", expect="SUPPORTED_LOWER_LOSS"),
                  Evidence("derived:T:legacy_halves-REML:profile_MSE", "delta", expect=-0.018003, tol=5e-6),
                  Evidence("derived:Q:legacy_halves-REML:profile_MSE", "delta", expect=-0.023639, tol=5e-6),
                  Evidence("derived:0:legacy_halves-REML:profile_MSE", "decision", expect="HIGHER_LOSS"),
                  Evidence("derived:H:legacy_halves-REML:profile_MSE", "decision", expect="HIGHER_LOSS"),
                  Evidence("derived:V:legacy_halves-REML:profile_MSE", "decision", expect="HIGHER_LOSS"),
                  Evidence("derived:S:legacy_halves-REML:profile_MSE", "decision", expect="HIGHER_LOSS"),
                  Evidence("derived:L:legacy_halves-REML:profile_MSE", "decision", expect="INCONCLUSIVE")],
                 information="per-world decisions are all in derived_metrics; no world is promoted as global"),
            Rule("nonlinear-misspecification", "T and Q worlds",
                 "Do not make a line stand in for every curve: four-cell profiles beat personal OLS by -.0203 (T) and -.0241 (Q), while linear REML gains only -.0022 (T) and is inconclusive (Q).",
                 "SUPPORTED_LOWER_LOSS",
                 [Evidence(pc, "delta", {"world": "T", "method": "four_cell", "metric": "profile_MSE"}, expect=-0.020261),
                  Evidence(pc, "delta", {"world": "Q", "method": "four_cell", "metric": "profile_MSE"}, expect=-0.024054),
                  Evidence(pc, "delta", {"world": "T", "method": "REML", "metric": "profile_MSE"}, expect=-0.002151)],
                 needs=["enough conditions to see curvature"]),
            Rule("dispersion-provenance", "T/Q linear vs four-cell, n=24",
                 "A misspecified mean inflates residual dispersion: linear prefix tail rates .050/.052 vs four-cell .029 and variance coverage .895/.893 vs .962/.945. Keep omitted-mean vs stochastic-error provenance; do not call residual removal noise reduction.",
                 "INFORMATION_LOSS",
                 [Evidence(tail, "prefix_true_threshold_tail", {"world": "T", "n": "24", "mode": "linear"}, expect=0.050208),
                  Evidence(tail, "prefix_true_threshold_tail", {"world": "T", "n": "24", "mode": "four_cell"}, expect=0.029375),
                  Evidence(tail, "variance_covered", {"world": "T", "n": "24", "mode": "linear"}, expect=0.895)]),
            Rule("tail-target", "H (t5 errors) vs L, n=24",
                 "Tail probability is its own target (t5 .0493 vs Gaussian .0455), and the Gaussian variance interval undercovers under heavy tails (.763 vs .935). The undercoverage is a retained negative, handed to F13.",
                 "HIGHER_LOSS",
                 [Evidence(tail, "variance_covered", {"world": "H", "n": "24", "mode": "linear"}, expect=0.763333),
                  Evidence(tail, "variance_covered", {"world": "L", "n": "24", "mode": "linear"}, expect=0.935),
                  Evidence(tail, "true_tail_probability", {"world": "H", "n": "24", "mode": "linear"}, expect=0.049313)],
                 information="HIGHER_LOSS here means interval undercoverage, not a loss difference"),
            Rule("s-pair-cluster", "S world application comparisons",
                 "S authors share intercepts in pairs; resampling pairs instead of authors changes no decision (interval width ratio 1.00-1.06).",
                 "SUPPORTED_FINITE",
                 [Evidence(pair, key=["decision_changes"], expect="[]"),
                  Evidence(pair, key=["width_ratio_range", 1], expect=1.0571, tol=5e-4)]),
        ],
        statuses={"implementation": "DONE", "protocol": "PARTIAL", "science": "SUPPORTED_FINITE"},
        gaps=[
            {"gap": "common-schema record adapter for F04 worlds", "next": "engineering; not needed for these rules"},
            {"gap": "heavy-tail interval calibration", "next": "F13 uncertainty rules"},
        ],
        non_claims=["synthetic linear/nonlinear response worlds; no personality labels", "H quantity is second-moment precision, not t5 likelihood Fisher information", "no parent card promoted"],
        handoffs=[{"to": "F06", "what": "endpoint allocation optimal only for linear slope targets"}, {"to": "F13", "what": "H undercoverage and dispersion provenance"}, {"to": "F16", "what": "declare conditional-contrast vs mean targets before scoring"}],
    )


SPECS["F04-06"] = f04_06


F05 = f"{R}/F05"


def f05_derived(root):
    """Paired forecast CIs (census gap): personal MAP vs frozen-source and vs source LMM."""
    import collections

    import numpy as np

    from grain_research.gptpro_foundation_v3.review_derived import decide, stratified

    path = f"{F05}/application_full_r1/forecast_metrics.csv"
    units = collections.defaultdict(dict)
    for r in load(root, path, phase="prefix"):
        units[(r["world"], r["n"], r["k"], r["seed"], r["author_id"])][r["method"]] = float(r["forecast_squared_error"])
    rows = []
    for world in ("core", "near_root", "shared_shock"):
        for n in ("16", "32", "48"):
            for k in ("1", "4"):
                keys = sorted(u for u in units if u[:3] == (world, n, k))
                for ref in ("source_only_dynamics", "source_LMM", "oracle_Kalman"):
                    diff = np.array([units[u]["personal_MAP"] - units[u][ref] for u in keys])
                    q = stratified(diff, np.array([u[3] for u in keys]))
                    rows.append({
                        "quantity": f"forecast:{world}:n{n}:k{k}:personal_MAP-{ref}",
                        "n_units": len(keys), "delta": float(diff.mean()), **q,
                        "decision": decide(q["lo95"], q["hi95"]),
                        "status": "descriptive_conditional_on_3_episodes" if world == "shared_shock" else "author_paired",
                    })
    return rows, [path]


def f05_07() -> Card:
    """F05-07 state rules."""
    pc = f"{F05}/application_full_r1/paired_comparisons.csv"
    mt = f"{F05}/missing_time_full_r1/paired_comparisons.csv"
    joint = f"{F05}/joint_full_r1/joint_summary.csv"
    inv = f"{F05}/invariants_r1/summary.json"
    return Card(
        card_id="F05-07",
        package="F05",
        artifact="state_rules.json",
        requirements=["R10", "R24", "A02"],
        depends_on=["F05-06"],
        derived=[f05_derived],
        extra_checks=[pytest_check("tests/test_grain_gptpro_f05_recovery.py", "tests/test_grain_gptpro_f05_application_recovery.py", "tests/test_grain_gptpro_f05_invariants.py", "tests/test_grain_gptpro_f05_joint.py", "tests/test_grain_gptpro_f05_missing_time.py")],
        rules=[
            Rule("estimand-separation", "core vs near_root, prefix 48, application",
                 "Level, inertia (rho), current state and forecasts are separate estimands. Personal MAP improves state recovery in both worlds but rho only in core (-.0121); in near_root rho worsens (+.0036) and the intercept is inconclusive, so a good state or forecast does not certify personal inertia.",
                 "COUNTEREXAMPLE",
                 [Evidence(pc, "delta", {"world": "core", "metric": "rho_squared_error"}, expect=-0.0121385, tol=5e-6),
                  Evidence(pc, "decision", {"world": "near_root", "metric": "rho_squared_error"}, expect="HIGHER_LOSS"),
                  Evidence(pc, "delta", {"world": "near_root", "metric": "rho_squared_error"}, expect=0.0035816, tol=5e-6),
                  Evidence(pc, "decision", {"world": "near_root", "metric": "state_squared_error"}, expect="SUPPORTED_LOWER_LOSS"),
                  Evidence(pc, "decision", {"world": "near_root", "metric": "a_squared_error"}, expect="INCONCLUSIVE")],
                 needs=["declared estimand per output field"]),
            Rule("forecast-vs-standard", "core and near_root, prefixes 16/32/48, 1- and 4-step",
                 "Against the standard static source LMM, personal MAP dynamics lower 1-step forecast error in core at every prefix (-.070 to -.101) and lower 4-step error at prefixes 16 and 48 (n48 -.0135 [-.0243,-.0032]) but not at 32 (inconclusive); in near_root the gains are large (4-step n48 -1.457). Frozen population dynamics are far worse than personal MAP everywhere. Gains are credited per world and horizon, not globally.",
                 "SUPPORTED_LOWER_LOSS",
                 [Evidence("derived:forecast:core:n48:k4:personal_MAP-source_LMM", "delta", expect=-0.0135, tol=5e-4),
                  Evidence("derived:forecast:core:n48:k4:personal_MAP-source_LMM", "decision", expect="SUPPORTED_LOWER_LOSS"),
                  Evidence("derived:forecast:core:n32:k4:personal_MAP-source_LMM", "decision", expect="INCONCLUSIVE"),
                  Evidence("derived:forecast:core:n16:k1:personal_MAP-source_LMM", "delta", expect=-0.0847, tol=5e-4),
                  Evidence("derived:forecast:core:n32:k1:personal_MAP-source_LMM", "delta", expect=-0.1006, tol=5e-4),
                  Evidence("derived:forecast:near_root:n48:k4:personal_MAP-source_LMM", "delta", expect=-1.4571, tol=5e-4),
                  Evidence("derived:forecast:core:n48:k4:personal_MAP-source_only_dynamics", "decision", expect="SUPPORTED_LOWER_LOSS")],
                 standard_method="source linear mixed model (static)", information="54 paired forecast contrasts in derived_metrics; shared_shock rows are descriptive (3 episodes)"),
            Rule("elapsed-time", "missing-time design, prefix 48",
                 "Use elapsed time in the dynamics: with oracle parameters the correct clock lowers state error (-.0090 core). With learned MAP parameters the correct clock can lose (near_root state +.1348); keep that reversal rather than adopt a wrong clock.",
                 "SUPPORTED_FINITE",
                 [Evidence(mt, "delta", {"world": "core", "budget": "48", "estimator": "oracle", "metric": "state_MSE"}, expect=-0.0090301, tol=5e-6),
                  Evidence(mt, "delta", {"world": "near_root", "budget": "48", "estimator": "personal_MAP", "metric": "state_MSE"}, expect=0.1347538, tol=5e-6),
                  Evidence(mt, "decision", {"world": "near_root", "budget": "48", "estimator": "personal_MAP", "metric": "state_MSE"}, expect="HIGHER_LOSS")],
                 needs=["time stamps / elapsed time"]),
            Rule("shared-shock", "shared_shock world, t=47, personal MAP",
                 "Model shared visible inputs when the design gives them: joint process covariance lowers present-state MSE .1789 -> .1236. It does not reveal future unobserved innovations, and with only 3 independent episodes the author-resampled intervals are conditional descriptions, not population confidence intervals.",
                 "SUPPORTED_FINITE",
                 [Evidence(joint, "state_MSE", {"world": "shared_shock", "t": "47", "parameter_prefix": "48", "method": "personal_MAP", "arm": "independent"}, expect=0.178882, tol=5e-6),
                  Evidence(joint, "state_MSE", {"world": "shared_shock", "t": "47", "parameter_prefix": "48", "method": "personal_MAP", "arm": "joint"}, expect=0.123640, tol=5e-6),
                  Evidence("derived:forecast:shared_shock:n48:k4:personal_MAP-source_LMM", "status", expect="descriptive_conditional_on_3_episodes")],
                 needs=["shared visible input", "independent episodes for population uncertainty"]),
            Rule("coordinate-gauge", "free 2-state linear Gaussian models",
                 "Rotations and scalings of the latent state leave the observation law unchanged (errors <= 1.1e-16); local ranks 5/5/4 for all/fixed-CQ/fixed-CQR. Arbitrary state axes are not named psychological constructs because a model fits.",
                 "EXACT_CERTIFICATE",
                 [Evidence(inv, key=["ranks", "all"], expect=5, tol=0),
                  Evidence(inv, key=["ranks", "fixed_CQR"], expect=4, tol=0),
                  Evidence(f"{F05}/invariants_r1/similarity_invariants.json", key=["transforms", 0, "errors", "covariance"], expect=1.11e-16, tol=1e-15)],
                 information="local rank is not global uniqueness"),
            Rule("prior-vs-data", "source profiles, 20 IDs per world/seed",
                 "Report which information comes from the observed sequence and which from the source prior: posterior and likelihood profiles are stored separately with data-only Hessians; a narrow posterior can be mostly prior. State intervals are plug-in and omit parameter uncertainty; they go to F13.",
                 "SUPPORTED_FINITE",
                 [Evidence(f"{F05}/profiles_full_r1/summary.json", key=["profile_rows"], expect=54000, tol=0),
                  Evidence(f"{F05}/profiles_full_r1/summary.json", key=["hessians"], expect=540, tol=0)],
                 needs=["prior_only flag when the sequence carries no identifying data"], information="prior share is reported per author, not summarised into a single number here"),
        ],
        statuses={"implementation": "DONE", "protocol": "PARTIAL", "science": "SUPPORTED_FINITE"},
        gaps=[
            {"gap": "episode-level uncertainty for shared shocks (3 episodes)", "next": "F13 (needs more independent episodes; simulation change = new protocol)"},
            {"gap": "state intervals omit parameter uncertainty", "next": "F13 uncertainty rules"},
            {"gap": "F05 source fits mis-split q/r (F13 recovery 2) while recovering stationary variance", "next": "F13 recovery 3 / review"},
        ],
        non_claims=["simulated latent state, not named emotion or trait", "no universal identification of the latent model", "no parent card promoted"],
        handoffs=[{"to": "F13", "what": "plug-in state intervals, shared-shock episode uncertainty, q/r split"}, {"to": "F16", "what": "dynamic targets carry estimand labels; prior_only status when no data"}],
    )


SPECS["F05-07"] = f05_07


F06 = f"{R}/F06"


def f06_06() -> Card:
    """F06-06 observation-policy rules and H04 handoff."""
    fam = f"{F06}/family_full_r2/paired_comparisons.csv"
    fam_sum = f"{F06}/family_full_r2/policy_summary.csv"
    irt = f"{F06}/irt_full_r1/paired_comparisons.csv"
    gts = f"{F06}/gaussian_full_summary_audit_r1/target_specific_comparisons.csv"
    gsum = f"{F06}/gaussian_full_r1/summary_by_policy.csv"
    opt = f"{F06}/policy_sensitivity_r1/conditional_optima.csv"
    return Card(
        card_id="F06-06",
        package="F06",
        artifact="policy_rules.json",
        requirements=["R11", "R26", "A07", "H04"],
        depends_on=["F06-05"],
        extra_checks=[pytest_check("tests/test_grain_gptpro_f06_policy_state.py")],
        rules=[
            Rule("family-target", "logistic family panel, budget 24",
                 "Choose conditions for the estimand they distinguish: information-targeted selection lowers family-probability MSE vs random by -.0031 and future Brier by -.0041, while coverage (visiting all 8 corners) is inconclusive vs random; a fixed corner leaves the three axes observationally identical (+.0300).",
                 "SUPPORTED_LOWER_LOSS",
                 [Evidence(fam, "delta", {"budget": "24", "policy": "expected_entropy_gain", "metric": "probability_MSE"}, expect=-0.0030855, tol=5e-7),
                  Evidence(fam, "delta", {"budget": "24", "policy": "expected_entropy_gain", "metric": "future_Brier"}, expect=-0.0040763, tol=5e-7),
                  Evidence(fam, "decision", {"budget": "24", "policy": "coverage", "metric": "probability_MSE"}, expect="INCONCLUSIVE"),
                  Evidence(fam, "delta", {"budget": "24", "policy": "fixed_corner", "metric": "probability_MSE"}, expect=0.0300344, tol=5e-7),
                  Evidence(fam_sum, "distinct_corners", {"budget": "24", "policy": "coverage"}, expect=8.0, tol=0),
                  Evidence(fam_sum, "distinct_corners", {"budget": "24", "policy": "expected_entropy_gain"}, expect=2.335, tol=5e-4)],
                 needs=["condition variation matched to the target"], standard_method="Bayesian adaptive design (expected information gain)"),
            Rule("variance-eig-redundant", "symmetric logistic family panel",
                 "Epistemic-variance and expected-entropy-gain selection produce identical trajectories in this symmetric panel (exact redundancy by the ordering theorem), so the panel cannot separate them; this is not universal equivalence.",
                 "NO_INCREMENT",
                 [Evidence(fam, "delta", {"budget": "24", "policy": "epistemic_variance", "metric": "probability_MSE"}, expect=-0.0030855, tol=5e-7),
                  Evidence(fam, "delta", {"budget": "24", "policy": "expected_entropy_gain", "metric": "entropy"}, expect=-0.1442321, tol=5e-7)],
                 information="exact redundancy within the frozen symmetric panel"),
            Rule("scalar-cat", "scalar IRT, 24 items",
                 "For a scalar parameter, EIG item selection lowers parameter MSE vs random (-.0271 [-.0430,-.0106]) but future Brier is inconclusive; a fixed central item is competitive (EIG vs fixed-centre probability MSE inconclusive). 'Always vary' and 'always fix' are both wrong as universal rules.",
                 "SUPPORTED_LOWER_LOSS",
                 [Evidence(irt, "delta", {"budget": "24", "policy": "expected_entropy_gain", "reference": "random", "metric": "a_MSE"}, expect=-0.0270858, tol=5e-7),
                  Evidence(irt, "decision", {"budget": "24", "policy": "expected_entropy_gain", "reference": "random", "metric": "future_Brier"}, expect="INCONCLUSIVE"),
                  Evidence(irt, "decision", {"budget": "24", "policy": "expected_entropy_gain", "reference": "fixed_center", "metric": "probability_MSE"}, expect="INCONCLUSIVE")],
                 standard_method="computerized adaptive testing (standard)"),
            Rule("state-target-adverse", "Gaussian reactive panel, distributed beta, budget 24",
                 "A state-targeted local information heuristic is an adverse counterexample: vs fixed it raises state MSE +.1464 [.1068,.1903] and squared displacement +24.13; vs random it is worse on both axes. The same failure occurs with zero physical displacement (nonreactive world state MSE .395 vs fixed .187), so it is not only reactivity.",
                 "COUNTEREXAMPLE",
                 [Evidence(gts, "delta", {"reactivity": "distributed", "budget": "24", "target": "state", "arm": "scalar_state", "reference": "fixed", "metric": "MSE"}, expect=0.1463771, tol=5e-7),
                  Evidence(gts, "delta", {"reactivity": "distributed", "budget": "24", "target": "state", "arm": "scalar_state", "reference": "fixed", "metric": "sum_squared_perturbation"}, expect=24.1305135, tol=5e-6),
                  Evidence(gts, "classification", {"reactivity": "distributed", "budget": "24", "target": "state", "arm": "scalar_state", "reference": "random", "metric": "MSE"}, expect="HIGHER"),
                  Evidence(gsum, "MSE", {"reactivity": "zero", "budget": "24", "target": "state", "arm": "scalar_state"}, expect=0.3951694, tol=5e-7),
                  Evidence(gsum, "sum_squared_perturbation", {"reactivity": "zero", "budget": "24", "target": "state", "arm": "scalar_state"}, expect=0.0, tol=0)],
                 needs=["per-target policy", "intervention-cost accounting"], information="causal attribution among approximate-posterior selection, opportunity variation and prior effects not established"),
            Rule("state-failure-attribution", "nonreactive world, state target, budget 24 (declared attribution arms)",
                 "The state-target failure is entirely the action channel: forcing actions under the fixed design reproduces it (+.219 [.185,.254]) while letting the criterion choose z without actions slightly helps (-.006); the working beta prior (.4) credits each action with a nonexistent state increment (bias +.456). Do not estimate state under an unverified intervention-effect prior while intervening.",
                 "SUPPORTED_FINITE",
                 [Evidence(f"{F06}/state_attribution_r1/contrasts.json", key=["fixed_all_h-fixed", "delta"], expect=0.2191, tol=5e-4),
                  Evidence(f"{F06}/state_attribution_r1/contrasts.json", key=["state_z_only-fixed", "decision"], expect="SUPPORTED_LOWER_LOSS"),
                  Evidence(f"{F06}/state_attribution_r1/summary.json", key=["arms", "fixed_all_h", "bias"], expect=0.4560, tol=5e-4),
                  Evidence(f"{F06}/state_attribution_r1/summary.json", key=["replay", "20260915:scalar_state", "sorted_max_abs_diff"], expect=0.0, tol=0)]),
            Rule("beta-cost", "Gaussian reactive panel, beta target, budget 24",
                 "Beta-targeting beats a design with no beta information (-.0137) but not random or coverage (inconclusive -.0023/-.0022) while adding intervention burden (+2.06/+2.77 squared displacement). The extra cost is not justified by this calculation.",
                 "INCONCLUSIVE",
                 [Evidence(gts, "delta", {"reactivity": "distributed", "budget": "24", "target": "beta", "arm": "scalar_beta", "reference": "fixed", "metric": "MSE"}, expect=-0.0137478, tol=5e-7),
                  Evidence(gts, "classification", {"reactivity": "distributed", "budget": "24", "target": "beta", "arm": "scalar_beta", "reference": "random", "metric": "MSE"}, expect="UNRESOLVED"),
                  Evidence(gts, "classification", {"reactivity": "distributed", "budget": "24", "target": "beta", "arm": "scalar_beta", "reference": "random", "metric": "sum_squared_perturbation"}, expect="HIGHER")],
                 needs=["burden/reactivity cost from real people (H04)"]),
            Rule("prior-only-beta", "Gaussian panel, fixed design (no interventions)",
                 "Without interventions beta never enters the data law; the estimate is the prior. It looks good when the population matches (MSE .0394, coverage .958) and fails when it does not (zero world MSE .160, coverage 0). Outputs must carry prior_only status; the sensitivity cases flag it in all 6 zero-action cases.",
                 "NOT_IDENTIFIED",
                 [Evidence(gsum, "MSE", {"reactivity": "distributed", "budget": "24", "target": "beta", "arm": "fixed"}, expect=0.0394186, tol=5e-7),
                  Evidence(gsum, "coverage", {"reactivity": "zero", "budget": "24", "target": "beta", "arm": "fixed"}, expect=0.0, tol=0),
                  Evidence(gsum, "MSE", {"reactivity": "zero", "budget": "24", "target": "beta", "arm": "fixed"}, expect=0.16, tol=1e-9),
                  Evidence(opt, "beta_status", {"case": "20260915_zero_zero", "gamma": ""}, expect="prior_only_no_data_information")],
                 needs=["prior_only flag"]),
            Rule("a-b-simple-design", "Gaussian panel, a and b targets, budget 24",
                 "For level and slope, targeted policies add no resolved gain over the simple alternating-z / no-intervention design (both UNRESOLVED vs fixed, zero extra displacement); they beat random/coverage, which also disturb the state.",
                 "INCONCLUSIVE",
                 [Evidence(gts, "classification", {"reactivity": "distributed", "budget": "24", "target": "a", "arm": "scalar_a", "reference": "fixed", "metric": "MSE"}, expect="UNRESOLVED"),
                  Evidence(gts, "classification", {"reactivity": "distributed", "budget": "24", "target": "b", "arm": "scalar_b", "reference": "fixed", "metric": "MSE"}, expect="UNRESOLVED"),
                  Evidence(gts, "classification", {"reactivity": "distributed", "budget": "24", "target": "a", "arm": "scalar_a", "reference": "random", "metric": "MSE"}, expect="LOWER")],
                 information="not an equivalence claim"),
        ],
        statuses={"implementation": "DONE", "protocol": "PARTIAL", "science": "SUPPORTED_FINITE"},
        gaps=[
            {"gap": "real questioning burden, reactivity and openness cost", "next": "H04 (human-stage preparation contract in F17-03)"},
            {"gap": "none for the state-target attribution (resolved in F06/state_attribution_r1)", "next": "-"},
            {"gap": "duplicate-event protection / F00 DAG packaging", "next": "engineering (recorded)"},
        ],
        non_claims=["simulated units, not psychological harm", "ADO is not shown to dominate coverage in general", "no parent card promoted"],
        handoffs=[
            {"to": "H04", "what": "needs: per-question burden, reactivity (state change caused by asking), refusal/openness cost, intervention timing; estimand: target-specific MSE/cost Pareto; columns listed in F17-03"},
            {"to": "F16", "what": "policy-dependent likelihood factors must be recorded with actions; prior_only status propagates"},
        ],
    )


SPECS["F06-06"] = f06_06


F08 = f"{R}/F08"


def f08_derived(root):
    """Layered loss attribution on identical authors: render, reader, source fit."""
    import collections

    import numpy as np

    from grain_research.gptpro_foundation_v3.review_derived import decide, stratified

    e2e = collections.defaultdict(dict)
    for r in load(root, f"{F08}/end_to_end_r1/metrics_by_author.csv"):
        e2e[(r["reader"], r["world"], r["candidate"], r["seed"], r["author_id"])][(r["panel"], r["arm"])] = float(r["brier"])
    mnar = collections.defaultdict(dict)
    for r in load(root, f"{F08}/mnar_source_r1/metrics_by_author.csv"):
        mnar[(r["reader"], r["world"], r["candidate"], r["seed"], r["author_id"])][r["arm"]] = float(r["brier"])
    layers = {
        "render": (("mixed", "oracle_channel_oracle_source"), ("complete", "oracle_channel_oracle_source")),
        "reader_application_mixed": (("mixed", "hard_channel_oracle_source"), ("mixed", "oracle_channel_oracle_source")),
        "reader_application_complete": (("complete", "hard_channel_oracle_source"), ("complete", "oracle_channel_oracle_source")),
        "source_fit_gap_aware": (("mixed", "hard_channel_read_source"), ("mixed", "hard_channel_oracle_source")),
    }
    rows = []
    for reader in ("lexical", "nli"):
        for world in ("c2_xor", "c2_iid"):
            for cand in ("b1", "b2", "m"):
                keys = sorted(k for k in e2e if k[:3] == (reader, world, cand))
                seeds = np.array([k[3] for k in keys])
                for name, (a, b) in layers.items():
                    diff = np.array([e2e[k][a] - e2e[k][b] for k in keys])
                    q = stratified(diff, seeds)
                    verdict = "IDENTICAL" if np.all(diff == 0) else decide(q["lo95"], q["hi95"])
                    rows.append({"quantity": f"layer:{reader}:{world}:{cand}:{name}", "n_units": len(keys), "delta": float(diff.mean()), **q, "decision": verdict})
                mk = sorted(k for k in mnar if k[:3] == (reader, world, cand))
                diff = np.array([mnar[k]["em_validation_law"] - mnar[k]["oracle_source"] for k in mk])
                q = stratified(diff, np.array([k[3] for k in mk]))
                rows.append({"quantity": f"layer:{reader}:{world}:{cand}:source_fit_em_validation_law", "n_units": len(mk), "delta": float(diff.mean()), **q, "decision": decide(q["lo95"], q["hi95"])})
    return rows, [f"{F08}/end_to_end_r1/metrics_by_author.csv", f"{F08}/mnar_source_r1/metrics_by_author.csv"]


def f08_07() -> Card:
    """F08-07 text-bridge rules and frozen interface."""
    fac = f"{F08}/factorial_full_r1/paired_decomposition.csv"
    dec = f"{F08}/source_read_decomposition_r1/summary.json"
    return Card(
        card_id="F08-07",
        package="F08",
        artifact="text_bridge_rules.json",
        requirements=["R14", "R15", "R16"],
        depends_on=["F08-06"],
        derived=[f08_derived],
        extra_checks=[pytest_check("tests/test_grain_gptpro_f08.py", "tests/test_grain_gptpro_f08_estimators.py", "tests/test_grain_gptpro_f08_filter.py", "tests/test_grain_gptpro_f08_materials.py")],
        rules=[
            Rule("render-layer", "controlled text, oracle reader, oracle source, application",
                 "Rendering events as mixed-status text (quoted/hypothetical events carry no evidence about the author) costs loss even with a perfect reader where history matters: B1 xor +.0383, B2 +.0345, M +.0329 Brier versus the complete panel. In the iid world the same rendering lowers B2/M loss (-.0042/-.0051) because fewer personal events mean less overfitting; the render cost is world- and estimator-specific.",
                 "INFORMATION_LOSS",
                 [Evidence("derived:layer:lexical:c2_xor:b1:render", "delta", expect=0.0383, tol=5e-4),
                  Evidence("derived:layer:lexical:c2_xor:b1:render", "decision", expect="HIGHER_LOSS"),
                  Evidence("derived:layer:lexical:c2_xor:m:render", "delta", expect=0.0329, tol=5e-4),
                  Evidence("derived:layer:lexical:c2_iid:b2:render", "delta", expect=-0.0042, tol=5e-4),
                  Evidence("derived:layer:lexical:c2_iid:m:render", "decision", expect="SUPPORTED_LOWER_LOSS")],
                 needs=["qualification (ownership/actuality) per event"]),
            Rule("reader-qualification", "NLI reader, mixed panel, factorial on identical histories",
                 "At application, almost all reader cost is evidence qualification, not stance: B1 xor qualification +.0231 [.0209,.0253] vs polarity -.000002 (inconclusive); B2 +.0379. A strong stance reader can fail at the qualification interface.",
                 "HIGHER_LOSS",
                 [Evidence(fac, "delta", {"world": "c2_xor", "candidate": "b1", "loss": "brier", "component": "qualification"}, expect=0.0231491, tol=5e-7),
                  Evidence(fac, "decision", {"world": "c2_xor", "candidate": "b1", "loss": "brier", "component": "polarity"}, expect="INCONCLUSIVE"),
                  Evidence(fac, "delta", {"world": "c2_xor", "candidate": "b2", "loss": "brier", "component": "qualification"}, expect=0.0379269, tol=5e-7)],
                 needs=["qualification channel separate from stance channel"]),
            Rule("reader-complete-panel", "complete panel (all events qualified)",
                 "When every event is qualified, both measured readers give application losses identical to the oracle channel; on the mixed panel the hard NLI channel costs B1 xor +.0292 and B2 +.0381 (lexical: identical). The reader matters through qualification.",
                 "SUPPORTED_FINITE",
                 [Evidence("derived:layer:lexical:c2_xor:b1:reader_application_complete", "decision", expect="IDENTICAL"),
                  Evidence("derived:layer:nli:c2_xor:b1:reader_application_complete", "decision", expect="IDENTICAL"),
                  Evidence("derived:layer:nli:c2_xor:b1:reader_application_mixed", "delta", expect=0.0292, tol=5e-4),
                  Evidence("derived:layer:nli:c2_xor:b2:reader_application_mixed", "delta", expect=0.0381, tol=5e-4),
                  Evidence("derived:layer:lexical:c2_xor:b1:reader_application_mixed", "decision", expect="IDENTICAL")],
                 information="the factorial (soft readings) and this hard-channel layer are different estimands of reader cost; both retained"),
            Rule("source-fit-layer", "mixed panel, source cohort read by the same reader",
                 "The reader also enters the source fit. Dropping unqualified reads under the MNAR NLI reader collapses the source table (B1 application +.1275 iid / +.1239 xor; table shift selection .242 + values .109); putting the measured emission law in the source likelihood recovers B1/B2 (B1 +.0007 / -.0004) but not M.",
                 "HIGHER_LOSS",
                 [Evidence("derived:layer:nli:c2_iid:b1:source_fit_gap_aware", "delta", expect=0.1275, tol=5e-4),
                  Evidence("derived:layer:nli:c2_xor:b1:source_fit_gap_aware", "delta", expect=0.1239, tol=5e-4),
                  Evidence(dec, key=["nli:selection_only", "mean_max_shift"], expect=0.2417, tol=5e-4),
                  Evidence(dec, key=["nli:values_only", "mean_max_shift"], expect=0.1089, tol=5e-4),
                  Evidence("derived:layer:nli:c2_iid:b1:source_fit_em_validation_law", "delta", expect=0.0007, tol=5e-4),
                  Evidence("derived:layer:nli:c2_xor:m:source_fit_em_validation_law", "decision", expect="HIGHER_LOSS")],
                 needs=["measured emission law in the source likelihood", "author-level E-step for M"]),
            Rule("source-fit-author-estep", "mixed panel, M source model, measured emission law",
                 "M needs an author-level E-step: imputing each source author under the pooled chain shrinks between-author variation and leaves M worse than the oracle-fitted source (NLI xor +.0060); imputing under each author's own chain improves M over the pooled E-step in all 8 reader x world x law contrasts (NLI xor -.0141) and reaches or beats the oracle-fitted source in 3 of 4 reader x world cells; NLI iid stays higher (+.0051 deployable law).",
                 "SUPPORTED_LOWER_LOSS",
                 [Evidence(f"{F08}/author_estep_r1/m_contrasts.csv", "delta", {"reader": "nli", "world": "c2_xor", "contrast": "em_author_validation_law-em_pooled_validation_law"}, expect=-0.01409, tol=5e-5),
                  Evidence(f"{F08}/author_estep_r1/m_contrasts.csv", "decision", {"reader": "nli", "world": "c2_iid", "contrast": "em_author_validation_law-oracle_source"}, expect="HIGHER_LOSS"),
                  Evidence(f"{F08}/author_estep_r1/m_contrasts.csv", "delta", {"reader": "nli", "world": "c2_iid", "contrast": "em_author_validation_law-oracle_source"}, expect=0.00512, tol=5e-5),
                  Evidence(f"{F08}/author_estep_r1/m_contrasts.csv", "decision", {"reader": "lexical", "world": "c2_iid", "contrast": "em_author_validation_law-oracle_source"}, expect="SUPPORTED_LOWER_LOSS"),
                  Evidence(f"{F08}/author_estep_r1/replay.json", key=["em_pooled_validation_law_vs_mnar_source_r1", "max_abs_diff"], expect=0.0, tol=0)],
                 needs=["author-level E-step for author-structure models"], information="author tables are overdispersed individually; not person estimates"),
            Rule("estimator-layer", "C2 worlds, oracle channel",
                 "Estimator choice is its own layer: with the oracle channel and oracle source, B1 is the best of B1/B2/M in the shared-law C2 worlds (F02-06 C2 rules); reader and source layers must be judged against the same estimator.",
                 "SUPPORTED_FINITE",
                 [Evidence(f"{R}/F02/recovery_full_r2/paired_comparisons.csv", "decision", {"world": "c2_xor", "candidate": "b2", "reference": "b1", "metric": "brier"}, expect="HIGHER_LOSS")]),
            Rule("interface-freeze", "F07/F11/F16 consumers",
                 "Frozen text-bridge interface: each event carries (pole reading, qualification reading, reader_id, emission law id); the scorer consumes a qualified history, not a semantic score; source models record their fit mode (oracle/controlled or emission-law EM; gap-aware dropping forbidden); outputs carry prior_only status when no qualified evidence exists.",
                 "SUPPORTED_FINITE",
                 [Evidence(f"{F08}/mnar_source_r1/replay.json", key=["oracle_source", "max_abs_diff"], expect=0.0, tol=0),
                  Evidence(f"{F08}/end_to_end_r1/replay.json", key=["nli:hard_channel_oracle_source", "max_abs_diff"], expect=0.0, tol=0)],
                 information="interface is a production contract; numeric mechanism and text implementation are judged separately; reader human validity is not required or claimed"),
        ],
        statuses={"implementation": "DONE", "protocol": "PARTIAL", "science": "SUPPORTED_FINITE"},
        gaps=[
            {"gap": "F08-06 evidence cards lack reader_id / contribution / uncertainty / allowed_interpretation fields", "next": "F16 explanation blocks carry these for served bundles; legacy cards retained"},
            {"gap": "NLI iid M residual cost after author-level E-step (+.0051 deployable law, +.0016 true law)", "next": "retained; partly law measured on other materials"},
            {"gap": "application filter remains MAR-approximate under MNAR readers", "next": "F07-05 measured-law MAP covers a different target; F16 text transport uses measured channels"},
        ],
        non_claims=["controlled synthetic materials and two measured readers; no natural-language validity", "no reader human validity claimed", "no parent card promoted"],
        handoffs=[
            {"to": "F07", "what": "qualification errors enter the likelihood directionally (MNAR)"},
            {"to": "F11", "what": "reader error law is a separately measured object; multi-reader agreement is not validity"},
            {"to": "F16", "what": "interface-freeze fields; source fit mode recorded in bundles"},
        ],
    )


SPECS["F08-07"] = f08_07


F09 = f"{R}/F09"


def f09_06() -> Card:
    """F09-06 rival-process rules."""
    rt = f"{F09}/recovery_tables_r3/recovery_table.csv"
    xb = f"{F09}/recovery_tables_r3/extra_b_positive_control.csv"
    pb = f"{F09}/m4_power_bound_r1/summary.json"
    f15 = f"{R}/F15/recovery_r2/F15-02/metrics.csv"
    return Card(
        card_id="F09-06",
        package="F09",
        artifact="rival_rules.json",
        requirements=["R18", "R19", "A08"],
        depends_on=["F09-05"],
        extra_checks=[pytest_check("tests/test_grain_gptpro_f09_rivals.py")],
        rules=[
            Rule("family-recovery", "in-family main panel, 12 development events (even corners), application",
                 "The response family is recovered well above chance (.791 [.758,.820] vs .333) and slope less so (.529); 22% of authors' development data are exactly uninformative between families at this design, so reports must carry ties rather than an argmax label.",
                 "SUPPORTED_FINITE",
                 [Evidence(rt, "value", {"metric": "family_accuracy"}, expect=0.790741, tol=5e-6),
                  Evidence(rt, "lower", {"metric": "family_accuracy"}, expect=0.758139, tol=5e-6),
                  Evidence(rt, "value", {"metric": "beta_accuracy"}, expect=0.528889, tol=5e-6),
                  Evidence(rt, "value", {"metric": "exact_family_tie_rate"}, expect=0.222222, tol=5e-6)],
                 needs=["tie-aware posterior reporting"]),
            Rule("names-not-identification", "process sub-world (belief / audience / negative-slope descriptions)",
                 "Process names do not identify processes: 18 descriptions give only 12 observable classes at binary corners; corner reports carry 0 nats about process class and Bayes accuracy is exactly .500 at 1-3 repetitions; intermediate levels add 2e-5 nats (.502-.503).",
                 "NOT_IDENTIFIED",
                 [Evidence(f15, "value", {"world_id": "F09_process_subworld", "method": "E1", "estimand": "mutual_information_process_nats"}, expect=0.0, tol=1e-12),
                  Evidence(f15, "value", {"world_id": "F09_process_subworld", "method": "E2", "estimand": "mutual_information_process_nats"}, expect=1.98e-5, tol=1e-7),
                  Evidence(xb, "bayes_process_accuracy", {"reps_per_level": "3", "design": "corners_only"}, expect=0.5, tol=1e-9),
                  Evidence(xb, "bayes_process_accuracy", {"reps_per_level": "3", "design": "plus_intermediate_levels"}, expect=0.503152, tol=5e-6)],
                 needs=["a breaking observation, not more of the same reports"]),
            Rule("belief-material", "process sub-world with the belief B observed",
                 "Belief and audience accounts become separable only when the material records the belief itself: observing B gives .1089 nats and Bayes accuracy .642/.690/.727 at 1/2/3 repetitions. The extra measurement carries a stated independence assumption.",
                 "SUPPORTED_FINITE",
                 [Evidence(f15, "value", {"world_id": "F09_process_subworld", "method": "E3", "estimand": "mutual_information_process_nats"}, expect=0.108857, tol=5e-6),
                  Evidence(xb, "bayes_process_accuracy", {"reps_per_level": "1", "design": "plus_observed_belief"}, expect=0.642482, tol=5e-6),
                  Evidence(xb, "bayes_process_accuracy", {"reps_per_level": "3", "design": "plus_observed_belief"}, expect=0.727320, tol=5e-6)],
                 needs=["belief (B) observation in the material design"]),
            Rule("out-of-family-scope", "application main panel with 25% M4 (sum-of-coordinates law)",
                 "The calibrated out-of-family check holds its level in-family (.038) but detects M4 at .027, below its level. Non-rejection therefore never certifies family membership. The exact Neyman-Pearson bound shows the budget is not the binding limit: at the same 24 events a test calibrated on the prior mixture could reach at most .207/.450/.716 power for M4 beta .5/1/1.5, so the omnibus statistic, not the data, lacks power; a targeted M4 contrast is the candidate repair.",
                 "INCONCLUSIVE",
                 [Evidence(rt, "value", {"metric": "in_family_false_rejection"}, expect=0.037778, tol=5e-6),
                  Evidence(rt, "value", {"metric": "M4_detection_rate"}, expect=0.026667, tol=5e-6),
                  Evidence(pb, key=["beta1.0:r3:mixture_bound"], expect=0.4503, tol=5e-4),
                  Evidence(pb, key=["beta1.5:r3:mixture_bound"], expect=0.7156, tol=5e-4),
                  Evidence(pb, key=["beta0.5:r3:mixture_bound"], expect=0.2069, tol=5e-4)],
                 information="NP bounds are for a test that knows M4's form; they bound, not predict, any composite test"),
            Rule("targeted-m4", "declared M4 alternative, same 24-event data",
                 "A pre-declared targeted likelihood-ratio contrast, calibrated exactly at every in-family atom (level <= .043), detects M4 with exact power .119/.374/.658 (40-79% of the NP bound) and on the stored application authors .360 [.288,.439] versus .027 for the omnibus check, with in-family false rejection .040. Out-of-family checks must be targeted at declared rivals.",
                 "SUPPORTED_FINITE",
                 [Evidence(f"{F09}/targeted_m4_r1/targeted_m4.json", key=["max_level_over_atoms"], expect=0.0430, tol=5e-4),
                  Evidence(f"{F09}/targeted_m4_r1/targeted_m4.json", key=["exact_power", "beta1.0"], expect=0.3738, tol=5e-4),
                  Evidence(f"{F09}/targeted_m4_r1/targeted_m4.json", key=["stored_author_rates", "application_M4", "targeted_rate"], expect=0.36, tol=1e-9),
                  Evidence(f"{F09}/targeted_m4_r1/targeted_m4.json", key=["stored_author_rates", "application_in_family", "targeted_rate"], expect=0.04, tol=1e-9)],
                 information="detects only the declared alternative"),
            Rule("m4-budget", "M4 vs nearest in-family atom, exact",
                 "Even a test that is level .05 at every in-family atom is bounded above by .297 (beta .5) at the design budget; the bound rises to .472 and .719 with 48 and 96 events. Weak-slope departures need several times the current budget.",
                 "BOUNDS_ONLY",
                 [Evidence(pb, key=["beta0.5:r3:min_atom_bound", "power"], expect=0.2972, tol=5e-4),
                  Evidence(pb, key=["beta0.5:r6:min_atom_bound", "power"], expect=0.4719, tol=5e-4),
                  Evidence(pb, key=["beta0.5:r12:min_atom_bound", "power"], expect=0.7191, tol=5e-4)]),
        ],
        statuses={"implementation": "DONE", "protocol": "PARTIAL", "science": "SUPPORTED_FINITE"},
        gaps=[
            {"gap": "general (untargeted) out-of-family power remains low", "next": "declare rival families explicitly (targeted checks)"},
            {"gap": "explicit M4-label-out-of-posterior test (F09-04)", "next": "test addition"},
        ],
        non_claims=["belief/audience labels are simulated process descriptions, not psychological processes", "no parent card promoted"],
        handoffs=[{"to": "F06", "what": "candidate method set: expected information gain over observable classes (not over synonymous descriptions); belief-observation designs for process questions; targeted out-of-family contrasts"}, {"to": "F16", "what": "report tie sets and out-of-family non-rejection without membership claims"}],
    )


SPECS["F09-06"] = f09_06


F10 = f"{R}/F10"


def f10_05() -> Card:
    """F10-05 action-metric rules (F10-04 blocked)."""
    pd_ = f"{F10}/decision_r1/paired_decisions.csv"
    geo = f"{R}/F15/recovery_r2/F15-04/metrics.csv"
    sh = f"{F10}/decision_r1/shrinkage_choices.json"
    return Card(
        card_id="F10-05",
        package="F10",
        artifact="action_metric_rules.json",
        requirements=["R20", "R21", "A05"],
        depends_on=["F10-04"],
        rules=[
            Rule("zero-retained", "PV01 native vector path, 72 application pairs, task-then-pair interval primary",
                 "Neither the action-weighted nor the Euclidean vector geometry beats the zero reference on any target; zero stays the reference. Not beating zero means no promotion.",
                 "INCONCLUSIVE",
                 [Evidence(pd_, "decision_task_pair", {"part": "pv01_action_vs_zero", "query": "extraversion", "target": "donor_effect", "contrast": "weighted_minus_zero"}, expect="INCONCLUSIVE"),
                  Evidence(pd_, "decision_task_pair", {"part": "pv01_action_vs_zero", "query": "extraversion", "target": "donor_effect", "contrast": "euclidean_minus_zero"}, expect="HIGHER_LOSS"),
                  Evidence(pd_, "decision_task_pair", {"part": "pv01_action_vs_zero", "query": "extraversion", "target": "full_contrast", "contrast": "weighted_minus_zero"}, expect="INCONCLUSIVE")],
                 information="3 source / 6 application tasks; synthetic persona pairs"),
            Rule("action-beats-euclidean", "PV01, donor-effect target",
                 "The action (J-weighted) metric is a better relative geometry than Euclidean on the donor target (E -.000297, both resampling levels) - a retained narrow mechanism, not a deployable predictor.",
                 "SUPPORTED_LOWER_LOSS",
                 [Evidence(pd_, "delta_mse", {"part": "pv01_action_vs_zero", "query": "extraversion", "target": "donor_effect", "contrast": "weighted_minus_euclidean"}, expect=-0.000297, tol=5e-6),
                  Evidence(pd_, "decision_task_pair", {"part": "pv01_action_vs_zero", "query": "extraversion", "target": "donor_effect", "contrast": "weighted_minus_euclidean"}, expect="SUPPORTED_LOWER_LOSS"),
                  Evidence(pd_, "decision_task_pair", {"part": "pv01_action_vs_zero", "query": "neuroticism", "target": "donor_effect", "contrast": "weighted_minus_euclidean"}, expect="SUPPORTED_LOWER_LOSS")]),
            Rule("crossfit-shrinkage", "direct text-contrast transport",
                 "Select transport shrinkage by cross-fitting over source tasks, never by in-sample source loss: in-sample picks c=1 (degenerate), cross-fitting picks c=.25 (E) and 0 (N), and the cross-fitted transport still does not beat zero.",
                 "SUPPORTED_FINITE",
                 [Evidence(sh, key=[{"query": "extraversion"}, "c_in_sample"], expect=1.0, tol=0),
                  Evidence(sh, key=[{"query": "extraversion"}, "c_crossfit"], expect=0.25, tol=0),
                  Evidence(sh, key=[{"query": "neuroticism"}, "c_crossfit"], expect=0.0, tol=0),
                  Evidence(pd_, "decision_task_pair", {"part": "direct_text_contrast_shrinkage", "query": "extraversion", "target": "full_contrast", "contrast": "crossfit_minus_in_sample"}, expect="SUPPORTED_LOWER_LOSS")]),
            Rule("action-vs-variance-chart", "F10 controlled worlds (recomputed in F15-04)",
                 "A variance (PCA) chart can miss the acting directions entirely (linear world: PCA rank-2 .0642 vs zero .0639, action rank-2 exact); a first-order action chart fails when effects enter through second order in its null directions (curvature world: action .1347 vs PCA .0642; null-direction second-order mean square .22-.33).",
                 "COUNTEREXAMPLE",
                 [Evidence(geo, "value", {"section": "f10_recomputed", "world_id": "f10_linear", "role": "application", "method": "pca_rank2"}, expect=0.064151, tol=5e-6),
                  Evidence(geo, "value", {"section": "f10_recomputed", "world_id": "f10_linear", "role": "application", "method": "action_rank2"}, expect=0.0, tol=1e-12),
                  Evidence(geo, "value", {"section": "f10_recomputed", "world_id": "f10_null_curvature", "role": "application", "method": "action_rank2"}, expect=0.134678, tol=5e-6),
                  Evidence(geo, "value", {"section": "f10_geometry", "seed": "20260915", "world_id": "f10_null_curvature", "method": "action_rank2"}, expect=0.329615, tol=5e-6)],
                 needs=["second-order check in null directions before using a local action chart"]),
            Rule("derivative-triage", "F10-04 r2 (supervisor-authorized repair; 90 forwards + 6 backward, cumulative ~169)",
                 "Frozen rules label all six PV01 finite-difference failures FORMULA_ERROR, but the rule premise failed: the nominal FP64 arm rounds to float32 inside every Qwen2 RMSNorm/rotary/softmax. The analytic gradients agree across precisions (<=3.3e-5 relative); nominal-FP64 FD at the original step is within the original tolerance in 6/6 while MPS FP32 FD fails in 6/6 and replays the original values within .63%. Formula error is not supported; a precision explanation is consistent but not certified.",
                 "INCONCLUSIVE",
                 [Evidence(f"{F10}/derivative_triage_r2/accounting.json", key=["r2_forward_attempts"], expect=90, tol=0),
                  Evidence(f"{F10}/derivative_triage_r2/accounting.json", key=["r2_backward_attempts"], expect=6, tol=0),
                  Evidence(f"{F10}/derivative_triage_r2/accounting.json", key=["cumulative_estimated_forwards"], expect=169, tol=0),
                  Evidence(f"{F10}/derivative_triage_r2/triage.json", key=[{"pair": "extraversion|described|advice|s11", "query": "extraversion"}, "classification"], expect="FORMULA_ERROR"),
                  Evidence(f"{F10}/derivative_triage_r2/diagnostic_posthoc.json", key=["summary", "fp64arm_fd_0.01_within_tolerance"], expect=6, tol=0),
                  Evidence(f"{F10}/derivative_triage_r2/diagnostic_posthoc.json", key=["summary", "mps_fails_original_tolerance"], expect=6, tol=0),
                  Evidence(f"{F10}/derivative_triage_r2/diagnostic_posthoc.json", key=["summary", "max_analytic_rel_disagreement"], expect=3.27e-5, tol=5e-7),
                  Evidence(f"{F10}/derivative_triage_r2/diagnostic_posthoc.json", key=["summary", "qwen2_rmsnorm_casts_to_float32"], expect=True)],
                 information="post-hoc diagnostic does not relabel; no third rerun; end-to-end FP64 check would be a new experiment"),
        ],
        statuses={"implementation": "DONE", "protocol": "PARTIAL", "science": "INCONCLUSIVE"},
        gaps=[
            {"gap": "F10-04 frozen rules assumed an end-to-end FP64 arm; certifying the precision reading needs a patched FP64 model", "next": "new experiment only if requested; no third rerun"},
            {"gap": "native four-edit signatures incl. missing audience edit", "next": "blocked: source native interventions and fitted weights were never saved"},
            {"gap": "c x PV01-native transport", "next": "blocked for the same reason"},
        ],
        non_claims=["vector path and GRAIN action meaning are separate objects; not beating zero is not a GRAIN-wide failure", "synthetic persona pairs; no human population", "no parent card promoted"],
        handoffs=[{"to": "F16", "what": "downstream object: zero reference plus action metric as a relative geometry only; cross-fitted shrinkage rule"}],
    )


SPECS["F10-05"] = f10_05


F11 = f"{R}/F11"


def f11_05() -> Card:
    """F11-05 reader-error rules and F12/F13 interface."""
    aud = f"{F11}/mtmm_audit_r1/summary.json"
    anc = f"{F11}/anchor_audit_r1/summary.json"
    ancs = f"{F11}/anchor_full_r2/summary.json"
    su = f"{F11}/source_uncertainty_r1/coverage_summary.csv"
    ag = f"{F11}/reader_interface_r1/agreement.csv"
    return Card(
        card_id="F11-05",
        package="F11",
        artifact="reader_error_rules.json",
        requirements=["R16", "R20", "R29", "A06"],
        depends_on=["F11-04"],
        extra_checks=[pytest_check("tests/test_grain_gptpro_f11_mtmm.py", "tests/test_grain_gptpro_f11_anchor.py", "tests/test_grain_gptpro_f11_boundary.py")],
        rules=[
            Rule("sum-only", "author-constant common method variance, loading 1",
                 "Author-constant method variance shared by all readers and occasions is observationally identical to extra trait covariance (covariance discrepancy exactly 0 at tau=.25): from text readings alone only the sum is identified; multi-reader agreement cannot separate it.",
                 "NOT_IDENTIFIED",
                 [Evidence(aud, key=["constant_method_equivalence_error"], expect=0.0, tol=0)],
                 needs=["independent anchor or design that breaks the sum"]),
            Rule("anchor-calibration", "anchored worlds A/B, source anchors assumed independent",
                 "An independent source anchor raises design rank 8 -> 9 and calibrates the split: where common variation is method (A) anchoring lowers trait MSE -.0526 [-.0718,-.0340]; where it is all trait (B) it changes nothing (-.0001, inconclusive). Anchors are design inputs with an independence assumption, never chosen from application data.",
                 "SUPPORTED_LOWER_LOSS",
                 [Evidence(ancs, key=["source_rank_without_anchor"], expect=8, tol=0),
                  Evidence(ancs, key=["source_rank_with_anchor"], expect=9, tol=0),
                  Evidence(anc, key=["paired_intervals", {"world": "A"}, "mse_delta"], expect=-0.0526, tol=5e-5),
                  Evidence(anc, key=["paired_intervals", {"world": "A"}, "ci_high"], expect=-0.0340, tol=5e-5),
                  Evidence(anc, key=["paired_intervals", {"world": "B"}, "ci_low"], expect=-0.000525, tol=5e-6),
                  Evidence(ancs, key=["independence_of_anchor_is_assumed"], expect=True)],
                 needs=["anchor independent of reader/method error"]),
            Rule("event-common-error", "shared-event world",
                 "Readers of the same event share that event's deviation: three readers of one occasion leave trait MSE at .270 versus .179 with two occasions (seed 20260915, shared-event world). Replication across events, not across readers of the same event, removes event-specific error.",
                 "SUPPORTED_FINITE",
                 [Evidence(f"{F11}/mtmm_full_r1/metrics.csv", "mse", {"seed": "20260915", "world": "shared_event", "method": "reml", "role": "application", "panel": "all_two_occasions"}, expect=0.179475, tol=5e-6),
                  Evidence(f"{F11}/mtmm_full_r1/metrics.csv", "mse", {"seed": "20260915", "world": "shared_event", "method": "reml", "role": "application", "panel": "same_event_three_readers"}, expect=0.270093, tol=5e-6)],
                 needs=["multiple occasions"], information="panels differ in cell budget (18 vs 9 cells), so this is a mechanism illustration, not an equal-cost comparison; rank-9 separability is from the F11 MTMM report"),
            Rule("reader-accuracy-not-validity", "two measured readers on 96 fixed materials (application role)",
                 "Readers agree strongly on stance (pole r .89) but weakly on qualification (r .22); agreement is a property of the readers on these materials, not evidence for a latent trait or external validity.",
                 "SUPPORTED_FINITE",
                 [Evidence(ag, "reader_r", {"role": "application", "target": "pole"}, expect=0.8939, tol=5e-4),
                  Evidence(ag, "reader_r", {"role": "application", "target": "qualification"}, expect=0.2217, tol=5e-4)],
                 information="multi-model consensus is never external validity"),
            Rule("source-uncertainty-small", "MTMM trait intervals, 300 source authors",
                 "Source-refit uncertainty is a small component of trait intervals (1.4-2.9% of variance; coverage +.001-.004). The conditional-loading under-coverage is shared with the exact oracle and is sampling variation (oracle .9498 at 200,000 authors), not source uncertainty.",
                 "SUPPORTED_FINITE",
                 [Evidence(su, "coverage", {"world": "conditional_loading", "panel": "all_two_occasions", "arm": "plugin"}, expect=0.9289, tol=5e-4),
                  Evidence(su, "coverage", {"world": "conditional_loading", "panel": "all_two_occasions", "arm": "augmented"}, expect=0.9322, tol=5e-4),
                  Evidence(su, "source_var_share", {"world": "base", "panel": "all_two_occasions", "arm": "augmented"}, expect=0.029, tol=5e-4),
                  Evidence(f"{F11}/source_uncertainty_r1/oracle_exactness_check.json", key=["conditional_loading:all_two_occasions", "coverage"], expect=0.9498, tol=5e-4)]),
            Rule("reml-robustness", "source bootstrap refits",
                 "The canonical REML fails on 6.6% of source resamples (non-convergence; eigenvalue-tolerance exceptions concentrated in one conditional cell). Report refit failure rates with every interval; cells above 10% are not used.",
                 "SUPPORTED_FINITE",
                 [Evidence(f"{F11}/source_uncertainty_r1/checks.json", key=["bootstrap_failures"], expect=118, tol=0),
                  Evidence(f"{F11}/source_uncertainty_r1/checks.json", key=["failures_by_cell", "20260917_conditional_loading"], expect=33, tol=0)]),
        ],
        statuses={"implementation": "DONE", "protocol": "PARTIAL", "science": "SUPPORTED_FINITE"},
        gaps=[
            {"gap": "protocol-form loading world lambda_r(z)=1+.3rz not run (implemented loading tied to occasion)", "next": "recorded deviation; a new world would be a new protocol"},
            {"gap": "GRAIN reader has no same-material output in approved caches", "next": "missing reader stays missing (not synthesized)"},
            {"gap": "robust REML re-parameterisation", "next": "new estimator revision if needed"},
        ],
        non_claims=["simulated MTMM worlds and two readers on fixed synthetic materials", "no human reliability or trait validity", "no parent card promoted"],
        handoffs=[
            {"to": "F12", "what": "which errors can be calibrated by common materials: reader-specific offsets and event-specific deviations (with occasions); author-constant method only with an independent anchor"},
            {"to": "F13", "what": "error interface: trait-interval components = posterior + source refit (small) + reader/event facets; failure rates reported"},
        ],
    )


SPECS["F11-05"] = f11_05


F12 = f"{R}/F12"


def f12_derived(root):
    """Three-seed mean application metrics per world x method from metrics.csv."""
    import collections

    import numpy as np

    rows = []
    cells = collections.defaultdict(list)
    for r in load(root, f"{F12}/transport_full_r1/metrics.csv", role="application"):
        cells[(r["world"], r["method"])].append(r)
    for (world, method), rs in sorted(cells.items()):
        def mean(k, rs=rs):
            v = [float(r[k]) for r in rs if r[k] not in ("", "nan")]
            return float(np.mean(v)) if v else ""

        rows.append({"quantity": f"app:{world}:{method}", "delta": mean("delta_mse"), "mean": mean("delta_coverage"),
                     "status": mean("delta_spearman"), "n_units": sum(int(r["supported_authors"]) for r in rs)})
    return rows, [f"{F12}/transport_full_r1/metrics.csv"]


def f12_05() -> Card:
    """F12-05 transport rules and H03/H05 contracts (handoff)."""
    dif = f"{F12}/dif_table_r1/summary.json"
    return Card(
        card_id="F12-05",
        package="F12",
        artifact="transport_rules.json",
        requirements=["R21", "R22", "R23", "R28", "H03", "H05"],
        depends_on=["F12-04"],
        derived=[f12_derived],
        extra_checks=[pytest_check("tests/test_grain_gptpro_f12_transport.py", "tests/test_grain_gptpro_f12_batch.py", "tests/test_grain_gptpro_f12_signatures.py")],
        rules=[
            Rule("direct", "identity channel",
                 "When the new reader/domain reads materials in the old unit, scores transport directly: calibration changes nothing (Delta MSE .0746, coverage .958 for all methods) and conditional DIF sits at the noise floor (.018).",
                 "SUPPORTED_FINITE",
                 [Evidence("derived:app:identity:uncalibrated", "delta", expect=0.07455, tol=5e-5),
                  Evidence("derived:app:identity:global", "delta", expect=0.07455, tol=5e-5),
                  Evidence(dif, key=["identity:uncalibrated", "mean_dif_range"], expect=0.0175, tol=5e-4)],
                 needs=["common materials showing the unit is unchanged"]),
            Rule("global-pilot", "global affine channel 1+1.5r",
                 "A global unit change needs a pilot calibration on common materials: uncalibrated coverage .43 (Delta MSE .887) is restored by the global inverse (.0746, .958). Its DIF (.208) is condition-free and removable by a global renorm (to the .045 floor); ranking is preserved (Spearman .985) while absolute error is not.",
                 "SUPPORTED_LOWER_LOSS",
                 [Evidence("derived:app:global:uncalibrated", "mean", expect=0.43, tol=5e-3),
                  Evidence("derived:app:global:global", "delta", expect=0.07455, tol=5e-5),
                  Evidence(dif, key=["global:uncalibrated", "mean_dif_range"], expect=0.2085, tol=5e-4),
                  Evidence(dif, key=["global:uncalibrated", "mean_dif_range_after_oracle_renorm"], expect=0.0450, tol=5e-4),
                  Evidence("derived:app:global:uncalibrated", "status", expect=0.9848, tol=5e-4)],
                 needs=["pilot pairs on common materials"]),
            Rule("conditional-pilot", "conditional channel z+(1+.25z)r",
                 "A condition-dependent channel needs a conditional pilot with anchors at every condition: global calibration leaves Delta MSE 5.246 (coverage .007) and the conditional DIF survives even the best possible global renorm (1.354 vs .045 floor); only the conditional inverse restores .0746. A pilot without condition variation returns MISSING_ANCHOR (rank 2 of 4).",
                 "SUPPORTED_LOWER_LOSS",
                 [Evidence("derived:app:conditional:global", "delta", expect=5.24618, tol=5e-4),
                  Evidence("derived:app:conditional:conditional", "delta", expect=0.07455, tol=5e-5),
                  Evidence(dif, key=["conditional:global", "mean_dif_range_after_oracle_renorm"], expect=1.3540, tol=5e-4),
                  Evidence(dif, key=["conditional:conditional", "mean_dif_range"], expect=0.0175, tol=5e-4),
                  Evidence(f"{F12}/transport_full_r1/calibration_20260915_conditional_missing_condition_certificate.json", key=["status"], expect="MISSING_ANCHOR"),
                  Evidence(f"{F12}/transport_full_r1/calibration_20260915_conditional_missing_condition_certificate.json", key=["rank"], expect=2, tol=0)],
                 needs=["condition-crossed pilot anchors"], information="a new-domain norm cannot erase conditional DIF"),
            Rule("not-transportable", "nonlinear channel r+.1r^3",
                 "With the declared estimators the nonlinear channel is not transportable: the prescribed cubic inverse is nonmonotone on the pilot range (min derivative -1.09) and is refused (0 supported scores); global/conditional linear inverses leave miscalibration (coverage .653/.628) and residual DIF after renorm (.145/.226 vs .045 floor).",
                 "NOT_IDENTIFIED",
                 [Evidence(f"{F12}/transport_full_r1/calibration_20260915_nonlinear_cubic.json", key=["status"], expect="NONMONOTONE_INVERSE"),
                  Evidence(f"{F12}/transport_full_r1/calibration_20260915_nonlinear_cubic.json", key=["min_derivative_on_pilot_range"], expect=-1.0949, tol=5e-4),
                  Evidence(dif, key=["nonlinear:cubic", "seeds_ok"], expect=0, tol=0),
                  Evidence("derived:app:nonlinear:global", "mean", expect=0.6533, tol=5e-4),
                  Evidence(dif, key=["nonlinear:conditional", "mean_dif_range_after_oracle_renorm"], expect=0.2258, tol=5e-4)],
                 information="refusal, not a fabricated inverse; a monotone nonlinear calibrator would be a new estimator"),
            Rule("batch-support", "F12 transport_score route",
                 "Scores are per-author functions of the frozen pilot calibration: a new batch never changes existing scores or support; unseen conditions and out-of-pilot-range cubic readings are unsupported, never extrapolated (tests); full score/support tables replay to 4.0e-15.",
                 "SUPPORTED_FINITE",
                 [Evidence(f"{F12}/table_audit_r1/summary.json", key=["max_difference"], expect=4.0e-15, tol=1e-15),
                  Evidence(f"{F12}/table_audit_r1/summary.json", key=["missing_scalars"], expect=23040, tol=0)]),
        ],
        statuses={"implementation": "DONE", "protocol": "PARTIAL", "science": "SUPPORTED_FINITE"},
        gaps=[
            {"gap": "source/pilot fitting uncertainty not in F12 intervals", "next": "F16 text/numeric transport carries calibration draws"},
            {"gap": "multilingual descriptive link (F14 X authors with >1 language tag) not executed", "next": "descriptive only; no same-human language evidence -> no invariance claim"},
            {"gap": "'four interventions' only span rank 2 (chosen post-audit)", "next": "recorded contradiction"},
        ],
        non_claims=["synthetic channel worlds; pilot pairs are deterministic transforms, not independent readers", "no cross-language invariance claimed", "no parent card promoted"],
        handoffs=[
            {"to": "H03", "what": "same-person cross-language: needed columns person_id, language, material_id, condition, reading, reader_id, pilot_flag; estimand = conditional DIF by language after pilot calibration; see F17-03"},
            {"to": "H05", "what": "use consequences/fairness: group_id, decision, score, support flag, DIF table by group; estimand = conditional DIF surviving global renorm; see F17-03"},
            {"to": "F16", "what": "transport rules: direct / global pilot / conditional pilot / refuse"},
        ],
    )


SPECS["F12-05"] = f12_05


F13 = f"{R}/F13"


def f13_null_bounds_check(root):
    """Deviation decision: code returns null (no finite bounds) without a declared domain."""
    src = (root / "grain_research/gptpro_foundation_v3/f13_history_contract.py").read_text()
    ok = "no finite bounds without an explicit parameter domain" in src
    return "null_without_declared_domain_in_code", ok, "f13_history_contract.py declares null + no finite bounds absent an explicit parameter domain"


def f13_05() -> Card:
    """F13-05 uncertainty rules and F16 publishable fields."""
    r3 = f"{F13}/recovery3_r1"
    r2 = f"{F13}/recovery2_r1"
    mf = f"{F13}/material_facet_r1/material_facet.json"
    return Card(
        card_id="F13-05",
        package="F13",
        artifact="uncertainty_rules.json",
        requirements=["R12", "R24", "R25", "R29"],
        depends_on=["F13-04"],
        extra_checks=[f13_null_bounds_check, pytest_check("tests/test_grain_gptpro_f13_contract.py", "tests/test_grain_gptpro_f13_history_contract.py", "tests/test_grain_gptpro_f13_uncertainty.py")],
        rules=[
            Rule("components-separate", "binary next-p (B1/B2/M), C2 worlds",
                 "Publish next-p intervals with separate state, personal and source components: complete-history B1 intervals (source only) under-cover the true next-p (.914 xor), while with missing data the lag-state component dominates and coverage rises (.960-.975).",
                 "SUPPORTED_FINITE",
                 [Evidence(f"{r3}/summary.csv", "cover", {"world": "c2_xor", "mechanism": "none", "candidate": "b1"}, expect=0.914, tol=5e-4),
                  Evidence(f"{r3}/summary.csv", "cover", {"world": "c2_xor", "mechanism": "MCAR", "candidate": "b1"}, expect=0.960, tol=5e-4),
                  Evidence(f"{r3}/summary.csv", "state", {"world": "c2_xor", "mechanism": "MCAR", "candidate": "b1"}, expect=0.0320, tol=5e-4)],
                 needs=["component-labelled variance fields"]),
            Rule("working-interval-not-reliability", "B2 ridge working posterior",
                 "B2's working personal variance (sigma^2=.25) gives nearly uninformative intervals (width .58-.86) that over-cover; such an interval must not be published as a reliability statement.",
                 "INFORMATION_LOSS",
                 [Evidence(f"{r3}/summary.csv", "width", {"world": "c2_xor", "mechanism": "none", "candidate": "b2"}, expect=0.578, tol=5e-4),
                  Evidence(f"{r3}/summary.csv", "cover", {"world": "c2_xor", "mechanism": "none", "candidate": "b2"}, expect=0.990, tol=5e-4)]),
            Rule("mnar-personal", "binary next-p under MNAR vs MCAR (matched missing rate)",
                 "Response-dependent missingness enters through personal parameters: bias vs MCAR B2 -.124, M -.066 (xor) while B1 -.019; MAR is effectively ignorable. The dynamic T3 forecast shows the same direction in F13 recovery 2 (see that report). Declare the missingness assumption as a published field.",
                 "SUPPORTED_FINITE",
                 [Evidence(f"{r3}/contrasts.csv", "delta", {"world": "c2_xor", "candidate": "b2", "contrast": "MNAR-MCAR:bias"}, expect=-0.1235, tol=5e-4),
                  Evidence(f"{r3}/contrasts.csv", "delta", {"world": "c2_xor", "candidate": "b1", "contrast": "MNAR-MCAR:bias"}, expect=-0.0194, tol=5e-4),
                  Evidence(f"{r3}/contrasts.csv", "decision", {"world": "c2_xor", "candidate": "m", "contrast": "MAR-MCAR:bias"}, expect="INCONCLUSIVE")],
                 needs=["missingness mechanism declaration"]),
            Rule("mdc-not-clinical", "F04/F05 T1 change",
                 "Minimum detectable change (MDC95 about .84 raw, .82-.89 reference SD) is a measurement-precision threshold, not clinical or psychological significance.",
                 "SUPPORTED_FINITE",
                 [Evidence(f"{r2}/minimum_change.csv", "mean_mdc95", {"seed": "20260915", "method": "noise_predictive"}, expect=0.8402, tol=5e-4),
                  Evidence(f"{r2}/minimum_change.csv", "mdc95_in_reference_sd", {"seed": "20260915", "method": "noise_predictive"}, expect=0.8860, tol=5e-4)]),
            Rule("material-facet", "text route (F16 pilot channel, NLI arm)",
                 "Template-level material sampling is already covered by the Jeffreys count uncertainty (design effect .81-.96 on informative cells), but the channel is a mixture over material subtypes: misread rates are 0 vs .625 (hypothetical vs quoted negatives) and 0 vs .5 (negated vs actual positives). A different material mix moves the channel anywhere in that range; publish the material-mix field and restrict scores to matched mixes or use subtype-stratified channels.",
                 "SUPPORTED_FINITE",
                 [Evidence(mf, key=["counts_match_stored_channel"], expect=True),
                  Evidence(mf, key=["design_effect_cluster_over_dirichlet", "max"], expect=0.9641, tol=5e-4),
                  Evidence(mf, key=["misread_sensitivity_to_material_mix", "pole0_qual0", "error_rate_by_status", "quoted"], expect=0.625, tol=1e-9),
                  Evidence(mf, key=["misread_sensitivity_to_material_mix", "pole1_qual1", "error_rate_by_status", "actual"], expect=0.5, tol=1e-9)],
                 needs=["material subtype mix recorded with every score"]),
            Rule("source-refit-small", "MTMM traits (F11)",
                 "Source-refit uncertainty is a small component at 300 source authors (1.4-2.9% of trait interval variance); report it with the refit failure rate.",
                 "SUPPORTED_FINITE",
                 [Evidence(f"{R}/F11/source_uncertainty_r1/coverage_summary.csv", "source_var_share", {"world": "base", "panel": "all_two_occasions", "arm": "augmented"}, expect=0.029, tol=5e-4)]),
            Rule("null-not-zero", "scores without evidence",
                 "Null is not zero: with no qualified evidence a score is null (or prior_only when a declared model supplies it); finite bounds are reported only when a parameter domain is declared (e.g. probabilities on [0,1]). Deviation decision: the code's 'no finite bounds without a declared domain' is accepted as the correct refinement of the protocol's 'null + partial bounds'.",
                 "SUPPORTED_FINITE",
                 [Evidence(f"{F13}/cold_contract_full_r1/summary.json", key=["score_annotation_counts", "FAIL"], expect=33800, tol=0),
                  Evidence(f"{F13}/cold_contract_full_r1/summary.json", key=["scores"], expect=86400, tol=0)],
                 information="33,800 of 86,400 cold-contract scores carry FAIL calibration annotations and are retained, not handed over as reliable"),
        ],
        statuses={"implementation": "DONE", "protocol": "PARTIAL", "science": "SUPPORTED_FINITE"},
        gaps=[
            {"gap": "cross-day / cross-occasion real validation of intervals", "next": "H01 (human stage; F17-03 contract)"},
            {"gap": "person-specific-law binary worlds; calibrated B2 personal variance", "next": "optional F13 packet"},
            {"gap": "subtype-stratified text channels not implemented", "next": "F16 production rule; implement if a mismatched material mix is used"},
        ],
        non_claims=["simulation coverage is not human reliability", "MDC is not clinical significance", "no parent card promoted"],
        handoffs=[{"to": "F16", "what": "publishable fields: point; variance components (state, personal, source/calibration draws, material-mix flag); support; prior_only/null status; missingness assumption; refit failure rate; H01 pending flag"},
                  {"to": "H01", "what": "same-person cross-occasion data to validate interval coverage; see F17-03"}],
    )


SPECS["F13-05"] = f13_05


F14 = f"{R}/F14"


def f14_register(root):
    """Human-pending register (separate from agent work) derived from F14 metadata."""
    d = __import__("json").loads((root / F14 / "metadata_full_r2/summary.json").read_text())["datasets"]
    rows = [
        {"quantity": "register:H01", "status": "HUMAN_PENDING", "mean": d["pandora"]["authors_multiple_days"],
         "delta": "same-person cross-occasion validation needs designed occasions; PANDORA/X have calendar-ordered posts (observed, not designed occasions)"},
        {"quantity": "register:H02", "status": "HUMAN_PENDING", "mean": 0,
         "delta": "behavioural criteria and blind coding: no dataset carries external behavioural criteria"},
        {"quantity": "register:H03", "status": "HUMAN_PENDING", "mean": d["x_market"]["authors_multiple_languages"],
         "delta": "same-person cross-language: X authors with >1 language tag exist but not same-content translations; descriptive link only"},
        {"quantity": "register:H04", "status": "HUMAN_PENDING", "mean": 0,
         "delta": "questioning reactivity/burden: no dataset records questioning or refusal"},
        {"quantity": "register:H05", "status": "HUMAN_PENDING", "mean": 0,
         "delta": "use consequences/fairness: no decisions or group outcomes recorded"},
        {"quantity": "register:H06", "status": "HUMAN_PENDING", "mean": 0,
         "delta": "external team rebuild: requires an independent team; SERIALIZE bundles/handbook are the handoff"},
    ]
    return rows, [f"{F14}/metadata_full_r2/summary.json"]


def f14_04() -> Card:
    """F14-04 natural support matrix review."""
    m = f"{F14}/metadata_full_r2/natural_support_matrix.csv"
    s = f"{F14}/metadata_full_r2/summary.json"
    rr = f"{F14}/reference_replay_r1/summary.json"
    return Card(
        card_id="F14-04",
        package="F14",
        artifact="natural_support_matrix_review.json",
        requirements=["R27", "H01", "H02", "H03"],
        depends_on=["F14-03"],
        derived=[f14_register],
        extra_checks=[pytest_check("tests/test_grain_gptpro_f14_metadata.py", "tests/test_grain_gptpro_f14_role.py", "tests/test_grain_gptpro_f14_cache_interface.py", "tests/test_grain_gptpro_f14_reference.py")],
        rules=[
            Rule("pandora-x-observed", "PANDORA and X metadata",
                 "PANDORA and X support observed author-condition profiles and calendar-ordered expression (PANDORA 10,199 authors with multiple conditions, 10,239 on multiple dates; X 35,641 / 49,582); these are observed selections, not designed occasions or causal contrasts.",
                 "SUPPORTED_FINITE",
                 [Evidence(m, "support", {"dataset": "pandora", "target": "observed_author_condition_profile"}, expect="METADATA_SUPPORT"),
                  Evidence(s, key=["datasets", "pandora", "authors_multiple_conditions"], expect=10199, tol=0),
                  Evidence(s, key=["datasets", "x_market", "authors_multiple_days"], expect=49582, tol=0)],
                 needs=["F16 text observation mapping before any numeric target"]),
            Rule("no-exposure", "all four datasets",
                 "No dataset records opportunities or non-response: opportunity-adjusted response targets are not supported anywhere (F03 published-only rule applies); do not invent exposure.",
                 "NOT_IDENTIFIED",
                 [Evidence(m, "support", {"dataset": "pandora", "target": "complete_opportunity_adjusted_response"}, expect="MISSING_SUPPORT"),
                  Evidence(m, "support", {"dataset": "x_market", "target": "complete_opportunity_adjusted_response"}, expect="MISSING_SUPPORT"),
                  Evidence(s, key=["datasets", "pandora", "all_opportunities_observed"], expect=False)]),
            Rule("no-direct-numeric", "all datasets, T1/T2/T3",
                 "Metadata give addresses, not calibrated continuous observations: T1/T2/T3 cannot be scored directly from natural data without the text observation mapping (F16-R02) and its reader law.",
                 "NOT_IDENTIFIED",
                 [Evidence(m, "support", {"dataset": "pandora", "target": "T1_T2_T3_gaussian_score_directly"}, expect="MISSING_SUPPORT"),
                  Evidence(m, "support", {"dataset": "meps", "target": "T1_T2_T3_gaussian_score_directly"}, expect="MISSING_SUPPORT")]),
            Rule("essays-meps-limits", "Essays and MEPS metadata",
                 "Essays carries single texts (0 authors with multiple conditions or dates); MEPS has 46 authors with multiple conditions but no dates. Neither supports cross-occasion targets; MEPS raw text/questionnaires were not read.",
                 "NOT_IDENTIFIED",
                 [Evidence(s, key=["datasets", "essays", "authors_multiple_conditions"], expect=0, tol=0),
                  Evidence(s, key=["datasets", "meps", "authors_multiple_conditions"], expect=46, tol=0),
                  Evidence(s, key=["datasets", "meps", "authors_multiple_days"], expect=0, tol=0)]),
            Rule("cold-addresses", "cached numeric readouts",
                 "Existing cached numeric readouts replay exactly under cold scoring (PANDORA 22,464 units, Essays 13,790; max error 0 / 6e-17) and are handed to F16 as cold-scoring addresses; held-author coverage of the full target is low (median .046).",
                 "SUPPORTED_FINITE",
                 [Evidence(rr, key=["datasets", {"dataset": "pandora"}, "units"], expect=22464, tol=0),
                  Evidence(rr, key=["datasets", {"dataset": "pandora"}, "score_replay_max_error"], expect=0.0, tol=0),
                  Evidence(rr, key=["datasets", {"dataset": "pandora"}, "coverage", "full", "median"], expect=0.0460, tol=5e-4)]),
            Rule("human-register", "H01-H06",
                 "Human-stage items are registered separately (derived_metrics register:H01-H06) and are not used to close agent-executable work: every agent item keeps a next card.",
                 "HUMAN_PENDING",
                 [Evidence("derived:register:H03", "mean", expect=5082, tol=0),
                  Evidence("derived:register:H01", "status", expect="HUMAN_PENDING")]),
        ],
        statuses={"implementation": "DONE", "protocol": "PARTIAL", "science": "SUPPORTED_FINITE"},
        gaps=[
            {"gap": "X loader dedupe migration", "next": "engineering; counts unaffected for these targets"},
            {"gap": "multilingual descriptive link (X 5,082 authors)", "next": "descriptive only (F12-05 gap)"},
        ],
        non_claims=["metadata only: no private text or questionnaire values read", "support is not validity", "no parent card promoted"],
        handoffs=[{"to": "F16", "what": "cold-scoring addresses (PANDORA/Essays cached readouts) and support flags"}, {"to": "F17-03", "what": "human-pending register H01-H06"}],
    )


SPECS["F14-04"] = f14_04


F16 = f"{R}/F16"


def f16_06() -> Card:
    """F16-06 construction acceptance: per-target verdicts, production vs human use."""
    t1 = f"{F16}/t1_source_audit_r1/summary.json"
    t2 = f"{F16}/t2_cold_audit_r1/summary.json"
    t3 = f"{F16}/t3_application_audit_r1/summary.json"
    ser = f"{F16}/serialize_audit_r4/summary.json"
    return Card(
        card_id="F16-06",
        package="F16",
        artifact="construction_acceptance.json",
        requirements=["R01", "R32", "R33", "R34"],
        depends_on=["F16-05"],
        extra_checks=[pytest_check("tests/test_grain_gptpro_f16_serialize.py", "tests/test_grain_gptpro_f16_c2_rejection.py")],
        rules=[
            Rule("T1-construction", "T1 conditional contrast, world L, 3 seeds x 200 application authors",
                 "T1 is constructed and cold-replayable (cold/warm error 0); its source-mixture interval passes the coverage gate in 2 of 3 seeds (.955, .980) and fails in seed 20260915 (.910, Wilson [.862,.942]); the failed seed is retained, and more identical resampling is not the next mechanism.",
                 "SUPPORTED_FINITE",
                 [Evidence(t1, key=["cells", {"seed": 20260915}, "analytic_source_mixture", "coverage"], expect=0.91, tol=1e-9),
                  Evidence(t1, key=["cells", {"seed": 20260915}, "analytic_source_mixture", "decision"], expect="FAIL"),
                  Evidence(t1, key=["cells", {"seed": 20260916}, "analytic_source_mixture", "decision"], expect="PASS"),
                  Evidence(t1, key=["cells", {"seed": 20260917}, "analytic_source_mixture", "decision"], expect="PASS")],
                 information="finite construction closure; not human usability"),
            Rule("T2-construction", "T2 current state, core world",
                 "T2 is constructed and cold-replayable; coverage gates pass in 2 of 3 seeds (seed 20260915 .925 PASS; one seed FAIL retained).",
                 "SUPPORTED_FINITE",
                 [Evidence(t2, key=["cells", {"seed": 20260915}, "coverage"], expect=0.925, tol=1e-9),
                  Evidence(t2, key=["cells", {"seed": 20260915}, "decision"], expect="PASS"),
                  Evidence(t2, key=["cells", {"seed": 20260916}, "decision"], expect="FAIL")]),
            Rule("T3-rebuild", "T3 four-step planned future, independent rebuild",
                 "A separate agent rebuilt T3 from brief, handbook, source data and bundle schema only; the application passes coverage (seed 20260915 .945) and the mismatched historical-outcome diagnostic fails (.705), showing the target plan matters. The rebuild is an engineering replication by another agent, not H06 external-team rebuild.",
                 "SUPPORTED_FINITE",
                 [Evidence(t3, key=["cells", {"seed": 20260915}, "planned_outcome", "coverage"], expect=0.945, tol=1e-9),
                  Evidence(t3, key=["cells", {"seed": 20260915}, "planned_outcome", "decision"], expect="PASS"),
                  Evidence(t3, key=["cells", {"seed": 20260915}, "mismatched_historical_outcome_diagnostic_only", "coverage"], expect=0.705, tol=1e-9)],
                 information="H06 (human external team) remains pending"),
            Rule("public-bundle", "SERIALIZE public calibrated bundles",
                 "162 public calibrated bundles replay the archived outputs (max 1.14e-13) under sealed cold scoring and match independent formulas (<= 6.97e-13); all conform to the derived bundle schema (consistency check, schema derived from serializer + bundles); an old C2 binary-history request is rejected by a T1 bundle (test).",
                 "SUPPORTED_FINITE",
                 [Evidence(ser, key=["replay_max_abs_diff"], expect=1.14e-13, tol=1e-15),
                  Evidence(ser, key=["sealing_pass"], expect=True),
                  Evidence(f"{F16}/bundle_schema_r1/conformance.json", key=["bundles"], expect=162, tol=0),
                  Evidence(f"{F16}/bundle_schema_r1/conformance.json", key=["violations"], expect="[]")]),
            Rule("briefs-consolidated", "F16-01",
                 "The three construction briefs are consolidated from frozen sources with hashes (T1 world L; T2/T3 core; GP06 wrong-world brief rejected); target functions and units precede application scores.",
                 "SUPPORTED_FINITE",
                 [Evidence(f"{F16}/construction_briefs_r2/construction_briefs.json", key=["T1", "target_id"], expect="T1"),
                  Evidence(f"{F16}/construction_briefs_r2/construction_briefs.json", key=["T3", "target_id"], expect="T3")]),
            Rule("human-usability", "all targets",
                 "Finite production closure is not human usability: no target has human validity, external rebuild or behavioural interpretation evidence; interpretations do not inherit unrelated historical results.",
                 "HUMAN_PENDING",
                 [Evidence(f"{F16}/t2_cold_full_r1/summary.json", key=["cells", 0, "human_validity"], expect="NOT_TESTED")]),
        ],
        statuses={"implementation": "DONE", "protocol": "PARTIAL", "science": "SUPPORTED_FINITE"},
        gaps=[
            {"gap": "evidence-card schema alignment (event_address/subject_status/reader_id/contribution) for legacy F16-04 cards", "next": "SERIALIZE explanation blocks carry these for served bundles; legacy cards retained"},
            {"gap": "real T1 selection card (legacy no-op)", "next": "recorded"},
            {"gap": "H06 external team rebuild; H01 validation", "next": "F17-03 contracts"},
        ],
        non_claims=["construction verdicts are finite simulation verdicts", "failed coverage seeds retained", "no parent card promoted"],
        handoffs=[{"to": "F17", "what": "per-target construction verdicts; production rules doc docs/GRAIN_F16_TARGET_PRODUCTION_RULES_V1.md (+ 2026-09-26 addendum) and handbook docs/GRAIN_F16_PUBLIC_CALIBRATED_BUNDLE_HANDBOOK_V1.md; bundle schema docs/GRAIN_F16_BUNDLE_SCHEMA_V1.json"}],
    )


SPECS["F16-06"] = f16_06
