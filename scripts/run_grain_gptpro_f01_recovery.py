#!/usr/bin/env python3
"""F01 recovery: full-source bibliography, primary cards, adoption, identification counterexamples."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

from grain_research.gptpro_foundation_v3 import f01_sources as src

ROOT = Path(__file__).resolve().parents[1]
COORD = Path("/Volumes/mobile3/projects/project persona")
SOURCES = {
    "P01": COORD / "reports/gptpro_seminar_20260915/sources/P01.md",
    "P02": COORD / "reports/gptpro_seminar_20260915/sources/P02.md",
    "P03": COORD / "reports/gptpro_seminar_20260915/sources/P03.md",
    "P04": COORD / "reports/gptpro_seminar_20260915/sources/P04.md",
    "P05": COORD / "gptproplans/思路汇总.md",
}
PRIMARY = ROOT / "results/grain_gptpro_foundation_v3/F01/primary_sources_r1"
CARDS = ROOT / "configs/grain_f01_primary_cards_v1.json"


def sha(path):
    """File hash."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    """Immutable JSON."""
    with Path(path).open("x") as f:
        f.write(
            json.dumps(
                value, sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False
            )
            + "\n"
        )


def write_csv(path, rows):
    """Immutable CSV."""
    with Path(path).open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def common(folder, card, reqs, inputs, summary, metrics, decision):
    """COMMON eight files."""
    write_json(
        folder / "protocol.json",
        {
            "card": card,
            "requirements": reqs,
            "protocol": "docs/GRAIN_F01_RECOVERY_PROTOCOL_V1.md",
        },
    )
    write_json(folder / "input_manifest.json", inputs)
    write_json(folder / "summary.json", summary)
    write_csv(
        folder / "metrics.csv",
        metrics or [{"reason": "review card without numeric metrics"}],
    )
    (folder / "decision.md").write_text(decision)
    (folder / "run_log.jsonl").write_text(
        json.dumps({"card": card, "status": summary.get("status")}) + "\n"
    )
    (folder / "tests.txt").write_text(
        "see tests/test_grain_gptpro_f01_recovery.py junit in parent\n"
    )


def finish(folder):
    """Output manifest last."""
    write_json(
        folder / "output_manifest.json",
        {
            p.name: sha(p)
            for p in sorted(folder.iterdir())
            if p.is_file() and not p.name.startswith(".")
        },
    )


def bibliography(out, inputs):
    """F01-02: every URL/author-year citation, deduplicated, classified, mapped to cards."""
    records = []
    for name, path in SOURCES.items():
        records += src.extract(path.read_text(), name)
    entries = src.deduplicate(records)
    cards = json.loads(CARDS.read_text())["cards"]
    for e in entries:
        blob = e["title"] + " " + " ".join(e["urls"])
        e["primary_cards"] = [
            c["id"]
            for c in cards
            if c["cited_pattern"] and re.search(c["cited_pattern"], blob, re.IGNORECASE)
        ]
        if e["primary_cards"] and e["class"] != "load_bearing":
            e["class_override"] = "load_bearing (matches a protocol load-bearing card)"
            e["class"] = "load_bearing"
    rows = [
        {
            "citation_id": e["citation_id"],
            "class": e["class"],
            "topics": ";".join(e["topics"]),
            "primary_cards": ";".join(e["primary_cards"]),
            "title": e["title"],
            "doi": e["doi"],
            "arxiv": e["arxiv"],
            "urls": " ".join(e["urls"]),
            "source_locators": " ".join(e["locators"]),
            "sources": ";".join(e["sources"]),
            "core_mark_in_source": e["core_mark"],
            "kind": e["kind"],
            "dedup_key": e["citation_key"],
        }
        for e in entries
    ]
    folder = out / "F01-02"
    folder.mkdir()
    write_csv(folder / "bibliography.csv", rows)
    card_rows = []
    for c in cards:
        file = c["retrieval"].get("file")
        card_rows.append(
            dict(
                c,
                retrieval=dict(
                    c["retrieval"],
                    sha256=sha(PRIMARY / file)
                    if file and (PRIMARY / file).exists()
                    else None,
                ),
                cited_by=[
                    r["citation_id"]
                    for r in rows
                    if c["id"] in r["primary_cards"].split(";")
                ],
            )
        )
    write_json(folder / "primary_equation_cards.json", card_rows)
    md = [
        "# Primary equation cards (F01-02)\n",
        "Locators verified from the cached primary files (hash in JSON) or explicit web fetch; UNAVAILABLE items are recorded, not reconstructed.\n",
    ]
    for c in card_rows:
        md.append(
            f"## {c['id']} {c['topic']} ({', '.join(c['requirements'])})\n\n- Work: {c['work']}\n- Retrieval: {c['retrieval']}\n- Locator: {c['locator']}\n- Assumptions: {'; '.join(c['assumptions']) or '—'}\n- Our equation / code: {c['our_equation']}\n- Difference: {c['difference'] or '—'}\n- Status: **{c['status']}**; next card {c['next_card']}\n- Cited in sources as: {', '.join(c['cited_by']) or 'not cited in the five sources (added by protocol requirement)'}\n"
        )
    (folder / "primary_equation_cards.md").write_text("\n".join(md))
    from collections import Counter

    classes = Counter(r["class"] for r in rows)
    status = Counter(c["status"].split("_")[0] for c in card_rows)
    summary = {
        "status": "COMPUTED",
        "raw_citation_records": len(records),
        "deduplicated_entries": len(rows),
        "classes": dict(classes),
        "per_source_records": dict(Counter(r["source"] for r in records)),
        "primary_cards": len(card_rows),
        "card_status_prefix": dict(status),
        "residual_duplicate_note": "dedup keys: DOI > arXiv > specific URL > normalized title; generic-host URLs and author-year vs URL forms can leave near-duplicates, listed separately rather than merged by guess",
    }
    metrics = [
        {
            "task_id": "F01-02",
            "method": "citation_extraction",
            "estimand": k,
            "value": v,
        }
        for k, v in (
            ("raw_records", len(records)),
            ("deduplicated", len(rows)),
            *classes.items(),
        )
    ]
    decision = f"""# F01-02 decision

- All five frozen sources scanned (P01–P04 PDF-page text, P05 sections): {len(records)} citation records -> {len(rows)} deduplicated entries; classes {dict(classes)}.
- {len(card_rows)} primary cards for protocol load-bearing topics (PSR/bisimulation, state-space identifiability, nonlinear ICA/iVAE, IB, Blackwell, Mori-Zwanzig, Bayesian design/ADO, ALBA, RSA, counterfactual explanation, selection bounds, MTMM/G-theory/invariance, validity). Locators are exact where retrieved; {sum(1 for c in card_rows if "UNAVAILABLE" in c["locator"])} cards record unavailable primaries (paywall/403/reCAPTCHA/connection reset) and point to our finite certificates instead.
- Classification is a declared keyword rule plus card-match overrides; it is not a judgment of each paper's quality. Context citations need no experiment (F17 rule).
"""
    common(
        folder,
        "F01-02",
        ["R03", "A01", "A02", "A03", "A04"],
        inputs,
        summary,
        metrics,
        decision,
    )
    finish(folder)
    return summary, card_rows


ADOPTION = [
    (
        "generalizability_theory",
        "variance components of facets (person, occasion, rater)",
        "F11/F13 variance decomposition",
        "finite_certificate",
        "person/condition/reader/residual decomposition computed in F11 and F13; material-resampling facet still open (F13 gap)",
        "F13-05",
    ),
    (
        "mtmm",
        "trait x method correlation matrix",
        "F11 3x3x2 MTMM covariance",
        "finite_certificate",
        "method/trait non-identification without anchor certified; anchor restores rank 8->9",
        "F11-05",
    ),
    (
        "measurement_invariance_dif",
        "configural/metric/scalar invariance across groups",
        "F12 conditional DIF / transport",
        "finite_certificate",
        "condition-dependent unit change breaks pooled calibration (coverage .005) and is repaired by conditional calibration; real DIF table + H03/H05 contracts open",
        "F12-05",
    ),
    (
        "cat_irt",
        "item selection by Fisher information on a scalar latent",
        "F06 scalar IRT-CAT panel",
        "adopt",
        "standard scalar CAT used as reference arm; SIM only",
        "F06-06",
    ),
    (
        "ado",
        "Bayesian design maximizing expected utility",
        "F06 EIG / target-specific policies",
        "adopt",
        "utility must be target-specific (Chaloner & Verdinelli p.275); variance and EIG coincide in panel A by symmetry",
        "F06-06",
    ),
    (
        "alba",
        "adaptive language-based assessment with IRT",
        "F06/F07 design only",
        "not_executed",
        "requires language-derived item bank and human criterion; design contract belongs to H04",
        "F17-03",
    ),
    (
        "rsa_pragmatics",
        "listener/speaker Bayesian reference model",
        "F08 subject-status qualification",
        "not_executed",
        "our reader is an NLI classifier; an RSA listener over subject-status alternatives is a declared extension contract",
        "F15-05",
    ),
    (
        "counterfactual_explanation",
        "nearest input change reaching a target output",
        "F16 evidence deletion effects",
        "adopt_with_difference",
        "exact withdrawal effects over observed evidence rather than nearest-input search",
        "F16-06",
    ),
    (
        "concept_bottleneck",
        "prediction through named concepts",
        "none",
        "not_applicable",
        "no named concept layer is claimed; naming is outside current targets",
        "F15-05",
    ),
    (
        "sparse_autoencoder",
        "dictionary features of model activations",
        "F10 action metric (vector path)",
        "not_executed",
        "no new model development allowed; action metric uses existing caches",
        "F15-05",
    ),
    (
        "nli_reader",
        "entailment probabilities as readouts",
        "F08/F16 readers",
        "adopt",
        "pinned cross-encoder/nli-deberta-v3-base readouts with joint error law",
        "F08-07",
    ),
    (
        "psr",
        "predictive state / system-dynamics rank",
        "F02/F15-01",
        "adopt",
        "exact closure and too-short-history counterexample",
        "F15-01",
    ),
    (
        "bisimulation_metric",
        "future-law metric between states",
        "F02 exact equivalence",
        "adopt_with_difference",
        "exact equivalence only; metric not needed for finite exact laws",
        "F15-05",
    ),
    (
        "information_bottleneck",
        "relevance-compression trade-off",
        "F15-02 exact MI",
        "not_executed",
        "IB curve not optimized; sufficiency/garbling answers the declared question",
        "F15-05",
    ),
    (
        "blackwell",
        "garbling order of experiments",
        "F15-02 certificates",
        "adopt",
        "explicit kernels + Farkas vectors",
        "F15-02",
    ),
    (
        "mori_zwanzig",
        "Markov + memory + noise decomposition",
        "F15-01 memory term",
        "adopt_with_difference",
        "discrete analog; memory deterministic in omitted lags",
        "F15-01",
    ),
    (
        "nonlinear_ica_ivae",
        "identifiable latent sources with auxiliary variable",
        "F15-03/04",
        "finite_certificate",
        "rank conditions verified; learned estimator not executed (no neural training)",
        "F15-04",
    ),
    (
        "state_space_identifiability",
        "observability of parameter-augmented state",
        "F05 invariants/profiles",
        "finite_certificate",
        "structural ranks and gauge invariants; practical weak identification near root",
        "F05-07",
    ),
    (
        "partial_identification",
        "worst-case bounds under selection",
        "F03 identified sets",
        "finite_certificate",
        "exact enumeration; primary not retrieved",
        "F03-06",
    ),
    (
        "blup_reml",
        "mixed-model shrinkage and variance estimation",
        "F04/F13",
        "adopt",
        "standard estimators used as references; not claimed as GRAIN novelty",
        "F04-06",
    ),
]


def adoption(out, inputs):
    """F01-03 predecessor adoption ledger."""
    folder = out / "F01-03"
    folder.mkdir()
    rows = [
        {
            "method": m,
            "statistical_object": o,
            "our_counterpart": c,
            "decision": d,
            "reason": r,
            "responsible_card": card,
        }
        for m, o, c, d, r, card in ADOPTION
    ]
    write_csv(folder / "predecessor_adoption.csv", rows)
    from collections import Counter

    summary = {
        "status": "COMPUTED",
        "methods": len(rows),
        "decisions": dict(Counter(r["decision"] for r in rows)),
    }
    decision = "# F01-03 decision\n\nEvery predecessor method family named by the protocol has an explicit adopt / adopt_with_difference / finite_certificate / not_applicable / not_executed decision with reason and responsible card; nothing is silently omitted. Decisions are provisional until F15-05 reviews them with the primary cards.\n"
    common(
        folder,
        "F01-03",
        ["R03", "A06", "A07", "A08"],
        inputs,
        summary,
        [
            {"task_id": "F01-03", "method": k, "estimand": "count", "value": v}
            for k, v in summary["decisions"].items()
        ],
        decision,
    )
    finish(folder)
    return summary


def counterexamples():
    """Four minimal same-observation / different-target counterexamples, computed exactly."""
    out = []
    # 1 unknown exposure: published indicator law depends only on f = pi*q.
    a, b = (0.8, 0.5), (0.4, 1.0)
    law_a = np.array([1 - a[0] * a[1], a[0] * a[1]])
    law_b = np.array([1 - b[0] * b[1], b[0] * b[1]])
    full_a = np.array([1 - a[0], a[0] * (1 - a[1]), a[0] * a[1]])
    full_b = np.array([1 - b[0], b[0] * (1 - b[1]), b[0] * b[1]])
    out.append(
        {
            "id": "CX1_unknown_exposure",
            "requirements": ["R05", "R06", "R07"],
            "worlds": {"A": {"pi": a[0], "q": a[1]}, "B": {"pi": b[0], "q": b[1]}},
            "observed_view": "selected_indicator (opportunities known, O unobserved)",
            "observed_law_max_difference": float(np.abs(law_a - law_b).max()),
            "target": "selection propensity q",
            "target_values": [a[1], b[1]],
            "breaking_observation": "all_opportunities view (O observed)",
            "broken_law_max_difference": float(np.abs(full_a - full_b).max()),
            "evidence_card": "F03-01 same_observation_counterexample.json",
        }
    )
    # 2 state shift: random walk with unknown initial state; (a, s0) -> (a+c, s0-c).
    t = np.arange(8)
    q = 0.25
    cov_rw = q * np.minimum.outer(t + 1, t + 1) + 0.25 * np.eye(8)
    mean_a = (1.0 + 0.5) * np.ones(8)
    mean_b = (1.7 + (-0.2)) * np.ones(8)
    rho = 0.6
    stat = q / (1 - rho**2) * rho ** np.abs(np.subtract.outer(t, t)) + 0.25 * np.eye(8)
    out.append(
        {
            "id": "CX2_state_shift_gauge",
            "requirements": ["R10", "R07", "R21"],
            "worlds": {"A": {"a": 1.0, "s0": 0.5}, "B": {"a": 1.7, "s0": -0.2}},
            "observed_view": "y_t = a + s_t + e_t, s_t random walk from unknown s0 (rho=1)",
            "observed_law_max_difference": float(
                max(np.abs(mean_a - mean_b).max(), np.abs(cov_rw - cov_rw).max())
            ),
            "target": "person level a",
            "target_values": [1.0, 1.7],
            "breaking_observation": "stationary rho<1 (s0 ~ N(0,q/(1-rho^2)) independent of a) or a known initial state",
            "broken_law_max_difference": float(abs(1.0 - 1.7)),
            "note": f"with rho={rho} the state mean is 0 so E[y]=a identifies a; stationary covariance diag {stat[0, 0]:.4f}",
            "evidence_card": "F05 invariants_r1 (rho=1 translation example 1.223907)",
        }
    )

    # 3 shared method: author-constant method m shared by all readers.
    def cov(var_f, var_m, readers=3, noise=0.25):
        return (var_f + var_m) * np.ones((readers, readers)) + noise * np.eye(readers)

    anchor_a, anchor_b = 1.0, 1.25
    out.append(
        {
            "id": "CX3_shared_method",
            "requirements": ["R16", "R20", "R29", "A06"],
            "worlds": {
                "A": {"trait_var": 1.0, "method_var": 0.25},
                "B": {"trait_var": 1.25, "method_var": 0.0},
            },
            "observed_view": "three readers of the same texts, y_r = f + m + e_r",
            "observed_law_max_difference": float(
                np.abs(cov(1.0, 0.25) - cov(1.25, 0.0)).max()
            ),
            "target": "trait variance",
            "target_values": [1.0, 1.25],
            "breaking_observation": "independent anchor z = f + e_z not sharing m: cov(z, y) = trait variance",
            "broken_law_max_difference": float(abs(anchor_a - anchor_b)),
            "evidence_card": "F11 anchor_full_r2 (anchor cross-cov difference .25)",
        }
    )
    # 4 equivalent rival: belief process equals a negative-slope family at binary corners.
    gamma, beta = 1.2, 0.5

    def belief(z):
        pb = expit(beta * z)
        return pb * expit(-gamma) + (1 - pb) * expit(gamma)

    beta_neg = float(logit(expit(gamma) - (2 * expit(gamma) - 1) * expit(beta)))

    def rival(z):
        return expit(beta_neg * z)

    corners = max(abs(belief(z) - rival(z)) for z in (-1.0, 1.0))
    middle = max(abs(belief(z) - rival(z)) for z in (-1 / 3, 1 / 3))
    out.append(
        {
            "id": "CX4_equivalent_rival",
            "requirements": ["R18", "R19", "R07"],
            "worlds": {
                "A": {
                    "process": "belief B~Bern(logistic(beta z1)), report logistic(gamma(1-2B))",
                    "beta": beta,
                    "gamma": gamma,
                },
                "B": {
                    "process": "single no-intercept logistic in z1 (protocol beta')",
                    "protocol_beta_prime": beta_neg,
                },
            },
            "observed_view": "binary report at z1 in {-1,+1}",
            "observed_law_max_difference": float(corners),
            "target": "process identity (belief vs direct response)",
            "target_values": ["belief", "direct"],
            "breaking_observation": "intermediate levels z1=+-1/3 (or observing B)",
            "broken_law_max_difference": float(middle),
            "evidence_card": "F09 panels_audit_r1/process_certificates.csv (gap .000290 at beta=.5)",
        }
    )
    return out


def identification(out, inputs):
    """F01-04 identification contract and computed counterexample registry."""
    folder = out / "F01-04"
    folder.mkdir()
    cx = counterexamples()
    write_json(folder / "counterexample_registry.json", cx)
    contract = """# Identification contract (F01-04)

1. **Declared target first.** A target is a functional tau(P_theta) of the full generative law (e.g. T1 m(+1)-m(-1), T2 E[s_T|prefix], T3 E[mean future|plan], trait variance, selection propensity). Identification is always of a declared target, never of "the person".
2. **Identified** means: for the declared observation design, any two parameter values giving the same law of observables give the same target value. **Not identified** is shown by an explicit pair with equal observable law and different target (registry CX1-CX4).
3. **Structural vs practical.** Structural identification uses the exact law (noise-free / infinite data). Practical identification asks whether finite data and priors concentrate the target (F05 near-root profiles, F13 coverage). Structural rank alone does not certify practical identification.
4. **Coordinate gauge vs function.** When a group of transformations leaves the observable law unchanged (similarity transforms, factor rotation, method/trait reallocation), only invariants of that group are identified; named coordinates are not (Locatello et al. 2019 Thm 1; F05-06; F11).
5. **Functional equivalence.** Two mechanisms that induce the same observable law on the design are equivalent for that design (F09 belief vs negative-slope family); adding design points or observables may break the equivalence.
6. **Identification is not validity.** An identified target is a well-defined function of the declared model; whether it measures a psychological attribute requires external criteria (H01–H06) and is not claimed here.
7. **Every counterexample lists the observation that would add identification**; production rules must require that observation or report only the identified invariant.
"""
    (folder / "identification_contract.md").write_text(contract)
    ok = all(
        c["observed_law_max_difference"] <= 1e-12
        and c["broken_law_max_difference"] > 1e-6
        for c in cx
    )
    summary = {
        "status": "COMPUTED" if ok else "COUNTEREXAMPLE_FAILED",
        "counterexamples": [
            {
                k: c[k]
                for k in (
                    "id",
                    "observed_law_max_difference",
                    "broken_law_max_difference",
                )
            }
            for c in cx
        ],
    }
    metrics = [
        {
            "task_id": "F01-04",
            "method": c["id"],
            "estimand": "observed_law_max_difference",
            "value": c["observed_law_max_difference"],
        }
        for c in cx
    ] + [
        {
            "task_id": "F01-04",
            "method": c["id"],
            "estimand": "broken_law_max_difference",
            "value": c["broken_law_max_difference"],
        }
        for c in cx
    ]
    decision = (
        "# F01-04 decision\n\n"
        + "\n".join(
            f"- {c['id']}: equal observable law (max difference {c['observed_law_max_difference']:.2e}) with different {c['target']} {c['target_values']}; broken by {c['breaking_observation']} (difference {c['broken_law_max_difference']:.6g})."
            for c in cx
        )
        + "\n"
    )
    common(folder, "F01-04", ["R07", "R21", "R32"], inputs, summary, metrics, decision)
    finish(folder)
    return summary


def main():
    """Run F01-02..04 into a new directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = {
        "sources": {k: {"path": str(v), "sha256": sha(v)} for k, v in SOURCES.items()},
        "cards_config": {"path": str(CARDS.relative_to(ROOT)), "sha256": sha(CARDS)},
        "primary_files": {
            p.name: sha(p)
            for p in sorted(PRIMARY.glob("*.pdf"))
            if not p.name.startswith(".")
        },
        "code": {
            p: sha(ROOT / p)
            for p in (
                "grain_research/gptpro_foundation_v3/f01_sources.py",
                "scripts/run_grain_gptpro_f01_recovery.py",
            )
        },
    }
    write_json(
        args.output / "protocol_before_run.json",
        {"protocol": "docs/GRAIN_F01_RECOVERY_PROTOCOL_V1.md", "inputs": inputs},
    )
    print(json.dumps(bibliography(args.output, inputs)[0], ensure_ascii=False))
    print(json.dumps(adoption(args.output, inputs)))
    print(json.dumps(identification(args.output, inputs)))


if __name__ == "__main__":
    main()
