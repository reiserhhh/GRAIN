#!/usr/bin/env python3
"""NEXT78: frozen endpoint query contrast, author variation and joint errors."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms import endpoint_query_contrast as fn
from grain_research.vector_mechanisms.stage_file_guard import install
from scripts import run_grain_source_endpoint_assembly as previous

old, prior = previous.old, previous.prior
OUT = old.BASE / "logic_endpoint_query_contrast_v1"
PLAN = ROOT / "docs/GRAIN_ENDPOINT_QUERY_CONTRAST_PLAN_V1.md"
TABLES = (
    "term_metrics.csv",
    "factor_metrics.csv",
    "joint_account.csv",
    "error_account.csv",
    "source_reader_geometry.csv",
)
COUNTS = [144, 432, 56, 48, 2]
CAP = 30


def limit(start, used):
    """Enforce the frozen numeric budget across application, analysis and replay."""
    if time.process_time() - start + used > CAP:
        raise TimeoutError("NEXT78 cumulative CPU budget exceeded")


def prepare(out):
    """Pin all predecessor evidence and this new no-fit decomposition before use."""
    start = time.process_time()
    m = old.verify(previous.OUT)
    if old.read(previous.OUT / "audit.json")["status"] != "PASS":
        raise ValueError("Audited NEXT77 required")
    out.mkdir(exist_ok=False)
    jobs = old.read(previous.OUT / "private_jobs.json")
    if len(jobs) != 12:
        raise ValueError("Exact twelve-author panel required")
    old.write(out / "private_jobs.json", jobs)
    inputs = dict(m["inputs"])
    inputs.update(
        {str((previous.OUT / n).relative_to(ROOT)): h for n, h in m["prepared"].items()}
    )
    for folder in ("predictions", "analysis"):
        f = previous.freeze(previous.OUT, folder)
        inputs.update(
            {
                str((previous.OUT / folder / n).relative_to(ROOT)): h
                for n, h in f["files"].items()
            }
        )
    paths = [
        PLAN,
        Path(__file__),
        Path(fn.__file__),
        ROOT / "tests/test_grain_endpoint_query_contrast.py",
        previous.OUT / "audit.json",
        ROOT
        / "reports/grain_vector_mechanisms/source_endpoint_assembly_v1/publication.json",
    ]
    inputs.update({str(p.relative_to(ROOT)): old.sha(p) for p in paths})
    result = {
        "protocol": "GRAIN_ENDPOINT_QUERY_CONTRAST_V1",
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "prepare_cpu_seconds": time.process_time() - start,
        "inputs": inputs,
        "prepared": {"private_jobs.json": old.sha(out / "private_jobs.json")},
        "max_cpu_seconds": CAP,
        "fits": 0,
        "new_neural_calls": 0,
        "human_labels_read": False,
    }
    old.write(out / "manifest.json", result)
    print(
        {"status": "PREPARED", "cpu_seconds": result["prepare_cpu_seconds"]}, flush=True
    )


def predict(out, folder):
    """Decompose only frozen source predictions; current native readers are denied."""
    old.verify(out)
    used, start = prior.prior_cpu(out), time.process_time()
    jobs = old.read(out / "private_jobs.json")
    dst = out / folder
    dst.mkdir(exist_ok=False)
    names = [j["key"] + ".npz" for j in jobs]
    reads = [previous.OUT / "private_readers" / n for n in names]
    reads += [
        previous.OUT / "predictions" / (k + ".npz")
        for j in jobs
        for k in j["query_keys"]
    ]
    probes = [
        previous.ENDPOINT / "capture_new/new8_a00_neuroticism.npz",
        previous.CAPTURE / "private_capture/new8_a00_neuroticism.npz",
        previous.OUT / "analysis/contrast_actions.csv",
    ]
    denied = install(
        reads,
        [dst / n for n in (*names, "freeze.json")],
        probes,
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    errors = []
    for j in jobs:
        source = old.arrays(previous.OUT / "private_readers" / (j["key"] + ".npz"))
        np.testing.assert_array_equal(
            source["members"], previous.source_set(j["cohort"], j["index"])
        )
        r = source["r"][:, 0]
        p = [
            old.arrays(previous.OUT / "predictions" / (k + ".npz"))
            for k in j["query_keys"]
        ]
        result = fn.split_packets(
            np.stack([x["primal"] for x in p]), np.stack([x["vectors"] for x in p]), r
        )
        for mode, term in ((0, 2), (1, 0)):
            errors.extend(
                [
                    float(
                        np.max(
                            abs(
                                result["scalar_terms"][:, term]
                                - (p[0]["scalar"][:, mode] - p[1]["scalar"][:, mode])
                            )
                        )
                    ),
                    float(
                        np.max(
                            abs(
                                result["action_terms"][:, :, term]
                                - (
                                    p[0]["actions"][:, :, mode]
                                    - p[1]["actions"][:, :, mode]
                                )
                            )
                        )
                    ),
                ]
            )
        g = fn.geometry(r)
        np.savez(
            dst / (j["key"] + ".npz"),
            **result,
            geometry=np.asarray(list(g.values())),
            geometry_fields=np.asarray(list(g)),
        )
        limit(start, used)
    if max(errors) > 1e-9:
        raise ValueError("NEXT77 paired-mode identity mismatch")
    previous.previous.finish(
        out,
        folder,
        names,
        start,
        mode_identity_max_error=max(errors),
        fits=0,
        native_reads=0,
        target_reads=0,
        denial_probes=len(probes),
        extra_denials=len(denied) - len(probes),
    )


def metric(y, p):
    """Describe fidelity without treating factor coordinates as participants."""
    return {
        k.replace("action_", "").replace("actions", "observations"): v
        for k, v in old.fidelity(np.asarray(y).ravel(), np.asarray(p).ravel()).items()
    }


def summarize(groups):
    """All scalar/action terms, factor counterparts and raw/centered joint accounts."""
    tm, fm, jm, em, gm = [], [], [], [], []
    for cohort, rows in groups.items():
        s, a, ys, ya = [
            np.stack([r[k] for r in rows])
            for k in (
                "scalar_terms",
                "action_terms",
                "target_scalar_terms",
                "target_action_terms",
            )
        ]
        for oi, obs in enumerate(fn.OBSERVABLES):
            p, y = (s, ys) if oi == 0 else (a[:, :, oi - 1], ya[:, oi - 1])
            base = {"cohort": cohort, "authors": len(rows), "observable": obs}
            jm.append(base | {"arm": "native_target"} | fn.moments(y))
            for ai, arm in enumerate(previous.fn.ARMS):
                meta = base | {"arm": arm}
                jm.append(meta | fn.moments(p[:, ai]))
                em.append(meta | fn.error_account(y, p[:, ai]))
                for ti, term in enumerate(fn.TERMS):
                    m = meta | {"term": term}
                    tm.append(m | metric(y[:, ti], p[:, ai, ti]))
                    if oi:
                        for f in range(4):
                            fm.append(
                                m
                                | {"factor": f"whole:F{f + 1:02d}"}
                                | metric(y[:, ti, f], p[:, ai, ti, f])
                            )
        g = pd.DataFrame([r["source_geometry"] for r in rows])
        meta = {
            "cohort": cohort,
            "authors": len(rows),
            "distinct_source_references": 4 if cohort == "source4" else 1,
        }
        for c in g:
            meta.update(
                {
                    c + "_mean": float(g[c].mean()),
                    c + "_std": float(g[c].std(ddof=0)),
                    c + "_min": float(g[c].min()),
                    c + "_max": float(g[c].max()),
                }
            )
        gm.append(meta)
    return [pd.DataFrame(r) for r in (tm, fm, jm, em, gm)]


def native_packets(job, weights, qweights):
    """Reuse NEXT77's cached native target construction, with no model forward."""
    packets = []
    for q, key in zip(old.QUERIES, job["query_keys"], strict=True):
        cap = old.arrays(previous.CAPTURE / "private_capture" / (key + ".npz"))
        e = old.arrays(previous.SOURCE / "private_inputs" / (key + ".npz"))
        sm = old.arrays(
            previous.SELECTOR / "models" / f"{prior.reference(job)}_{q}.npz"
        )
        v, dv = previous.fn.value.values(cap["x"], cap["d"], weights)
        np.testing.assert_array_equal(v, e["v"])
        rows = []
        for anchor in (sm["anchor"], previous.source_query(cap["x"][-1], qweights)):
            ell, dell = previous.fn.value.scalar.logits(
                anchor, e["k"], e["dk"], e, float(e["scale"])
            )
            rows.append(previous.fn.vectors(ell, dell, e["mask"], v, dv, weights["o"]))
        packets.append(rows)
    return {
        k: np.stack([np.stack([p[k] for p in rows]) for rows in packets])
        for k in packets[0][0]
    }


def old_metric_replay(tables):
    """Independently reproduce both old endpoint modes, with their own targets."""
    c = pd.read_csv(previous.OUT / "analysis/contrast_actions.csv")
    s = pd.read_csv(previous.OUT / "analysis/scalar_metrics.csv")
    errors = []
    for row in tables[0].to_dict("records"):
        if row["term"] not in ("evidence_change", "total"):
            continue
        mode = (
            "query_pair_mean" if row["term"] == "evidence_change" else "query_specific"
        )
        scalar = row["observable"] == "scalar"
        table = s if scalar else c
        mask = (
            (table.cohort == row["cohort"])
            & (table.arm == row["arm"])
            & (table["mode"] == mode)
        )
        if scalar:
            mask &= table["query"] == "N_minus_confirmation"
        else:
            mask &= table.component == (
                "total" if row["observable"] == "total_action" else row["observable"]
            )
        matched = table[mask]
        if len(matched) != 1:
            raise ValueError("Old target metric lookup is not unique")
        prefix = "scalar" if scalar else "action"
        for key in ("r", "mse_over_zero", "rmse"):
            errors.append(abs(row[key] - float(matched.iloc[0][prefix + "_" + key])))
    if len(errors) != 288 or max(errors) > 1e-8:
        raise ValueError("NEXT77 scalar/action target reproduction failed")
    return max(errors)


def evaluate(out, folder):
    """Open target readers after prediction freeze and account for both query terms."""
    old.verify(out)
    previous.freeze(out, folder)
    used, start = prior.prior_cpu(out), time.process_time()
    dst = out / ("analysis" if folder == "predictions" else "replay_analysis")
    dst.mkdir(exist_ok=False)
    weights, qweights = (
        old.arrays(previous.STATE / "private_weights.npz"),
        old.arrays(previous.CAPTURE / "private_weights.npz"),
    )
    groups = {"source4": [], "new8": []}
    names = []
    for j in old.read(out / "private_jobs.json"):
        p = old.arrays(out / folder / (j["key"] + ".npz"))
        rb = old.arrays(previous.OUT / "private_readers" / (j["key"] + ".npz"))["r"][
            :, 0
        ]
        rc = np.stack(
            [
                previous.selected(
                    previous.ENDPOINT
                    / ("capture_source" if j["cohort"] == "source4" else "capture_new")
                    / (k + ".npz"),
                    ("r",),
                )["r"]
                for k in j["query_keys"]
            ]
        )
        native = native_packets(j, weights, qweights)
        oracle = fn.split_packets(native["primal"], native["vectors"], rb)
        target = fn.split_packets(
            native["primal"][:, 1:2], native["vectors"][:, 1:2], rc
        )
        row = {
            "scalar_terms": np.concatenate((p["scalar_terms"], oracle["scalar_terms"])),
            "action_terms": np.concatenate((p["action_terms"], oracle["action_terms"])),
            "target_scalar_terms": target["scalar_terms"][0],
            "target_action_terms": target["action_terms"][0],
        }
        name = "private_" + j["key"] + ".npz"
        names.append(name)
        np.savez(dst / name, **row)
        row["source_geometry"] = {
            str(k): float(v)
            for k, v in zip(p["geometry_fields"], p["geometry"], strict=True)
        }
        groups[j["cohort"]].append(row)
        limit(start, used)
    tables = summarize(groups)
    replay_error = old_metric_replay(tables)
    if [len(t) for t in tables] != COUNTS:
        raise ValueError("Complete term grid required")
    identity = max(
        tables[2].raw_identity_error.max(),
        tables[2].centered_identity_error.max(),
        tables[3].raw_identity_error.max(),
        tables[3].centered_identity_error.max(),
    )
    if identity > 1e-9:
        raise ValueError("Raw/centered error identity failed")
    for name, table in zip(TABLES, tables, strict=True):
        table.to_csv(dst / name, index=False)
    limit(start, used)
    previous.previous.finish(
        out,
        dst.name,
        [*TABLES, *names],
        start,
        counts=COUNTS,
        old_mode_metric_max_error=replay_error,
        energy_identity_max_error=float(identity),
    )


def audit(out):
    """Check the same finite decomposition twice; this is not an efficacy gate."""
    old.verify(out)
    a, b, aa, bb = [
        previous.freeze(out, n)
        for n in ("predictions", "replay", "analysis", "replay_analysis")
    ]
    if len(a["files"]) != 12 or len(aa["files"]) != 17 or aa["counts"] != COUNTS:
        raise ValueError("Fixed packet/table inventory failed")
    for first, second, fa, fb in (
        ("predictions", "replay", a, b),
        ("analysis", "replay_analysis", aa, bb),
    ):
        if set(fa["files"]) != set(fb["files"]):
            raise ValueError("Replay inventory mismatch")
        for name in fa["files"]:
            if name.endswith(".npz"):
                x, y = [old.arrays(out / folder / name) for folder in (first, second)]
                if set(x) != set(y):
                    raise ValueError("Replay field mismatch")
                for k in x:
                    np.testing.assert_array_equal(x[k], y[k])
            elif fa["files"][name] != fb["files"][name]:
                raise ValueError("Exact table replay failed")
    if (
        any(
            f["extra_denials"] or f["fits"] or f["target_reads"] or f["native_reads"]
            for f in (a, b)
        )
        or prior.prior_cpu(out) > CAP
    ):
        raise ValueError("Declared access/budget exceeded")
    result = {
        "status": "PASS",
        "fits": 0,
        "new_neural_calls": 0,
        "human_labels_read": False,
        "application_packets": 12,
        "private_evaluation_packets": 12,
        "public_tables": 5,
        "total_cpu_seconds": prior.prior_cpu(out),
        "mode_identity_max_error": a["mode_identity_max_error"],
        "old_mode_metric_max_error": aa["old_mode_metric_max_error"],
        "energy_identity_max_error": aa["energy_identity_max_error"],
    }
    old.write(out / "audit.json", result)
    print(result, flush=True)


def main():
    """Run each immutable stage in a fresh one-thread process."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stage", required=True, choices=("prepare", "predict", "evaluate", "audit")
    )
    p.add_argument("--folder", default="predictions", choices=("predictions", "replay"))
    p.add_argument("--output", type=Path, default=OUT)
    a = p.parse_args()
    with threadpool_limits(limits=1):
        if a.stage in ("predict", "evaluate"):
            globals()[a.stage](a.output, a.folder)
        else:
            globals()[a.stage](a.output)


if __name__ == "__main__":
    main()
