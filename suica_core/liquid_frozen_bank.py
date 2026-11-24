#!/usr/bin/env python3
"""Read old factor coordinates through a frozen LTC's causal time kernel, without refitting."""

import argparse
import hashlib
import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import liquid_instrument as li
from . import v7_multiview as mv
from . import v7_observations as vo
from . import v7_psychometric as ps

ROOT = Path(__file__).resolve().parents[1]
FA = ROOT / "results/v7_operator_smoke/initial_full_final"
E4 = ROOT / "results/v7_operator_family/e4_v72_full_20260715_r3"
REP_HASH = {
    "WORD12_TFIDF_SVD24": "5f30e4e0a00eb2b2b0c1f818dec07695818cb7b61c2759494369788e917b2a27",
    "CHAR35_TFIDF_SVD24": "2e6766e66d4068fdb0046ebca40dc8183b8cb1aba58f47d0df6704a7b6b302d8",
    "whole": "675ff65243ea61bd971580a1e23e0ff8574dd9708bfdb05277949dae235b2477",
    "native": "e13248d9f0558fa7181abe229c7247cb8eed2ef48e998ab755c7706bd70fc9c7",
    "fixed_128": "6706c9afee7015b5851bcf4d93648f2e464b63824d5642a20bdda5ca183bc9f4",
    "semantic": "52cd127cf78bd1745eba9e1b265493a0f93bd8ef62126a1c1ccb3b7ae2a4b34e",
}


def old_runtime(name, path):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != REP_HASH[name]:
        raise ValueError("not the known released numerical artifact")
    return joblib.load(io.BytesIO(raw))


def mix(values, carry):
    """Equal mixture of cell traces, not a recurrence using the average cell gate."""
    v = np.asarray(values, float)
    r = np.asarray(carry, float)
    if v.ndim != 2 or r.ndim != 2 or len(v) != len(r) or not len(v) or not r.shape[1]:
        raise ValueError("misaligned carry and value paths")
    if (
        not np.isfinite(v).all()
        or not np.isfinite(r).all()
        or np.any((r < 0) | (r > 1))
    ):
        raise ValueError("invalid path")
    traces = np.tile(v[0], (r.shape[1], 1))
    result = [v[0]]
    for ri, vi in zip(r[1:], v[1:], strict=True):
        traces = ri[:, None] * traces + (1 - ri[:, None]) * vi
        result.append(traces.mean(0))
    return np.array(result)


def clocks(units, ref, transport, cache):
    if (
        ref["worker"] != li.digest(li.WORKER.read_text())
        or ref["version"] != li.VERSION
    ):
        raise ValueError("reference and source reader are incompatible")
    if ref["width"] != 0 or ref.get("cut_policy", {}).get(
        "kind", "fixed_or_turn"
    ) not in ("fixed_or_turn", "turn"):
        raise ValueError(
            "use the whole-opportunity reader; the archive specifies its own cuts"
        )
    code = (
        li.WORKER.read_text().split('if __name__ == "__main__":')[0]
        + Path(__file__).with_name("liquid_clock_worker.py").read_text()
    )
    groups = list(units.groupby("user_id", sort=False))
    ids = [li.digest(list(g.text)) for _, g in groups]
    missing = {
        key: g for key, (_, g) in zip(ids, groups, strict=True) if key not in cache
    }
    err = 0.0
    keys = list(missing)
    for start in range(0, len(keys), 32):
        chunk = keys[start : start + 32]
        pp = [
            li.locate(ref["atlas"], missing[key].text.tolist())[0].tolist()
            for key in chunk
        ]
        got = li.call(
            {"paths": pp, "weights": ref["weights"]}, transport, worker_code=code
        )
        err = max(err, got["official_state_max_error"])
        cache.update(zip(chunk, got["paths"], strict=True))
    return [
        (g.index.to_numpy(), cache[key])
        for key, (_, g) in zip(ids, groups, strict=True)
    ], err


def features(u, x, clocks_by_group):
    out = {}
    for kind in ("original", "fixed_half", "liquid"):
        z = x.copy()
        for ids, r in clocks_by_group:
            if kind != "original":
                z[ids] = mix(
                    x[ids], r if kind == "liquid" else np.full((len(ids), 1), 0.5)
                )
        out[kind] = ps.author_features_from_embeddings(u, z)
    return out


def records(t):
    return t.to_dict(orient="records")


def score(panel, ref, transport, families=("factor", "consensus")):
    if panel.empty or not panel.user_id.notna().all():
        raise ValueError("empty panel or missing IDs")
    out = {
        "version": "released-clock-bank-v1",
        "reader_digest": li.digest(ref),
        "factor": {},
        "consensus": {},
        "coverage": {},
        "official_state_max_error": 0.0,
    }
    cache = {}

    def observe(family, spec, panel):
        u = vo.build_observations(panel, spec).reset_index(drop=True)
        if u.empty:
            out["coverage"][family + ":" + spec.name] = {}
            return u, []
        order = u.groupby("user_id").unit_index.apply(
            lambda x: x.is_monotonic_increasing and x.is_unique
        )
        if not order.all():
            raise ValueError("unordered or ambiguous unit address")
        c, err = clocks(u, ref, transport, cache)
        out["official_state_max_error"] = max(out["official_state_max_error"], err)
        out["coverage"][family + ":" + spec.name] = {
            k: {
                "n": len(g),
                "source_rows": g.source_rows.tolist(),
                "unit_digest": li.digest(g.text.tolist()),
                "clock": "one observation unit",
            }
            for k, g in u.groupby("user_id", sort=False)
        }
        return u, c

    if "factor" in families:
        tables = {}
        for v in ("whole", "native", "fixed_128", "semantic", "nested"):
            path = FA / "bundles" / (v + "_factor_bundle.json")
            b = ps.read_factor_bundle(path)
            if v == "nested":
                members = tuple(b.runtime_artifact["nested_members"])
                tab = {}
                for kind in ("original", "fixed_half", "liquid"):
                    if all(member in tables for member in members):
                        tab[kind] = ps.combine_nested_author_features(
                            {m: tables[m][kind] for m in members}, members
                        )
            else:
                u, clk = observe("factor", vo.ObservationSpec(**b.operator), panel)
                if u.empty:
                    out["factor"][v] = {
                        "bundle": b.bundle_id,
                        "rows": {},
                        "status": "no_eligible_units",
                    }
                    continue
                runtime = old_runtime(v, FA / b.runtime_artifact["representation_path"])
                tab = features(u, runtime.transform(u.text), clk)
            tables[v] = tab
            out["factor"][v] = {
                "bundle": b.bundle_id,
                "bundle_digest": ps.bundle_sha256(path),
                "rows": {
                    k: records(ps.score_author_features(b, f))
                    for k, f in tab.items()
                    if not f.empty
                },
            }
    if "consensus" in families:
        cfg = li.load(ROOT / "configs/v7_operator_family_registry.json")
        specs = [
            vo.ObservationSpec(
                min_tokens=cfg["min_tokens_per_unit"],
                max_units_per_user=cfg["max_units_per_user"],
                max_source_comments_per_user=cfg["max_source_comments_per_user"],
                max_source_tokens_per_user=cfg["max_source_tokens_per_user"],
                **spec,
            )
            for spec in cfg["operators"]
        ]
        prepared = vo.prepare_source_panel(panel, specs[0])
        cuts = {s.name: observe("consensus", s, prepared) for s in specs}
        for rep in ("WORD12_TFIDF_SVD24", "CHAR35_TFIDF_SVD24"):
            bank = old_runtime(rep, E4 / ("runtime_" + rep + ".joblib"))
            out["consensus"][rep] = {}
            for view in bank["views"]:
                u, clk = cuts[view]
                if u.empty:
                    out["consensus"][rep][view] = {}
                    continue
                x = bank["svd"].transform(bank["vectorizer"].transform(u.text))
                tabs = features(u, x, clk)
                rows = {}
                for k, tab in tabs.items():
                    ids = list(tab.user_id.astype(str))
                    block = mv.transform_feature_block(
                        tab, scaler=bank["scalers"][view], user_ids=ids
                    )
                    m = bank["model"]
                    z = m.encoders[view].predict(block)
                    shared = z @ m.decoders[view]
                    rows[k] = {
                        key: {
                            "coordinates": zz.tolist(),
                            "private_residual": res.tolist(),
                        }
                        for key, zz, res in zip(ids, z, block - shared, strict=True)
                    }
                out["consensus"][rep][view] = rows
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("input", "reference", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--transport", choices=("local", "ssh"), default="local")
    a = p.parse_args()
    if a.out.exists():
        p.error("output already exists")
    doc = li.load(a.input)
    if doc.get("language") != "en" or doc.get("clock") != "opportunity":
        p.error("English observation clock required")
    episodes = doc["episodes"]
    if len({e["id"] for e in episodes}) != len(episodes):
        p.error("duplicate independent history IDs")
    # Keep separate episodes separate, even for the same author.
    rows = [
        (e["id"], t, i, e["condition"])
        for e in episodes
        for i, t in enumerate(e["events"])
    ]
    raw = pd.DataFrame(rows, columns=["author", "body", "created_utc", "subreddit"])
    with threadpool_limits(1):
        panel = vo.canonicalize_comments(raw)
        panel["split"] = "application"
        ref = li.load(a.reference)
        ans = (
            score(panel, ref, a.transport)
            if not panel.empty
            else {
                "version": "released-clock-bank-v1",
                "reader_digest": li.digest(ref),
                "factor": {},
                "consensus": {},
                "coverage": {},
                "status": "no_eligible_events",
            }
        )
        ans.update(
            input_digest=li.digest(doc),
            episode_authors={e["id"]: e["author"] for e in episodes},
            no_eligible_events=sorted(set(raw.author) - set(panel.user_id)),
        )
        li.save(a.out, ans)


if __name__ == "__main__":
    main()
