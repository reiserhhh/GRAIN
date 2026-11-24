#!/usr/bin/env python3
"""GRAIN Liquid: freeze a native text atlas, then measure ordered episodes on that atlas."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from . import factor_discovery as fd
from . import suica as native
from .liquid_slices import cut, fit_policy

VERSION = "grain-liquid-node-v2"
WORKER = Path(__file__).with_name("liquid_worker.py")


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    payload = json.dumps(
        obj, allow_nan=False, ensure_ascii=False, separators=(",", ":")
    )
    with Path(path).open("x", encoding="utf8") as f:
        f.write(payload + "\n")


def episodes(doc, width, policy=None):
    if doc.get("clock") != "opportunity" or doc.get("language") != "en" or width < 0:
        raise ValueError("this frontend requires English and an opportunity clock")
    out = []
    ids = set()
    for e in doc["episodes"]:
        if not all(
            isinstance(e.get(k), str) and e[k]
            for k in ("id", "author", "condition", "carrier", "view")
        ):
            raise ValueError("supply nonempty id, author, condition, carrier and view")
        if e["id"] in ids or e["view"] not in ("A", "B"):
            raise ValueError("duplicate episode or invalid measurement view")
        ids.add(e["id"])
        rows = []
        for i, txt in enumerate(e["events"]):
            if (
                not isinstance(txt, str)
                or not txt.strip()
                or native.PERSONALITY_LEAK_RE.search(txt)
            ):
                raise ValueError(
                    f"{e['id']}:{i}: empty or contains a personality-test clue; supply a clean episode"
                )
            chunks = cut(txt, width, policy or {})
            if not chunks:
                raise ValueError("event contains no tokens")
            rows.extend((i, j, t) for j, t in enumerate(chunks))
        if not rows:
            raise ValueError("empty episode")
        if len(rows) > 128:
            raise ValueError(
                "full deletion tensor limited to 128 opportunities; supply shorter independent episodes explicitly"
            )
        out.append(
            ({k: e[k] for k in ("id", "author", "condition", "carrier", "view")}, rows)
        )
    if not out:
        raise ValueError("no episodes")
    return out


def vectorizer(vocab=None):
    return TfidfVectorizer(
        vocabulary=vocab,
        ngram_range=(1, 2),
        strip_accents="unicode",
        min_df=2,
        sublinear_tf=True,
        max_features=22000,
    )


def tree_fit(x, k, minimum):
    nodes = {}
    leaves = []

    def rec(ids, parent, depth):
        count = min(k if depth == 1 else 3, len(ids) // minimum)
        if count < 2:
            return
        m = MiniBatchKMeans(
            n_clusters=count,
            random_state=42 + depth * 1009 + len(leaves),
            batch_size=min(2048, max(256, len(ids))),
            n_init=5,
            max_iter=100,
            reassignment_ratio=0.01,
        )
        chosen = m.fit_predict(x[ids])
        children = []
        nodes[parent] = {"centers": m.cluster_centers_.tolist(), "children": children}
        for c in range(count):
            sub = ids[chosen == c]
            name = f"{parent}.{c:02d}" if parent else f"n{c:02d}"
            if len(sub) < minimum:
                children.append(None)
            else:
                children.append(name)
                if depth == 2 or len(sub) < 2 * minimum:
                    leaves.append(name)
                else:
                    rec(sub, name, depth + 1)

    rec(np.arange(len(x)), "", 1)
    if len(leaves) < 2:
        raise ValueError("insufficient calibration slices for a native tree")
    return {"tree": nodes, "labels": [*leaves, "unassigned"]}


def fit_atlas(texts, k, method):
    tf = vectorizer()
    mat = tf.fit_transform(texts)
    dim = min(16, mat.shape[0] - 1, mat.shape[1] - 1)
    if dim < 2 or len(texts) < 2 * k or k < 2:
        raise ValueError("reference too small")
    sv = TruncatedSVD(n_components=dim, random_state=42)
    dense = sv.fit_transform(mat)
    sc = StandardScaler().fit(dense)
    x = sc.transform(dense)
    atlas = {
        "vocab": {k: int(v) for k, v in tf.vocabulary_.items()},
        "idf": tf.idf_.tolist(),
        "svd": sv.components_.tolist(),
        "mean": sc.mean_.tolist(),
        "sd": sc.scale_.tolist(),
        "method": method,
    }
    if method == "gmm_soft":
        m = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            max_iter=120,
            reg_covar=1e-5,
            random_state=42,
        ).fit(x)
        atlas.update(
            labels=[f"gmm_{j:02d}" for j in range(k)],
            weights=m.weights_.tolist(),
            means=m.means_.tolist(),
            variances=m.covariances_.tolist(),
        )
    else:
        atlas.update(tree_fit(x, k, max(4, len(texts) // (k * 8))))
    atlas["labels"].append("outside_vocabulary")
    return atlas


def locate(atlas, texts):
    tf = vectorizer(atlas["vocab"])
    tf.idf_ = np.asarray(atlas["idf"])
    mat = tf.transform(texts)
    x = (mat @ np.asarray(atlas["svd"]).T - atlas["mean"]) / atlas["sd"]
    if atlas["method"] == "gmm_soft":
        var = np.array(atlas["variances"])
        logp = np.log(atlas["weights"]) - 0.5 * (
            np.log(var).sum(-1) + ((x[:, None] - atlas["means"]) ** 2 / var).sum(-1)
        )
        probs = softmax(logp, axis=1)
    else:
        p = []
        labels = atlas["labels"][:-1]
        for v in x:
            name = ""
            while name in atlas["tree"]:
                node = atlas["tree"][name]
                name = node["children"][
                    np.argmin(((np.array(node["centers"]) - v) ** 2).sum(1))
                ]
            p.append(labels.index(name) if name in labels else len(labels) - 1)
        probs = np.eye(len(labels))[p]
    covered = np.asarray(mat.getnnz(axis=1) > 0)
    # Coverage is itself observed; never route a zero TF-IDF vector to a semantic leaf.
    return np.column_stack([probs * covered[:, None], ~covered]), covered


def call(req, transport, *, worker_code=None):
    code = WORKER.read_text() if worker_code is None else worker_code
    if transport == "ssh":
        host = os.environ["GRAIN_LIQUID_HOST"]
        python = shlex.quote(os.environ["GRAIN_LIQUID_PYTHON"])
        lib = shlex.quote(os.environ.get("GRAIN_LIQUID_NCPS_PATH", ""))
        cmd = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=6",
            *(
                ["-S", os.environ["GRAIN_LIQUID_SSH_SOCKET"]]
                if os.getenv("GRAIN_LIQUID_SSH_SOCKET")
                else []
            ),
            host,
            f"PYTHONPATH={lib} {python} -c " + shlex.quote(code),
        ]
    else:
        cmd = [sys.executable, "-c", code]
    r = subprocess.run(
        cmd,
        input=json.dumps(req).encode(),
        stdout=subprocess.PIPE,
        check=True,
        timeout=600,
    )
    ans = json.loads(r.stdout)
    if ans.pop("request_digest") != digest(req):
        raise ValueError("worker/input mismatch")
    return ans


def native_paths(atlas, items):
    paths = []
    for meta, rows in items:
        p, _ = locate(atlas, [r[2] for r in rows])
        paths.append(p.tolist())
    return paths


def trajectory(p, obs):
    p = np.array(p)
    y = np.array(obs["readout"])
    reset = np.array(obs["reset"])
    effect = np.array(obs["missing_effect"])
    if (
        y.shape != p.shape
        or effect.shape != (len(p), *p.shape)
        or not np.isfinite(effect).all()
    ):
        raise ValueError("invalid traced sequence")
    if any(np.max(np.abs(effect[k, :k])) > 1e-5 for k in range(1, len(p))):
        raise ValueError("future source changed prefix")
    strength = np.linalg.norm(effect, axis=-1)
    # Input-origin nodes and readout-destination nodes use the SAME frozen dictionary.
    signed = np.einsum("ka,ktb->tab", p, effect)
    origin = strength.T @ p
    joint = y - obs["all_missing"]

    def rms(v):
        return float(np.sqrt(np.mean(np.square(v))))

    hist = y - reset
    ema = p.copy()
    for j in range(1, len(p)):
        ema[j] = 0.5 * ema[j - 1] + 0.5 * p[j]
    return {
        "native": p.tolist(),
        "liquid": y.tolist(),
        "history": hist.tolist(),
        "fixed_half": ema.tolist(),
        "anchored_signed": (p + hist).tolist(),
        "missing_effect": effect.tolist(),
        "source_strength": strength.tolist(),
        "origin_mass": origin.tolist(),
        "source_node_to_readout_node": signed.tolist(),
        "joint_minus_sum": (joint - effect.sum(axis=0)).tolist(),
        "native_stir_level_rms": rms(p.mean(0) - p[::-1].mean(0)),
        "liquid_stir_endpoint_rms": rms(y[-1] - obs["reverse_endpoint"]),
        "same_ends_middle_swap_rms": None
        if obs["middle_swap_endpoint"] is None
        else rms(y[-1] - obs["middle_swap_endpoint"]),
    }


def participant_features(items, paths, conditions):
    groups = {}
    for (meta, _), series in zip(items, paths, strict=True):
        groups.setdefault((meta["author"], meta["view"]), []).append(
            (meta, np.asarray(series))
        )
    out = {}
    for key, records in groups.items():
        seq = [x for _, x in records]
        pool = np.concatenate(seq)
        base = pool.mean(0)
        desc = fd._pooled_path_descriptors(seq)
        conditional = []
        shares = []
        for cond in conditions:
            cell = [x for m, x in records if m["condition"] == cond]
            shares.append(sum(map(len, cell)) / len(pool))
            conditional.extend(
                (np.concatenate(cell).mean(0) - base)
                if cell
                else np.full(len(base), np.nan)
            )
        order = (
            "volatility",
            "inertia",
            "roughness",
            "trend",
            "changepoint",
            "multiscale",
        )
        out[key] = {
            "static": np.r_[base, shares],
            "dynamic": np.concatenate(
                [desc.get(s, np.full(len(base), np.nan)) for s in order]
            ),
            "hybrid": np.array(conditional),
        }
    return out


def fit_axes(groups):
    ids = sorted({u for u, v in groups if (u, "A") in groups and (u, "B") in groups})
    if len(ids) < 4:
        return {"status": "need_four_source_authors_with_both_views", "n": len(ids)}
    out = {"status": "source_candidates", "n": len(ids)}
    for fam in ("static", "dynamic", "hybrid"):
        a, b = [np.stack([groups[u, v][fam] for u in ids]) for v in ("A", "B")]
        dims = np.flatnonzero(
            np.isfinite(a).all(0)
            & np.isfinite(b).all(0)
            & (np.vstack([a, b]).std(0) > 1e-8)
        )
        if len(dims) < 2:
            out[fam] = {"indices": [], "k": 0}
            continue
        m = fd.fit_stable_crossview(
            a[:, dims],
            b[:, dims],
            n_permutations=99,
            max_factors=min(8, len(ids) - 1),
            seed=431,
        )
        out[fam] = {
            "indices": dims.tolist(),
            "k": m.n_factors,
            "center": m.center.tolist(),
            "scale": m.scale.tolist(),
            "loadings": m.rotated_directions.tolist(),
        }
    return out


def scores(groups, model):
    def nullable(a):
        return [float(v) if np.isfinite(v) else None for v in a]

    rows = []
    for (user, view), f in groups.items():
        row = {"author": user, "view": view}
        for fam, arr in f.items():
            ax = model.get(fam, {"k": 0})
            val = arr[ax["indices"]] if ax["k"] else np.array([])
            row[fam] = {
                "features": nullable(arr),
                "factor_scores": None
                if not ax["k"] or not np.isfinite(val).all()
                else (
                    (val - ax["center"]) / ax["scale"] @ np.array(ax["loadings"])
                ).tolist(),
            }
        rows.append(row)
    return rows


def run(
    mode,
    doc,
    ref,
    transport,
    width=0,
    k=8,
    method="gmm_soft",
    cut_kind="fixed_or_turn",
    rind_dir=None,
):
    fitting = mode == "fit"
    if not fitting and (
        ref["version"] != VERSION or ref["worker"] != digest(WORKER.read_text())
    ):
        raise ValueError("incompatible reference or worker")
    policy = (
        fit_policy(episodes(doc, 0), cut_kind, rind_dir)
        if fitting
        else ref.get("cut_policy", {})
    )
    items = episodes(doc, width if fitting else ref["width"], policy)
    if fitting:
        atlas = fit_atlas([r[2] for _, ev in items for r in ev], k, method)
        ref = {
            "version": VERSION,
            "width": width,
            "cut_policy": policy,
            "atlas": atlas,
            "worker": digest(WORKER.read_text()),
            "conditions": sorted({m["condition"] for m, _ in items}),
            "source_digest": digest(doc),
        }
    paths = native_paths(ref["atlas"], items)
    if fitting:
        ref.update(call({"mode": "fit", "paths": paths}, transport))
    observed = call(
        {"mode": "score", "weights": ref["weights"], "paths": paths}, transport
    )["paths"]
    tracks = [trajectory(p, o) for p, o in zip(paths, observed, strict=True)]
    families = {}
    for channel in ("native", "fixed_half", "liquid", "anchored_signed"):
        feats = participant_features(
            items, [t[channel] for t in tracks], ref["conditions"]
        )
        if fitting:
            ref.setdefault("axes", {})[channel] = fit_axes(feats)
        else:
            families[channel] = scores(feats, ref["axes"][channel])
    if fitting:
        return ref
    slice_frame = pd.DataFrame(
        {"user_id": [m["author"] for m, rr in items for _ in rr]}
    )
    batch_scores = native.soft_node_scores(
        slice_frame, np.concatenate(paths), ref["atlas"]["labels"]
    )
    rows = {
        m["id"]: {
            **m,
            "slots": [list(r[:2]) for r in ev],
            "unit_digests": [digest(r[2]) for r in ev],
            **t,
        }
        for (m, ev), t in zip(items, tracks, strict=True)
    }
    pairs = []
    for spec in doc.get("pairs", []):
        a, b = rows[spec["left"]], rows[spec["right"]]
        if "source_event" in spec:
            available = [
                [
                    j
                    for j, (event, _) in enumerate(e["slots"])
                    if event == spec["source_event"]
                ]
                for e in (a, b)
            ]
            if not all(available):
                raise ValueError("source event absent from paired branches")
            ia, ib = [positions[-1] for positions in available]
        else:
            ia = ib = spec["source_index"]
        if (
            spec["kind"] not in ("withdrawal", "carrier")
            or a["id"] == b["id"]
            or a["author"] != b["author"]
        ):
            raise ValueError("invalid paired branches")
        if not (0 <= ia < len(a["slots"]) and 0 <= ib < len(b["slots"])):
            raise ValueError("source_index is a zero-based slice position")
        if spec["kind"] == "withdrawal" and (
            a["unit_digests"][: ia + 1] != b["unit_digests"][: ib + 1]
            or a["carrier"] != b["carrier"]
            or a["unit_digests"] == b["unit_digests"]
        ):
            raise ValueError(
                "withdrawal requires a shared actual prefix and distinct later observations"
            )
        if spec["kind"] == "carrier" and a["carrier"] == b["carrier"]:
            raise ValueError("supply different carriers")
        pair = dict(spec)
        for side, e, i in (("left", a, ia), ("right", b, ib)):
            s = e["source_strength"][i]
            pair[side + "_trace"] = {
                "write": s[i],
                "path": s[i:],
                "retention": s[-1] / s[i] if s[i] > 1e-9 else None,
                "elapsed_opportunities": len(s) - 1 - i,
            }
            slot = e["slots"][i][0]
            pre = (
                next(j for j, (event, _) in enumerate(e["slots"]) if event == slot) - 1
            )
            if spec["kind"] == "withdrawal" and pre >= 0:
                return_paths = {}
                for channel in ("native", "fixed_half", "liquid", "anchored_signed"):
                    path = np.array(e[channel])
                    dist = np.linalg.norm(path[i:] - path[pre], axis=1)
                    return_paths[channel] = {
                        "distance_to_pre_source": dist.tolist(),
                        "fraction_returned_at_end": 1 - float(dist[-1] / dist[0])
                        if dist[0] > 1e-9
                        else None,
                    }
                pair[side + "_trace"]["observed_return"] = return_paths
        pair["endpoint_difference"] = (
            np.array(a["liquid"][-1]) - b["liquid"][-1]
        ).tolist()
        pairs.append(pair)
    labels = ref["atlas"]["labels"]
    descriptors = (
        "volatility",
        "inertia",
        "roughness",
        "trend",
        "changepoint",
        "multiscale",
    )
    names = {
        "static": [*labels, *["choice::" + c for c in ref["conditions"]]],
        "dynamic": [s + "::" + k for s in descriptors for k in labels],
        "hybrid": [c + "::" + k for c in ref["conditions"] for k in labels],
    }
    return {
        "version": VERSION,
        "reference_digest": digest(ref),
        "input_digest": digest(doc),
        "clock": "one opportunity per retained slice",
        "node_names": ref["atlas"]["labels"],
        "feature_names": names,
        "unknown_conditions": sorted(
            {m["condition"] for m, _ in items} - set(ref["conditions"])
        ),
        "conditions": ref["conditions"],
        "episodes": rows,
        "families": families,
        "contrasts": pairs,
        "legacy_batch_relative_node_scores": batch_scores.to_dict(orient="records"),
    }


def main():
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("mode", choices=["fit", "score"])
    pa.add_argument("--input", required=True)
    pa.add_argument("--reference")
    pa.add_argument("--out", required=True)
    pa.add_argument("--transport", choices=["local", "ssh"], default="local")
    pa.add_argument(
        "--width", type=int, default=0, help="fit only; 0 preserves whole utterances"
    )
    pa.add_argument("--nodes", type=int, default=8)
    pa.add_argument("--method", choices=["gmm_soft", "kmeans_tree"], default="gmm_soft")
    pa.add_argument(
        "--cut", choices=["fixed_or_turn", "rind", "semantic"], default="fixed_or_turn"
    )
    pa.add_argument("--rind-dir", type=Path)
    args = pa.parse_args()
    if args.mode == "score" and not args.reference:
        pa.error("score requires a frozen --reference")
    if Path(args.out).exists():
        pa.error("--out must be new")
    with threadpool_limits(limits=1):
        save(
            args.out,
            run(
                args.mode,
                load(args.input),
                load(args.reference) if args.reference else None,
                args.transport,
                args.width,
                args.nodes,
                args.method,
                args.cut,
                args.rind_dir,
            ),
        )


if __name__ == "__main__":
    main()
