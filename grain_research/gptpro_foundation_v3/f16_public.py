#!/usr/bin/env python3
"""Target-specific public contracts over immutable, source-fitted F16 engines."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from . import f16_cutoff as clock
from . import f16_dynamic_text as dynamic
from . import f16_t1_source as numeric_t1
from . import f16_t1_text as text_t1

VERSION = "F16_public_source_compilation_v1"
CLASSES = {
    "T1": "conditional_response_function",
    "T2": "dynamic_state",
    "T3": "predictive_state",
}
ENGINES = {"text_t1", "text_dynamic", "numeric_t1", "numeric_clock"}
CODE_FILES = (
    "f16_public.py",
    "f16_cutoff.py",
    "f16_dynamic_text.py",
    "f16_t1_text.py",
    "f16_t1_source.py",
    "f16_state_instrument.py",
    "f16_independent_t3.py",
    "f05_joint.py",
    "f05_recovery.py",
    "f13_dynamic_parameter.py",
)


def json_bytes(value):
    """Serialize canonical finite JSON without pickle or executable payloads."""
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def object_hash(value):
    """Bind the full JSON object, not only its target label."""
    return hashlib.sha256(json_bytes(value)).hexdigest()


def file_hash(path):
    """Hash a declared local construction artifact."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_manifest():
    """Record current adapter and explicit numerical engine dependencies."""
    root = Path(__file__).parent
    return {name: file_hash(root / name) for name in CODE_FILES}


def require_hash(value, label):
    """Reject absent or malformed identities before any inference."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"invalid {label} hash")


def describe(base, target):
    """Derive canonical metadata from a validated known source-fitted engine."""
    if target not in CLASSES:
        raise ValueError("unknown target")
    text = base.get("version") in (text_t1.VERSION, dynamic.VERSION)
    if base.get("version") == text_t1.VERSION:
        if target != "T1" or base.get("target_id") != "T1":
            raise ValueError("T1 text target mismatch")
        if base.get("object_class") != CLASSES["T1"]:
            raise ValueError("T1 text object class mismatch")
        text_t1.verify_bundle(base)
        engine = "text_t1"
        source = base["source_bundle_hash"]
        require_hash(source, "source bundle")
        count = base["reference_authors"]
    elif base.get("version") == dynamic.VERSION:
        if (
            target not in ("T2", "T3")
            or base.get("target_ids") != ["T2", "T3"]
            or base.get("object_classes") != ["dynamic_state", "future_response"]
            or base.get("qualification_rate") != 0.5
        ):
            raise ValueError("dynamic text target/class/design mismatch")
        dynamic.validate_bundle(base)
        if base["reader_hash"] != object_hash(
            {"materials": base["materials"], "channels": base["channel_draws"]}
        ):
            raise ValueError("dynamic cached reader identity mismatch")
        engine = "text_dynamic"
        source = {
            k: base[k]
            for k in ("source_hash", "source_bundle_sha256", "reader_bundle_sha256")
        }
        count = base.get("source_reference_authors", base.get("source_authors"))
    elif base.get("formula_version") == "T1_four_cell_source_mixture_v1":
        if target != "T1" or base.get("object_class") != CLASSES["T1"]:
            raise ValueError("numeric T1 target/class mismatch")
        numeric_t1.verify_bundle(base)
        engine = "numeric_t1"
        source = {k: base[k] for k in ("source_hash", "source_fits_hash")}
        count = base.get("source_reference_authors", base.get("source_authors"))
    elif base.get("target_id") in ("T2", "T3"):
        if base["target_id"] != target or base.get("object_class") != CLASSES[target]:
            raise ValueError("numeric dynamic target/class mismatch")
        clock.validate_base(base)
        engine = "numeric_clock"
        source = base["source_hash"]
        count = base.get("source_reference_authors", base.get("source_authors"))
    else:
        raise ValueError("unsupported source-fitted engine")
    if type(count) is not int or count < 2:
        raise ValueError("source reference population required")
    if not source:
        raise ValueError("source provenance required")
    if text:
        reader = object_hash(
            {
                "materials": base["materials"],
                "channels": base["channel_draws"],
                "version": base["version"],
                "qualification_rate": base["qualification_rate"],
                "alphabet": "2*(pole>=.5)+(qualification>=.5)",
            }
        )
    else:
        reader = base["reader_hash"]
        require_hash(reader, "numeric reader")
    if engine == "text_dynamic":
        ref = base["reference"][target]
        mu, sigma = ref["mu"], ref["sigma"]
    else:
        mu, sigma = base["reference_mu"], base["reference_sigma"]
    if not np.isfinite([mu, sigma]).all() or sigma <= 0:
        raise ValueError("finite frozen reference required")
    for key in (
        "unit",
        "invalidating_result",
        "support_rule",
        "error_model",
        "explanation_rule",
    ):
        if not isinstance(base.get(key), str) or not base[key].strip():
            raise ValueError(f"missing semantic declaration: {key}")
    conditions = [-1.0, -1 / 3, 1 / 3, 1.0] if target == "T1" else [-1.0, 1.0]
    return {
        "target_id": target,
        "object_class": CLASSES[target],
        "invalidating_result": base["invalidating_result"],
        "formula_version": base.get("version", base.get("formula_version")),
        "observed_role": "controlled_text_reader" if text else "numeric_observation",
        "unit": base["unit"],
        "reference_mu": mu,
        "reference_sigma": sigma,
        "source_hash": object_hash(source),
        "source_reference_authors": count,
        "reader_hash": reader,
        "world_contract": base["world_contract"],
        "support_rule": base["support_rule"],
        "error_model": base["error_model"],
        "explanation_rule": base["explanation_rule"],
        "time_contract": {
            "unit": "simulator_step",
            "target_time": 31 if target == "T1" else 47,
            "rule": "requested cutoff;retain missing steps;not calendar days",
        },
        "condition_contract": {
            "support": conditions,
            "future": [1, -1, 1, -1] if target == "T3" else [],
        },
        "observation_contract": "controlled_text_history_v1"
        if text
        else "numeric_prefix_v1",
        "engine": engine,
    }


def build_instrument(brief, source_manifest):
    """Compile a source-fitted, hash-pinned engine without refitting or app data.

    Existing source constructors own estimation. This public compilation stage
    owns target semantics, immutable packaging and cold execution consistency.
    """
    if (
        set(source_manifest) != {"path", "sha256", "role"}
        or source_manifest["role"] != "source_fitted"
    ):
        raise ValueError("declared source-fitted manifest required")
    require_hash(source_manifest["sha256"], "manifest")
    payload = Path(source_manifest["path"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != source_manifest["sha256"]:
        raise ValueError("source-fitted bundle bytes changed")
    base = json.loads(payload)
    metadata = describe(base, brief.get("target_id"))
    fields = ("target_id", "object_class", "unit", "world_contract", "observed_role")
    if brief != {k: metadata[k] for k in fields}:
        raise ValueError("brief and source engine disagree")
    codes = code_manifest()
    return {
        "schema_version": VERSION,
        **metadata,
        "scorer_code_hash": object_hash(codes),
        "code_manifest": codes,
        "base_bundle": base,
        "base_bundle_hash": object_hash(base),
        "provenance": {
            "source_fitted_file_sha256": source_manifest["sha256"],
            "construction": "compile already source-fitted model;not new source fitting",
        },
    }


def verify_bundle(bundle):
    """Re-derive, rather than trust, every public semantic declaration."""
    if not isinstance(bundle, dict) or bundle.get("schema_version") != VERSION:
        raise ValueError("wrong public bundle version")
    expected = describe(bundle["base_bundle"], bundle.get("target_id"))
    extras = {
        "schema_version",
        "scorer_code_hash",
        "code_manifest",
        "base_bundle",
        "base_bundle_hash",
        "provenance",
    }
    if set(bundle) != set(expected) | extras:
        raise ValueError("missing or undeclared public bundle fields")
    if any(bundle[k] != v for k, v in expected.items()):
        raise ValueError("public metadata inconsistent with source engine")
    codes = code_manifest()
    if bundle["code_manifest"] != codes or bundle["scorer_code_hash"] != object_hash(
        codes
    ):
        raise ValueError("public engine code changed")
    if bundle["base_bundle_hash"] != object_hash(bundle["base_bundle"]):
        raise ValueError("nested source bundle changed")
    provenance = bundle["provenance"]
    if set(provenance) != {"source_fitted_file_sha256", "construction"}:
        raise ValueError("source compilation provenance required")
    require_hash(provenance["source_fitted_file_sha256"], "compiled source")


def request_contract(bundle):
    """Expose only the declared input contract; never application reference norms."""
    return {
        k: copy.deepcopy(bundle[k])
        for k in (
            "target_id",
            "object_class",
            "unit",
            "reader_hash",
            "world_contract",
            "time_contract",
            "condition_contract",
            "observation_contract",
        )
    }


def validate_request(request, bundle):
    """Reject wrong targets, readers, units, clocks and hidden outcome fields."""
    if set(request) != {"contract", "data"} or request["contract"] != request_contract(
        bundle
    ):
        raise ValueError("request contract mismatch")
    data, target = request["data"], bundle["target_id"]
    if not isinstance(data, dict):
        raise TypeError("observed data envelope required")
    if (
        bundle["engine"].startswith("text") or bundle["engine"] == "numeric_clock"
    ) and (
        type(data.get("target_time")) is not int
        or data["target_time"] != bundle["time_contract"]["target_time"]
    ):
        raise ValueError("requested time outside source-frozen public target")
    if bundle["engine"] == "numeric_t1" and any(
        type(r.get("t")) is not int or not 0 <= r["t"] <= 31
        for r in data.get("records", [])
    ):
        raise ValueError("numeric T1 observation outside target clock")
    if target not in CLASSES:
        raise ValueError("unknown request target")
    return data


def cold_scores(requests, bundle):
    """Score from bundle and observed requests only, keeping source norms fixed."""
    verify_bundle(bundle)
    inputs = [validate_request(r, bundle) for r in requests]
    if not inputs:
        return []
    base, engine, target = bundle["base_bundle"], bundle["engine"], bundle["target_id"]
    if engine == "text_t1":
        outputs = text_t1.cold_scores(inputs, base)
    elif engine == "text_dynamic":
        outputs = dynamic.cold_scores(inputs, base)
    elif engine == "numeric_t1":
        outputs = [numeric_t1.cold_score(raw, base) for raw in inputs]
    else:
        adapter = clock.build_adapter(base)
        outputs = [clock.cold_score(raw, adapter) for raw in inputs]
    result = []
    for row in outputs:
        projected = row["targets"][target] if engine == "text_dynamic" else row
        result.append(
            {
                "target_id": target,
                "object_class": CLASSES[target],
                "score": projected["score"],
                "variance": projected["variance"],
                "reference_score": projected["reference_score"],
                "unit": bundle["unit"],
                "target_time": bundle["time_contract"]["target_time"],
                "engine_output": row,
            }
        )
    return result


def cold_score(request, bundle):
    """Single-author variant of the identical batch-independent computation."""
    return cold_scores([request], bundle)[0]


def main():
    """Load externally authenticated finite JSON and write a fresh score file."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "input", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    payload = args.bundle.read_bytes()
    require_hash(args.sha256, "trusted public bundle")
    if hashlib.sha256(payload).hexdigest() != args.sha256:
        raise ValueError("untrusted public bundle")
    requests = [json.loads(line) for line in args.input.read_text().splitlines()]
    outputs = cold_scores(requests, json.loads(payload))
    with args.output.open("x") as stream:
        for row in outputs:
            stream.write(json_bytes(row).decode() + "\n")


if __name__ == "__main__":
    main()
