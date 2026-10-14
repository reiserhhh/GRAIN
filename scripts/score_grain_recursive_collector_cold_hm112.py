#!/usr/bin/env python3
"""Cold numeric current compilation: no tokenizer, model object or joint answers."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_qualified_composition as M
from grain_research.construction import math_recursive_collector as C


def main():
    """Authenticate the closed numeric interface and execute both fixed variants."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.input.read_text())
    for path, expected in spec["hashes"].items():
        if C.stream_sha(path) != expected:
            raise ValueError("Changed sealed numeric/compiler dependency: " + path)
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError("Preserve prior numeric outputs")
    if "transformers" in sys.modules:
        raise RuntimeError("Cold compiler must not load transformers")
    torch.set_num_threads(2)
    if spec["device"] == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Declared device unavailable; no silent platform change")
    started = time.perf_counter()
    with np.load(spec["library"], allow_pickle=False) as source:
        library = {name: source[name] for name in source.files}
    weights = C.load_weights(spec["weights"], spec["blocks"], spec["device"])
    loading_seconds = time.perf_counter() - started
    keys, values, scores = [], [], []
    q, d = np.asarray(spec["queries"]), np.asarray(spec["directions"])
    execution_started = time.perf_counter()
    with torch.no_grad():
        for variant in spec["variants"]:
            kk, vv, ss = [], [], []
            for start in range(0, len(spec["cases"]), spec["batch_size"]):
                if time.perf_counter() - execution_started > spec["max_seconds"]:
                    raise TimeoutError("Fixed cold budget exhausted; do not reset")
                batch = spec["cases"][start : start + spec["batch_size"]]
                k, v = C.compile_current(
                    library, batch, weights, spec["config"], variant, spec["device"]
                )
                kk.append(k)
                vv.append(v)
                ss.extend(M.attention(a, b, q, d) for a, b in zip(k, v, strict=True))
                print(
                    "HM112 numeric",
                    spec["bank"],
                    variant,
                    start + len(batch),
                    flush=True,
                )
            keys.append(np.concatenate(kk))
            values.append(np.concatenate(vv))
            scores.append(np.stack(ss))
    with args.output.open("xb") as stream:
        np.savez_compressed(
            stream, k=np.stack(keys), v=np.stack(values), scores=np.stack(scores)
        )
    receipt = {
        "bank": spec["bank"],
        "cases": len(spec["cases"]),
        "variants": spec["variants"],
        "current_only_neural_arithmetic_executions": len(spec["cases"])
        * len(spec["variants"]),
        "full_model_forwards": 0,
        "neural_parameter_dependency": True,
        "transformers_loaded": "transformers" in sys.modules,
        "tokenizer_or_raw_text_input": False,
        "joint_reference_input": False,
        "fitted_parameters": 0,
        "loading_seconds": loading_seconds,
        "execution_seconds": time.perf_counter() - execution_started,
        "input_sha256": C.stream_sha(args.input),
        "output_sha256": C.stream_sha(args.output),
        "weight_keys": [
            [f"model.layers.{i}.{key}" for key in sorted(w)]
            for i, w in enumerate(weights)
        ],
    }
    with args.output.with_suffix(".json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    print(
        "HM112 cold complete",
        receipt["cases"],
        receipt["execution_seconds"],
        flush=True,
    )


if __name__ == "__main__":
    main()
