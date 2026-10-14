#!/usr/bin/env python3
"""Cold conditional response construction, without tokenizer/model/answer input."""

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
from grain_research.construction import math_expression_response as M
from grain_research.construction import math_history_routing as R
from grain_research.construction import math_recursive_collector as C


def main():
    """Construct every registered focus before opening any compatibility reference."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.input.read_text())
    for path, expected in spec["hashes"].items():
        if C.stream_sha(path) != expected:
            raise ValueError("Changed sealed constructor dependency: " + path)
    if args.output.exists():
        raise FileExistsError("Preserve prior cold outputs")
    if "transformers" in sys.modules:
        raise RuntimeError("Cold process must not load transformers")
    torch.set_num_threads(2)
    if spec["device"] == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Declared device unavailable")
    begun = time.perf_counter()
    weights = C.load_weights(spec["weights"], spec["blocks"], spec["device"])
    with np.load(spec["coordinates"], allow_pickle=False) as z:
        coordinates = {name: z[name] for name in z.files}
    keys, values, readings = [], [], []
    compute = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(spec["contexts"]), spec["batch_size"]):
            if time.perf_counter() - compute > spec["max_seconds"]:
                raise TimeoutError("Finite cold budget; no automatic renewal")
            paths = spec["contexts"][start : start + spec["batch_size"]]
            histories = {name: [] for name in ("k", "v")}
            for path in paths:
                with np.load(path, allow_pickle=False) as z:
                    for name, values in histories.items():
                        values.append(z[name])
            k, v = M.compile_histories(
                {name: np.stack(a) for name, a in histories.items()},
                coordinates["seeds"],
                coordinates["cos"],
                coordinates["sin"],
                np.asarray(spec["valid"])[start : start + len(paths)],
                weights,
                spec["config"],
                spec["device"],
            )
            keys.extend(k)
            values.extend(v)
            readings.extend(
                np.stack(
                    [
                        R.read_current(kk, vv, spec["endpoint"], spec["reader"])
                        for kk, vv in zip(krow, vrow, strict=True)
                    ]
                )
                for krow, vrow in zip(k, v, strict=True)
            )
            print("HM119 cold", start + len(paths), flush=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(
            stream, k=np.stack(keys), v=np.stack(values), readings=np.stack(readings)
        )
    receipt = {
        "histories": len(keys),
        "constructed_focuses": len(keys) * 2,
        "pretrained_arithmetic_dependency": True,
        "full_model_calls": 0,
        "transformers_loaded": "transformers" in sys.modules,
        "application_answers_or_current_reference_input": False,
        "fits": 0,
        "loading_seconds": compute - begun,
        "execution_seconds": time.perf_counter() - compute,
        "input_sha256": C.stream_sha(args.input),
        "output_sha256": C.stream_sha(args.output),
    }
    with args.output.with_suffix(".json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    print("HM119 complete", receipt, flush=True)


if __name__ == "__main__":
    main()
