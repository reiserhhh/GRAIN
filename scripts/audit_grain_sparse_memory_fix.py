#!/usr/bin/env python3
"""Compare pre-fix and current nested selectors on fixed synthetic fixtures."""

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms import external_heads as heads
from scripts.run_grain_vector_mechanism_controls import write_json


def main():
    """Replay only trusted repository code and synthetic labels, never source data."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if any(x for x in a.output_dir.iterdir() if not x.name.startswith("._")):
        raise ValueError("Fresh synthetic audit directory required")
    ref = "14505a7:grain_research/vector_mechanisms/external_heads.py"
    old = subprocess.check_output(["git", "show", ref], cwd=ROOT).decode()
    definition = next(
        n
        for n in ast.parse(old).body
        if isinstance(n, ast.FunctionDef) and n.name == "select_head"
    )
    namespace = dict(vars(heads))
    # Only the named function from our fixed, trusted repository commit is loaded.
    exec(  # noqa: S102
        compile(ast.Module(body=[definition], type_ignores=[]), ref, "exec"), namespace
    )
    previous = namespace["select_head"]
    rng = np.random.default_rng(731)
    dense = rng.normal(size=(60, 6))
    ids = np.asarray([str(i) for i in range(60)])
    binary = (dense[:, 0] > 0).astype(int)
    continuous = dense[:, :5] * 0.1 + rng.normal(0, 0.03, size=(60, 5))
    text = np.asarray(
        [
            f"{'hello group happy' if y else 'quiet alone reflective'} example repeat {i % 7}"
            for i, y in enumerate(binary)
        ]
    )
    records = []
    with threadpool_limits(limits=1):
        for kind, x in (("dense", dense), ("word_char", text)):
            for task, y in (("EI_cont", binary), ("big5", continuous)):
                before, bs, bp = previous(x, y, ids, kind, task, pca=False)
                after, search, parameter = heads.select_head(
                    x, y, ids, kind, task, pca=False
                )
                error = float(
                    np.max(
                        abs(
                            heads.predict_head(before, x, task)
                            - heads.predict_head(after, x, task)
                        )
                    )
                )
                if bs != search or bp != parameter or error != 0:
                    raise ValueError("Matrix lifetime changed selection or predictions")
                records.append(
                    {
                        "feature": kind,
                        "task": task,
                        "parameter": parameter,
                        "max_prediction_error": error,
                        "all_inner_scores_equal": True,
                    }
                )
    result = {
        "status": "COMPLETE_SYNTHETIC_NUMERICAL_EQUIVALENCE",
        "reference": ref,
        "reference_sha256": hashlib.sha256(old.encode()).hexdigest(),
        "current_sha256": hashlib.sha256(Path(heads.__file__).read_bytes()).hexdigest(),
        "fixtures": records,
        "private_data_read": False,
    }
    write_json(a.output_dir / "summary.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
