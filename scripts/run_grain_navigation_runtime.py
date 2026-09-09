#!/usr/bin/env python3
"""Score JSONL excerpts with a self-contained GRAIN navigation bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    """Load a selected frozen program and write results and dependency evidence."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--program", choices=("token", "context4", "state1", "state2"), default="token"
    )
    p.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    a = p.parse_args()
    sys.path.insert(0, str(a.bundle.resolve()))
    from grain_runtime.navigation_runtime import NavigationRuntime
    from threadpoolctl import threadpool_limits

    a.output.mkdir(parents=True, exist_ok=False)
    results = []
    with threadpool_limits(limits=2):
        runtime = NavigationRuntime(a.bundle, a.program, a.device)
        with a.input.open(encoding="utf-8") as stream:
            for i, line in enumerate(stream):
                row = json.loads(line)
                if set(row) - {"id", "texts", "query"}:
                    raise ValueError(
                        "Only id/texts/query allowed; do not supply labels or teacher states"
                    )
                result = runtime.score(row["texts"], row["query"])
                results.append({"id": row.get("id", str(i)), **result})
    (a.output / "predictions.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (a.output / "dependencies.json").write_text(
        json.dumps(runtime.dependency_report(), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps({"status": "COMPLETE", "program": a.program, "rows": len(results)})
    )


if __name__ == "__main__":
    main()
