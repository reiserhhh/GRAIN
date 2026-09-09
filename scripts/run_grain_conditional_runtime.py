#!/usr/bin/env python3
"""Self-contained raw-text CLI for selected conditional-navigation programs."""

import argparse
import json
import sys
from pathlib import Path


def main():
    """Write one score per fixed method, with a single forward for each input."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--program", choices=("conditional",), default="conditional")
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    a = parser.parse_args()
    sys.path.insert(0, str(a.bundle.resolve()))
    from grain_runtime.conditional_runtime import ConditionalRuntime
    from threadpoolctl import threadpool_limits

    a.output.mkdir(parents=True, exist_ok=False)
    rows = []
    with threadpool_limits(limits=2):
        runtime = ConditionalRuntime(a.bundle, a.device)
        for line in a.input.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            if set(d) != {"id", "texts", "query"}:
                raise ValueError("Only id/texts/query permitted")
            for method, score in runtime.score(d["texts"], d["query"]).items():
                rows.append(
                    {
                        "id": d["id"] + "::" + method,
                        "program": "conditional",
                        "method": method,
                        "query": d["query"],
                        "score": score,
                    }
                )
    for name, data in (
        ("predictions.json", rows),
        ("dependencies.json", runtime.dependency_report()),
    ):
        (a.output / name).write_text(
            json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    print({"status": "COMPLETE", "rows": len(rows), "calls": runtime.calls})


if __name__ == "__main__":
    main()
