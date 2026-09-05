#!/usr/bin/env python3
"""VM00: explicitly scoped local data inventory, without model or label fitting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.inventory import data_inventory


def main():
    """Write an aggregate schema report to an ignored run directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = data_inventory(args.data_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "data_inventory.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# VM00 Data Schema",
        "",
        "PANDORA schema/ID inventory only; no external labels analyzed.",
        "",
    ]
    for name, info in result.items():
        if isinstance(info, dict):
            lines += [
                f"## {name}",
                "",
                f"Status: {info['status']}",
                f"Fields: {', '.join(info.get('fields', []))}",
                f"Rows: {info.get('rows', 'not scanned')}; authors: {info.get('unique_authors', 'not scanned')}",
                "",
            ]
    (args.output_dir / "data_schema.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                k: {p: v for p, v in d.items() if p != "fields"}
                if isinstance(d, dict)
                else d
                for k, d in result.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
