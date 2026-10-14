#!/usr/bin/env python3
"""Archive public source text and installed-package metadata without executing it."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd
import requests

MEH_REVISION = "3b306a74490b5f99c171ac683e6cb8fe8f3ea3dc"
MEH_FILES = [
    "LICENSE",
    "README.MD",
    "MEH 2/MainForm.cs",
    "MEH 2/BGWorkerData.cs",
    "MEH 2/NgramDictionaryBuilder.cs",
    "MEH 2/TextConversionClass.cs",
    "MEH 2/LemmatizerChooser.cs",
    "MEH 2/Properties/Settings.Designer.cs",
    "MEH 2/Properties/AssemblyInfo.cs",
    "MEH 2/Resources/stoplists/stoplist_english.txt",
    "MEH 2/Resources/default_conversion_lists/conversions.txt",
    "MEH 2/Resources/lemmatization-lists-master/README.md",
]


def sha(path):
    """Hash the exact downloaded or package-supplied bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_registry(path):
    """Find the CSV record header, not a physical line inside quoted prose."""
    with path.open(newline="", encoding="utf-8-sig") as stream:
        records = csv.reader(stream)
        header = next((row for row in records if row and row[0] == "Name"), None)
        if header is None or "Path" not in header:
            raise ValueError("L-BAM registry header not found")
        rows = [row for row in records if row and any(row)]
    if any(len(row) != len(header) for row in rows):
        raise ValueError("L-BAM registry width changed")
    return pd.DataFrame(rows, columns=header).replace("", None)


def run(args):
    """Retain successes and missing files separately; fetch no participant data."""
    if args.output.exists():
        raise ValueError("Use a fresh archive")
    args.output.mkdir(parents=True)
    records = []
    for rel in MEH_FILES:
        url = f"https://raw.githubusercontent.com/ryanboyd/MEHv2/{MEH_REVISION}/{rel}"
        response = requests.get(url, timeout=30)
        record = {
            "repo": "ryanboyd/MEHv2",
            "revision": MEH_REVISION,
            "path": rel,
            "url": url,
            "status_code": response.status_code,
            "executed": False,
        }
        if response.ok:
            path = args.output / "mehv2" / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
            record["sha256"] = sha(path)
        records.append(record)
    registry = args.rlib / "text/extdata/The_L-BAM_Library.csv"
    d = read_registry(registry)
    valid = d[d.Name.notna()]
    valid.to_csv(args.output / "lbam_nonempty_registry.csv", index=False)
    for rel in ("DESCRIPTION", "extdata/The_L-BAM_Library.csv"):
        path = args.rlib / "text" / rel
        dest = args.output / "text_1_9" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        records.append(
            {
                "installed_package": "text",
                "version": "1.9",
                "path": rel,
                "sha256": sha(dest),
                "executed": False,
            }
        )
    data = {
        "resources": records,
        "public_registry_rows": len(d),
        "named_registry_rows": len(valid),
        "registry_origin": "installed CRAN text1.9 bundled snapshot, not live Google Sheet",
        "license_note": "MEHv2 root MIT does not establish licensing for all bundled resources; registry open is not a precise license",
    }
    (args.output / "manifest.json").write_text(json.dumps(data, indent=2) + "\n")
    print(
        json.dumps(
            {
                "archived": sum("sha256" in r for r in records),
                "failed": [r["path"] for r in records if "sha256" not in r],
                "named_models": len(valid),
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rlib", type=Path, required=True)
    run(p.parse_args())
