#!/usr/bin/env python3
"""Fetch pinned public Stanza English reference models, without participant text."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/grain_vector_mechanisms/resources"
PACKAGES = BASE / "stanza_py111"
MODELS = BASE / "stanza_111_ewt"
PROCESSORS = {
    "tokenize": "ewt",
    "mwt": "ewt",
    "pos": "ewt_nocharlm",
    "lemma": "ewt_nocharlm",
    "depparse": "ewt_nocharlm",
}


def sha(path):
    """Hash model files without deserializing them."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    """Download known Stanford weights and persist a local reproducibility contract."""
    sys.path.insert(0, str(PACKAGES))
    import stanza

    if stanza.__version__ != "1.11.0":
        raise ValueError("Install isolated stanza==1.11.0 and emoji==2.15.0 first")
    if (MODELS / "grain_inventory.json").exists():
        raise ValueError("Inventory already exists; verify it, do not silently refresh")
    stanza.download(
        "en",
        model_dir=str(MODELS),
        processors=PROCESSORS,
        package=None,
        resources_version="1.11.0",
        verbose=True,
    )
    files = {
        str(p.relative_to(MODELS)): {"sha256": sha(p), "bytes": p.stat().st_size}
        for p in sorted(MODELS.rglob("*"))
        if p.is_file() and not p.name.startswith(".")
    }
    result = {
        "stanza": stanza.__version__,
        "resources_version": "1.11.0",
        "processors": PROCESSORS,
        "resource_url": "https://raw.githubusercontent.com/stanfordnlp/stanza-resources/main/resources_1.11.0.json",
        "source": "https://github.com/stanfordnlp/stanza",
        "files": files,
        "total_bytes": sum(f["bytes"] for f in files.values()),
    }
    (MODELS / "grain_inventory.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "files"}), flush=True)


if __name__ == "__main__":
    main()
