#!/usr/bin/env python3
"""Create a V7.3 provenance manifest before a result-generating run."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grain_core.v7_governance import write_run_manifest  # noqa: E402


def main() -> int:
    """Build a manifest without printing a pseudonymization secret."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimand-id", required=True)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--code", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--external-labels-read", action="store_true")
    parser.add_argument("--raw-identifiers-persisted", action="store_true")
    parser.add_argument("--pseudonym-salt-env", default="GRAIN_PSEUDONYM_SALT")
    args = parser.parse_args()
    secret = os.environ.get(args.pseudonym_salt_env, "")
    if not secret and args.pseudonym_salt_env == "GRAIN_PSEUDONYM_SALT":
        secret = os.environ.get("SUICA_V7_PSEUDONYM_SALT", "")
    fingerprint = hashlib.sha256(secret.encode("utf-8")).hexdigest() if secret else None
    payload = write_run_manifest(
        args.output,
        repository_root=ROOT,
        input_paths=args.input,
        config_path=args.config,
        code_paths=args.code,
        estimand_id=args.estimand_id,
        external_labels_read=args.external_labels_read,
        raw_identifiers_persisted=args.raw_identifiers_persisted,
        pseudonym_salt_fingerprint=fingerprint,
    )
    print(json.dumps({"output": str(args.output), "estimand_id": payload["estimand_id"], "git": payload["repository"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
