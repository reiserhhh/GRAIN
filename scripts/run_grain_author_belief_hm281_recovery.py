#!/usr/bin/env python3
"""One status-only adapter; retain the frozen zero-outcome HM281 primary."""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_author_belief_hm281 as E
from scripts import run_grain_prospective_evidence_hm218 as H

DEFAULT = E.DEFAULT / "runtime_io_recovery_r1"
SCIENCE = (
    "docs/GRAIN_AUTHOR_BELIEF_RUNTIME_RECOVERY_HM281.md",
    "scripts/run_grain_author_belief_hm281_recovery.py",
    "scripts/run_grain_prospective_evidence_hm218.py",
    "tests/test_grain_author_belief_runtime.py",
)


def runtime_writer(original, out):
    """Only status is mutable; all scientific output files remain write-once."""

    def write(path, value):
        if Path(path) == out / "runtime.json":
            H.status(out, value)
        else:
            original(path, value)

    return write


def prepare(out):
    """Copy exact zero-outcome inputs and tighten, rather than renew, the cap."""
    if out.exists():
        raise FileExistsError("Only one mechanical companion")
    manifest = E.R.check(E.DEFAULT, ("prepared_seal.json",))
    failure = E.R.read(E.DEFAULT / "runtime.json")
    if (
        failure["state"] != "FAILED_RUNTIME_STATUS_IO"
        or failure["scientific_outputs"] != 0
        or list(E.DEFAULT.glob("source_*.npz"))
        or (E.DEFAULT / "trajectories.npz").exists()
    ):
        raise ValueError("Only this pre-science I/O failure can be repaired")
    names = tuple(E.R.read(E.DEFAULT / "prepared_seal.json"))
    out.mkdir(parents=True)
    for name in names:
        shutil.copyfile(E.DEFAULT / name, out / name)
    manifest["config"] = {**manifest["config"], "compute_seconds_cap": 179}
    manifest["code"] = {
        **manifest["code"],
        **{p: E.R.digest(E.ROOT / p) for p in SCIENCE},
    }
    manifest["inputs"] = {
        **manifest["inputs"],
        **{
            str((E.DEFAULT / p).relative_to(E.ROOT)): E.R.digest(E.DEFAULT / p)
            for p in names + ("manifest.json", "prepared_seal.json", "runtime.json")
        },
    }
    manifest["repair"] = {
        "primary_failure": failure,
        "original_cap_seconds": 180,
        "remaining_integer_cap_seconds": 179,
        "science_changed": False,
    }
    E.R.write(out / "manifest.json", manifest)
    E.R.seal(out, "prepared_seal.json", names)
    H.status(out, {"state": "PREPARED", "pid": None})
    print("HM281 one status companion prepared; remaining cap179s")


def main():
    """Reuse unchanged stages and a scoped administrative writer adapter."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage", choices=("prepare", "compute", "analyze", "verify", "check")
    )
    a = p.parse_args()
    if a.stage == "prepare":
        prepare(DEFAULT)
    elif a.stage == "check":
        E.R.check(
            DEFAULT,
            (
                "prepared_seal.json",
                "computed_seal.json",
                "analyzed_seal.json",
                "verification_seal.json",
            ),
        )
        print("HM281 primary/companion seals PASS")
    else:
        original = E.R.write
        E.R.write = runtime_writer(original, DEFAULT)
        try:
            getattr(E, a.stage)(DEFAULT)
        finally:
            E.R.write = original


if __name__ == "__main__":
    main()
