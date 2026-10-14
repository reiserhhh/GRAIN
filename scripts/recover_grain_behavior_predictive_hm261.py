#!/usr/bin/env python3
"""Read only sealed HM261 addresses; keep primary science and arrays unchanged."""

import argparse
import sys
from pathlib import Path, PosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_behavior_predictive_hm261 as F
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_source_meaning_bridge_hm233 as S

BASE = F.DEFAULT
DEFAULT = BASE / "analysis_recovery_r1"
SCIENCE = (
    "docs/GRAIN_BEHAVIOR_PREDICTIVE_READER_RECOVERY_HM261.md",
    "scripts/recover_grain_behavior_predictive_hm261.py",
    "tests/test_grain_behavior_predictive_recovery.py",
)


class RegisteredCapturePath(PosixPath):
    """Replace only capture discovery with the originally registered address set."""

    def glob(self, pattern, **kwargs):
        """Never pass AppleDouble or unregistered files to the scientific reader."""
        if self.name == "captures" and pattern == "*.npz":
            return (self / f"{i:04d}.npz" for i in range(672))
        return super().glob(pattern, **kwargs)


def prepare(out, base):
    """Freeze a separate reader after authenticating all completed observations."""
    if out.exists():
        raise FileExistsError("Do not replace an old recovery")
    manifest = R.check(base, ("collected_seal.json",))
    runtime = R.read(base / "runtime.json")
    seal = R.read(base / "collected_seal.json")
    expected = [f"captures/{i:04d}.npz" for i in range(672)]
    if sorted(n for n in seal if n.startswith("captures/")) != expected:
        raise ValueError("Every registered capture must remain in the original seal")
    if (runtime["state"], runtime["done_batches"], runtime["prefixes"]) != (
        "COMPLETE",
        672,
        96,
    ):
        raise ValueError("No partial collection may be treated as complete")
    if any(
        (base / n).exists()
        for n in ("cold_packet.json", "summary.json", "analyzed_seal.json")
    ):
        raise ValueError("Expected primary failure before any analyzed outputs")
    metadata = list((base / "captures").glob("._*.npz"))
    for path in metadata:
        with path.open("rb") as stream:
            header = stream.read(4)
        if (
            header != bytes.fromhex("00051607")
            or not (path.parent / path.name[2:]).is_file()
        ):
            raise ValueError("Unexpected metadata; do not ignore unknown files")
    out.mkdir(parents=True, exist_ok=False)
    for name in ("cases.json", "bank.json", "collection_quality.json"):
        R.write(out / name, R.read(base / name))
    R.archive(
        out / "source_observations.npz",
        **S.selected(
            base / "source_observations.npz", ("fields", "choice", "coordinates")
        ),
    )
    for name in ("captures", "basis"):
        (out / name).symlink_to(Path("..") / name, target_is_directory=True)
    inputs = dict(manifest["inputs"])
    for name in (
        "manifest.json",
        "prepared_seal.json",
        "collected_seal.json",
        "cases.json",
        "bank.json",
        "source_observations.npz",
        "collection_quality.json",
        "runtime.json",
    ):
        path = base / name
        inputs[str(path.relative_to(ROOT))] = R.digest(path)
    inputs.update(
        {str((base / name).relative_to(ROOT)): digest for name, digest in seal.items()}
    )
    R.write(
        out / "manifest.json",
        {
            **manifest,
            "code": {**manifest["code"], **{p: R.digest(ROOT / p) for p in SCIENCE}},
            "inputs": inputs,
        },
    )
    R.write(
        out / "recovery.json",
        {
            "primary_analysis_handle": 9717,
            "primary_analysis_exit": 1,
            "primary_error": "ValueError: This file contains pickled (object) data",
            "cause": "Wildcard reader visited macOS AppleDouble metadata, not numerical arrays",
            "appledouble_metadata_files": len(metadata),
            "registered_capture_files": 672,
            "changed_only": "Offline capture discovery uses registered sealed addresses",
            "primary_analysis_no_outputs": True,
            "primary_science_collection_equations_unchanged": True,
            "new_model_API_training_calls": 0,
            "original_scope_complete": False,
        },
    )
    R.seal(
        out,
        "prepared_seal.json",
        ("cases.json", "bank.json", "source_observations.npz", "recovery.json"),
    )
    R.seal(out, "collected_seal.json", ("collection_quality.json",))
    print("REGISTERED_READER_PREPARED", flush=True)


def main():
    """Delegate unchanged analysis and verification; never start model collection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "analyze", "verify", "check"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--base", type=Path, default=BASE)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.out, args.base)
    elif args.stage == "check":
        R.check(
            args.out,
            ("collected_seal.json", "analyzed_seal.json", "verification_seal.json"),
        )
        print("ALL_SEALS_PASS")
    else:
        getattr(F, args.stage)(RegisteredCapturePath(args.out))


if __name__ == "__main__":
    main()
