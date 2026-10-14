#!/usr/bin/env python3
"""Hash-checked stable-basis companion; retain the complete frozen HM36 primary."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_target_active_support_stable as STABLE
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts import run_grain_math_target_active_support as RUN

EXPECTED = {
    "grain_research/construction/math_target_active_support.py": "f8d21572746ef6cb8c3efbf0449b2935e4a686a2cbcd672c04cfe1bdb953e321",
    "scripts/run_grain_math_target_active_support.py": "e972cd5095fabbb01e1c7835ebe18395b205b7593cf7417c4c56d1eae39dc80c",
    "scripts/verify_grain_math_target_active_support.py": "9f855435873078d62df4f8071c5622e60143dc01cd407272515d4cc7c20f6cc8",
}
PRIMARY = RUN.OUT
RESERVED_PRIOR_SECONDS = 120.0
OUT = ROOT / "results/grain_math_response_v1/target_active_support_r2"


def install():
    """Change only the numerical basis constructor in this new Python process."""
    for name, expected in EXPECTED.items():
        if digest(ROOT / name) != expected:
            raise ValueError("frozen primary code changed " + name)
    old_budget, old_fit = RUN.Budget, RUN.fit_source

    class CompanionBudget(old_budget):
        """Charge the failed primary conservatively against the unchanged900s cap."""

        def checkpoint(self, status):
            super().checkpoint(status)
            if (
                status["analysis_seconds"] + RESERVED_PRIOR_SECONDS
                > self.spec["maximum_analysis_seconds"]
            ):
                raise TimeoutError("combined primary/companion900-second bound reached")

    def fit(out):
        """Require every learned rank to match the frozen primary before application."""
        old_fit(out)
        for query in read_json(RUN.SPEC)["queries"]:
            for space in ("D", "V", "DV"):
                name = f"01/models/{query}_{space}.json"
                if read_json(out / name)["rank"] != read_json(PRIMARY / name)["rank"]:
                    raise ValueError("companion changed a source-selected rank")

    RUN.Budget, RUN.fit_source = CompanionBudget, fit
    RUN.T.source_basis = STABLE.source_basis
    RUN.OUT = OUT
    RUN.CODE += (
        "grain_research/construction/math_target_active_support_stable.py",
        "scripts/run_grain_math_target_active_support_r2.py",
        "scripts/verify_grain_math_target_active_support_r2.py",
        "tests/test_grain_math_target_active_support_stable.py",
        "docs/GRAIN_TARGET_ACTIVE_SUPPORT_NUMERICAL_REPAIR_V1.md",
        "results/grain_math_response_v1/target_active_support_verification_r1/failure.json",
        *(
            str(p.relative_to(ROOT))
            for p in sorted((PRIMARY / "01/models").glob("*.json"))
            if not p.name.startswith("._")
            and not p.stem.endswith("_raw")
            and "random" not in p.stem
        ),
    )


if __name__ == "__main__":
    install()
    RUN.main()
