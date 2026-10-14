"""Bounded replay adapters, run only as a child of revision_stage.run_adapter after a passed preflight.

Every file an adapter reads comes from `adapter_contract.json` (role -> pinned path) in its fresh output directory.
Where a reused library function reads a path internally, the adapter asserts that path equals the declared contract
path before calling it, so no second input path exists outside the contract. Sockets are disabled for the whole
child process.
"""
from __future__ import annotations

import filecmp
import hashlib
import importlib
import json
import socket
import subprocess
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


def _deny(*_a, **_k):
    raise RuntimeError("network access is disabled inside revision adapters")


socket.socket.connect = _deny  # type: ignore[assignment]
socket.create_connection = _deny  # type: ignore[assignment]


class ContractError(RuntimeError):
    pass


def _same(declared: str, used: Path) -> None:
    if Path(declared).resolve() != Path(used).resolve():
        raise ContractError(f"library reads {used}, contract declares {declared}")


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _redirected_analysis(module_name: str, inputs: dict, work: Path, links: dict, compare: dict) -> dict:
    """links: name in work dir -> role; compare: produced file -> role of the stored reference."""
    work.mkdir()
    for name, role in links.items():
        (work / name).symlink_to(inputs[role])
    mod = importlib.import_module(module_name)
    mod.OUT = work
    rc = mod.main()
    same = {f: filecmp.cmp(work / f, inputs[role], shallow=False) for f, role in compare.items()}
    return {"main_return": rc, "byte_identical": same, "status": "PASS" if (rc in (0, None) and all(same.values())) else "FAIL"}


def r24_r1_analysis(out: Path, inputs: dict) -> dict:
    rr = importlib.import_module("run_grain_r24_personal_law")
    _same(inputs["bootstrap_dir"], rr.BOOT)
    files = ["summary.csv", "paired_contrasts.csv", "coverage_intervals.csv", "source_validation_summary.csv", "homogeneous_reduction.json",
             "tau2_truth_vs_source.csv", "analysis.json"]
    return _redirected_analysis("analyze_grain_r24_personal_law", inputs, out / "work",
                                {"author_metrics.csv": "author_metrics", "source_validation_calibration.csv": "source_validation", "source_side.json": "source_side"},
                                {f: f"stored_{f}" for f in files})


def r10_episode_analysis(out: Path, inputs: dict) -> dict:
    mod = importlib.import_module("analyze_grain_r10_common_shock_episodes")
    _same(inputs["joint_full_r1_metrics"], mod.ROOT / "results/grain_gptpro_foundation_v3/F05/joint_full_r1/joint_state_metrics.csv")
    files = ["levels.csv", "paired_contrasts.csv", "source_component.csv", "source_component.json", "archived_three_episode_reference.csv", "analysis.json"]
    return _redirected_analysis("analyze_grain_r10_common_shock_episodes", inputs, out / "work", {"episodes": "episodes"},
                                {f: f"stored_{f}" for f in files})


def r24_r2_cold_verify(out: Path, inputs: dict) -> dict:
    import pandas as pd
    mod = importlib.import_module("analyze_grain_r24_personal_law_r2")
    cold = mod.cold_verify(pd.read_parquet(inputs["event_trace"]), pd.read_parquet(inputs["histories"]), pd.read_parquet(inputs["truth"]),
                           pd.read_csv(inputs["author_metrics_r2"], float_precision="round_trip"))
    (out / "cold_verification.json").write_text(json.dumps(cold, indent=1) + "\n")
    stored = json.loads(Path(inputs["stored_cold_verification"]).read_text())
    return {"cold": cold["status"], "identical_to_stored": cold == stored, "rows": cold["rows_compared"],
            "status": "PASS" if cold["status"] == "PASS" and cold == stored else "FAIL"}


def r10_one_episode_replay(out: Path, inputs: dict) -> dict:
    import numpy as np
    import pandas as pd
    rr = importlib.import_module("run_grain_r10_common_shock_episodes")
    res = {}
    for world in ("core", "shared_shock"):
        info, auth = rr.run_episode(world, 20261001, None)
        saved = pd.read_csv(inputs[f"stored_{world}_author_metrics"], float_precision="round_trip")
        keys = ["author_id", "budget", "method", "arm"]
        a, s = auth.sort_values(keys).reset_index(drop=True), saved.sort_values(keys).reset_index(drop=True)
        cols = [c for c in s.columns if c.endswith(("_err2", "_cover", "_width"))]
        err = float(np.nanmax(np.abs(a[cols].to_numpy(float) - s[cols].to_numpy(float))))
        old = json.loads(Path(inputs[f"stored_{world}_episode"]).read_text())
        res[world] = {"keys_equal": bool((a[keys].values == s[keys].values).all()), "max_abs_err": err,
                      "source_rows_sha_equal": info["source_rows_sha256"] == old["source_rows_sha256"],
                      "common_variance_equal": info["moment_common_variance"] == old["moment_common_variance"]}
        auth.to_csv(out / f"{world}_20261001_author_metrics.csv", index=False)
    ok = all(v["keys_equal"] and v["max_abs_err"] <= 1e-12 and v["source_rows_sha_equal"] and v["common_variance_equal"] for v in res.values())
    return {"episodes": res, "status": "PASS" if ok else "FAIL"}


def r24_r1_canonical_replay(out: Path, inputs: dict) -> dict:
    r1 = importlib.import_module("run_grain_r24_personal_law")
    _same(inputs["bootstrap_dir"], r1.BOOT)
    _same(inputs["recovery3_author_metrics"], r1.R / "F13/recovery3_r1/author_metrics.csv")
    got = r1.replay_check()
    (out / "replay_check.json").write_text(json.dumps(got, indent=1) + "\n")
    stored = json.loads(Path(inputs["stored_replay_check"]).read_text())
    return {"replay": got, "identical_to_stored": got == stored, "status": "PASS" if got["status"] == "PASS" and got == stored else "FAIL"}


def f16_sealed_cold_subset(out: Path, inputs: dict) -> dict:
    index = json.loads(Path(inputs["bundle_index"]).read_text())
    names = sorted(k[len("bundle__"):] for k in inputs if k.startswith("bundle__"))
    results = {}
    for name in names:
        e = next(x for x in index if x["name"] == name)
        bundle = Path(inputs[f"bundle__{name}"])
        if _sha(bundle) != e["sha256"]:
            results[name] = {"error": "bundle hash mismatch against bundle_index"}
            continue
        target = out / f"{name}.scores.jsonl"
        cmd = [sys.executable, "-m", "grain_research.gptpro_foundation_v3.f16_serialize", "score", "--bundle", str(bundle),
               "--sha256", e["sha256"], "--input", inputs[f"requests__{name}"], "--output", str(target)]
        done = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        stored_path = Path(inputs[f"stored_scores__{name}"])
        stored = [json.loads(x) for x in stored_path.read_text().splitlines()]
        got = [json.loads(x) for x in target.read_text().splitlines()] if target.exists() else None
        results[name] = {"exit": done.returncode, "rows": len(stored), "parsed_equal": got == stored,
                         "byte_identical": target.exists() and filecmp.cmp(target, stored_path, shallow=False)}
    ok = bool(results) and all(v.get("exit") == 0 and v.get("parsed_equal") for v in results.values())
    return {"bundles": results, "status": "PASS" if ok else "FAIL"}


def r08_gp01_collision(out: Path, inputs: dict) -> dict:
    mod = importlib.import_module("verify_grain_r08_collision_provenance")
    _same(inputs["gp01_dir"], mod.GP01)
    _same(inputs["snapshot_20260915"], mod.SNAPSHOT)
    _same(inputs["f00_01_legacy_input_manifest"], mod.F00_01)
    _same(inputs["future_equivalence"], mod.REPO / "results/grain_gptpro_foundation_v3/F02/recovery_full_r2/future_equivalence.csv")
    got = mod.verify(out)
    stored = json.loads(Path(inputs["stored_provenance"]).read_text())
    same = {k: got[k] == stored[k] for k in ("hashes", "equations", "regeneration")}
    return {"verdicts": {k: got[k]["status"] for k in ("hashes", "equations", "regeneration")}, "identical_to_stored": same,
            "status": "PASS" if all(same.values()) and got["equations"]["status"] == "PASS" else "FAIL"}


ADAPTERS = {f.__name__: f for f in (r24_r1_analysis, r10_episode_analysis, r24_r2_cold_verify, r10_one_episode_replay,
                                     r24_r1_canonical_replay, f16_sealed_cold_subset, r08_gp01_collision)}


def main() -> int:
    name, out = sys.argv[1], Path(sys.argv[2])
    try:
        contract = json.loads((out / "adapter_contract.json").read_text())
        if contract["adapter"] != name:
            raise ContractError("contract/adapter mismatch")
        res = ADAPTERS[name](out, contract["inputs"])
    except Exception:  # noqa: BLE001 - the parent treats any failure as FAIL
        res = {"status": "FAIL", "error": traceback.format_exc()[-3000:]}
    res["network_guard"] = True
    (out / "adapter_result.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    return 0 if res["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
