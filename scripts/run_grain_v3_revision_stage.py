#!/usr/bin/env python
"""Revision-aware stage CLI (packet 12 Part C3; F00-04 / R35). Exit 0 = pass, 2 = refused, 3 = checked and failed.

  integrity  --root R | --card C | --all        re-hash declared manifests (declared append-only amendments verified)
  readiness  --root R | --card C [--run-tests]  dependency/evidence readiness; rejected legacy never grants readiness
  reproduce  --adapter A --output DIR           observed reproduction of stored derived outputs (fresh DIR)
  execute    --adapter A --output DIR --authorization DOC --cap-seconds N   authorized bounded generating replay
  coverage   --output DIR                        full mapping/coverage table for every registered and unregistered root
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from grain_research.gptpro_foundation_v3 import revision_stage as rs  # noqa: E402

PLAN = Path("/Volumes/mobile3/projects/project persona/configs/grain_gptpro_atomic_v3.json")


def plan() -> dict:
    return {t["id"]: t for t in json.loads(PLAN.read_text())["tasks"]}


def unregistered(reg: dict) -> list[str]:
    out = []
    for pkg in sorted(p for p in rs.RES.iterdir() if p.is_dir() and not p.name.startswith("._")):
        for d in sorted(p for p in pkg.iterdir() if p.is_dir() and not p.name.startswith("._")):
            key = f"{pkg.name}/{d.name}"
            if d.name in ("r1", "r2", "r3"):
                for c in sorted(x for x in d.iterdir() if x.is_dir() and not x.name.startswith("._")):
                    if f"{key}/{c.name}" not in reg["revisions"]:
                        out.append(f"{key}/{c.name}")
            elif key not in reg["revisions"]:
                out.append(key)
    return out


def coverage(reg: dict, out: Path) -> dict:
    if out.exists():
        raise rs.StageRefusal("coverage output must be fresh")
    out.mkdir(parents=True)
    icache, cache = {}, {}
    adapters_by_root = {}
    for aid, ad in reg["adapters"].items():
        adapters_by_root.setdefault(ad["root"], []).append(f"{aid}:{ad['mode']}")
    rows = []
    for key, e in sorted(reg["revisions"].items()):
        r = rs.root_readiness(reg, key, icache=icache)
        cache[key] = r
        integ = icache.get(key) or rs.integrity(key, e)
        if key in adapters_by_root:
            gap = ""
        elif e["kind"] == "REJECTED_LEGACY":
            gap = "rejected legacy history: no replay adapter by design"
        elif integ["integrity"] == "NO_MANIFEST":
            gap = "no output_manifest/COMMON record/analysis hashes: integrity unverifiable; needs a manifest before any adapter"
        else:
            gap = "integrity verifiable; no bounded replay adapter implemented in packet 12"
        rows.append({"root": key, "kind": e["kind"], "curation": e["curation"], "status": e.get("status"), "cards": ";".join(e.get("cards", [])),
                     "requirements": ";".join(e.get("requirements", [])), "integrity": integ["integrity"], "integrity_basis": integ.get("integrity_basis", ""), "provenance_basis": r.get("provenance_basis", ""), "files_checked": integ["checked"],
                     "declared_amendments": ";".join(integ.get("declared_amendments_verified", [])), "problems": " | ".join(integ["problems"][:3]),
                     "evidence_doc": (e.get("evidence") or {}).get("decision") or "", "readiness": r["state"],
                     "readiness_reasons": " | ".join(r.get("reasons", [])[:3]), "claims": json.dumps(e.get("claims", {}), ensure_ascii=False) if e.get("claims") else "", "adapters": ";".join(adapters_by_root.get(key, [])), "adapter_gap": gap})
    for key in unregistered(reg):
        rows.append({"root": key, "kind": "UNREGISTERED", "curation": "", "status": "", "cards": "", "requirements": "", "integrity": "NOT_CHECKED", "integrity_basis": "", "provenance_basis": "",
                     "files_checked": 0, "declared_amendments": "", "problems": "", "evidence_doc": "", "readiness": "NOT_READY_UNKNOWN_MAPPING",
                     "readiness_reasons": "root not in registry", "claims": "", "adapters": "", "adapter_gap": "unregistered: map it (or declare it legacy) before use"})
    with (out / "root_coverage.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    p = plan()
    crow = []
    for cid in sorted(p):
        c = rs.card_readiness(reg, p, cid, cache=cache, icache=icache)
        crow.append({"card": cid, "readiness": c["state"], "own_evidence_state": c["own"]["state"], "current_evidence": ";".join(reg["cards"][cid]["current_evidence"]),
                     "own_reasons": " | ".join(c["own"].get("reasons", [])[:3]), "blocked_dependencies": ";".join(c["blocked_dependencies"]),
                     "legacy_root": c["legacy_root"]})
    with (out / "card_readiness.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(crow[0]))
        w.writeheader()
        w.writerows(crow)

    def count(rows_, k):
        d = {}
        for r in rows_:
            d[r[k]] = d.get(r[k], 0) + 1
        return d
    summary = {"roots": len(rows), "by_kind": count(rows, "kind"), "integrity": count(rows, "integrity"), "integrity_basis_of_ready": count([r for r in rows if r["readiness"] == rs.READY], "integrity_basis"), "readiness": count(rows, "readiness"),
               "roots_with_adapter": sum(bool(r["adapters"]) for r in rows), "adapters": len(reg["adapters"]),
               "cards": len(crow), "card_readiness": count(crow, "readiness"),
               "registry_sha256": rs.sha(rs.REGISTRY), "registry_built_at": reg.get("built_at"),
               "note": "readiness is bookkeeping for dependent use; it is not scientific acceptance, and adapter coverage is partial by design"}
    (out / "coverage_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["integrity", "readiness", "reproduce", "execute", "coverage"])
    ap.add_argument("--root")
    ap.add_argument("--card")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--adapter")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--authorization", type=Path)
    ap.add_argument("--cap-seconds", type=float)
    ap.add_argument("--run-tests", action="store_true")
    ap.add_argument("--registry", type=Path, default=rs.REGISTRY)
    a = ap.parse_args()
    try:
        reg = rs.load_registry(a.registry)
        if a.mode == "integrity":
            keys = sorted(reg["revisions"]) if a.all else ([a.root] if a.root else reg["cards"].get(a.card, {}).get("current_evidence", []) if a.card else [])
            if not keys:
                raise rs.StageRefusal("nothing to check: unknown card or no target")
            res = [rs.integrity(k, reg["revisions"].get(k)) for k in keys]
            print(json.dumps(res if len(res) > 1 else res[0], indent=1))
            return 0 if all(r["integrity"] in ("OK", "OK_DECLARED_AMENDMENTS") for r in res) else 3
        if a.mode == "readiness":
            if a.card:
                res = rs.card_readiness(reg, plan(), a.card)
            elif a.root:
                res = rs.root_readiness(reg, a.root, run_tests=a.run_tests)
            else:
                raise rs.StageRefusal("readiness needs --root or --card")
            print(json.dumps(res, indent=1))
            return 0 if res["state"] == rs.READY else 3
        if a.mode in ("reproduce", "execute"):
            if not a.adapter or not a.output:
                raise rs.StageRefusal("adapter and fresh output directory required")
            ad = reg["adapters"].get(a.adapter)
            if ad is None:
                raise rs.StageRefusal(f"unknown adapter {a.adapter}")
            if ad["mode"] != a.mode:
                raise rs.StageRefusal(f"adapter {a.adapter} is registered for mode {ad['mode']}, not {a.mode}")
            res = rs.run_adapter(reg, a.adapter, a.output, a.authorization, a.cap_seconds)
            print(json.dumps({k: res[k] for k in ("adapter", "mode", "status", "child_exit", "seconds", "source_artifacts_unchanged")}, indent=1))
            return 0 if res["status"] == "PASS" else 3
        if a.mode == "coverage":
            if not a.output:
                raise rs.StageRefusal("coverage needs a fresh --output")
            print(json.dumps(coverage(reg, a.output), indent=1))
            return 0
    except rs.StageRefusal as e:
        print(json.dumps({"decision": "REFUSED", "reason": str(e)}))
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
