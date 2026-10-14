#!/usr/bin/env python
"""R08 legacy collision-world provenance (packet 12 Part C6). CPU only, 0 neural calls.

1. Re-hash the legacy GP01 files and compare them to contemporaneous hashes: the 2026-09-15 seminar snapshot, the
   F00-01 input manifest, and the GP01 manifest's recorded code hashes. A new hash is a current snapshot, not a
   proof of pre-registration or of immutability since creation.
2. Independently verify the finite collision and equivalence equations in exact rational arithmetic, and check the
   stored records against the exact laws.
3. Regenerate the GP01 records and certificate with the current code (hash-identical to the recorded inputs only if
   the comparison says so), and compare hashes.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter
from fractions import Fraction as Fr
from itertools import product
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
GP01 = REPO / "results/grain_measurement_discovery_v1/gp01/r1"
SNAPSHOT = Path("/Volumes/mobile3/projects/project persona/reports/gptpro_seminar_20260915/full_coverage_audit/frozen_artifacts_snapshot.json")
F00_01 = REPO / "results/grain_gptpro_foundation_v3/F00/r1/F00-01/input_manifest.json"
OUT = REPO / "results/grain_gptpro_foundation_v3/F02/collision_provenance_r1"
FILES = ("certificate.json", "decision.md", "manifest.json", "metrics.csv", "records.jsonl", "run_log.jsonl", "summary.json", "tests.txt", "truth.json")


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def hashes() -> dict:
    snap = json.loads(SNAPSHOT.read_text())
    f00 = {x["path"]: x.get("sha256") for x in json.loads(F00_01.read_text())["inputs"]}
    rows = {}
    for name in FILES:
        cur = sha_bytes((GP01 / name).read_bytes())
        s = snap.get(f"results/grain_measurement_discovery_v1/gp01/r1/{name}")
        f = f00.get(str(GP01 / name))
        rows[name] = {"current_sha256": cur, "snapshot_20260915_sha256": s, "matches_snapshot": s == cur,
                      "f00_01_manifest_sha256": f, "matches_f00_01": (f == cur) if f else None}
    man = json.loads((GP01 / "manifest.json").read_text())
    code = {Path(i["path"]).name: {"recorded_sha256": i["sha256"], "current_sha256": sha_bytes(Path(i["path"]).read_bytes()),
                                   "matches": i["sha256"] == sha_bytes(Path(i["path"]).read_bytes())} for i in man["inputs"]}
    ok = all(r["matches_snapshot"] for r in rows.values()) and all(r["matches_f00_01"] in (True, None) for r in rows.values()) and all(c["matches"] for c in code.values())
    return {"files": rows, "code": code, "gp01_created_at": man["created_at"],
            "provenance_label": ("current snapshot, compared with a contemporaneous 2026-09-15 seminar snapshot that is untracked in git "
                                 "and with the F00-01 input manifest; neither is a pre-registration or immutability proof"),
            "status": "MATCH" if ok else "DRIFT"}


def xor_p(a, b, c):
    return Fr(3, 4) if c == (a ^ b) else Fr(1, 4)


def iid_p(a, b, c):
    return Fr(1, 2)


def future_law(hist, h, p):
    law = {(): Fr(1)}
    for _ in range(h):
        nxt = {}
        for suf, pr in law.items():
            seq = hist + suf
            for v in (0, 1):
                nxt[suf + (v,)] = nxt.get(suf + (v,), Fr(0)) + pr * p(seq[-2], seq[-1], v)
        law = nxt
    return law


def rank_fraction(rows):
    m = [list(r) for r in rows]
    rank, col = 0, 0
    ncol = len(m[0])
    while rank < len(m) and col < ncol:
        piv = next((i for i in range(rank, len(m)) if m[i][col] != 0), None)
        if piv is None:
            col += 1
            continue
        m[rank], m[piv] = m[piv], m[rank]
        for i in range(len(m)):
            if i != rank and m[i][col] != 0:
                f = m[i][col] / m[rank][col]
                m[i] = [x - f * y for x, y in zip(m[i], m[rank])]
        rank += 1
        col += 1
    return rank


def equations() -> dict:
    cert = json.loads((GP01 / "certificate.json").read_text())
    out, ok = {}, True
    # C1
    c1 = {t: {"marginal": (Fr(1, 2) + Fr(t, 4) * -1 + Fr(1, 2) + Fr(t, 4)) / 2, "half_diff": ((Fr(1, 2) + Fr(t, 4)) - (Fr(1, 2) - Fr(t, 4))) / 2} for t in (-1, 1)}
    c1_ok = all(float(c1[t]["marginal"]) == cert["c1"]["marginal_p_y1_by_type"][str(t)] and float(c1[t]["half_diff"]) == cert["c1"]["conditional_half_difference_by_type"][str(t)] for t in (-1, 1))
    out["c1"] = {"marginal": {str(t): str(c1[t]["marginal"]) for t in c1}, "half_diff": {str(t): str(c1[t]["half_diff"]) for t in c1}, "matches_certificate": c1_ok}
    # C2 pair stationarity and laws
    pair = {(a, b): Fr(1, 4) for a, b in product((0, 1), repeat=2)}
    nxt = {(b, c): sum(pair[(a, b)] * xor_p(a, b, c) for a in (0, 1)) for b, c in product((0, 1), repeat=2)}
    trip = {w: {f"{a}{b}{c}": Fr(1, 4) * p(a, b, c) for a, b, c in product((0, 1), repeat=3)} for w, p in (("xor", xor_p), ("iid", iid_p))}
    single = {w: sum(v for k, v in t.items() if k[2] == "1") for w, t in trip.items()}
    pairs = {w: {bc: sum(v for k, v in t.items() if k[1:] == bc) for bc in ("00", "01", "10", "11")} for w, t in trip.items()}
    tv_trip = sum(abs(trip["xor"][k] - trip["iid"][k]) for k in trip["xor"]) / 2

    def lag3(p):
        return sum(Fr(1, 4) * p(a, b, c) * p(b, c, d) for a, b, c, d in product((0, 1), repeat=4) if a == d)
    l3 = {"xor": lag3(xor_p), "iid": lag3(iid_p)}
    c2_ok = (nxt == pair and single["xor"] == single["iid"] == Fr(1, 2) and pairs["xor"] == pairs["iid"]
             and all(float(trip["xor"][k]) == cert["c2"]["triple_xor"][k] and float(trip["iid"][k]) == cert["c2"]["triple_iid"][k] for k in trip["xor"])
             and float(l3["xor"]) == cert["c2"]["lag3_equal_xor"] and float(l3["iid"]) == cert["c2"]["lag3_equal_iid"])
    # finite-future collision: histories 00 and 11
    tv = []
    for h in range(1, 7):
        f0, f1 = future_law((0, 0), h, xor_p), future_law((1, 1), h, xor_p)
        tv.append(str(sum(abs(f0[k] - f1[k]) for k in f0) / 2))
    # Hankel ranks: histories = last pairs; futures of length <= h
    hank = {}
    for h in (1, 2, 3):
        futs = [f for L in range(1, h + 1) for f in product((0, 1), repeat=L)]
        for w, p in (("xor", xor_p), ("iid", iid_p)):
            rows = [[future_law(hist, len(f), p)[f] for f in futs] for hist in product((0, 1), repeat=2)]
            hank[f"{w}_h{h}"] = rank_fraction(rows)
    stored_tv = [line.split(",")[1] for line in (REPO / "results/grain_gptpro_foundation_v3/F02/recovery_full_r2/future_equivalence.csv").read_text().splitlines()[1:]]
    tv_ok = [float(Fr(x)) for x in tv] == [float(x) for x in stored_tv]
    out["c2"] = {"pair_law_stationary_uniform": nxt == pair, "single_equal": str(single["xor"]), "adjacent_pairs_equal": pairs["xor"] == pairs["iid"],
                 "triple_tv_xor_vs_iid": str(tv_trip), "lag3_equal": {w: str(v) for w, v in l3.items()}, "matches_certificate": c2_ok,
                 "future_tv_00_vs_11_by_horizon": tv, "matches_F02_recovery_full_r2_future_equivalence": tv_ok, "hankel_rank": hank,
                 "reading": "one-step predictions of histories 00 and 11 coincide while their finite futures differ (TV 1/2 from horizon 2); "
                            "the order-2 pair is a sufficient predictive state (xor Hankel rank 4 with 2-step futures, 2 with 1-step only; iid rank 1)"}
    # C3, distance, local rank
    c3 = [{"pi": str(Fr(pi).limit_denominator()), "q": str(Fr(q).limit_denominator()), "p_selected": str(Fr(pi).limit_denominator() * Fr(q).limit_denominator())} for pi, q in ((0.8, 0.5), (0.4, 1.0))]
    c3_ok = all(Fr(r["p_selected"]) == Fr(2, 5) for r in c3) and all(float(Fr(r["p_selected"])) == c["p_selected"] and c["q_bounds_given_f"] == [0.4, 1.0] for r, c in zip(c3, cert["c3"]))
    pts = [Fr(0), Fr(3, 4), Fr(3, 2)]
    dist_ok = abs(pts[0] - pts[1]) <= 1 and abs(pts[1] - pts[2]) <= 1 and abs(pts[0] - pts[2]) > 1 and cert["distance"]["transitive_at_epsilon"] is False
    grid = [i / 1000 for i in range(-2000, 2001)]
    cube = [x ** 3 for x in grid]
    rank_ok = all(b > a for a, b in zip(cube, cube[1:])) and 3 * 0.0 ** 2 == cert["local_rank"]["jacobian"] and cert["local_rank"]["global_injective"] is True
    out["c3"] = {"pairs": c3, "q_lower_bound_equals_selected_rate": True, "matches_certificate": c3_ok}
    out["distance"] = {"ab": "3/4", "bc": "3/4", "ac": "3/2", "non_transitive": dist_ok}
    out["local_rank"] = {"jacobian_at_0": 0.0, "strictly_increasing_on_grid": rank_ok}
    # empirical check of the stored records against the exact xor law (records are c2 world)
    seqs = {}
    for line in (GP01 / "records.jsonl").read_text().splitlines():
        r = json.loads(line)
        seqs.setdefault(r["author_id"], []).append((r["t"], r["observation"]))
    trip_c, lag_c, n_trip, n_lag, ones, n_obs = Counter(), 0, 0, 0, 0, 0
    for s in seqs.values():
        ys = [y for _t, y in sorted(s)]
        ones += sum(ys); n_obs += len(ys)
        for i in range(2, len(ys)):
            trip_c[f"{ys[i-2]}{ys[i-1]}{ys[i]}"] += 1; n_trip += 1
        for i in range(3, len(ys)):
            lag_c += ys[i] == ys[i - 3]; n_lag += 1
    emp = {k: trip_c[k] / n_trip for k in trip["xor"]}
    z = {k: (emp[k] - float(trip["xor"][k])) / math.sqrt(float(trip["xor"][k]) * (1 - float(trip["xor"][k])) / n_trip) for k in emp}
    zl = (lag_c / n_lag - 0.625) / math.sqrt(0.625 * 0.375 / n_lag)
    out["records_vs_exact"] = {"authors": len(seqs), "triples": n_trip, "empirical_triple": {k: round(v, 5) for k, v in emp.items()},
                               "max_abs_z_triple_naive": round(max(abs(v) for v in z.values()), 3), "lag3_empirical": round(lag_c / n_lag, 5),
                               "z_lag3_naive": round(zl, 3), "single_mean": round(ones / n_obs, 5),
                               "note": "z uses a naive iid SE; triples within an author overlap, so this is a descriptive consistency check"}
    ok = c1_ok and c2_ok and tv_ok and c3_ok and dist_ok and rank_ok and tv == ["0"] + ["1/2"] * 5 and hank["xor_h2"] == 4 and hank["xor_h1"] == 2 and hank["iid_h3"] == 1
    out["status"] = "PASS" if ok else "FAIL"
    return out


def regeneration() -> dict:
    from grain_research.measurement_discovery.collision_worlds import (
        c2_lag3_equal_probability, c2_triple_law, c3_exposure_bounds, distance_chain, generate_records)
    counts = {"source": 300, "validation": 100, "application": 200}
    records, truth = generate_records("c2", {}, 91501, n_authors=counts, events=48)
    rec_bytes = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode()
    truth_bytes = (json.dumps(truth, ensure_ascii=False, indent=2) + "\n").encode()
    cert = json.loads((GP01 / "certificate.json").read_text())
    regen_c2 = {"triple_xor": c2_triple_law("xor"), "triple_iid": c2_triple_law("iid"), "lag3_equal_xor": c2_lag3_equal_probability("xor"),
                "lag3_equal_iid": c2_lag3_equal_probability("iid")}
    same_cert = all(cert["c2"][k] == v for k, v in regen_c2.items()) and cert["distance"] == distance_chain() and \
        [dict(c, q_bounds_given_f=list(c["q_bounds_given_f"])) for c in c3_exposure_bounds()] == cert["c3"]
    r = {"records_sha256": sha_bytes(rec_bytes), "records_equal_stored": sha_bytes(rec_bytes) == sha_bytes((GP01 / "records.jsonl").read_bytes()),
         "truth_sha256": sha_bytes(truth_bytes), "truth_equal_stored": sha_bytes(truth_bytes) == sha_bytes((GP01 / "truth.json").read_bytes()),
         "certificate_values_equal": same_cert}
    r["status"] = "REPRODUCED" if all(v for k, v in r.items() if k.endswith(("equal_stored", "values_equal"))) else "NOT_REPRODUCED"
    return r


def verify(out: Path) -> dict:
    res = {"hashes": hashes(), "equations": equations(), "regeneration": regeneration()}
    res["hashes"]["status"] = res["hashes"]["status"]
    (out / "provenance.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    return res


def main() -> int:
    if OUT.exists():
        raise FileExistsError("collision_provenance_r1 exists; preserve earlier runs")
    OUT.mkdir(parents=True)
    res = verify(OUT)
    print(json.dumps({k: v["status"] for k, v in res.items()}))
    return 0 if res["equations"]["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
