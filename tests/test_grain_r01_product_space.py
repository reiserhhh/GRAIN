"""R01 product-space identification: primal reconstruction, witness rebuilt from saved factors, exposure identity."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "results/grain_gptpro_foundation_v3"
OUT = V3 / "P05/product_space_identification_r1"
R = json.loads((OUT / "results.json").read_text())
ROWS = pd.read_csv(OUT / "exposure_rows.csv")
IDS = json.loads((ROOT / "docs/GRAIN_P05_NATIVE_ACTION_PACKET_IDS_V1.json").read_text())
OP = ROOT / "results/grain_vector_mechanisms/conditional_operator_v1/r2"
ORDER = json.loads((OP / "pair_order.json").read_text())
ARR = np.load(OP / "action_arrays.npz"); PREP = np.load(V3 / "P05/native_action_r1/private_prepared.npz")
SL = {"extraversion": slice(0, 7552), "neuroticism": slice(7552, 15232)}


def _load(q):
    si = np.flatnonzero(ARR["source"])
    Js = np.stack([np.load(V3 / "P05/native_action_r1/private_jacobians" / f"{i:03d}_{q}.npy").ravel() for i in si])
    s = Js.shape[1] / np.mean((Js ** 2).sum(1))
    return si, Js, s


def test_checks_and_counts():
    c = R["checks"]
    assert c["reconstruction_pass"] and c["witness_pass"] and R["neural_calls"] == 0 and R["fits"] == 0
    assert len(ROWS) == 72 * 2 * 2 and all(v["K_rank"] == 36 and v["dropped_positive_modes"] == 0 for v in R["spectra"].values())
    loo = pd.read_csv(OUT / "source_loo_rows.csv")
    assert len(loo) == 36 * 4


def test_primal_J24_reconstruction_independent_of_kernel():
    q = "extraversion"; si, Js, s = _load(q)
    Us = np.sqrt(s) * Js; Z = PREP["z24"][si].astype(float); Y = ARR["dh"][si, SL[q]].astype(float)
    K = (Z @ Z.T) * (Us @ Us.T); c = np.linalg.solve(K + np.eye(36), (Us * Y).sum(1))
    W = Z.T @ (c[:, None] * Us) / np.sqrt(s)          # native map: a(z) = W'z * ... (U already carries sqrt(s))
    for p in IDS["application_pairs"][:4]:
        j = ORDER.index(p)
        Ja = np.load(V3 / "P05/application_j_diagnostic_r1/private_app_jacobians" / f"{j:03d}_{q}.npy").ravel()
        a = (np.sqrt(s) * W).T @ PREP["z24"][j]       # compiled change sum_i c_i (z_i.z) U_i
        ref = ROWS[(ROWS.pair == p) & (ROWS["query"] == q) & (ROWS.input == 24)].L_ref_J.iloc[0]
        assert abs(Ja @ a - ref) <= 1e-8 * abs(ref) + 1e-14


def test_witness_rebuilt_from_saved_factors():
    q = "neuroticism"; si, Js, s = _load(q)
    Us = np.sqrt(s) * Js; Z = PREP["z24"][si].astype(float)
    B = np.load(OUT / "witness_compact_factors.npz")
    for p in IDS["application_pairs"][10:13]:
        j = ORDER.index(p)
        Ua = np.sqrt(s) * np.load(V3 / "P05/application_j_diagnostic_r1/private_app_jacobians" / f"{j:03d}_{q}.npy").ravel()
        zs = PREP["z24"][j].astype(float)
        beta = B[f"24|{q}|{p}".replace("|", "__")]
        dW = np.outer(zs, Ua) - Z.T @ (beta[:, None] * Us)
        n2 = (dW ** 2).sum()
        src = np.einsum("ip,pd,id->i", Z, dW, Us)
        assert np.abs(src).max() <= 1e-8 * np.sqrt(n2) * np.linalg.norm(Z, axis=1).max() * np.linalg.norm(Us, axis=1).max()
        assert abs(zs @ dW @ Ua - n2) <= 1e-8 * n2
        row = ROWS[(ROWS.pair == p) & (ROWS["query"] == q) & (ROWS.input == 24)].iloc[0]
        assert abs(1 - n2 / row.q - row.exposure) < 1e-9     # exposure = 1 - ||dW||^2 / q


def test_span_vs_product_coverage_reported_separately():
    for k, v in R["summary"].items():
        assert v["Z_span_share"]["cell_mean"] > .999 and v["J_span_share"]["cell_mean"] > .95
        assert 0 < v["application_exposure"]["cell_mean"] <= 1
