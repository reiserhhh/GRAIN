"""Fixed-layer attention-edge interventions on complete response tokens."""

from __future__ import annotations

import numpy as np

from .response_context import paired_interval

CUTS = (-1, 10, 16, 22, 27)
SITES = (10, 16, 22)


def cut_mask(mask, start, stop, *, sham=False):
    """Clone an additive mask; change only allowed body-query/prefix-key edges."""
    import torch

    if (
        not isinstance(mask, torch.Tensor)
        or mask.dtype != torch.float32
        or mask.ndim != 4
        or mask.shape[:2] != (1, 1)
        or mask.shape[2] != mask.shape[3]
        or not 0 < start < stop <= mask.shape[2]
    ):
        raise ValueError("Unsupported float32 square additive attention mask")
    result = mask.clone()
    block = result[..., start:stop, :start]
    if torch.isnan(block).any() or (block > 0).any():
        raise ValueError("Invalid additive prefix edges")
    allowed = block == 0
    edges = int(allowed.sum())
    if not sham:
        block[allowed] = torch.finfo(result.dtype).min
    return result, edges


def access_forward(observer, prefix, body, *, cut=27, sham=False, **kwargs):
    """Hook only later blocks, assert complete coverage, and always remove hooks."""
    if cut not in CUTS or len(observer.model.model.layers) != 28:
        raise ValueError("Unsupported frozen cutoff or decoder depth")
    if kwargs.get("compact", False):
        raise ValueError("Access intervention requires fixed body origin")
    handles, visited = [], []
    expected = list(range(cut + 1, 28))

    def hook(index):
        def apply(_module, inputs, keywords):
            mask = keywords.get("attention_mask")
            if mask is None or tuple(mask.shape) != (1, 1, 1024, 1024):
                raise ValueError("Missing full attention mask in decoder hook")
            changed, edges = cut_mask(mask, 768, 768 + len(body), sham=sham)
            if edges != len(prefix) * len(body):
                raise ValueError("Prefix/body edge coverage differs from input mask")
            visited.append(index)
            return inputs, {**keywords, "attention_mask": changed}

        return apply

    try:
        for index in expected:
            handles.append(
                observer.model.model.layers[index].register_forward_pre_hook(
                    hook(index), with_kwargs=True
                )
            )
        result = observer.forward(prefix, body, **kwargs)
        if visited != expected:
            raise ValueError("Incomplete or repeated attention hook execution")
        result["access"] = {
            "cut": cut,
            "layers": len(visited),
            "edges_per_layer": len(prefix) * len(body) if visited else 0,
            "sham": sham,
        }
        return result
    finally:
        for handle in handles:
            handle.remove()


def conditions(forward, pa, pm, body):
    """Run all eighteen fixed arms; instrument identities fail before inference."""
    values, rows = {}, []

    def run(condition, prefix, **kwargs):
        value = forward(prefix, body, **kwargs)
        values[condition] = value
        rows.append(
            {
                "condition": condition,
                "mean_logp": value["mean_logp"],
                "tail_logp": value["tail_logp"],
                "first_logp": float(value["token_logp"][0]),
                "prefix_tokens": len(prefix),
                "response_tokens": len(body),
                **value["access"],
            }
        )
        return value

    va, vm = run("actual_27", pa), run("matched_27", pm)
    for cut in CUTS[:-1]:
        for role, prefix in (("actual", pa), ("matched", pm)):
            run(f"{role}_{cut}", prefix, cut=cut)
    patch_replay = {}
    for site in SITES:
        donor = va["trace"][f"block_{site:02}"]["states"]
        for cut, label in ((27, "full"), (site, "cut")):
            v = run(
                f"patch_{site}_{label}",
                pm,
                cut=cut,
                patch_site=f"block_{site:02}",
                patch=donor,
            )
            if label == "cut":
                patch_replay[str(site)] = float(
                    abs(
                        v["token_logp"][1:] - values[f"actual_{site}"]["token_logp"][1:]
                    ).max()
                )
    mask_sham = run("mask_sham", pm, cut=10, sham=True)
    self_sham = run(
        "self_sham",
        pm,
        cut=16,
        patch_site="block_16",
        patch=vm["trace"]["block_16"]["states"],
    )
    embedding = va["trace"]["input_embedding"]["mean"]
    if any(
        not np.array_equal(v["trace"]["input_embedding"]["mean"], embedding)
        for v in values.values()
    ):
        raise ValueError("Body input embedding changed")
    checks = {
        "cut0_tail_error": float(
            abs(
                values["actual_-1"]["token_logp"][1:]
                - values["matched_-1"]["token_logp"][1:]
            ).max()
        ),
        "mask_sham_error": float(abs(mask_sham["token_logp"] - vm["token_logp"]).max()),
        "self_sham_error": float(
            abs(self_sham["token_logp"] - values["matched_16"]["token_logp"]).max()
        ),
        "first_token_error": max(
            float(
                abs(
                    v["token_logp"][0]
                    - (va if c.startswith("actual_") else vm)["token_logp"][0]
                )
            )
            for c, v in values.items()
        ),
        "patch_cut_errors": patch_replay,
    }
    if (
        checks["cut0_tail_error"] > 1e-5
        or max(patch_replay.values()) > 1e-5
        or any(
            checks[k] != 0
            for k in ("mask_sham_error", "self_sham_error", "first_token_error")
        )
    ):
        raise ValueError(f"Access instrument identity failed: {checks}")
    return rows, checks


def summarize(frame):
    """Summarize all fixed access and state contrasts without mediation fractions."""
    p = frame.pivot(index="participant_code", columns="condition", values="tail_logp")
    native = p.actual_27 - p.matched_27
    effects = {}
    for cut in CUTS:
        d = p[f"actual_{cut}"] - p[f"matched_{cut}"]
        effects[str(cut)] = {
            "retained_context_difference": paired_interval(d),
            "differential_access_effect": paired_interval(native - d),
            "actual_access_effect": paired_interval(p.actual_27 - p[f"actual_{cut}"]),
            "matched_access_effect": paired_interval(
                p.matched_27 - p[f"matched_{cut}"]
            ),
        }
    patches = {}
    for site in SITES:
        full = p[f"patch_{site}_full"] - p.matched_27
        cut = p[f"patch_{site}_cut"] - p[f"matched_{site}"]
        patches[str(site)] = {
            "full_patch_effect": paired_interval(full),
            "cut_patch_effect": paired_interval(cut),
            "full_minus_cut_patch_effect": paired_interval(full - cut),
        }
    return {"effects": effects, "patch_effects": patches}
