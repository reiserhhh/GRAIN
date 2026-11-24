#!/usr/bin/env python3
"""Appended to the archived numeric Reader; an idle step never contains the coming observation."""

import hashlib
import json
import sys

import torch


def simulate(req, cls):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cases = req["cases"]
    x = torch.tensor([s["x"] for s in cases], dtype=torch.float32)
    n, T, d = x.shape
    if (
        not torch.isfinite(x).all()
        or (x < 0).any()
        or not torch.allclose(x.sum(-1), torch.ones(n, T), atol=1e-5)
    ):
        raise ValueError("expected calibrated observation memberships")
    r = cls(d)
    r.load_state_dict({k: torch.tensor(v) for k, v in req["weights"].items()})
    r.eval()
    zero = torch.zeros(n, d + 1)
    out = {}
    with torch.inference_mode():
        for arm in cases[0]["gaps"]:
            times = torch.tensor([s["gaps"][arm] for s in cases], dtype=torch.float32)
            if (
                times.shape != (n, T)
                or not torch.isfinite(times).all()
                or (times < 0).any()
                or (times[:, 0] != 0).any()
            ):
                raise ValueError("invalid pre-arrival gaps")

            def roll(times, parts, mask_first=False):
                h = torch.zeros(n, r.cell.state_size)
                post = []
                pre = []
                for t in range(T):
                    dt = times[:, t]
                    use = dt > 0
                    if use.any():
                        # Zero-duration intervals bypass the solver; an unobserved interval is a masked input.
                        sliced = h[use]
                        for _ in range(parts):
                            _, sliced = r.cell(zero[use], sliced, dt[use, None] / parts)
                        h = h.clone()
                        h[use] = sliced
                    pre.append(h.clone())
                    event = (
                        torch.cat([x[:, t], torch.ones(n, 1)], 1)
                        if not (mask_first and t == 0)
                        else zero
                    )
                    y, h = r.cell(event, h, 1.0)
                    post.append(r.head(y).softmax(-1))
                return torch.stack(post, 1), torch.stack(pre, 1)

            parts = 2 if arm == "wall_two_steps" else 1
            state, pre = roll(times, parts)
            erased, _ = roll(times, parts, True)
            if arm == "event":
                official = r(x, torch.ones(n, T, 1))
                torch.testing.assert_close(official, state)
            out[arm] = {
                "post": state.tolist(),
                "source_effect": (state - erased).tolist(),
                "pre_last": pre[:, -1].tolist(),
            }
    return out


if __name__ == "__main__":
    req = json.load(sys.stdin)
    answer = simulate(req, Reader)  # noqa: F821
    answer["request_digest"] = hashlib.sha256(
        json.dumps(req, sort_keys=True).encode()
    ).hexdigest()
    json.dump(answer, sys.stdout, allow_nan=False)
