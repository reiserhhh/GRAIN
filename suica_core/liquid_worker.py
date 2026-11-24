#!/usr/bin/env python3
"""Numeric-only GRAIN node reader. May be run over stdin on a separate CPU host."""

import hashlib
import json
import sys

import torch
from ncps.torch.ltc_cell import LTCCell
from ncps.wirings import AutoNCP
from torch import nn


class Reader(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.cell = LTCCell(
            AutoNCP(24, 8, seed=431),
            d + 1,
            ode_unfolds=6,
            implicit_param_constraints=True,
        )
        self.head = nn.Linear(8, d)

    def forward(self, x, seen):
        h = x.new_zeros(len(x), 24)
        series = []
        for p, mask in zip(x.unbind(1), seen.unbind(1)):
            o, h = self.cell(torch.cat([p * mask, mask], dim=1), h, 1.0)
            series.append(self.head(o).softmax(-1))
        return torch.stack(series, 1)


def process(req):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(431)
    seq = [torch.tensor(s, dtype=torch.float32) for s in req["paths"]]
    d = seq[0].shape[-1]
    if any(
        x.ndim != 2
        or len(x) < 1
        or x.shape[-1] != d
        or not torch.isfinite(x).all()
        or (x < 0).any()
        or not torch.allclose(x.sum(-1), torch.ones(len(x)), atol=1e-5)
        for x in seq
    ):
        raise ValueError("expected complete sequences of native node memberships")
    net = Reader(d)
    if req["mode"] == "fit":
        seq = [x for x in seq if len(x) > 1]
        if not seq:
            raise ValueError("calibration needs at least one observed transition")
        # One fixed recipe, no labels, no evaluation-based parameter selection.
        opt = torch.optim.Adam(net.parameters(), lr=0.008)
        for _ in range(64):
            batch = [seq[i] for i in torch.randint(len(seq), (16,)).tolist()]
            x = nn.utils.rnn.pad_sequence(batch, batch_first=True)
            valid = (
                torch.arange(x.size(1))[None, :]
                < torch.tensor([len(s) for s in batch])[:, None]
            )
            y = net(x[:, :-1], valid[:, :-1, None].float())
            loss = -(x[:, 1:] * y.clamp_min(1e-10).log()).sum(-1)[valid[:, 1:]].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return {
            "weights": {k: v.detach().tolist() for k, v in net.state_dict().items()},
            "train_paths": len(seq),
            "train_transitions": sum(len(x) - 1 for x in seq),
        }
    if req["mode"] != "score":
        raise ValueError("unknown mode")
    net.load_state_dict({k: torch.tensor(v) for k, v in req["weights"].items()})
    out = []
    with torch.inference_mode():
        for x in seq:
            n = len(x)
            # At most 16 interventions in flight, including long documents.
            seen = x.new_ones(1, n, 1)
            actual = net(x[None], seen)[0]
            reset = net(x[:, None], x.new_ones(n, 1, 1))[:, 0]
            missing = []
            for start in range(0, n, 16):
                ids = torch.arange(start, min(n, start + 16))
                masks = seen.repeat(len(ids), 1, 1)
                masks[torch.arange(len(ids)), ids] = 0
                missing.extend(
                    (actual - net(x.expand(len(ids), -1, -1), masks)).tolist()
                )
            rev = net(x.flip([0])[None], seen)[0, -1]
            order = list(range(n))
            if n >= 4:
                order[1], order[2] = order[2], order[1]
            fixed_end = net(x[order][None], seen)[0, -1] if n >= 4 else None
            empty = net(x[None], seen * 0)[0]
            out.append(
                {
                    "readout": actual.tolist(),
                    "reset": reset.tolist(),
                    "missing_effect": missing,
                    "reverse_endpoint": rev.tolist(),
                    "all_missing": empty.tolist(),
                    "middle_swap_endpoint": None
                    if fixed_end is None
                    else fixed_end.tolist(),
                }
            )
    return {"paths": out}


if __name__ == "__main__":
    req = json.load(sys.stdin)
    result = process(req)
    result["request_digest"] = hashlib.sha256(
        json.dumps(req, sort_keys=True).encode()
    ).hexdigest()
    json.dump(result, sys.stdout, allow_nan=False)
