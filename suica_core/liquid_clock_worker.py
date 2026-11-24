#!/usr/bin/env python3
"""Second half of the numeric-only worker, sent after the frozen Reader definition."""

import hashlib
import json
import sys

import torch


def coefficients(cell, x, h):
    p = cell._params
    positive = cell.make_positive_fn

    def conduct(v, prefix):
        g = (
            cell._sigmoid(v, p[prefix + "mu"], p[prefix + "sigma"])
            * positive(p[prefix + "w"])
            * p[prefix + "sparsity_mask"]
        )
        return g.sum(1), (g * p[prefix + "erev"]).sum(1)

    sensory, drive = conduct(cell._map_inputs(x), "sensory_")
    cap = positive(p["cm"]) * cell._ode_unfolds
    leak = positive(p["gleak"])
    carry = torch.ones_like(h)
    for _ in range(cell._ode_unfolds):
        g, v = conduct(h, "")
        den = cap + leak + (g + sensory) + cell._epsilon
        carry = carry * (cap / den)
        h = (cap * h + leak * p["vleak"] + (v + drive)) / den
    return h, carry


def main(reader_class):
    req = json.load(sys.stdin)
    torch.set_num_threads(1)
    net = reader_class(len(req["paths"][0][0]))
    net.load_state_dict({k: torch.tensor(v) for k, v in req["weights"].items()})
    result = []
    err = 0.0
    with torch.inference_mode():
        seqs = [torch.tensor(p, dtype=torch.float32) for p in req["paths"]]
        if any(
            len(p) == 0 or p.dim() != 2 or not torch.isfinite(p).all() for p in seqs
        ):
            raise ValueError("invalid observation path")
        batch = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
        h = torch.zeros(len(batch), net.cell.state_size)
        steps = []
        for points in batch.unbind(1):
            x = torch.cat([points, torch.ones(len(h), 1)], dim=1)
            _, official = net.cell(x, h, 1.0)
            observed, carry = coefficients(net.cell, x, h)
            err = max(err, (observed - official).abs().max().item())
            torch.testing.assert_close(observed, official, rtol=1e-5, atol=5e-7)
            if not ((carry > 0) & (carry < 1)).all():
                raise ValueError("invalid substep carry")
            h = official
            steps.append(carry)
        stacked = torch.stack(steps, dim=1)
        result = [stacked[i, : len(seq)].tolist() for i, seq in enumerate(seqs)]
    json.dump(
        {
            "paths": result,
            "official_state_max_error": err,
            "request_digest": hashlib.sha256(
                json.dumps(req, sort_keys=True).encode()
            ).hexdigest(),
        },
        sys.stdout,
        allow_nan=False,
    )


if __name__ == "__main__":
    main(Reader)  # noqa: F821 - supplied by the content-bound first half of this worker
