# SPDX-License-Identifier: Apache-2.0
"""Test-only independent numerical oracle; never a production fallback."""

import torch

from rl_engine.contracts.operators.indexer_icv import ICV_BACKEND_ID

FP4_TABLE = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def decode_fp4(data, scales, dim):
    # Independent CPU float64 oracle includes every nibble and exponent.
    raw, scale = data.cpu(), scales.cpu()
    codes = torch.stack((raw & 15, raw >> 4), dim=-1).flatten(-2)
    table = torch.tensor(FP4_TABLE, dtype=torch.float64)
    values = table[codes.long()]
    return (values * torch.exp2(scale.double() - 127).repeat_interleave(32, -1))[..., :dim]


def decode(data, scales, contract):
    if contract.precision == "mxfp4":
        return decode_fp4(data, scales, contract.head_dim)
    return data.cpu().double() * scales.cpu().double()


class ReferenceICVBackend:
    backend_id = ICV_BACKEND_ID
    device_type = "cpu"

    def __init__(self):
        self.calls = 0

    @staticmethod
    def plan(contract):
        return {"test_only": True}

    def prepare(self, data, scales, contract):
        return decode(data, scales, contract).bfloat16(), None

    def dot(self, q, k, q_scale, k_scale, contract):
        self.calls += 1
        return torch.einsum("mhd,nd->mhn", q.float(), k.float())


def payload(contract, rows, *, q=False, seed=0, device="cpu"):
    generator = torch.Generator().manual_seed(seed)
    leading = (rows, contract.num_heads) if q else (rows,)
    if contract.precision == "mxfp4":
        data = torch.randint(
            0, 256, (*leading, contract.padded_dim // 2), generator=generator, dtype=torch.uint8
        )
        scales = torch.randint(
            123, 131, (*leading, contract.padded_dim // 32), generator=generator, dtype=torch.uint8
        )
    else:
        dtype = getattr(torch, "float8_" + contract.fp8_encoding)
        data = torch.randn((*leading, contract.head_dim), generator=generator).to(dtype)
        scales = torch.exp2(torch.randint(-3, 4, (*leading, 1), generator=generator).float())
    return data.to(device), scales.to(device)


def ids(request, rows, *, start=0, device="cpu"):
    return torch.tensor(
        [[request, i] for i in range(start, start + rows)], dtype=torch.int64, device=device
    ).reshape(rows, 2)


def byte_equal(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
