# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Bit-preserving H3 video patch permutation on CUDA and ROCm."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from rl_engine.reference.packing.video_patch_pack_unpack import (
    _CHANNELS,
    _WIDTH,
    _validate_pack,
    _validate_unpack,
)

_BLOCK = 256


@triton.jit
def _copy_patch(
    src,
    dst,
    C: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    PACK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One launch row handles one (batch, frame) pair. A lane names one value
    # in that frame's packed output, making stores contiguous for pack.
    frame_id = tl.program_id(1).to(tl.int64)
    batch = frame_id // T
    frame = frame_id % T
    packed_offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    values_per_frame: tl.constexpr = C * H * W
    valid = packed_offset < values_per_frame

    # Packed feature order is [channel, patch row, patch column]. Patch order
    # within a frame is row-major, exactly as in Qwen-Image's 2x2 permutation.
    feature = packed_offset % (4 * C)
    patch = packed_offset // (4 * C)
    channel = feature // 4
    patch_row = (feature % 4) // 2
    patch_col = feature % 2
    row = patch // (W // 2) * 2 + patch_row
    col = patch % (W // 2) * 2 + patch_col

    # Input is contiguous [B,C,T,H,W]; output is contiguous [B,T*P,4*C].
    # Their frame bases differ when T>1, so they cannot share Qwen's base.
    latent_index = (((batch * C + channel) * T + frame) * H + row) * W + col
    token_index = frame_id * values_per_frame + packed_offset

    # The caller passes integer views (int16/int32), so even NaN payloads and
    # signed zero move without floating-point conversion or arithmetic.
    if PACK:
        value = tl.load(src + latent_index, valid, other=0)
        tl.store(dst + token_index, value, valid)
    else:
        value = tl.load(src + token_index, valid, other=0)
        tl.store(dst + latent_index, value, valid)


def _launch(src: torch.Tensor, shape: tuple[int, int, int, int, int], pack: bool) -> torch.Tensor:
    b, c, t, h, w = shape
    tokens_shape = (b, t * (h // 2) * (w // 2), _WIDTH)
    dst = torch.empty(tokens_shape if pack else shape, dtype=src.dtype, device=src.device)
    bits = torch.int32 if src.element_size() == 4 else torch.int16
    with torch.cuda.device(src.device):
        _copy_patch[(triton.cdiv(c * h * w, _BLOCK), b * t)](
            src.view(bits), dst.view(bits), c, t, h, w, pack, _BLOCK
        )
    return dst


class _VideoPatchPermutation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src: torch.Tensor, shape: tuple[int, int, int, int, int], pack: bool):
        ctx.shape = shape
        ctx.pack = pack
        ctx.input_shape = tuple(src.shape)
        return _launch(src, shape, pack)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        # Re-enter apply instead of calling _launch directly. This records the
        # inverse permutation when create_graph=True, so gradgrad is correct.
        result = _VideoPatchPermutation.apply(grad.contiguous(), ctx.shape, not ctx.pack)
        return result.reshape(ctx.input_shape), None, None


class TritonVideoPatchPackUnpackOp:
    backend_id = "triton-video-patch-pack-unpack-v1"

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Triton H3 video patch requires a CUDA or ROCm GPU")

    def pack(self, x: torch.Tensor) -> torch.Tensor:
        b, t, h, w = _validate_pack(x)
        if x.device.type != "cuda":
            raise ValueError("Triton H3 video patch requires a CUDA or ROCm tensor")
        return _VideoPatchPermutation.apply(x, (b, _CHANNELS, t, h, w), True)

    def unpack(self, tokens: torch.Tensor, shape: tuple[int, int, int, int, int]) -> torch.Tensor:
        dims = _validate_unpack(tokens, shape)
        if tokens.device.type != "cuda":
            raise ValueError("Triton H3 video patch requires a CUDA or ROCm tensor")
        return _VideoPatchPermutation.apply(tokens, dims, False)
