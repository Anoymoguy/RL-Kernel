# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CUDA implementation of the MiniMax-H3 video patch permutation."""

from __future__ import annotations

import torch

from rl_engine.backends.extension import _C, _EXT_AVAILABLE
from rl_engine.reference.packing.video_patch_pack_unpack import (
    _CHANNELS,
    _validate_pack,
    _validate_unpack,
)


class _CudaVideoPatchPermutation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src: torch.Tensor, shape: tuple[int, int, int, int, int], unpack: bool):
        ctx.shape = shape
        ctx.unpack = unpack
        return _C.video_patch_pack_unpack(src, *shape, unpack)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        # The inverse permutation is also differentiable for create_graph=True.
        inverse = _CudaVideoPatchPermutation.apply(grad.contiguous(), ctx.shape, not ctx.unpack)
        return inverse, None, None


class CudaVideoPatchPackUnpackOp:
    backend_id = "cuda-video-patch-pack-unpack-v1"

    def __init__(self) -> None:
        if torch.version.hip is not None or not torch.cuda.is_available():
            raise RuntimeError("CUDA video patch requires an NVIDIA GPU")
        if not _EXT_AVAILABLE or _C is None or not hasattr(_C, "video_patch_pack_unpack"):
            raise RuntimeError(
                "Rebuild rl_engine._C with csrc/cuda/packing/video_patch_pack_unpack.cu"
            )

    def pack(self, x: torch.Tensor) -> torch.Tensor:
        b, t, h, w = _validate_pack(x)
        if x.device.type != "cuda":
            raise ValueError("CUDA video patch requires a CUDA tensor")
        return _CudaVideoPatchPermutation.apply(x, (b, _CHANNELS, t, h, w), False)

    def unpack(self, tokens: torch.Tensor, shape: tuple[int, int, int, int, int]) -> torch.Tensor:
        dims = _validate_unpack(tokens, shape)
        if tokens.device.type != "cuda":
            raise ValueError("CUDA video patch requires a CUDA tensor")
        return _CudaVideoPatchPermutation.apply(tokens, dims, True)
