// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

// Adapt #410's 2x2 pair transpose to H3's channel-major [B,C,T,H,W] latents.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cstdint>

namespace {

// H3 has 48 pairs per token. A 16-wide tile covers them in three full passes
// with 128 threads per block; the 32-wide tile needs a half-full second pass.
constexpr int kTile = 16;
constexpr int kThreadsY = 8;
constexpr int64_t kChannels = 24;
constexpr int64_t kPatchWidth = 2;
constexpr int64_t kFeatureWidth = kChannels * kPatchWidth * kPatchWidth;

template <typename Bits>
struct Pair { Bits left, right; };

template <typename Bits, bool Unpack>
__global__ void video_patch_permute(
    const Pair<Bits>* __restrict__ input,
    Pair<Bits>* __restrict__ output,
    int64_t channels, int64_t frames, int64_t height, int64_t width) {
  __shared__ Pair<Bits> tile[kTile][kTile + 1];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int64_t frame_id = blockIdx.z;
  const int64_t batch = frame_id / frames;
  const int64_t frame = frame_id % frames;
  const int64_t token_cols = width / 2;
  const int64_t pairs_per_token = channels * 2;
  const int64_t token_row = blockIdx.y;
  const int64_t token_col_start = static_cast<int64_t>(blockIdx.x) * kTile;

  for (int64_t pair_start = 0; pair_start < pairs_per_token; pair_start += kTile) {
    // Read coalesced pairs, transpose in shared memory, then write packed pairs.
    for (int offset = 0; offset < kTile; offset += kThreadsY) {
      if constexpr (Unpack) {
        const int64_t col = token_col_start + ty + offset;
        const int64_t pair = pair_start + tx;
        if (col < token_cols && pair < pairs_per_token) {
          const int64_t index = frame_id * channels * height * token_cols
              + (token_row * token_cols + col) * pairs_per_token + pair;
          tile[ty + offset][tx] = input[index];
        }
      } else {
        const int64_t col = token_col_start + tx;
        const int64_t pair = pair_start + ty + offset;
        if (col < token_cols && pair < pairs_per_token) {
          const int64_t channel = pair / 2;
          const int64_t row = token_row * 2 + pair % 2;
          const int64_t index = ((batch * channels + channel) * frames + frame)
              * height * token_cols + row * token_cols + col;
          tile[ty + offset][tx] = input[index];
        }
      }
    }
    __syncthreads();
    for (int offset = 0; offset < kTile; offset += kThreadsY) {
      if constexpr (Unpack) {
        const int64_t col = token_col_start + tx;
        const int64_t pair = pair_start + ty + offset;
        if (col < token_cols && pair < pairs_per_token) {
          const int64_t channel = pair / 2;
          const int64_t row = token_row * 2 + pair % 2;
          const int64_t index = ((batch * channels + channel) * frames + frame)
              * height * token_cols + row * token_cols + col;
          output[index] = tile[tx][ty + offset];
        }
      } else {
        const int64_t col = token_col_start + ty + offset;
        const int64_t pair = pair_start + tx;
        if (col < token_cols && pair < pairs_per_token) {
          const int64_t index = frame_id * channels * height * token_cols
              + (token_row * token_cols + col) * pairs_per_token + pair;
          output[index] = tile[tx][ty + offset];
        }
      }
    }
    if (pair_start + kTile < pairs_per_token) {
      __syncthreads();
    }
  }
}

template <typename Bits, bool Unpack>
void launch(const torch::Tensor& input, torch::Tensor& output,
            int64_t batch, int64_t channels, int64_t frames,
            int64_t height, int64_t width) {
  const dim3 grid((width / 2 + kTile - 1) / kTile, height / 2, batch * frames);
  const dim3 block(kTile, kThreadsY);
  video_patch_permute<Bits, Unpack><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const Pair<Bits>*>(input.data_ptr()),
      reinterpret_cast<Pair<Bits>*>(output.data_ptr()), channels, frames, height, width);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

torch::Tensor video_patch_pack_unpack_cuda(
    torch::Tensor input, int64_t batch, int64_t channels, int64_t frames,
    int64_t height, int64_t width, bool unpack) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "expected contiguous CUDA input");
  TORCH_CHECK(input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf ||
              input.scalar_type() == at::kBFloat16, "expected fp32, fp16, or bf16");
  TORCH_CHECK(batch > 0 && channels == kChannels && frames > 0 && height > 0 && width > 0 &&
              height % 2 == 0 && width % 2 == 0 && batch <= 65535 / frames &&
              height / 2 <= 65535, "invalid H3 video latent dimensions");
  const int64_t sequence = frames * (height / 2) * (width / 2);
  if (unpack) {
    TORCH_CHECK(input.dim() == 3 && input.size(0) == batch &&
                input.size(1) == sequence && input.size(2) == kFeatureWidth,
                "expected packed [B,S,96] input");
  } else {
    TORCH_CHECK(input.dim() == 5 && input.size(0) == batch &&
                input.size(1) == channels && input.size(2) == frames &&
                input.size(3) == height && input.size(4) == width,
                "expected H3 latent [B,24,T,H,W] input");
  }
  const c10::cuda::CUDAGuard guard(input.device());
  auto output = unpack ? torch::empty({batch, channels, frames, height, width}, input.options())
                       : torch::empty({batch, sequence, kFeatureWidth}, input.options());
  if (input.scalar_type() == at::kFloat) {
    if (unpack) launch<uint32_t, true>(input, output, batch, channels, frames, height, width);
    else launch<uint32_t, false>(input, output, batch, channels, frames, height, width);
  } else {
    if (unpack) launch<uint16_t, true>(input, output, batch, channels, frames, height, width);
    else launch<uint16_t, false>(input, output, batch, channels, frames, height, width);
  }
  return output;
}
