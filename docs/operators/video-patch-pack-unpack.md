# MiniMax-H3 video patch pack/unpack

## Summary

Rearrange video latents from `(B,24,T,H,W)` into Transformer patch tokens
`(B,S,96)`, where `S=T*(H/2)*(W/2)`. `unpack` restores the original layout
exactly. Each token contains the 24 channels of one `2x2` region in a single
frame: `24*1*2*2=96` values.

The operator copies values without arithmetic. Backward applies the inverse
permutation. It supports MiniMax-H3 video layout conversion; the caller owns
video projections, multimodal sequence assembly, and full model execution.

The public name is `video_patch_pack_unpack`. Its current contract fixes H3's
24 channels and `(1,2,2)` patch; other model configurations are unsupported.

## Entry Point

Run this example from the repository root using the CPU PyTorch backend:

```python
import torch

from rl_engine.runtime.registry import KernelRegistry

x = torch.arange(24 * 2 * 4 * 6, dtype=torch.float32).reshape(1, 24, 2, 4, 6)
op = KernelRegistry().get_op("video_patch_pack_unpack", device=x.device)
tokens = op.pack(x)
recovered = op.unpack(tokens, tuple(x.shape))

assert tokens.shape == (1, 12, 96)
assert torch.equal(recovered, x)
print(op.backend_id, tuple(tokens.shape))
```

For CUDA inputs, the same entry point selects a backend using the priorities
below. `unpack` requires the original latent shape because the token shape
alone does not uniquely determine `T,H,W`.

## Backends

| Backend | Wrapper | Native symbol | Status |
| --- | --- | --- | --- |
| CUDA | `rl_engine.backends.cuda.packing.video_patch_pack_unpack.CudaVideoPatchPackUnpackOp` | `rl_engine._C.video_patch_pack_unpack` | Validated on one H20; requires the compiled extension |
| Triton on CUDA | `rl_engine.backends.shared.triton.video_patch_pack_unpack.TritonVideoPatchPackUnpackOp` | `_copy_patch` | Validated on one H20 |
| Triton on ROCm | Same Triton wrapper | `_copy_patch` | Implementation and dispatch provided; AMD hardware validation pending |
| PyTorch fallback | `rl_engine.reference.packing.video_patch_pack_unpack.NativeVideoPatchPackUnpackOp` | None; PyTorch tensor operations | CPU reference; also used as an independent GPU benchmark reference |

The CUDA autograd wrapper calls the binding in `csrc/bindings/ops.cpp`, which
launches `csrc/cuda/packing/video_patch_pack_unpack.cu`. The source is selected
by `build_tools/extensions.py` for NVIDIA builds and excluded from ROCm builds.

The CUDA implementation follows #410's shared-memory pair transpose. It groups
horizontally adjacent values into pairs and transposes a `16x16` tile using
128 threads per block. H3 has 48 pairs per token, covered in three passes.
Input addresses follow `(b,c,t,h,w)`; output follows `(b,t,patch,feature)`.
The time dimension changes the input frame base, so Qwen-Image's single-frame
`batch_base` cannot be reused directly. This implementation does not depend
on #410 being merged.

### Existing Implementations and Reuse

Investigation date: 2026-10-11. The table covers the listed entry points only.

| Source | Existing behavior and applicability | Reuse decision |
| --- | --- | --- |
| [Diffusers 0.40.0](https://github.com/huggingface/diffusers/tree/v0.40.0/src/diffusers/modular_pipelines/minimax_h3) | H3 packing and decoder inverse-permutation statements | Execute the pinned upstream tensor statements unchanged for correctness and performance comparisons |
| [SGLang](https://github.com/sgl-project/sglang/blob/ad8b1e1f851a6445e1211f7f0cf2130028ecae0c/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/packed_tokens.py) | H3 pack/unpack uses reshape, einsum without reduction, and contiguous; preserves values and returns 2D token rows | Reuse the permutation semantics through equivalent PyTorch dimension operations without a full SGLang runtime dependency |
| [vLLM-Omni](https://github.com/vllm-project/vllm-omni/blob/b1fc5fcfa3e1a4f76c6686eaf09242f0fc6464bc/vllm_omni/diffusion/models/minimax_h3/packed_tokens.py) | Native tensor pack/unpack for H3 | Reuse the same semantics; the existing native path is not classified as non-BI |
| [Qwen-Image #410](https://github.com/RL-Align/RL-Kernel/pull/410) | Shared-memory pair transpose for single-frame latents | Reuse the CUDA memory-access design, add time indexing and 48-pair tokens, and use a tile width of 16 |
| [CUTLASS/CuTe](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html) | Generic layout mapping and copy components; H3's permutation and launch still need to be expressed | Use #410's existing design without an additional CuTe runtime dependency |
| [FlashInfer packbits](https://docs.flashinfer.ai/generated/flashinfer.quantization.packbits.html) | Compresses boolean arrays into bits; does not permute floating-point values | This entry point cannot replace video patch pack/unpack |

The native tensor statements contain no reductions or arithmetic shared across
batch samples, suggesting that they can serve as BI references. This source
analysis does not establish hardware validation on every device. The reference
backend reuses the native permutation; CUDA and Triton provide optional performance
implementations of the same contract. Select an implementation based on measurements
for the relevant layout.

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `x` in `pack(x)` | `(B,24,T,H,W)` | FP16, BF16, FP32 | Contiguous; `B,T,H,W>0`; even `H,W`; `B*T<=65535` |
| `pack` output / `tokens` in `unpack(tokens, shape)` | `(B,T*(H/2)*(W/2),96)` | Same as latents | Contiguous unpack input with dimensions matching `shape` |
| `shape` in `unpack` | `(B,24,T,H,W)` | Five integers in a `tuple` or `torch.Size` | Same shape restrictions as `pack` |
| `unpack` output | `(B,24,T,H,W)` | Same as `tokens` | Contiguous; same device as the input |

Tokens are ordered by batch, frame, spatial patch row, and spatial patch column.
Features within a token are ordered by channel, patch row, and patch column:

```text
tokens[b, t*(H/2)*(W/2) + hi*(W/2) + wi, c*4 + dy*2 + dx]
    = latents[b, c, t, 2*hi + dy, 2*wi + dx]
```

All backends reuse the input validation in the PyTorch reference module and
raise explicit errors for invalid inputs. The upstream Diffusers H3 path returns
`(B*S,96)`; this interface preserves the batch dimension. Comparisons change
only the batch view, preserving element order.

CUDA additionally requires `H/2<=65535`, checked by the C++ entry point because
spatial patch rows use `grid.y`.

## Dispatch Behavior

`KernelRegistry.get_op("video_patch_pack_unpack", device=...)` uses these priorities:

| Device | Priority |
| --- | --- |
| NVIDIA CUDA | CUDA → Triton → PyTorch |
| ROCm | Triton → PyTorch |
| CPU | PyTorch |

If a backend cannot load, the registry tries the next entry. Invalid-input errors
from the selected backend propagate to the caller. ROCm does not select the NVIDIA
CUDA extension. NPU dispatch uses the PyTorch fallback but has not been validated.

Strict execution resolves and instantiates a full backend ID through the existing
semantic registry session. This CPU example is runnable; use the CUDA device and
backend ID to validate CUDA:

```python
from rl_engine.runtime.registry import KernelRegistry
from rl_engine.runtime.semantic_registry import OperatorRequirements

session = KernelRegistry().semantic.session()
resolution = session.resolve(
    semantic_op="video_patch_pack_unpack",
    requested_backend="pytorch-video-patch-pack-unpack-v1",
    target="training",
    requirements=OperatorRequirements(
        device="cpu",
        dtype="float32",
        topology={"world_size": 1, "sharding": "unsharded"},
        alignment_properties={
            "channels": 24,
            "patch_size": {"t": 1, "h": 2, "w": 2},
            "bitwise": True,
        },
    ),
)
op = session.instantiate(resolution)
print(session.instance_provenance(resolution, op).to_dict())
```

The backend IDs start with `pytorch`, `cuda`, or `triton`, followed by
`-video-patch-pack-unpack-v1`. Strict requests do not change backends automatically.
An unavailable extension, unsupported dtype, or incompatible WS1 topology raises
an error. Strict declarations cover the CPU reference and NVIDIA paths. ROCm retains
ordinary dispatch for future validation and is not registered as a validated strict
capability. GPU contract tests and benchmarks use this session path and verify the
actual implementation identity.

## Accuracy

The independent PyTorch gold uses `reshape -> permute -> reshape -> contiguous`
for the permutation and its inverse. Each output corresponds to one input value,
with no floating-point arithmetic or dtype conversion. Outputs and first-order
gradients must therefore match bitwise with zero numerical tolerance.

Triton copies through integer views of the same width; CUDA copies integer pairs.
Tests compare raw tensor bytes, including signed zero, infinities, and NaN payloads.
Ordinary floating-point equality does not cover all these cases. GPU backward makes
the upstream gradient contiguous before applying the inverse permutation and retains
the higher-order gradient path for `create_graph=True`.

The second reference comes from `patchify_video_latents` and the video inverse
statements in `MiniMaxH3AfterDenoiseStep.__call__`, installed from
`diffusers==0.40.0`. Tests and benchmarks share the benchmark's `DiffusersVideoPatch`
adapter. It executes the upstream tensor statements unchanged, records source
SHA-256 hashes, and does not call RL-Kernel's reference permutation. The upstream
decoder entry uses `no_grad`; the unpack gradient reference is autograd applied
to the extracted tensor statements.

The shared `OP_SPECS` harness registers `video_patch_pack_unpack` as an
`elementwise` operator with PyTorch, CUDA, and Triton candidates. A validation-only
adapter maps its callable interface to `.pack()` or `.unpack()`; the runtime API
is unchanged. The gold path promotes the independently generated input to FP32
before applying the PyTorch permutation. Both directions declare `x` as the
gradient input and use the shared forward/backward accuracy profile. The focused
pytest tests additionally require exact bits and cover special floating-point values.

Select the direction and original latent geometry explicitly:

```bash
python tools/validation/operators/check_operator.py \
  --op video_patch_pack_unpack --candidate pytorch --device cpu \
  --dtype bf16 --patch-direction unpack \
  --batch 2 --frames 3 --height 6 --width 10 --check-grad --json
```

For unpack, the input builder generates `(B,S,96)` tokens directly, rather than
packing a latent input first. `--frames/--height/--width` describe the original
latent shape in either direction; they do not change the fixed 24 channels or
the `(1,2,2)` patch. `--seq` is unused for this operator.

## Performance Notes

After completing the NVIDIA setup, build, and correctness checks below, run
from the repository root:

```bash
python benchmarks/operators/packing/benchmark_video_patch_pack_unpack.py \
  --output-dir _dev_notes/h3-benchmark
```

Default shapes use latent dimensions for `H,W`:

| Preset | `(T,H,W)` | Tokens per sample |
| --- | --- | --- |
| `small` | `(3,6,10)` | 45 |
| `medium` | `(8,24,42)` | 2016 |
| `768p` | `(32,48,84)` | 32256 |

Reproduce the workload report's 28 layouts across three dtypes:

```bash
python benchmarks/operators/packing/benchmark_video_patch_pack_unpack.py \
  --workloads --output-dir _dev_notes/h3-benchmark/workloads
```

`--workloads` covers six aspect ratios at `T=37/72/102, B=1`, landscape batch
sweeps at `T=37/102, B=2/3/4/8`, and landscape/portrait conditions at `T=1, B=1`.
It replaces `--shapes` and `--batches`. The pinned upstream spatial compression
factor is 16. Aligned video frame counts `F=17n+5` map to latent `T=5n+2`, so
124/243/345 frames map to `T=37/72/102`. Geometry and frame alignment come from
the pinned
[modular_pipeline.py](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/modular_pipelines/minimax_h3/modular_pipeline.py).
Use `--shapes 16x9_t37,9x16_t72,16x9_t102,16x9_t1 --batches 1` for selected layouts.

Each default preset sweeps `B=1,2,4,8` and FP32/FP16/BF16, comparing CUDA, Triton,
the PyTorch reference, and the pinned Diffusers tensor path. Pack and unpack each
measure forward, backward, and forward plus backward. Outputs and first-order
gradients are checked for bitwise equality before timing.

Eager CUDA events and synchronized wall time measure individual calls. CUDA Graph
timing divides replay time by the number of captured calls; report it separately.
Defaults are 10 warmups, 48 samples, and 32 calls per graph. Backward-only timing
uses a forward graph prepared outside the measured interval.

`results.json` records raw samples, device/runtime versions, source revision and
workspace state, strict resolution traces, actual implementation identities,
loaded extension paths and hashes, upstream source hashes, and peak allocated
VRAM. Keep reports, figures, and raw results under the local `_dev_notes/`
directory and provide them through PR attachments or accessible links. Performance
claims must identify the shape, dtype, timing method, and source revision.
Operator speedups do not establish full model speedups.

H20 FP32 measurements at `0542141` recorded geometric-mean CUDA Graph speedups
of 1.437x/1.621x for generation pack/unpack over Diffusers. Single-frame condition
pack measured 0.748x, about 34% longer latency, and 1.000x in synchronized eager
wall time. The upstream native permutation is already BI; the optimized GPU
backends target memory-copy performance, not a numerical defect in that path. Attach
the final validation and source fingerprints to the PR; historical numbers do
not establish that a new commit was retested. Ordinary dispatch prefers available
CUDA implementations without automatically selecting by T or measured performance.

## Tests

### Local Development Checks

Use Python >=3.10 with PyTorch installed for the target device. Run these commands
from the repository root; CPU development does not require the CUDA extension:

```bash
python -m pip install --no-build-isolation -e ".[dev]"
python -m pytest tests/ops/packing/test_video_patch_pack_unpack.py \
  tests/ops/packing/test_video_patch_pack_unpack_diffusers.py \
  tests/runtime/test_kernel_registry.py tests/validation/operators/test_operator_inputs.py tests/validation/operators/test_op_checks.py -q -rs
```

Tests explicitly skip with a reason if the optional pinned Diffusers version,
GPU, or requested backend is unavailable. Other CPU contract tests still run.
With the correct Diffusers version installed, source extraction or comparison
errors fail the tests. Local runs with GPU skips do not establish GPU acceptance.

| Test file | Coverage |
| --- | --- |
| `tests/ops/packing/test_video_patch_pack_unpack.py` | Independent coordinate order, invalid inputs, ordinary/strict dispatch, missing-extension rejection, both directions and gradients, special bits, batch invariance, CUDA tile boundaries, noncontiguous gradients, and higher-order gradients |
| `tests/ops/packing/test_video_patch_pack_unpack_diffusers.py` | PyTorch/Triton/CUDA outputs and first-order gradients against upstream pack/unpack; six shapes including T=37/72/102 videos and T=1 conditions across three dtypes |

### NVIDIA GPU Operator Acceptance

The target GPU must be visible, with a CUDA toolkit and `nvcc` compatible with
PyTorch. Install dependencies, require the extension build, and run the tests:

```bash
python -m pip install pytest ninja triton 'diffusers==0.40.0'
RL_KERNEL_REQUIRE_EXT=1 python -m pip install --no-build-isolation -e ".[dev]"
mkdir -p _dev_notes/h3-benchmark
python -m pytest tests/ops/packing/test_video_patch_pack_unpack.py \
  tests/ops/packing/test_video_patch_pack_unpack_diffusers.py \
  -q -rs --junitxml=_dev_notes/h3-benchmark/validation.xml
```

Pytest must exit successfully with zero skips. Check the XML so a missing
dependency or backend cannot produce a passing acceptance result through skips:

```bash
python - <<'PY'
import xml.etree.ElementTree as ET

root = ET.parse('_dev_notes/h3-benchmark/validation.xml').getroot()
assert root.findall('.//testcase'), 'No tests executed'
for status in ('skipped', 'failure', 'error'):
    assert not root.findall(f'.//{status}'), f'GPU acceptance contains {status}'
print('GPU acceptance: all tests executed and passed')
PY
```

Run the shared accuracy harness for each backend, dtype, and direction. Each JSON
must report `passed=true`; the focused pytest suite remains the bitwise gate:

```bash
for backend in pytorch cuda triton; do
  for dtype in fp32 bf16 fp16; do
    for direction in pack unpack; do
      python tools/validation/operators/check_operator.py \
        --op video_patch_pack_unpack --candidate "$backend" --device cuda \
        --dtype "$dtype" --patch-direction "$direction" \
        --batch 2 --frames 3 --height 6 --width 10 --check-grad --json \
        > "_dev_notes/h3-benchmark/gtest-${backend}-${dtype}-${direction}.json"
    done
  done
done
```

These commands target NVIDIA. Validate the ROCm Triton path separately on AMD
hardware; the CUDA extension tests do not apply. ROCm acceptance is pending.

### Formatting and Documentation Checks

```bash
python -m pip install pre-commit mypy -r requirements/docs.txt
pre-commit run --all-files
mypy --ignore-missing-imports rl_engine/
mkdocs build --strict -f mkdocs.yaml
```

## Known Limitations

- Fixed `C=24`, patch `(1,2,2)`, and token width 96; other configurations are unsupported.
- Contiguous inputs required; zero dimensions, odd `H/W`, FP64, and integer dtypes are rejected.
- All backends require `B*T<=65535`; frames use Triton's second grid axis and CUDA `grid.z`.
- CUDA additionally requires `H/2<=65535` because spatial patch rows use `grid.y`.
- The CUDA extension is NVIDIA-only. ROCm Triton and NPU fallback lack target-hardware validation.
- The benchmark targets one NVIDIA GPU and measures an operator. It does not load model weights,
  the VAE, or the full Transformer; the Diffusers comparison does not establish model-level
  end-to-end validation.
