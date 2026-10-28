# SPDX-License-Identifier: Apache-2.0
"""Collect GDN instance/state facts without claiming a kernel was authorized.

The bridge owns state lineage. These observations are candidates: a Python
callable name does not establish the native/JIT artifact actually launched.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch

from rl_engine.contracts.strict_provider import StrictProviderError, fingerprint


def implementation_name(value: Any) -> str:
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _nonnegative(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise StrictProviderError("invalid_observation", "Expected a nonnegative integer", field)
    return value


def tensor_observation(tensor: torch.Tensor) -> dict[str, Any]:
    """Read the actual kernel-facing view; do not copy/hash the state contents."""
    if not isinstance(tensor, torch.Tensor):
        raise StrictProviderError("invalid_observation", "Expected an actual tensor")
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "device_type": tensor.device.type,
        "storage_offset": tensor.storage_offset(),
        "contiguous": tensor.is_contiguous(),
        "requires_grad": tensor.requires_grad,
    }


def observe_gdn_instance(provider: Any) -> dict[str, Any]:
    """Read effective attributes from a bound vLLM GDN instance."""
    fields = ("num_k_heads", "num_v_heads", "head_k_dim", "head_v_dim", "conv_kernel_size")
    config = {}
    for field in fields:
        value = getattr(provider, field, None)
        if type(value) is not int or value <= 0:
            raise StrictProviderError("unobservable_binding", "Missing GDN dimension", field)
        config[field] = value
    for field in (
        "tp_size",
        "gqa_interleaved_layout",
        "gdn_prefill_backend",
        "gdn_decode_kernel",
        "enable_packed_recurrent_decode",
        "enable_fused_gdn_decode",
    ):
        value = getattr(provider, field, None)
        if isinstance(value, (str, bool, int)):
            config[field] = value
    norm = getattr(provider, "norm", None)
    if norm is not None:
        config["norm_implementation"] = implementation_name(norm)
    chunk = getattr(provider, "chunk_gated_delta_rule", None)
    if chunk is not None:
        config["prefill_implementation"] = implementation_name(chunk)
    return {
        "implementation": implementation_name(provider),
        "effective_config": config,
        "evidence_level": "instance_candidate",
        "enforcement_ready": False,
    }


def observe_gdn_metadata(metadata: Any) -> dict[str, Any]:
    """Snapshot small dispatch metadata, including real sequence/slot indices."""
    get = (
        metadata.get if isinstance(metadata, Mapping) else lambda key: getattr(metadata, key, None)
    )
    result = {}
    for key in ("num_prefills", "num_decodes", "num_actual_tokens"):
        result[key] = _nonnegative(get(key), key)
    prefill, decode = result["num_prefills"], result["num_decodes"]
    result["phase"] = (
        "mixed"
        if prefill and decode
        else "prefill"
        if prefill
        else "decode"
        if decode
        else "profile"
    )
    for key in (
        "has_initial_state",
        "prefill_has_initial_state",
        "spec_sequence_masks",
        "non_spec_query_start_loc",
        "prefill_query_start_loc",
        "non_spec_state_indices_tensor",
        "prefill_state_indices",
    ):
        value = get(key)
        if value is None:
            continue
        if not isinstance(value, torch.Tensor) or value.ndim > 2:
            raise StrictProviderError("invalid_observation", "Expected metadata tensor", key)
        if value.dtype not in (torch.bool, torch.int32, torch.int64):
            raise StrictProviderError("invalid_observation", "Expected integer/bool metadata", key)
        result[key] = {**tensor_observation(value), "values": value.detach().cpu().tolist()}
    return result


@dataclass(frozen=True)
class GDNStateBinding:
    """Bridge-supplied lineage, not something inferred from a tensor address."""

    logical_sequence_id: str
    weight_snapshot_id: str
    processed_tokens: int
    slot: int
    slot_generation: int

    def __post_init__(self):
        for field in ("logical_sequence_id", "weight_snapshot_id"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise StrictProviderError("invalid_observation", "Missing state identity", field)
        for field in ("processed_tokens", "slot", "slot_generation"):
            _nonnegative(getattr(self, field), field)

    def to_dict(self):
        return asdict(self)

    @property
    def logical_boundary_id(self):
        # Slots and their generations are local to a worker, not pairing keys.
        return fingerprint(
            {
                "logical_sequence_id": self.logical_sequence_id,
                "weight_snapshot_id": self.weight_snapshot_id,
                "processed_tokens": self.processed_tokens,
            }
        )


def observe_gdn_states(*, conv_state, recurrent_state, bindings):
    if not isinstance(conv_state, torch.Tensor) or not isinstance(recurrent_state, torch.Tensor):
        raise StrictProviderError("invalid_observation", "Expected actual state tensors")
    if conv_state.ndim != 3 or recurrent_state.ndim != 4:
        raise StrictProviderError("invalid_observation", "Expected conv rank 3/recurrent rank 4")
    if conv_state.device != recurrent_state.device:
        raise StrictProviderError("invalid_observation", "State devices differ")
    if conv_state.shape[0] != recurrent_state.shape[0]:
        raise StrictProviderError("invalid_observation", "State banks have different capacities")
    bindings = tuple(bindings)
    if not bindings or any(not isinstance(item, GDNStateBinding) for item in bindings):
        raise StrictProviderError("unobservable_state", "Bridge state bindings are required")
    if len({item.slot for item in bindings}) != len(bindings):
        raise StrictProviderError("invalid_observation", "Active sequences must own distinct slots")
    for item in bindings:
        if item.slot >= conv_state.shape[0]:
            raise StrictProviderError("invalid_observation", "State slot is out of range")
    return {
        "convolution": tensor_observation(conv_state),
        "recurrent": tensor_observation(recurrent_state),
        "bindings": [item.to_dict() for item in bindings],
        "lineage_evidence": "bridge_declared",
    }
