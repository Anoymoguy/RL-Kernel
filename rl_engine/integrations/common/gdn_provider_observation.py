# SPDX-License-Identifier: Apache-2.0
"""Collect shared GDN state facts without claiming a kernel was authorized.

The bridge owns state lineage. These observations are candidates: a Python
callable name does not establish the native/JIT artifact actually launched.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from rl_engine.contracts.strict_provider import StrictProviderError, fingerprint


def implementation_name(value: Any) -> str:
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def nonnegative_integer(value: Any, field: str) -> int:
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
            nonnegative_integer(getattr(self, field), field)

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
