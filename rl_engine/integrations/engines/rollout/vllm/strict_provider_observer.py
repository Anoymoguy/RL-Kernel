# SPDX-License-Identifier: Apache-2.0
"""vLLM GDN candidate collection; no implicit qualification of opaque dispatch."""

from collections.abc import Mapping
from typing import Any

import torch

from rl_engine.contracts.strict_provider import StrictProviderError
from rl_engine.integrations.common.gdn_provider_observation import (
    implementation_name,
    nonnegative_integer,
    tensor_observation,
    observe_gdn_states,
)


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
        result[key] = nonnegative_integer(get(key), key)
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


class VllmGDNCandidateCollector:
    """Inspect a bound instance and the actual state views supplied by its hook.

    This collector does not implement the executable ProviderObserver protocol.
    It intentionally cannot produce an executable strict call:
    the version-specific native/JIT dispatch adapter has not been qualified.
    """

    def observe_bound_provider(self, provider):
        return observe_gdn_instance(provider)

    def observe_call(self, provider, metadata, *, conv_state, recurrent_state, bindings):
        return {
            "provider": observe_gdn_instance(provider),
            "dispatch_metadata": observe_gdn_metadata(metadata),
            "state": observe_gdn_states(
                conv_state=conv_state, recurrent_state=recurrent_state, bindings=bindings
            ),
            "enforcement_ready": False,
        }

    def prepare_call(self, provider, inputs, metadata):
        raise StrictProviderError(
            "unobservable_plan",
            "GDN candidate metadata does not identify an authorized native/JIT dispatch",
        )
