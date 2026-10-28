# SPDX-License-Identifier: Apache-2.0
"""Collect engine-independent replay bridge state candidates.

Framework adapters may call this collector with their actual bridge/state views.
It does not implement the executable ProviderObserver protocol or install hooks.
"""

from rl_engine.contracts.strict_provider import StrictProviderError
from rl_engine.integrations.common.gdn_provider_observation import (
    implementation_name,
    observe_gdn_states,
)


class ReplayGDNCandidateCollector:
    """Read a replay bridge's explicit state; do not borrow rollout caches."""

    def observe_call(self, bridge, state, *, indices, cu_seqlens, bindings):
        import torch

        if not isinstance(indices, torch.Tensor) or indices.ndim != 1:
            raise StrictProviderError("invalid_observation", "Expected one state slot per sequence")
        if indices.dtype not in (torch.int32, torch.int64):
            raise StrictProviderError("invalid_observation", "State indices must be integer")
        if not isinstance(cu_seqlens, torch.Tensor) or cu_seqlens.ndim != 1:
            raise StrictProviderError("invalid_observation", "Expected sequence boundaries")
        if cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise StrictProviderError("invalid_observation", "Sequence boundaries must be integer")
        slots, boundaries = indices.detach().cpu().tolist(), cu_seqlens.detach().cpu().tolist()
        bindings = tuple(bindings)
        state_facts = observe_gdn_states(
            conv_state=state.convolution, recurrent_state=state.recurrent, bindings=bindings
        )
        if slots != [item.slot for item in bindings]:
            raise StrictProviderError(
                "state_binding_mismatch", "Actual slots differ from bridge lineage"
            )
        if len(boundaries) != len(slots) + 1 or not boundaries or boundaries[0] != 0:
            raise StrictProviderError("invalid_observation", "Invalid packed sequence boundaries")
        if any(end < start for start, end in zip(boundaries, boundaries[1:], strict=False)):
            raise StrictProviderError("invalid_observation", "Sequence boundaries must be ordered")
        return {
            "implementation": implementation_name(bridge),
            "state": state_facts,
            "indices": slots,
            "cu_seqlens": boundaries,
            "evidence_level": "replay_state_candidate",
            "enforcement_ready": False,
        }

    def prepare_call(self, provider, inputs, metadata):
        raise StrictProviderError(
            "unobservable_plan",
            "Replay state observation cannot qualify a GDN native/JIT provider",
        )
