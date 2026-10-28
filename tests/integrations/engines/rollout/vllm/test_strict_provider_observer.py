# SPDX-License-Identifier: Apache-2.0
"""Use real CPU tensor views to check GDN candidate evidence, not GPU parity."""

from types import SimpleNamespace

import pytest
import torch

from rl_engine.contracts.strict_provider import StrictProviderError
from rl_engine.integrations.common.gdn_provider_observation import (
    GDNStateBinding,
    observe_gdn_states,
)
from rl_engine.integrations.engines.rollout.vllm.strict_provider_observer import VllmGDNObserver


def provider():
    return SimpleNamespace(
        num_k_heads=16,
        num_v_heads=32,
        head_k_dim=128,
        head_v_dim=128,
        conv_kernel_size=4,
        tp_size=1,
        gqa_interleaved_layout=True,
        gdn_prefill_backend="fla",
        gdn_decode_kernel="triton",
    )


def test_reads_effective_provider_attributes_and_stays_candidate():
    actual = provider()
    observer = VllmGDNObserver()
    first = observer.observe_bound_provider(actual)
    actual.gdn_decode_kernel = "other-qualified-candidate"
    second = observer.observe_bound_provider(actual)
    assert first["effective_config"]["gdn_decode_kernel"] == "triton"
    assert second["effective_config"]["gdn_decode_kernel"] != "triton"
    assert second["enforcement_ready"] is False
    with pytest.raises(StrictProviderError) as exc:
        observer.prepare_call(actual, {}, {})
    assert exc.value.code == "unobservable_plan"


def test_observes_actual_state_view_and_copies_small_metadata():
    conv = torch.zeros(8, 3, 16).transpose(-1, -2)
    recurrent = torch.zeros(8, 32, 128, 128)
    slots = torch.tensor([7])
    metadata = SimpleNamespace(
        num_prefills=0,
        num_decodes=1,
        num_actual_tokens=1,
        non_spec_state_indices_tensor=slots,
        non_spec_query_start_loc=torch.tensor([0, 1]),
    )
    result = VllmGDNObserver().observe_call(
        provider(),
        metadata,
        conv_state=conv,
        recurrent_state=recurrent,
        bindings=[GDNStateBinding("sequence-A", "weights-1", 4, 7, 0)],
    )
    slots[0] = 2
    assert result["dispatch_metadata"]["phase"] == "decode"
    assert result["dispatch_metadata"]["non_spec_state_indices_tensor"]["values"] == [7]
    assert result["state"]["convolution"]["shape"] == [8, 16, 3]
    assert result["state"]["convolution"]["stride"] == list(conv.stride())
    assert result["state"]["convolution"]["contiguous"] is False
    assert result["enforcement_ready"] is False


def test_logical_boundary_allows_different_local_slots_and_generations():
    rollout = GDNStateBinding("sequence-A", "weights-1", 4, 7, 2)
    replay = GDNStateBinding("sequence-A", "weights-1", 4, 23, 9)
    assert rollout.logical_boundary_id == replay.logical_boundary_id
    assert (
        rollout.logical_boundary_id
        != GDNStateBinding("sequence-A", "weights-1", 3, 23, 9).logical_boundary_id
    )
    assert (
        rollout.logical_boundary_id
        != GDNStateBinding("sequence-B", "weights-1", 4, 23, 9).logical_boundary_id
    )
    assert (
        rollout.logical_boundary_id
        != GDNStateBinding("sequence-A", "weights-2", 4, 23, 9).logical_boundary_id
    )


@pytest.mark.parametrize("slots", [(7, 7), (7, 8)])
def test_state_slots_are_distinct_and_in_range(slots):
    bindings = [GDNStateBinding(str(i), "weights-1", 4, slot, 0) for i, slot in enumerate(slots)]
    with pytest.raises(StrictProviderError):
        observe_gdn_states(
            conv_state=torch.zeros(8, 16, 3),
            recurrent_state=torch.zeros(8, 2, 4, 4),
            bindings=bindings,
        )


def test_missing_instance_fields_are_not_inferred_from_model_name():
    with pytest.raises(StrictProviderError) as exc:
        VllmGDNObserver().observe_bound_provider(SimpleNamespace())
    assert exc.value.code == "unobservable_binding"
