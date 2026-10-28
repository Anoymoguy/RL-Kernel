# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from rl_engine.contracts.strict_provider import StrictProviderError
from rl_engine.integrations.common.gdn_provider_observation import (
    GDNStateBinding,
    observe_gdn_states,
)
from rl_engine.integrations.common.gdn_replay_observation import (
    ReplayGDNCandidateCollector,
)


def test_replay_observes_its_own_explicit_state_and_packing():
    state = SimpleNamespace(
        convolution=torch.zeros(24, 16, 3),
        recurrent=torch.zeros(24, 2, 4, 4),
    )
    observation = ReplayGDNCandidateCollector().observe_call(
        SimpleNamespace(),
        state,
        indices=torch.tensor([23]),
        cu_seqlens=torch.tensor([0, 4]),
        bindings=[GDNStateBinding("sequence-A", "weights-1", 0, 23, 1)],
    )
    assert observation["indices"] == [23]
    assert observation["state"]["recurrent"]["shape"] == [24, 2, 4, 4]
    assert observation["enforcement_ready"] is False


@pytest.mark.parametrize("slots,boundaries", [([1], [0, 4]), ([23], [1, 4]), ([23], [0, -1])])
def test_replay_rejects_bad_actual_slot_mapping_or_packing(slots, boundaries):
    state = SimpleNamespace(
        convolution=torch.zeros(24, 16, 3),
        recurrent=torch.zeros(24, 2, 4, 4),
    )
    with pytest.raises(StrictProviderError):
        ReplayGDNCandidateCollector().observe_call(
            SimpleNamespace(),
            state,
            indices=torch.tensor(slots),
            cu_seqlens=torch.tensor(boundaries),
            bindings=[GDNStateBinding("sequence-A", "weights-1", 0, 23, 1)],
        )


def test_replay_candidate_cannot_authorize_native_dispatch():
    with pytest.raises(StrictProviderError) as exc:
        ReplayGDNCandidateCollector().prepare_call(SimpleNamespace(), {}, {})
    assert exc.value.code == "unobservable_plan"


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
