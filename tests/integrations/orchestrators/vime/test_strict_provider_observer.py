# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from rl_engine.contracts.strict_provider import StrictProviderError
from rl_engine.integrations.common.gdn_provider_observation import GDNStateBinding
from rl_engine.integrations.orchestrators.vime.providers.strict_provider_observer import (
    VimeGDNObserver,
)


def test_replay_observes_its_own_explicit_state_and_packing():
    state = SimpleNamespace(
        convolution=torch.zeros(24, 16, 3),
        recurrent=torch.zeros(24, 2, 4, 4),
    )
    observation = VimeGDNObserver().observe_call(
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
        VimeGDNObserver().observe_call(
            SimpleNamespace(),
            state,
            indices=torch.tensor(slots),
            cu_seqlens=torch.tensor(boundaries),
            bindings=[GDNStateBinding("sequence-A", "weights-1", 0, 23, 1)],
        )


def test_replay_candidate_cannot_authorize_native_dispatch():
    with pytest.raises(StrictProviderError) as exc:
        VimeGDNObserver().prepare_call(SimpleNamespace(), {}, {})
    assert exc.value.code == "unobservable_plan"
