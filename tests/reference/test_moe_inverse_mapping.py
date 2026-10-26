# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from rl_engine.contracts.operators.moe import (
    MoEOutputCombineContractError,
    MoEOutputCombineInputs,
    MoEOutputCombinePlan,
)
from rl_engine.reference.moe import inverse_map_expert_output, inverse_mapping_provenance


def _request(*, index_dtype=torch.int64, expert_output=None, tokens=None, slots=None):
    plan = MoEOutputCombinePlan(
        top_k=2,
        hidden_size=3,
        token_index_dtype=str(index_dtype).removeprefix("torch."),
    )
    if expert_output is None:
        expert_output = torch.tensor(
            [[20, 21, 22], [0, 1, 2], [30, 31, 32], [10, 11, 12]],
            dtype=torch.bfloat16,
        )
    if tokens is None:
        tokens = [1, 0, 1, 0]
    if slots is None:
        slots = [0, 0, 1, 1]
    inputs = MoEOutputCombineInputs(
        expert_output=expert_output,
        token_indices=torch.tensor(tokens, dtype=index_dtype),
        expert_indices=torch.tensor([4, 4, 5, 5], dtype=index_dtype),
        slot_indices=torch.tensor(slots, dtype=index_dtype),
        routing_weights=torch.ones(4, dtype=torch.float32),
    )
    return plan, inputs


def test_inverse_mapping_restores_token_major_slot_order():
    plan, inputs = _request()

    restored = inverse_map_expert_output(plan, inputs, num_tokens=2)

    expected = torch.tensor(
        [[[0, 1, 2], [10, 11, 12]], [[20, 21, 22], [30, 31, 32]]],
        dtype=torch.bfloat16,
    )
    assert torch.equal(restored, expected)


def test_inverse_mapping_supports_non_contiguous_expert_output_and_int32_indices():
    base = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    expert_output = base.t()
    plan, inputs = _request(
        index_dtype=torch.int32,
        expert_output=expert_output,
        tokens=[0, 1, 0, 1],
        slots=[0, 0, 1, 1],
    )

    restored = inverse_map_expert_output(plan, inputs, num_tokens=2)

    assert restored.shape == (2, 2, 3)
    assert restored.is_contiguous()
    assert restored[0, 0].tolist() == [0, 4, 8]


@pytest.mark.parametrize(
    ("tokens", "slots", "num_tokens", "message"),
    [
        ([0, 0, 0, 1], [0, 0, 1, 1], 2, "duplicate"),
        ([0, 0, 1, 1], [0, 1, 0, 1], 3, "slot count"),
        ([0, 0, 1, 2], [0, 1, 0, 1], 2, "outside"),
    ],
)
def test_inverse_mapping_rejects_invalid_mapping(tokens, slots, num_tokens, message):
    plan, inputs = _request(tokens=tokens, slots=slots)

    with pytest.raises(MoEOutputCombineContractError, match=message):
        inverse_map_expert_output(plan, inputs, num_tokens=num_tokens)


def test_inverse_mapping_provenance_records_canonical_layout():
    plan, inputs = _request()

    provenance = inverse_mapping_provenance(plan, inputs, num_tokens=2)

    assert provenance["inverse_mapping"] == "token_major_slot_minor"
    assert provenance["restored_shape"] == [2, 2, 3]

