# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from rl_engine.contracts.operators.moe import (
    MoEOutputCombineContractError,
    MoEOutputCombineInputs,
    MoEOutputCombinePlan,
)
from rl_engine.reference.moe import prepare_output_combine


def _plan(**kwargs):
    values = {"top_k": 2, "hidden_size": 4}
    values.update(kwargs)
    return MoEOutputCombinePlan(**values)


def _inputs(plan, **kwargs):
    values = {
        "expert_output": torch.zeros(4, plan.hidden_size, dtype=torch.bfloat16),
        "token_indices": torch.tensor([0, 0, 1, 1], dtype=torch.int64),
        "expert_indices": torch.tensor([2, 3, 2, 3], dtype=torch.int64),
        "slot_indices": torch.tensor([0, 1, 0, 1], dtype=torch.int64),
        "routing_weights": torch.ones(4, dtype=torch.float32),
    }
    values.update(kwargs)
    return MoEOutputCombineInputs(**values)


def test_plan_has_stable_fingerprint_and_fixed_slot_order():
    first = _plan()
    second = _plan()
    assert first.merge_order == (0, 1)
    assert first.fingerprint == second.fingerprint
    assert first.to_dict()["accumulation_dtype"] == "float32"


def test_plan_rejects_bool_dimensions_and_mutable_metadata():
    with pytest.raises(MoEOutputCombineContractError):
        _plan(top_k=True)
    plan = _plan(metadata={"aux": [torch.bfloat16]})
    fingerprint = plan.fingerprint
    assert plan.fingerprint == fingerprint
    with pytest.raises(TypeError):
        plan.metadata["new"] = "value"
    with pytest.raises(MoEOutputCombineContractError):
        _plan(expert_weight_applied=1)
    with pytest.raises(MoEOutputCombineContractError):
        _plan(merge_order=(1.0, 0.0))


def test_plan_rejects_non_fixed_merge_order():
    with pytest.raises(MoEOutputCombineContractError, match="merge_order"):
        _plan(merge_order=(1, 0))


def test_inputs_validate_against_plan_and_record_shape_provenance():
    plan = _plan()
    inputs = _inputs(plan)
    inputs.validate(plan)
    provenance = inputs.provenance(plan)
    assert provenance["semantic_op"] == "moe_output_combine"
    assert provenance["plan_fingerprint"] == plan.fingerprint
    assert provenance["num_slots"] == 4


def test_weights_must_be_explicitly_owned_by_one_boundary():
    plan = _plan(expert_weight_applied=True)
    with pytest.raises(MoEOutputCombineContractError, match="already applied"):
        _inputs(plan).validate(plan)
    provenance = _inputs(plan, routing_weights=None).provenance(plan)
    assert provenance["plan_fingerprint"] == plan.fingerprint


def test_schema_and_shape_mismatches_fail_closed():
    plan = _plan()
    with pytest.raises(MoEOutputCombineContractError, match="mapping schema"):
        _inputs(plan, mapping_schema_version="p4-token-slot.v0").validate(plan)
    with pytest.raises(MoEOutputCombineContractError, match="one float32 value per expert output slot"):
        _inputs(plan, routing_weights=torch.ones(3)).validate(plan)
    with pytest.raises(MoEOutputCombineContractError, match="slot_indices must be within top_k"):
        _inputs(plan, slot_indices=torch.tensor([0, 1, 0, 2])).validate(plan)


def test_separate_shared_output_uses_declared_token_shape():
    plan = _plan(shared_expert_mode="separate")
    inputs = _inputs(
        plan,
        num_tokens=3,
        shared_output=torch.zeros(3, plan.hidden_size, dtype=torch.bfloat16),
    )
    inputs.validate(plan)
    with pytest.raises(MoEOutputCombineContractError, match="num_tokens"):
        _inputs(
            plan,
            num_tokens=3,
            shared_output=torch.zeros(4, plan.hidden_size, dtype=torch.bfloat16),
        ).validate(plan)


def test_prepare_output_combine_returns_versioned_provenance():
    plan = MoEOutputCombinePlan(top_k=1, hidden_size=3)
    inputs = MoEOutputCombineInputs(
        expert_output=torch.zeros(2, 3, dtype=torch.bfloat16),
        token_indices=torch.tensor([0, 1], dtype=torch.int64),
        expert_indices=torch.tensor([4, 4], dtype=torch.int64),
        slot_indices=torch.tensor([0, 0], dtype=torch.int64),
        routing_weights=torch.ones(2, dtype=torch.float32),
        producer_backend="test.reference",
        producer_build_fingerprint="test-build-v1",
    )

    prepared = prepare_output_combine(plan, inputs)

    assert prepared.provenance["semantic_op"] == "moe_output_combine"
    assert prepared.provenance["producer_backend"] == "test.reference"
    assert prepared.provenance["producer_build_fingerprint"] == "test-build-v1"
    assert prepared.provenance["expert_output_stride"] == [3, 1]

