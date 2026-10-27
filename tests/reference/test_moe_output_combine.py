# SPDX-License-Identifier: Apache-2.0
"""P6 contracts, request preparation, inverse mapping and explicit dispatch."""

import pytest
import torch

from rl_engine.contracts.operators.moe import (
    MoEOutputCombineContractError,
    MoEOutputCombineInputs,
    MoEOutputCombinePlan,
)
from rl_engine.reference.moe import (
    MoEOutputCombineReferenceOp,
    inverse_map_expert_output,
    inverse_mapping_provenance,
    prepare_output_combine,
)
from rl_engine.runtime.registry import kernel_registry
from rl_engine.runtime.semantic_registry import OperatorRequirements



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


def test_plan_freezes_nested_metadata_without_reordering_sequences():
    plan = _plan(
        metadata={"ordered": ["second", "first"], "nested": (({"value": 1},),)}
    )

    assert plan.to_dict()["metadata"]["ordered"] == ["second", "first"]
    with pytest.raises(TypeError):
        plan.metadata["nested"][0][0]["value"] = 2


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


def test_inputs_validate_against_plan():
    plan = _plan()
    inputs = _inputs(plan)

    inputs.validate(plan)
    provenance = inputs.provenance(plan)
    assert provenance["semantic_op"] == "moe_output_combine"
    assert provenance["plan_fingerprint"] == plan.fingerprint


def test_weights_must_be_explicitly_owned_by_one_boundary():
    plan = _plan(expert_weight_applied=True)
    inputs = _inputs(plan)

    with pytest.raises(MoEOutputCombineContractError, match="already applied"):
        inputs.validate(plan)


def test_schema_and_shape_mismatches_fail_closed():
    plan = _plan()
    inputs = _inputs(plan, mapping_schema_version="p4-token-slot.v0")

    with pytest.raises(MoEOutputCombineContractError, match="mapping schema"):
        inputs.validate(plan)
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
    with pytest.raises(MoEOutputCombineContractError, match="shared_output must have shape"):
        _inputs(
            plan,
            num_tokens=3,
            shared_output=torch.zeros(4, plan.hidden_size, dtype=torch.bfloat16),
        ).validate(plan)
    with pytest.raises(MoEOutputCombineContractError, match="num_tokens is required"):
        _inputs(
            plan,
            shared_output=torch.zeros(2, plan.hidden_size, dtype=torch.bfloat16),
        ).validate(plan)


def test_external_residual_uses_declared_token_shape():
    plan = _plan(residual_mode="external")
    valid = _inputs(
        plan,
        num_tokens=3,
        residual=torch.zeros(3, plan.hidden_size, dtype=torch.bfloat16),
    )
    valid.validate(plan)

    with pytest.raises(MoEOutputCombineContractError, match="residual must have shape"):
        _inputs(
            plan,
            num_tokens=3,
            residual=torch.zeros(2, plan.hidden_size, dtype=torch.bfloat16),
        ).validate(plan)
    with pytest.raises(MoEOutputCombineContractError, match="residual is required"):
        _inputs(plan, num_tokens=3).validate(plan)
    with pytest.raises(MoEOutputCombineContractError, match="residual must be omitted"):
        excluded = _plan()
        _inputs(
            excluded,
            residual=torch.zeros(2, excluded.hidden_size, dtype=torch.bfloat16),
        ).validate(excluded)
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
        num_tokens=2,
    )

    prepared = prepare_output_combine(plan, inputs)

    assert prepared.provenance["semantic_op"] == "moe_output_combine"
    assert prepared.provenance["producer_backend"] == "test.reference"
    assert prepared.provenance["producer_build_fingerprint"] == "test-build-v1"
    assert prepared.provenance["expert_output_stride"] == [3, 1]
    assert prepared.provenance["num_tokens"] == 2


def test_reference_preparation_requires_explicit_token_count():
    plan = MoEOutputCombinePlan(top_k=1, hidden_size=3)
    inputs = MoEOutputCombineInputs(
        expert_output=torch.zeros(2, 3, dtype=torch.bfloat16),
        token_indices=torch.tensor([0, 1], dtype=torch.int64),
        expert_indices=torch.tensor([4, 4], dtype=torch.int64),
        slot_indices=torch.tensor([0, 0], dtype=torch.int64),
        routing_weights=torch.ones(2, dtype=torch.float32),
    )

    with pytest.raises(MoEOutputCombineContractError, match="explicit num_tokens"):
        prepare_output_combine(plan, inputs)


def test_reference_preparation_rejects_non_cpu_inputs():
    plan = _plan()
    meta = torch.empty(4, plan.hidden_size, dtype=torch.bfloat16, device="meta")
    inputs = _inputs(
        plan,
        expert_output=meta,
        token_indices=torch.zeros(4, dtype=torch.int64, device="meta"),
        expert_indices=torch.zeros(4, dtype=torch.int64, device="meta"),
        slot_indices=torch.tensor([0, 1, 0, 1], dtype=torch.int64, device="meta"),
        routing_weights=torch.ones(4, dtype=torch.float32, device="meta"),
    )

    with pytest.raises(MoEOutputCombineContractError, match="on CPU"):
        prepare_output_combine(plan, inputs)
    with pytest.raises(MoEOutputCombineContractError, match="meta tensors"):
        inputs.validate(plan)


def _request(*, index_dtype=torch.int64, expert_output=None, tokens=None, slots=None, run_id="run-1"):
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
        mapping_run_id=run_id,
    )
    return plan, inputs


def test_inverse_mapping_restores_token_major_slot_order():
    plan, inputs = _request()

    restored = inverse_map_expert_output(plan, inputs, num_tokens=2, num_experts=8)

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

    restored = inverse_map_expert_output(plan, inputs, num_tokens=2, num_experts=8)

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

    provenance = inverse_mapping_provenance(
        plan, inputs, num_tokens=2, num_experts=8, expected_mapping_run_id="run-1"
    )

    assert provenance["inverse_mapping"] == "token_major_slot_minor"
    assert provenance["restored_shape"] == [2, 2, 3]


def test_inverse_mapping_rejects_wrong_run_id_and_expert_overflow():
    plan, inputs = _request()

    with pytest.raises(MoEOutputCombineContractError, match="run id"):
        inverse_map_expert_output(
            plan, inputs, num_tokens=2, expected_mapping_run_id="other-run"
        )

    bad_expert = inputs.expert_indices.clone()
    bad_expert[0] = 8
    bad_inputs = inputs.__class__(
        expert_output=inputs.expert_output,
        token_indices=inputs.token_indices,
        expert_indices=bad_expert,
        slot_indices=inputs.slot_indices,
        routing_weights=inputs.routing_weights,
        mapping_run_id=inputs.mapping_run_id,
    )
    with pytest.raises(MoEOutputCombineContractError, match="expert index"):
        inverse_map_expert_output(plan, bad_inputs, num_tokens=2, num_experts=8)


def test_p6_reference_backend_is_explicitly_registered():
    descriptor = kernel_registry.semantic.backend_descriptor(
        "moe_output_combine", "rlkernel.reference.moe_output_combine.v1"
    )

    assert descriptor is not None
    assert descriptor.is_strictly_observable
    assert descriptor.fallback_policy.value == "error"


def test_p6_reference_backend_can_be_resolved_and_instantiated():
    session = kernel_registry.semantic.session()
    resolution = session.resolve(
        semantic_op="moe_output_combine",
        requested_backend="rlkernel.reference.moe_output_combine.v1",
        target="training",
        requirements=OperatorRequirements(device="cpu", dtype="bfloat16"),
    )
    instance = session.instantiate(resolution)

    assert isinstance(instance, MoEOutputCombineReferenceOp)
    assert instance.backend_id == "rlkernel.reference.moe_output_combine.v1"
