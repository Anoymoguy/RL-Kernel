# SPDX-License-Identifier: Apache-2.0

from rl_engine.reference.moe import MoEOutputCombineReferenceOp
from rl_engine.runtime.registry import kernel_registry
from rl_engine.runtime.semantic_registry import OperatorRequirements


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
