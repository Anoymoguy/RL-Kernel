# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""P6 output-combine request preparation.

The first P6 implementation intentionally stops at a validated request. It
does not perform inverse permutation or reduction; later P6 tasks consume the
same plan for those operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rl_engine.contracts.operators.moe import (
    MoEOutputCombineContractError,
    MoEOutputCombineInputs,
    MoEOutputCombinePlan,
)


@dataclass(frozen=True)
class PreparedMoEOutputCombine:
    """Validated P6 request shared by reference and future ROCm backends."""

    plan: MoEOutputCombinePlan
    inputs: MoEOutputCombineInputs
    provenance: dict[str, Any]


def prepare_output_combine(
    plan: MoEOutputCombinePlan,
    inputs: MoEOutputCombineInputs,
) -> PreparedMoEOutputCombine:
    """Validate and freeze the semantic boundary before backend execution."""

    if not isinstance(plan, MoEOutputCombinePlan):
        raise MoEOutputCombineContractError("plan must be a MoEOutputCombinePlan")
    if not isinstance(inputs, MoEOutputCombineInputs):
        raise MoEOutputCombineContractError("inputs must be a MoEOutputCombineInputs")
    if plan.plan_version != "p6-output-combine.v1":
        raise MoEOutputCombineContractError(
            f"unsupported P6 plan version: {plan.plan_version}"
        )

    provenance = inputs.provenance(plan)
    return PreparedMoEOutputCombine(plan=plan, inputs=inputs, provenance=provenance)


__all__ = ["PreparedMoEOutputCombine", "prepare_output_combine"]

