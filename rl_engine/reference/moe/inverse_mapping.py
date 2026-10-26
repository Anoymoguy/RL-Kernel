# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Reference inverse mapping for the P6 expert-output boundary."""

from __future__ import annotations

from typing import Any

import torch

from rl_engine.contracts.operators.moe import (
    MoEOutputCombineContractError,
    MoEOutputCombineInputs,
    MoEOutputCombinePlan,
)


def inverse_map_expert_output(
    plan: MoEOutputCombinePlan,
    inputs: MoEOutputCombineInputs,
    *,
    num_tokens: int,
) -> torch.Tensor:
    """Restore shuffled expert slots to ``[token, slot, hidden]`` order.

    This is a correctness-first reference implementation.  It validates the
    complete ``(token, slot)`` domain on the host, then gathers rows in a
    canonical token-major order.  A future HIP/Triton implementation must
    preserve this exact mapping and failure contract.
    """

    if not isinstance(num_tokens, int) or num_tokens <= 0:
        raise MoEOutputCombineContractError("num_tokens must be a positive integer")
    inputs.validate(plan)

    token_indices = inputs.token_indices.detach().cpu().tolist()
    slot_indices = inputs.slot_indices.detach().cpu().tolist()
    expected_slots = num_tokens * plan.top_k
    if len(token_indices) != expected_slots:
        raise MoEOutputCombineContractError(
            "expert output slot count must equal num_tokens * top_k"
        )

    order: dict[tuple[int, int], int] = {}
    for row, (token, slot) in enumerate(zip(token_indices, slot_indices, strict=True)):
        token = int(token)
        slot = int(slot)
        if token < 0 or token >= num_tokens:
            raise MoEOutputCombineContractError(
                f"token index {token} is outside [0, {num_tokens})"
            )
        if slot < 0 or slot >= plan.top_k:
            raise MoEOutputCombineContractError(
                f"slot index {slot} is outside [0, {plan.top_k})"
            )
        key = (token, slot)
        if key in order:
            raise MoEOutputCombineContractError(
                f"duplicate expert result for token={token}, slot={slot}"
            )
        order[key] = row

    expected_keys = {
        (token, slot)
        for token in range(num_tokens)
        for slot in range(plan.top_k)
    }
    missing = sorted(expected_keys.difference(order))
    if missing:
        preview = ", ".join(f"({token},{slot})" for token, slot in missing[:4])
        suffix = "..." if len(missing) > 4 else ""
        raise MoEOutputCombineContractError(f"missing expert results: {preview}{suffix}")

    canonical_rows = [order[(token, slot)] for token in range(num_tokens) for slot in range(plan.top_k)]
    row_index = torch.tensor(canonical_rows, dtype=torch.int64, device=inputs.expert_output.device)
    restored = inputs.expert_output.index_select(0, row_index)
    return restored.reshape(num_tokens, plan.top_k, plan.hidden_size)


def inverse_mapping_provenance(
    plan: MoEOutputCombinePlan,
    inputs: MoEOutputCombineInputs,
    *,
    num_tokens: int,
) -> dict[str, Any]:
    """Return provenance for a successful inverse-map reference execution."""

    output = inverse_map_expert_output(plan, inputs, num_tokens=num_tokens)
    provenance = inputs.provenance(plan)
    provenance.update(
        {
            "inverse_mapping": "token_major_slot_minor",
            "num_tokens": num_tokens,
            "restored_shape": list(output.shape),
            "restored_stride": list(output.stride()),
        }
    )
    return provenance


__all__ = ["inverse_map_expert_output", "inverse_mapping_provenance"]

