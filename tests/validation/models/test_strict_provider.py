# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import struct

import pytest

from rl_engine.contracts.strict_provider import (
    KernelPlan,
    ProviderLock,
    SnapshotActivationReceipt,
    StrictProviderError,
    fingerprint,
)
from rl_engine.validation.models.strict_provider import (
    ARTIFACTS_SCHEMA,
    CASES_SCHEMA,
    OUTPUT_SCHEMA,
    SUMMARY_SCHEMA,
    artifact_reference,
    build_expectation,
    validate_run,
)


def make_plan():
    return KernelPlan.from_dict(
        {
            "node": "gdn",
            "phase": "prefill",
            "implementation": "tests.MockGDN",
            "entrypoint": "tests.MockGDN.recurrent",
            "code": {
                "source_inventory_sha256": "a" * 64,
                "binary_inventory_sha256": "b" * 64,
                "compiled_kernel_id": "cpu_mock_v1",
                "compile_key": "fixed_shape_v1",
            },
            "io": {
                "input_dtype": "float64",
                "output_dtype": "float64",
                "layout": "row_major",
                "shape_class": "two_rows",
                "workspace_policy": "none",
            },
            "numerics": {
                "accumulator_dtype": "float64",
                "split_policy": "disabled",
                "split_count": 1,
                "reduction_spec": "left_fold_v1",
                "cast_spec": "no_cast",
                "atomic_policy": "forbidden",
            },
            "launch": {"tile_config": [1], "graph_state": "eager", "tuning": "fixed"},
        }
    )


class Bundle:
    """Trusted predeclared runner plus separately persisted mock worker evidence."""

    def __init__(self, root: Path):
        self.root = root
        self.directory = root / "run1" / "attempt1"
        self.directory.mkdir(parents=True)
        self.original = self.directory / "provider.bin"
        self.original.write_bytes(b"original mock provider")
        original_ref = artifact_reference(self.original, root)
        self.artifacts_id = fingerprint(
            [
                {
                    "path": "provider.bin",
                    "size": original_ref["size_bytes"],
                    "sha256": original_ref["sha256"],
                }
            ]
        )
        plan = make_plan().to_dict()
        plan["code"]["source_inventory_sha256"] = self.artifacts_id
        plan["code"]["binary_inventory_sha256"] = self.artifacts_id
        plan.pop("plan_id")
        plan.pop("alignment_id")
        self.plan = KernelPlan.from_dict(plan)
        binding = {"implementation": self.plan.implementation, "artifacts_ref": self.artifacts_id}
        self.lock = ProviderLock.from_dict(
            {
                "schema_version": "rl_kernel.strict_provider.v1",
                "profile_id": "cpu_mock",
                "scope": "component.gdn",
                "model": {
                    "model_id": "mock_model",
                    "revision": "revision1",
                    "config_sha256": "1" * 64,
                    "weight_index_sha256": "2" * 64,
                    "shard_inventory_sha256": "3" * 64,
                },
                "platform": {
                    "kind": "cpu",
                    "gpu_arch": "cpu_mock",
                    "torch_build": "cpu_mock",
                    "hip_runtime": "not_applicable",
                    "driver_identity": "cpu_mock",
                    "triton_build": "not_applicable",
                },
                "execution": {
                    "topology": {"tp": 1, "ep": 1, "cp": 1, "pp": 1},
                    "activation_dtype": "float64",
                    "graph_mode": "eager",
                    "tuning_policy": "fixed",
                    "fallback_policy": "error",
                    "runtime_settings": {},
                },
                "nodes": {
                    "gdn": {
                        "semantics_ref": "mock_gdn_v1",
                        "bindings": {
                            "rollout": binding,
                            "replay": binding,
                        },
                        "allowed_plans": {
                            "rollout": [self.plan.plan_id],
                            "replay": [self.plan.plan_id],
                        },
                        "alignment": {"policy": "identical_plan"},
                    }
                },
            }
        )
        self.inventory = {"0": ["gdn"], "1": ["gdn"]}
        self.scope = {
            "scope_id": "component.gdn",
            "kind": "component",
            "required_nodes": ["gdn"],
            "inventory_fingerprint": fingerprint(self.inventory),
        }
        self.case = {
            "case_id": "prompt_boundary",
            "comparison_context": {
                "tokens": [1, 2],
                "positions": [0, 1],
                "mask": [1, 1],
                "state_boundary": "after_token_1",
                "logprob_semantics": "raw_policy",
            },
            "required_calls": [
                {
                    "node": "gdn",
                    "phase": "prefill",
                    "layer_id": str(layer),
                    "call_id": "prompt_A",
                    "role": role,
                    "rank": 0,
                    "count": 1,
                }
                for role in ("rollout", "replay")
                for layer in range(2)
            ],
            "output_names": ["recurrent_state"],
            "numerical_policy": "json_exact",
        }
        self.roster = [
            {"worker_id": role + "0", "role": role, "rank": 0} for role in ("rollout", "replay")
        ]
        self.artifact_inventory = {
            "schema_version": ARTIFACTS_SCHEMA,
            **self.context(),
            "inventory_sha256": self.artifacts_id,
            "files": [{"logical_path": "provider.bin", "reference": original_ref}],
        }
        inventory_ref = self.write("artifact_inventory.json", self.artifact_inventory)
        self.expectation = build_expectation(
            self.lock,
            self.scope,
            [self.case],
            self.roster,
            self.inventory,
            run_id="run1",
            attempt="attempt1",
            weight_snapshot_id="snapshot1",
            identity_artifacts=[inventory_ref],
        )
        self.summaries = []
        self.outputs = []
        for worker in self.roster:
            observation = {
                **worker,
                "model": self.lock.to_dict()["model"],
                "platform": self.lock.to_dict()["platform"],
                "execution": self.lock.to_dict()["execution"],
                "bindings": {"gdn": binding},
                "weight_snapshot_id": "snapshot1",
            }
            receipt = SnapshotActivationReceipt.from_dict(
                {
                    "snapshot_id": "snapshot1",
                    "generation": 0,
                    "worker_id": worker["worker_id"],
                    "expected_shards": ["weight0"],
                    "verified_shards": ["weight0"],
                    "required_refresh": [],
                    "refreshed_components": [],
                    "content_digest": "d" * 64,
                    "completed": True,
                    "evidence_kind": "worker_load",
                }
            ).to_dict()
            summary = {
                "schema_version": SUMMARY_SCHEMA,
                **self.context(),
                **worker,
                "state": "finished",
                "snapshot_generation": 0,
                "snapshot_history": [receipt],
                "bindings": {"gdn": binding},
                "observations": {"gdn": observation},
                "calls": [],
            }
            for layer in range(2):
                summary["calls"].append(
                    {
                        "context": {
                            **{k: v for k, v in self.context().items() if k != "lock_id"},
                            "case_id": self.case["case_id"],
                            "node": "gdn",
                            "phase": "prefill",
                            "layer_id": str(layer),
                            "call_id": "prompt_A",
                            "role": worker["role"],
                            "worker_id": worker["worker_id"],
                        },
                        "plan": self.plan.to_dict(),
                        "plan_id": self.plan.plan_id,
                        "alignment_id": self.plan.alignment_id,
                        "snapshot_generation": 0,
                        "runtime_id": f"runtime_{worker['role']}_{layer}",
                        "call_ticket_id": f"ticket_{worker['role']}_{layer}",
                        "status": "completed",
                        "states": ["prepared", "authorized", "launched", "completed"],
                        "completion": {"kind": "synchronous", "status": "completed"},
                    }
                )
            self.summaries.append(summary)
            self.outputs.append(
                {
                    "schema_version": OUTPUT_SCHEMA,
                    **self.context(),
                    **worker,
                    "case_id": self.case["case_id"],
                    "comparison_context": deepcopy(self.case["comparison_context"]),
                    "outputs": {
                        "recurrent_state": {"dtype": "float64", "shape": [2], "data": [1.25, -0.0]}
                    },
                }
            )
        self.report = {
            "schema_version": CASES_SCHEMA,
            **self.context(),
            "cases": [
                {
                    "case_id": self.case["case_id"],
                    "outcome": "passed",
                    "tests": 1,
                    "failures": 0,
                    "errors": 0,
                    "skipped": 0,
                }
            ],
        }
        self.save()

    def context(self):
        return {
            "lock_id": self.lock.lock_id,
            "run_id": "run1",
            "attempt": "attempt1",
            "weight_snapshot_id": "snapshot1",
        }

    def write(self, name, payload):
        path = self.directory / name
        path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")
        return artifact_reference(path, self.root)

    def save(self, *, update_footer=True):
        for summary in self.summaries:
            if update_footer:
                summary["footer"] = {"complete": True, "call_count": len(summary["calls"])}
            summary.pop("summary_id", None)
            for call in summary["calls"]:
                call["context_id"] = fingerprint(call["context"])
            summary["summary_id"] = fingerprint(summary)
        self.summary_refs = [
            self.write(f"worker_{i}.json", s) for i, s in enumerate(self.summaries)
        ]
        self.output_refs = [self.write(f"output_{i}.json", o) for i, o in enumerate(self.outputs)]
        self.report_refs = [self.write("cases.json", self.report)]

    def validate(self, *, numerical=True):
        return validate_run(
            self.lock,
            self.expectation,
            self.summary_refs,
            self.root,
            output_artifacts=self.output_refs if numerical else [],
            case_results=self.report_refs,
            require_numerical=numerical,
        )

    def cli_input(self):
        return {
            "lock": self.lock.to_dict(),
            "expectation": self.expectation,
            "worker_summaries": self.summary_refs,
            "output_artifacts": self.output_refs,
            "case_results": self.report_refs,
        }


@pytest.fixture
def bundle(tmp_path):
    return Bundle(tmp_path)


def test_nonempty_actual_outputs_and_manifest_status_are_separate(bundle):
    result = bundle.validate()
    assert result["manifest_enforcement"] == result["numerical_parity"] == "passed"
    assert result["compared_elements"] == 2
    assert result["call_count"] == 4
    assert result["hardware_validation"] == "not_evaluated"
    result = bundle.validate(numerical=False)
    assert result["manifest_enforcement"] == "passed"
    assert result["numerical_parity"] == "not_evaluated"
    assert result["compared_elements"] == 0


def test_independent_coverage_catches_missing_layer(bundle):
    for summary in bundle.summaries:
        summary["calls"] = summary["calls"][:1]
    bundle.save()
    with pytest.raises(StrictProviderError, match="multiplicities"):
        bundle.validate()


def test_expectation_cannot_omit_an_entire_inventory_layer(bundle):
    expectation = deepcopy(bundle.expectation)
    expectation["cases"][0]["required_calls"] = [
        c for c in expectation["cases"][0]["required_calls"] if c["layer_id"] == "0"
    ]
    expectation.pop("expectation_id")
    expectation["expectation_id"] = fingerprint(expectation)
    with pytest.raises(StrictProviderError, match="omits a selected"):
        validate_run(bundle.lock, expectation, bundle.summary_refs, bundle.root)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "rank", "phase", "duplicate_call"])
def test_worker_roster_multiplicity_and_branch_coverage(bundle, mutation):
    if mutation == "missing":
        bundle.summaries.pop()
    elif mutation == "duplicate":
        bundle.summaries.append(deepcopy(bundle.summaries[0]))
    elif mutation == "rank":
        bundle.summaries[0]["rank"] = 1
    elif mutation == "phase":
        bundle.summaries[0]["calls"][0]["context"]["phase"] = "decode"
    else:
        bundle.summaries[0]["calls"].append(deepcopy(bundle.summaries[0]["calls"][0]))
    bundle.save()
    with pytest.raises(StrictProviderError):
        bundle.validate()


@pytest.mark.parametrize("field", ["lock_id", "run_id", "attempt", "weight_snapshot_id"])
def test_stale_execution_context_is_rejected(bundle, field):
    bundle.summaries[0][field] = "previous_execution"
    bundle.save()
    with pytest.raises(StrictProviderError) as exc:
        bundle.validate()
    assert exc.value.code == "stale_evidence"


@pytest.mark.parametrize(
    "mutation", ["footer", "state", "pending", "unauthorized", "generation", "disk_receipt"]
)
def test_incomplete_and_unauthorized_execution_is_rejected(bundle, mutation):
    summary = bundle.summaries[0]
    if mutation == "footer":
        summary["footer"] = {"complete": False, "call_count": 2}
    elif mutation == "state":
        summary["state"] = "failed"
    elif mutation == "pending":
        summary["calls"][0]["completion"]["status"] = "pending"
    elif mutation == "unauthorized":
        summary["calls"][0]["states"].remove("authorized")
    elif mutation == "generation":
        summary["calls"][0]["snapshot_generation"] = 1
    else:
        receipt = summary["snapshot_history"][-1]
        receipt["evidence_kind"] = "disk_checkpoint"
        receipt.pop("receipt_id")
    bundle.save(update_footer=mutation != "footer")
    with pytest.raises(StrictProviderError):
        bundle.validate()


def test_original_artifact_content_change_cannot_be_hidden_by_same_size(bundle):
    original = bundle.original.read_bytes()
    bundle.original.write_bytes(b"X" + original[1:])
    with pytest.raises(StrictProviderError) as exc:
        bundle.validate()
    assert exc.value.code == "evidence_hash_mismatch"


@pytest.mark.parametrize(
    "mutation", ["truncate", "outside", "other_attempt", "duplicate_reference"]
)
def test_evidence_content_and_attempt_ownership(bundle, mutation):
    if mutation == "truncate":
        path = bundle.root / bundle.summary_refs[0]["path"]
        path.write_bytes(path.read_bytes()[:-2])
    elif mutation in {"outside", "other_attempt"}:
        path = bundle.root / ("outside.json" if mutation == "outside" else "run1/attempt0/old.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(bundle.summaries[0]), encoding="utf-8")
        bundle.summary_refs[0] = artifact_reference(path, bundle.root)
    else:
        bundle.summary_refs.append(bundle.summary_refs[0])
    with pytest.raises(StrictProviderError):
        bundle.validate()


@pytest.mark.parametrize(
    "field,value",
    [("tests", 0), ("skipped", 1), ("errors", 1), ("failures", 1), ("outcome", "skipped")],
)
def test_mandatory_tests_no_empty_or_skip_pass(bundle, field, value):
    bundle.report["cases"][0][field] = value
    bundle.save()
    with pytest.raises(StrictProviderError) as exc:
        bundle.validate()
    assert exc.value.code == "mandatory_case_failed"


def test_zero_cases_and_duplicate_cases_are_rejected(bundle):
    for cases in ([], [bundle.case, bundle.case]):
        with pytest.raises(StrictProviderError):
            build_expectation(
                bundle.lock,
                bundle.scope,
                cases,
                bundle.roster,
                bundle.inventory,
                run_id="run1",
                attempt="attempt1",
                weight_snapshot_id="snapshot1",
                identity_artifacts=bundle.expectation["identity_artifacts"],
            )


@pytest.mark.parametrize(
    "mutation", ["tokens", "positions", "mask", "state_boundary", "logprob_semantics"]
)
def test_pair_context_must_match_predeclared_case(bundle, mutation):
    bundle.outputs[1]["comparison_context"][mutation] = "different_context"
    bundle.save()
    with pytest.raises(StrictProviderError) as exc:
        bundle.validate()
    assert exc.value.code == "pair_context_mismatch"


@pytest.mark.parametrize("mutation", ["one_ulp", "empty", "nan", "passed_boolean", "shape"])
def test_actual_numeric_payload_required(bundle, mutation):
    payload = bundle.outputs[1]["outputs"]["recurrent_state"]
    if mutation == "one_ulp":
        payload["data"][0] = struct.unpack("!d", (0x3FF4000000000001).to_bytes(8, "big"))[0]
    elif mutation == "empty":
        payload["data"] = []
        payload["shape"] = [0]
    elif mutation == "nan":
        payload["data"][0] = float("nan")
    elif mutation == "passed_boolean":
        bundle.outputs[1]["outputs"]["recurrent_state"] = {"passed": True}
    else:
        payload["shape"] = [1, 2]
    bundle.save()
    with pytest.raises(StrictProviderError):
        bundle.validate()


def test_json_policy_explicitly_defines_signed_zero(bundle):
    bundle.outputs[1]["outputs"]["recurrent_state"]["data"][1] = 0.0
    bundle.save()
    assert bundle.validate()["numerical_parity"] == "passed"
    bundle.expectation["cases"][0]["numerical_policy"] = "json_float64_bits"
    bundle.expectation.pop("expectation_id")
    bundle.expectation["expectation_id"] = fingerprint(bundle.expectation)
    with pytest.raises(StrictProviderError) as exc:
        bundle.validate()
    assert exc.value.code == "numeric_mismatch"


def test_component_cannot_claim_full_model_by_changing_label(bundle):
    inventory = {**bundle.inventory, "2": ["moe"]}
    scope = {**bundle.scope, "kind": "full_model", "inventory_fingerprint": fingerprint(inventory)}
    with pytest.raises(StrictProviderError) as exc:
        build_expectation(
            bundle.lock,
            scope,
            [bundle.case],
            bundle.roster,
            inventory,
            run_id="run1",
            attempt="attempt1",
            weight_snapshot_id="snapshot1",
            identity_artifacts=bundle.expectation["identity_artifacts"],
        )
    assert exc.value.code == "unsupported_scope"


def test_supplied_missing_numeric_comparison_cannot_pass_parity(bundle):
    with pytest.raises(StrictProviderError) as exc:
        validate_run(
            bundle.lock,
            bundle.expectation,
            bundle.summary_refs,
            bundle.root,
            case_results=bundle.report_refs,
            require_numerical=True,
        )
    assert exc.value.code == "missing_numeric_evidence"


def test_real_session_finish_is_consumable_without_reconstructing_coverage(bundle):
    from rl_engine.contracts.strict_provider import CallContext, WorkerObservation
    from rl_engine.integrations.common.provider_observer import ResolvedCall
    from rl_engine.runtime.strict_provider import StrictProviderSession

    summaries = []
    for template in bundle.summaries:
        worker = {k: template[k] for k in ("worker_id", "role", "rank")}
        receipt = SnapshotActivationReceipt.from_dict(template["snapshot_history"][-1])
        session = StrictProviderSession(
            bundle.lock,
            **worker,
            run_id="run1",
            attempt="attempt1",
            weight_snapshot_id="snapshot1",
            snapshot_receipt=receipt,
        )
        session.bind(
            "gdn", worker["role"], WorkerObservation.from_dict(template["observations"]["gdn"])
        )
        for call in template["calls"]:
            call["context"]["call_id"] = f"prompt_layer_{call['context']['layer_id']}"
            context = CallContext.from_dict(call["context"])
            resolved = ResolvedCall(
                bundle.plan,
                lambda: [1.25, -0.0],
                call_context=context,
                recheck=lambda: {"actual_shape": [2]},
            )
            assert session.execute(resolved, context) == [1.25, -0.0]
        summaries.append(session.finish())
    bundle.summaries = summaries
    for call in bundle.expectation["cases"][0]["required_calls"]:
        call["call_id"] = f"prompt_layer_{call['layer_id']}"
    bundle.expectation.pop("expectation_id")
    bundle.expectation["expectation_id"] = fingerprint(bundle.expectation)
    bundle.save()
    assert bundle.validate()["manifest_enforcement"] == "passed"


@pytest.mark.parametrize("mutation", ["inventory_id", "unrelated_file", "logical_path"])
def test_artifact_inventory_must_cover_actual_locked_binding(bundle, mutation):
    inventory = deepcopy(bundle.artifact_inventory)
    if mutation == "inventory_id":
        inventory["inventory_sha256"] = "f" * 64
    elif mutation == "logical_path":
        inventory["files"][0]["logical_path"] = "../provider.bin"
    else:
        path = bundle.directory / "unrelated.bin"
        path.write_bytes(b"different provider")
        reference = artifact_reference(path, bundle.root)
        inventory["files"] = [{"logical_path": "unrelated.bin", "reference": reference}]
        inventory["inventory_sha256"] = fingerprint(
            [
                {
                    "path": "unrelated.bin",
                    "size": reference["size_bytes"],
                    "sha256": reference["sha256"],
                }
            ]
        )
    ref = bundle.write("artifact_inventory.json", inventory)
    bundle.expectation["identity_artifacts"] = [ref]
    bundle.expectation.pop("expectation_id")
    bundle.expectation["expectation_id"] = fingerprint(bundle.expectation)
    with pytest.raises(StrictProviderError):
        bundle.validate()


def test_plan_code_identity_must_have_original_file_evidence(bundle):
    call = bundle.summaries[0]["calls"][0]
    plan = deepcopy(call["plan"])
    plan["code"]["binary_inventory_sha256"] = "f" * 64
    plan.pop("plan_id")
    plan.pop("alignment_id")
    updated = KernelPlan.from_dict(plan)
    call.update(plan=updated.to_dict(), plan_id=updated.plan_id, alignment_id=updated.alignment_id)
    bundle.save()
    with pytest.raises(StrictProviderError) as exc:
        bundle.validate()
    assert exc.value.code == "missing_artifact_evidence"


def test_full_model_scope_rejected_even_with_matching_self_declared_inventory(bundle):
    lock_data = bundle.lock.to_dict()
    lock_data.pop("lock_id")
    lock_data["scope"] = "full_model"
    lock = ProviderLock.from_dict(lock_data)
    scope = {**bundle.scope, "scope_id": "full_model", "kind": "full_model"}
    with pytest.raises(StrictProviderError) as exc:
        build_expectation(
            lock,
            scope,
            [bundle.case],
            bundle.roster,
            bundle.inventory,
            run_id="run1",
            attempt="attempt1",
            weight_snapshot_id="snapshot1",
            identity_artifacts=bundle.expectation["identity_artifacts"],
        )
    assert exc.value.code == "unsupported_scope"


def test_multiplicity_two_rejected_until_runner_assigns_distinct_call_ids(bundle):
    case = deepcopy(bundle.case)
    case["required_calls"][0]["count"] = 2
    with pytest.raises(StrictProviderError) as exc:
        build_expectation(
            bundle.lock,
            bundle.scope,
            [case],
            bundle.roster,
            bundle.inventory,
            run_id="run1",
            attempt="attempt1",
            weight_snapshot_id="snapshot1",
            identity_artifacts=bundle.expectation["identity_artifacts"],
        )
    assert exc.value.code == "invalid_coverage"


def test_repeated_logical_call_rejected_even_with_new_ticket(bundle):
    copied = deepcopy(bundle.summaries[0]["calls"][0])
    copied["call_ticket_id"] = "new_ticket"
    bundle.summaries[0]["calls"].append(copied)
    bundle.save()
    with pytest.raises(StrictProviderError) as exc:
        bundle.validate()
    assert exc.value.code == "duplicate_call"


def test_gpu_lock_rejects_synchronous_completion_from_worker_summary(bundle):
    lock_data = bundle.lock.to_dict()
    lock_data.pop("lock_id")
    lock_data["platform"]["kind"] = "rocm"
    bundle.lock = ProviderLock.from_dict(lock_data)
    bundle.artifact_inventory["lock_id"] = bundle.lock.lock_id
    inventory_ref = bundle.write("artifact_inventory.json", bundle.artifact_inventory)
    bundle.expectation["lock_id"] = bundle.lock.lock_id
    bundle.expectation["identity_artifacts"] = [inventory_ref]
    bundle.expectation.pop("expectation_id")
    bundle.expectation["expectation_id"] = fingerprint(bundle.expectation)
    for summary in bundle.summaries:
        summary["lock_id"] = bundle.lock.lock_id
        summary["observations"]["gdn"]["platform"]["kind"] = "rocm"
    bundle.save()
    with pytest.raises(StrictProviderError) as exc:
        bundle.validate()
    assert exc.value.code == "incomplete_calls"


def test_graph_lock_cannot_be_accepted_without_a_graph_execution_protocol(bundle):
    lock = bundle.lock.to_dict()
    lock.pop("lock_id")
    lock["execution"]["graph_mode"] = "graph"
    graph_lock = ProviderLock.from_dict(lock)
    with pytest.raises(StrictProviderError) as exc:
        build_expectation(
            graph_lock,
            bundle.scope,
            [bundle.case],
            bundle.roster,
            bundle.inventory,
            run_id="run1",
            attempt="attempt1",
            weight_snapshot_id="snapshot1",
            identity_artifacts=bundle.expectation["identity_artifacts"],
        )
    assert exc.value.code == "unsupported_graph"
