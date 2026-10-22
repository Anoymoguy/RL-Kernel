# RFC: Consistency-Preserving Profile-Guided Operator Dispatch

| Field | Value |
| --- | --- |
| Status | Proposed; documentation only, not implemented or certified |
| Date | 2026-10-22 |
| Initial model | Qwen3-8B Dense |
| Reviewed repository baseline | `5ba769f681b7fa765ea6b29f9939454a858390ba` |
| Primary requirement | Preserve strict train-rollout numerical consistency |
| Initial selection mode | Offline profiling and cached, guarded dispatch |
| Initial platform scope | One explicitly validated hardware/software/topology matrix; CUDA and ROCm require separate evidence |

## 1. Summary and decision

Introduce a **contract-constrained, profile-guided operator selector**. It chooses
among explicitly registered implementations or configurations according to the
workload and a memory budget, but only after numerical eligibility is established.

**Consistency is a hard constraint, not a term in a performance score.** A faster
implementation with even one disallowed bit mismatch is ineligible. Profiling
cannot grant numerical eligibility, lower a precision threshold, or authorize a
fallback.

The proposed pipeline is:

```text
Logical operator request and exact numerical contract
  -> capability and certified-domain checks
  -> certified candidate set
  -> measured time/memory frontier
  -> whole-plan admission under per-rank memory budgets
  -> versioned, coordinated execution plan
  -> cached dispatch with exact runtime guards and provenance
```

The first implementation MUST NOT benchmark alternative candidates on live
training or rollout calls. "Dynamic" initially means that different certified
workload shapes/phases can select different premeasured routes, not that kernels
are repeatedly raced or numerics are changed at runtime.

This RFC specifies the conditions under which an implementation may claim
consistency. **Approving this document does not certify any kernel, configuration,
or performance improvement.** Actual certification requires the evidence and
acceptance gates below; unsupported configurations remain unsupported.

## 2. Motivation and existing integration points

A fixed priority list cannot express the trade-off between a low-overhead decode
implementation, a throughput-oriented prefill implementation, and a lower-memory
training execution strategy. A profile-guided selector can exploit those
differences without making every operator independently optimize itself.

At the reviewed baseline:

- `KernelRegistry.get_op()` uses platform priorities, availability checks, and
  instance caching. Attention and logprob also have contract-aware resolution
  paths. See [Runtime Dispatch](runtime-dispatch.md).
- `SemanticOperatorCatalog`, `OperatorBackendDescriptor`, and `OperatorSession`
  provide exact backend resolution, capabilities, lifecycle metadata, and
  implementation provenance. They do not by themselves certify numerical
  interchangeability between implementations.
- Qwen3's `ProfileOps` loads declared per-kind candidates for the model
  reference/validation path and verifies observed provenance. It is not the only
  execution path through the production engines.
- The benchmark profiler and `run_profile_suite.py` provide a starting point for
  measurements, not a ready-made numerical certificate or runtime policy store.
- `runtime/selector/` and `runtime/performance/` are reserved extension points.
  Their intended placement is described in the
  [repository layout](repository-layout.md).

An operator with one eligible implementation has no dispatch choice. The initial
candidate inventory MUST distinguish "registered", "loadable", "supported",
"certified", and "measured". A vendor backend or a new tile configuration is not
eligible merely because its symbol exists.

## 3. Goals and non-goals

### Goals

1. Reduce measured end-to-end time or memory within an unchanged strict numerical
   contract.
2. Select by hardware, exact capabilities, workload phase, layout, and performance
   shape buckets.
3. Preserve independently recomputed train-rollout selected LogP and the existing
   forward, backward, invariance, and model-chain requirements.
4. Make every selection, rejection, implementation, and plan change reproducible.
5. Keep the execution hot path bounded and cheap: guards and a cached route,
   without compilation, benchmarking, fingerprinting, or device synchronization.
6. Fail closed when eligibility, memory admission, or plan agreement is unknown.

### Non-goals

- Approximate, tolerance-relaxed, stochastic, FP8, or quantized substitution to
  rescue performance or an OOM.
- Claims of cross-vendor bitwise equality or unvalidated TP/CP combinations.
- Automatic changes to model weights, sampling semantics, sequence truncation,
  optimizer behavior, topology, or microbatch/accumulation semantics.
- General graph rewriting, automatic offload, or activation checkpoint placement
  in the first implementation.
- Replacing the existing numerical contracts or redesigning existing
  `STRICT`, `AUDIT`, `AUTO`, and `OFF` mode semantics.
- Creating new kernels just by profiling, or promising a speedup before
  representative measurements exist.

## 4. Strict consistency contract

The normative words MUST, MUST NOT, and SHOULD below apply to implementations of
this proposal. They do not claim that the feature already exists.

### 4.1 Accuracy, determinism, and interchangeability are different

The selector MUST retain all four judgments in the
[WS1 numerical precision standard](../contracts/ws1-numerical-precision-standard.md):
forward accuracy, gradient accuracy, forward/gradient invariance, and chain-level
parity. Accuracy against an FP32 reference retains the applicable existing
tolerances; this RFC does not demand BF16 outputs equal FP32 outputs.

For route substitution, however:

```text
deterministic(A) and deterministic(B)
    does not imply bits(A(x)) == bits(B(x))

same backend name or route family
    does not imply equality across batching, phases, or TP/CP layouts

allclose(A(x), B(x))
    is not permission for strict substitution
```

Changing tile geometry, split-K/split-KV, fusion boundaries, reduction trees,
accumulation precision, FMA/rounding behavior, or compiler configuration MAY change
bits. Every such change requires a new implementation identity and appropriate
validation. Tuning parameters are part of the candidate, not an untracked detail.

### 4.2 Required comparisons and scope

For the same model-weight version, logical active tokens, masks, positions,
sampling/scoring definition, and certified environment:

1. Training MUST independently recompute selected LogP and match the corresponding
   rollout values with **zero raw-byte/bit mismatches**. Reusing rollout LogP as the
   training result is forbidden.
2. Candidate substitutions MUST match the certified baseline at the declared
   observable operator or fused-region boundaries, after the contract's logical
   alignment. Hidden states, attention outputs/LSE where observable, masks,
   logits or selected LogP, and state updates are included when applicable.
3. Under the same training inputs and upstream gradients, the candidate MUST
   preserve the certified baseline's gradient bits at declared boundaries.
   Fixed-replay loss, accumulated gradients, and optimizer updates MUST be checked
   for the claimed end-to-end training scope.
4. Existing configuration-invariance obligations remain in force. No runtime
   bucket, new phase, or topology is implicitly covered by an old certificate.

Train-rollout equality is checked before comparing executions with different
weight versions. Old-policy, reference-policy, and updated-policy scores MUST
carry their actual weight identities; different weights are not expected to
produce equal LogP.

Use the comparison semantics of the
[Dense acceptance checklist](../validation/refactor-acceptance.md): logical
sample/token alignment, zero raw-bit mismatches, and no skipped required cells.
Positive and negative zero are distinct. Non-finite values at compared public
outputs invalidate a pass; any mathematically intended internal sentinel must be
declared separately and cannot excuse a non-finite public result. Missing tokens,
unobserved execution, and silent fallback also invalidate a pass.

The applicable operator and WS1/WS2 contracts remain authoritative. When their
existing supported scope is narrower than an intended route, that route remains
ineligible until a separate scope extension is approved and validated.

### 4.3 Certified interchangeability, not a boolean flag

Introduce a versioned `NumericalCertificate` associated with an exact candidate or
explicitly validated candidate combination. It MUST bind:

| Field | Required meaning |
| --- | --- |
| Contract identity | Schema/version and content hash of the applicable numerical contracts |
| Candidate identity | Backend, executable/build fingerprints, tuning parameters, math flags, forward/backward pair |
| Canonical baseline | Exact approved implementation/plan used for bitwise substitution comparisons |
| Coverage domain | Exact supported shape predicates, dtypes, layouts, masks, phases, topology, alignment and state requirements |
| Numerical boundary | Observable outputs, gradients, aliasing/mutation semantics, and persistent state representation |
| Environment | Hardware target and relevant compiler, driver, framework, engine, and companion-patch identities |
| Evidence | Commands, test input/seed identities, raw mismatch counts, rejected/skipped cases, immutable artifact hashes |
| Review status | Approved scope, certificate ID/version, and revocation status |

Self-reported `deterministic=True`, capability descriptors, successful profiling,
or passing a few random tensors MUST NOT create a certificate.

A certificate combines a reviewed numerical/implementation argument with the
required empirical evidence for a declared domain. Finite testing is not a
mathematical proof for every possible tensor or future software build. The system
MUST state that limit and reject extrapolation outside the declared domain. In
particular, a data-dependent algorithm needs explicit reasoning and coverage for
its branches; profiling inputs cannot define its numerical guarantee.

A shared `equivalence_family_id` is only an index into this evidence. Within that
family, every selectable member must satisfy the same canonical boundary contract
over its domain. If that compositional condition cannot be established, certify
explicit rollout/training candidate pairs or full plans instead. Never infer
untested compatibility from a family label.

### 4.4 Mixed-route and stateful execution

Rollout decode and training prefill can legitimately use different kernels and
physical shapes. Therefore "use the same kernel ID on both sides" is neither
necessary nor sufficient.

The certificate MUST cover the combinations the policy can actually produce:
rollout A / training A, rollout A / training B, and the relevant configuration
transformations. It MUST also cover forward/backward pairing and, when applicable,
activation-checkpoint recomputation. If the policy can produce B / A or B / B,
those combinations need evidence too.

The first release SHOULD constrain mixed combinations to a small, explicit
matrix. Unlisted combinations MUST fail eligibility. When only one combination is
certified, selection degenerates to that combination without bypassing checks.

For persistent KV caches, RNG state, or other mutable state, a candidate MUST
declare its representation and transition semantics. Switching writers/readers
mid-sequence is prohibited unless that transition is certified. Otherwise pin the
route for the cache lifetime and drain/rebuild the cache at an approved boundary.

## 5. Candidate scope and priorities

Start with a measured inventory, then enable only operators with at least two
eligible alternatives that offer a useful trade-off.

| Operator/region | Potential alternatives to investigate | Consistency risks requiring certification |
| --- | --- | --- |
| LM head + selected LogP | Materialized, chunked, streamed, or fused scoring | Vocab reduction order, rounding boundaries, real/padded vocab masks, temperature, TP merge, backward |
| Deterministic GEMM / FFN | Fixed-arithmetic CUDA/Triton variants; launch/tiling configurations | K-reduction order, precision, split-K, fused SwiGLU boundaries, parameter-gradient accumulation |
| Attention | Prefill/decode-specific schedules; memory-bounded implementations | Softmax/LSE merge, split-KV, causal/packed masks, RoPE, KV cache, CP reduction, backward |
| RMSNorm, RoPE, elementwise ops | Low-overhead or fused alternatives | Rounding, aliasing, changed fusion boundaries; optimize only if end-to-end relevant |

These are research candidates, not declarations that existing implementations
are mutually bitwise compatible.

Treat `lm_head -> logprob` as a semantic region when a candidate avoids materializing
logits. It cannot replace an API whose consumer requires full logits, full-vocab
entropy, or another unavailable intermediate. A fused candidate must expose every
required observable result and preserve the specified gradient boundary. It may
be ineligible for some sampling/scoring requests even if it fits memory.

Memory pressure caused primarily by persistent model/optimizer state or KV cache
cannot necessarily be fixed by swapping a temporary-workspace kernel. The
profiler MUST identify the source of pressure rather than imply a universal OOM
solution.

## 6. Architecture and data model

### 6.1 Separate correctness, performance, and execution

| Component | Responsibility |
| --- | --- |
| Existing semantic registry and operator contracts | Exact candidate identity and semantic/capability requirements |
| Proposed numerical eligibility layer | Validate certificate coverage, compatibility, revocation, and plan invariants |
| Proposed performance store | Measurements and uncertainty for already eligible candidates |
| Proposed selector | Rank feasible choices without changing their numerical eligibility |
| Proposed plan coordinator | Per-rank admission, route agreement, publication, lifetime, and rollback |
| Existing engine adapters plus guarded handles | Execute the published plan and report actual implementations |

Keep measurements out of immutable semantic descriptors. A performance record
references a candidate fingerprint and certificate ID; it cannot modify either.
Likewise, loading a policy must not import arbitrary factories from a JSON file:
resolve only locally registered, allowlisted exact implementations.

### 6.2 Operator request

Use common metadata plus operator-specific signatures, rather than requiring
unrelated dimensions on every operator:

```text
OperatorRequest
  semantic_op_or_region, callsite_contract, target, phase
  exact_signature: shapes, strides, alignment, layout, dtype, output requirements
  operator_metadata: M/N/K or q/kv lengths, head layout, masks, vocab metadata, etc.
  topology: TP/CP groups, partitions, logical ownership and collective contract
  numerical_contract_hash, required_observables, backward_requirements
  persistent_state_format, graph_mode, determinism_scope
```

Maintain a separate `PlanContext`:

```text
PlanContext
  environment_fingerprint, model_config_identity, policy_epoch
  weight_version, sampling_contract, RNG_policy
  per_rank_memory_budget, persistent_reservations, concurrency_class
```

Weight versions belong in execution/replay provenance and the consistency checks.
They need not fragment a shape-based performance cache if the implementation's
performance and certification are value-independent. Any value-dependent choice
needs a declared guard and validation.

### 6.3 Performance buckets are not correctness domains

A profile key can include:

```text
environment fingerprint + semantic region + numerical contract
  + target/phase + performance shape bucket + layout/dtype
  + topology/partition signature + output/backward requirements
  + graph mode + concurrency class + objective
```

Bucket examples such as `M <= 32` or `2048 < kv_len <= 4096` are only performance
indices. The chosen candidate MUST still pass guards against the **actual**
shape, stride, alignment, masks, topology, and certificate domain. A measurement
at a bucket endpoint does not establish memory safety or correctness for every
member. Boundary, odd/tail, padding, and alignment cases require explicit coverage.

Do not copy tensors to the CPU to construct a hot-path key. Use metadata already
owned by the scheduler/engine. Expensive fingerprints and device-state sampling
belong at initialization or admission boundaries.

### 6.4 Memory model and selection objective

Track candidate latency distributions, temporary workspace, outputs, saved
activations, backward/recompute workspace, layout conversion costs, and graph
resources. Persistent state and allocator headroom are plan-level inputs.

For a feasible plan, require at every relevant execution point and on every rank:

```text
live persistent state
  + live intermediate/saved tensors
  + overlapping kernel/collective workspace
  + graph pools and other reservations not already accounted for
  + safety headroom
  <= assigned device budget
```

The accounting must avoid double-counting allocator reservations. Isolated
operator peaks cannot simply be summed or treated as proof of the full-plan peak.
Include external-library allocations and concurrent streams where applicable.

Maintain a latency/memory Pareto frontier. Choose the fastest admitted plan under
the budget, or the lowest-memory plan under a declared latency target. Include
backward, conversion, communication, and setup amortization when relevant.

`mem_get_info()` alone is not an admission guarantee. Allocated/reserved counters,
allocator behavior, external usage, and graph reservations inform a conservative
budget established with the engine. Measured peaks are observations; a hard bound
needs exact-shape workspace/liveness accounting and integration validation.

If no certified plan fits, reject/defer the workload or request an explicit
higher-level scheduling change. Do not silently change precision, TP/CP,
microbatching, or numerical contracts.

## 7. Profiling and policy lifecycle

### 7.1 Offline measurement

Only numerically eligible candidates enter performance comparison. Run compilation
and warmup separately, then collect repeated measurements on a pinned environment.
Report device event timing and synchronized end-to-end timing separately; neither
is a substitute for the other.

Measurements MUST include:

- Candidate, certificate, exact request, environment, and actual execution identity.
- Warmup/repetition counts, raw samples, median and dispersion; tail percentiles
  only with adequate samples and documented sampling methodology.
- Forward, backward, and complete-region timing as applicable.
- Incremental workspace and full-plan peak allocated/reserved memory, with the
  measurement boundaries and external-allocation limitations stated.
- Eager versus graph replay mode, conversion/communication costs, and compile,
  autotune, capture, and startup costs reported separately.
- Failure/skip reasons and confidence; absence of a sample is not a pass.

Use isolated replay or a disposable worker for candidate trials. Snapshot or
recreate inputs, RNG, gradients, optimizer state, and KV cache where needed.
Profiling MUST NOT advance a live RNG stream, accumulate into live gradients,
mutate a production cache, or run collectives against production peers.

Alternate candidate order to reduce systematic warm-cache/clock bias. Do not
benchmark simultaneous competing candidates on the same GPU and interpret the
result as an isolated ranking. Validate finalists in representative whole-model
replay and real execution, including their intended concurrency.

### 7.2 Policy artifact

Use a versioned local JSON artifact initially; a database is optional later. Store
it in an explicitly configured cache/artifact directory, not as machine-specific
results committed to the source tree.

An artifact MUST include its schema version, environment and contract fingerprints,
candidate/certificate identities, guarded rules, baseline route, measurements,
objective, memory-admission assumptions, and publication epoch.

Use atomic writes and content hashes. Validate schema, artifact integrity,
allowlisted implementations, certificate availability, and exact runtime guards
before use. A checksum detects corruption; it is not proof that an untrusted
author supplied valid evidence. Certificate approval is a separate trust decision.

Invalidate or quarantine artifacts after relevant changes to hardware, compiler,
math flags, extension binary, frameworks/engines/patches, numerical contracts,
state layouts, or candidate parameters. TTL is only a performance freshness
mechanism, not numerical certification.

### 7.3 Runtime lookup and cache misses

Illustrative control flow, not a new existing API:

```python
def resolve_strict_route(request, plan_context):
    candidates = catalog.supported_candidates(request)
    eligible = certificates.filter_exact(request, candidates)
    eligible = certificates.filter_plan_compatibility(plan_context, eligible)
    feasible = admission.filter_exact_budget(request, plan_context, eligible)

    if not feasible:
        raise NoCertifiedFeasibleRoute()

    choice = profiles.choose_measured_feasible(request, plan_context, feasible)
    if choice is None:
        choice = approved_baseline_if_feasible(request, plan_context, feasible)
    if choice is None:
        raise MissingApprovedRoute()

    return coordinator.bind_or_reject(plan_context, request, choice)
```

A profile miss can use an explicitly certified and feasible baseline. If no such
baseline exists, fail closed. Do not delegate a miss to an unrestricted legacy
priority fallback or start live autotuning.

After binding, operator calls check lightweight exact guards and use the cached
implementation. Violated guards return to admission at an allowed boundary or
fail; they never "round up" numerical eligibility from a bucket.

## 8. Coordinated plans, graphs, and failure handling

### 8.1 Policy epochs and asymmetric topologies

Publish an immutable `ExecutionPlan` with a content hash, certificate references,
phase/bucket rules, explicit rollout/training compatibility, per-rank mappings,
and resource reservations.

The coordinator gathers relevant requirements and resource feasibility from all
participating groups. It publishes a plan only after acknowledgements; disagreement
or timeout aborts admission. Do not add a broadcast on every operator invocation.

Ranks need a coherent logical plan, not necessarily an identical local kernel:
shards may have different shapes, and rollout/train may have different validated
TP/CP layouts. The plan specifies each rank's mapping and preserves collective
participants, ordering, and arithmetic contract. No rank chooses independently
based on its free-memory sample.

### 8.2 Rollout, backward, recomputation, and replay

Pin a policy epoch for each rollout/training lifecycle. Persist it with trajectory
metadata, logical token IDs, model/weight identity, sampling parameters, RNG policy,
and state format. Training independently rescoring that trajectory MUST use the
recorded compatible plan or an explicitly certified equivalent plan.

Pending asynchronous rollouts keep their epoch alive. Do not globally replace a
policy and assume all outstanding trajectories can use it.

The forward pass records its candidate and backward binding in the autograd
context. Backward and activation-checkpoint recomputation MUST NOT reselect based
on changed free memory or a newer policy. Checkpoints retain sufficient plan
identity to resume or explicitly reject unavailable/revoked plans.

### 8.3 CUDA Graphs and compiled execution

Resolve and validate a route before compilation/capture; pin it for graph lifetime.
Include plan and candidate identity in graph-cache keys. A plan change requires
new capture/compilation and fresh memory admission at a quiescent boundary.

Graph pools and multiple retained graphs consume budget. Never benchmark, allocate
an unplanned workspace, or switch the callable inside an existing captured graph.
Graph/eager compatibility must be part of the claimed certificate and test matrix.

### 8.4 OOM, mismatch, and rollback

Memory prediction is conservative but cannot eliminate every runtime failure.
An OOM, certificate failure, unexpected implementation, non-finite public result,
or bit mismatch MUST mark the affected execution invalid and stop publication of
its results or optimizer update.

There is no automatic "catch OOM and try another kernel" after mutable state or
collectives have been touched. Coordinate failure across ranks, drain/abort the
affected work, and restore a known-good checkpoint/state when required. Retrying
is allowed only from a clean boundary under a newly admitted certified plan.

Quarantine the scoped failing candidate/plan and retain diagnostics. Do not use a
transient, shape-specific OOM to blacklist every use of a backend globally.
Rollback means a coordinated move to a still-valid certified plan, not an
untracked numerical fallback. If no safe rollback exists, stop.

## 9. Repository integration and compatibility

Proposed implementation responsibilities:

| Location | Change in a future implementation PR |
| --- | --- |
| `rl_engine/runtime/selector/` | Exact eligibility, guarded selection, plan compatibility and coordination interfaces |
| `rl_engine/runtime/performance/` | Request/profile schemas, measurement-store access, frontier/ranking logic, telemetry |
| `rl_engine/contracts/` and `rl_engine/validation/` | Certificate schema, coverage rules, numerical evidence and negative controls |
| `benchmarks/common/`, `benchmarks/tuning/`, `tools/benchmarking/` | Candidate replay, measurement collection and policy generation |
| `rl_engine/runtime/registry.py` and `semantic_registry.py` | Additive integration with existing identity/capability checks; no bypass |
| Qwen3 model/reference and production engine adapters | Explicit opt-in binding, execution readback, epoch propagation and tests |
| `tests/runtime/`, `tests/validation/`, `tests/integrations/` | Eligibility, rejection, consistency, lifecycle and actual-route coverage |

Keep semantic descriptors separate from performance records. Reuse existing
provenance facilities; a dispatch wrapper must report the concrete candidate
actually executed, not masquerade as the previous fixed class.

`ProfileOps` is part of the validation/reference path. Modifying it alone MUST NOT
be presented as enabling dispatch in vLLM/Megatron. Inventory direct backend calls,
canonical wrappers, provider hooks, and compiled/captured callsites in the real
Qwen3 path. Every enabled callsite needs an observed binding; remaining fixed
routes are explicitly reported.

Keep legacy interfaces and default selection unchanged until opt-in validation.
Add a separate selector setting, conceptually `static` or `profiled`, rather than
overloading the existing runtime mode names:

| Existing runtime mode | Proposed selector behavior |
| --- | --- |
| `STRICT` | `profiled` may select only certified feasible routes; no numerical downgrade |
| `AUDIT` | May collect diagnostics; report-only/fallback behavior cannot be marketed as the strict guarantee |
| `AUTO` / `OFF` | Existing behavior is not promoted to strict certification by this feature |

A strict opt-in combined with incompatible report-only/native behavior MUST be
rejected or require an explicit mode correction, never silently accepted.

## 10. Validation and release gates

### 10.1 CPU-safe control-plane tests

- Reject unsupported exact shapes, layouts, masks, topology, observables, and
  forward/backward pairs even when a performance bucket matches.
- Reject absent, expired-by-environment, revoked, malformed, or forged/unapproved
  certificates; reject policy data that attempts arbitrary factory loading.
- Verify numerical filtering precedes ranking, and faster invalid candidates
  never win.
- Verify baseline handling for profile misses, single-candidate no-ops, stale
  performance caches, and no-feasible-route errors.
- Test atomic publication, version/hash mismatch, rank acknowledgements/timeouts,
  asynchronous epoch retention, and checkpoint/graph identity.
- Test per-rank memory admission, saved-activation lifetime accounting, and
  prohibition of reselection during backward/recompute.
- Test that a wrapper records actual dispatch, not just requested dispatch.

### 10.2 Accelerator numerical matrix

Start from a baseline-supported platform and topology. Declare the matrix before
running it; do not remove failing cells after observing results.

Cover exact bucket boundaries and tails, odd shapes, stride/alignment constraints,
supported dtypes, padding/inactive tokens, packing, causal masking, query/KV
lengths, and numerical edge cases such as large finite values and near-tied logits.
Include gradient accumulation, supported forward/backward and recompute paths,
and persistent state transitions.

For each permitted route combination:

1. Run existing FP32-reference accuracy gates with unchanged thresholds.
2. Compare candidate/baseline boundary outputs and gradients by raw bits.
3. Run required batch/partition/phase/TP/CP invariance cases.
4. Run mixed rollout/training and cross-bucket route cases.
5. Repeat to check determinism, including graph/eager paths when claimed.

Include negative controls: deliberately perturb a reduction, output bit, mask,
token alignment, weight identity, or recorded backend; every corresponding gate
must fail. Reusing rollout LogP and reporting an unexecuted candidate must also be
detected.

### 10.3 Full Qwen3 and real engine acceptance

Use the pinned real Qwen3-8B weights/configuration and the actual rollout/train
engine integration. Keep reference-model evidence separate from production-engine
evidence.

- Fixed replay: same weights, tokens, masks, sampling/scoring contract, RNG,
  topology, optimizer state, and declared physical configurations.
- Independently recomputed selected LogP: zero raw-bit mismatches on aligned
  active tokens; no rollout-score reuse.
- Compare the existing required intermediates, loss, gradients, and updates; no
  contract-private tolerance relaxation.
- Exercise multiple real updates, recorded phase/bucket transitions, memory
  admission changes at allowed boundaries, and outstanding rollout epochs.
- Assert every selected implementation actually executed on the declared device;
  skipped cells, missing observations, and silent fallback fail the gate.

Different hardware vendors require separate acceptance matrices. Passing an
operator microbenchmark or CPU control-plane test is not end-to-end certification.

### 10.4 Performance and observability

Only after numerical gates pass, compare the static certified baseline with
profiled selection using operator, region/layer, fixed model replay, and real
rollout/training measurements. Report latency distributions, tokens/s, peak memory,
startup/capture cost, selector overhead, cache hit/miss rate, and whole-step impact.

Do not invent a project-wide speedup threshold. Agree on workload-specific targets
before measuring and report regressions. If the measured advantage does not
justify dispatch complexity, retain the static certified route.

Record plan/certificate/candidate fingerprints, requested and observed backend,
guard/rejection reason, target/phase, topology, memory-admission class, and epoch.
Keep telemetry bounded and avoid synchronization or full tensor logging on every
call. Monitoring detects problems; it does not replace pre-execution eligibility.

## 11. Incremental delivery

| Stage | Deliverable | Exit criterion |
| --- | --- | --- |
| 0: inventory and contracts | Callsite/candidate map, coverage domains, certificate schema, fixed baseline | No runtime behavior change; reviewers agree on the strict invariants and unsupported scope |
| 1: offline evidence | One operator/region with two candidate variants; isolated profiling and numerical evidence | Both variants and permitted cross-route combinations pass; a useful measured trade-off exists |
| 2: guarded lookup MVP | One platform, finite shape domain, local policy store, coordinated cached selection | CPU negative controls and full Qwen3/engine gates pass; default remains static |
| 3: broaden coverage | Additional eligible GEMM/FFN, scoring, and attention routes/topologies | Separate certificate and end-to-end evidence for each new scope |
| 4: controlled calibration | Optional startup/maintenance-window measurement of already certified variants | No live-state mutation; bounded cost and no expansion of numerical eligibility |

Stage 1 SHOULD target whichever of LM-head/scoring or deterministic GEMM first
demonstrates two bitwise-compatible candidates with a useful trade-off. Attention
can follow when its phase, cache, and distributed semantics are certified. This
avoids requiring three difficult operator families to land together.

Online exploratory execution on live traffic is out of scope. A future proposal
may refresh performance rankings using isolated replay, but it still cannot
promote uncertified arithmetic or change a published plan in flight.

## 12. Alternatives and open questions

### Alternatives considered

- **Keep static priorities:** simplest and correct when there is no measured
  benefit or only one certified candidate; retain this as the baseline.
- **Unconstrained per-call autotuning:** rejected because numerical equivalence,
  mutable state, collective coordination, and graph lifetime cannot be assumed.
- **Pure latency scoring with an error penalty:** rejected; consistency is not
  exchangeable for speed.
- **Select every operator independently:** rejected as the sole strategy because
  liveness, fusion boundaries, memory, and cross-route contracts are plan-level.
- **One kernel ID for all phases:** insufficient to prove batch/phase/topology
  invariance, and unnecessarily restrictive when different implementations have
  certified matching boundaries.
- **Immediately add a learned/bandit selector:** unnecessary for the finite MVP;
  a deterministic guarded table is easier to audit and reproduce.

### Questions for reviewers

1. Which two existing or narrowly modified candidates should establish the first
   useful strict time/memory frontier?
2. What is the smallest already-supported hardware/topology/shape matrix that can
   provide operator, mixed-route, and real-engine evidence?
3. Which component owns certificate approval/revocation and immutable artifact
   retention, reusing existing CI/provenance infrastructure?
4. Which engine/orchestrator boundary can coordinate plan epochs without
   interfering with in-flight rollouts and existing weight publication?
5. Which measured end-to-end targets justify enabling profiled selection by
   default in a later PR?

These questions can refine scope and ownership. They MUST NOT weaken the hard
consistency constraints, authorize silent fallback, or convert untested coverage
into a supported configuration.

## 13. Related contracts and reading

- [Runtime Dispatch](runtime-dispatch.md)
- [Repository Layout and Ownership](repository-layout.md)
- [Contributor Guide](../contributing/contributor-guide.md)
- [WS1 Numerical Precision Standard](../contracts/ws1-numerical-precision-standard.md)
- [WS2 CP-Aware Attention Contract](../contracts/ws2-cp-attention-contract.md)
- [Cross-Configuration Logprob Drift Contract](../contracts/cross_config_logprob_drift_contract.md)
- [Dense CUDA/ROCm Acceptance Checklist](../validation/refactor-acceptance.md)
- [Hardware Profiling Guide](../getting_started/hardware-profiling.md)
