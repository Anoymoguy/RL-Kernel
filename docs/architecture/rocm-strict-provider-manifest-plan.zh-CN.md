# ROCm strict provider manifest 实现计划草案

本计划对应 [RFC 428](https://github.com/RL-Align/RL-Kernel/issues/428) 的 ROCm `strict_provider_manifest` 行。目标是在 VIME replay 与 vLLM rollout 中固定 checkpoint、实际 provider、kernel/config、归约设置和执行环境，并对缺失证据、未核准实现及配置漂移执行 fail-closed 检查。

基线为 `test-qwennext` 的 `a078298fc93854913588b79de5a0917344eb5fc1`，计划日期为 2026 年 10 月 28 日。开发分支为 `feat/rocm-strict-provider-manifest`，最终 PR 目标为 upstream `test-qwennext`。以下里程碑是同一 claim PR 的内部提交顺序；RFC 要求一行乘一个平台对应一个 PR。

## 范围和验收边界

这项工作提供身份固定、实际 dispatch 观测、调用授权和验证证据关联，不重新实现 attention、GDN、MoE、GEMM 或 collective 数学。状态桥、MoE 排序、分布式 logprob 和整链资格仍由相应 claim 负责；本行记录它们的规约与实现身份，并拒绝使用不满足要求的路径。

第一轮先做 GPU 无关的契约与检查，再接通一个 ROCm provider 的两侧集成。以单卡、TP1/EP1/CP1/PP1、BF16、eager 作为待验证的初始范围。具体 GPU 架构由实际可用设备决定；`gfx942` 仅是现有 ROCm 实现可参考的目标，不能作为已获得资格的声明。

必须区分三种结果：

- `manifest_enforcement=passed`：身份、执行计划和覆盖检查通过。
- `numerical_parity=not_evaluated`：未进行声明输出的数值比较。
- `numerical_parity=passed`：同一比较上下文下，声明范围的输出通过严格相等检查。

CPU mock、CUDA 报告和单个组件结果均不能证明 Qwen3-Next ROCm 全模型通过。组件接入期间使用 `component.<op>` scope；缺少任何全模型必需节点时，拒绝生成全模型通过结果。

## 可复用代码和设计依据

| 来源 | 可复用内容 | 实现选择 |
| --- | --- | --- |
| `rl_engine/runtime/semantic_registry.py` | `OperatorSession`、resolution trace、instance provenance、strict fallback policy | 复用现有 registry，不创建第二套 backend 选择器 |
| `rl_engine/runtime/provenance/` | 稳定身份和执行来源记录 | 扩展 native/JIT artifact 身份，不以类名代替全部执行身份 |
| `rl_engine/contracts/operators/attention.py` | AttentionContract、SplitKV 及跨 rank 规约 | 引用既有数值契约，不平行定义冲突字段 |
| `rl_engine/backends/rocm/attention/flash_attn.py` | strict AITER/CK 的实际入口、source hash 和 provenance | 优先作为 ROCm 观测接入候选，实际支持形状仍需核验 |
| CUDA PR 543 | immutable run、checkpoint/environment 身份、worker profile 与结果验收 | 提炼共用机制；该 PR 未合入的文件是设计参考，不是当前基线可直接 import 的依赖 |
| vLLM v0.30.0 | selector capability 检查、实际 backend/impl 绑定、配置 hash | 在真实实例及最终配置上采集，不根据 CLI 参数推断实际执行 |

vLLM 的 ROCm AITER backend 对外名称可以为 `FLASH_ATTN`，因此需要完整 implementation class 和 artifact 身份。同一 implementation 的 decode 分支还可能根据 `num_decodes` 推荐 split，所以初始化检查之外需要调用前计划检查。

vLLM 的 `VllmConfig.compute_hash()` 主要面向 input ids/embeddings 到 final hidden states 的计算图配置。它可以作为辅助字段，但不能替代 checkpoint 内容、LM head/logprob 和实际 native/JIT 产物身份。

## 数据和执行流程

持久化数据分为三类：

| 数据 | 内容 | 产生位置 |
| --- | --- | --- |
| `provider-lock.json` | 版本化执行要求、角色绑定、支持范围、允许计划、lock_id | collect/seal 工具 |
| `worker-observed.json` | 当前 worker 实际加载身份、有效配置、角色和 rank | worker 初始化 |
| `validation-result.json` | 本次 lock/run/attempt/权重身份、覆盖、比较范围、结果及证据引用 | 验收器 |

调用记录可以单独写入 trace 文件，由结果文件引用其内容 hash，不把每次调用追加回不可变 lock。

```text
checkpoint + 有效配置 + provider/artifact 身份
                      |
               collect 候选记录
                      |
           schema / 完整性 / 支持检查
                      |
               seal 不可变 lock
                      |
          +-----------+------------+
          |                        |
    rollout worker            replay worker
          |                        |
     创建实际 provider 并采集绑定身份
          |                        |
     bind 检查：身份是否符合各自角色要求
          |                        |
     对每次调用：resolve plan -> authorize -> launch
          |                        |
       completed trace + 输出/状态引用
          |                        |
          +-----------+------------+
                      |
       身份与比较上下文 / 覆盖 / 数值比较
                      |
           validation-result.json
```

检查失败时产生字段级错误并拒绝相应执行或验收。采集模式允许发现候选，但不得自动扩充 enforce 模式的允许集合。

## 类型和文件落点

以下为拟新增文件，编码前先检查是否可合并到现有模块，避免同名能力重复实现。

| 类型或能力 | 建议文件 | 职责 |
| --- | --- | --- |
| `ProviderLock`、`KernelPlan`、`WorkerObservation` | `rl_engine/contracts/strict_provider.py` | 版本、必填字段、不可变记录、规范化及兼容规则 |
| `StrictProviderSession` | `rl_engine/runtime/strict_provider.py` | bind、authorize、完成记录、覆盖与生命周期 |
| 身份采集 | `rl_engine/runtime/provenance/provider_identity.py` | checkpoint、源码、native/JIT artifact、环境身份；优先复用已有工具 |
| vLLM observer | `rl_engine/integrations/engines/rollout/vllm/strict_provider_observer.py` | 读取实际 backend/impl，在真实 dispatch 边界生成 plan |
| VIME 侧接入 | `rl_engine/integrations/orchestrators/vime/providers/strict_provider_observer.py` | 对接 replay；训练引擎专属逻辑下放对应 train adapter |
| lock 工具 | `tools/validation/models/qwen3_next_provider_manifest.py` | collect、seal、validate 和报告检查 |
| 共享验收逻辑 | `rl_engine/validation/models/strict_provider.py` | 比较上下文、覆盖与结果身份；不把逻辑全部留在脚本 |

接口草案：

```python
observation = observer.observe_bound_provider(provider)
session.bind(node, role, observation)

call = observer.prepare_call(provider, inputs, metadata)
session.authorize(node, phase, call.plan)
output = call.launch()
session.record_completed(node, phase, call.plan, output_ref)

result = session.finish(required_coverage, comparison_evidence)
```

`prepare_call()` 必须返回同一个已解析配置对应的 plan 与 launch。不能先检查 `split=1`，再让 provider 内部重新自动选择 `split=4`。无法约束或观测的分支报 `unobservable_plan`，不通过扩大白名单掩盖缺口。

## 字段和匹配规则

### Lock

- schema version、profile ID、scope、规范化 lock_id。
- 模型 ID、基础 revision、config hash、index hash、shard 文件名/大小/内容 hash 清单。
- ROCm/GPU 架构及相关运行库、编译器、provider 构建身份。
- 有效 dtype、拓扑、graph 模式、动态 tuning 与 fallback 策略。
- 必需节点及阶段，每个角色的实现绑定、规约引用和允许计划。
- 环境变量只覆盖影响执行的明确集合；记录最终有效配置，不能只记录用户输入。

### KernelPlan

- 节点、阶段、implementation、实际入口。
- source/artifact 清单、binary/build ID、实际 kernel/algorithm ID、compile key。
- 输入输出 dtype、layout、shape 支持规则和 workspace 条件。
- accumulator dtype、cast 边界、reduction spec、split/merge tree、atomic policy。
- tile/launch 配置、graph 状态、固定算法设置。

必需字段为未知时拒绝 strict；不适用的字段使用 schema 明确定义的表示，不能把未知和不适用都写成空值。一个入口内含多 kernel 时，记录其有序执行计划或可审计的组合规约。第三方库无法暴露关键配置时，应限制支持范围或补 provider 接口。

### 身份和上下文

lock 的数值身份不包含时间戳、日志目录、cache 路径和 GPU UUID 等非数值属性；这些保留在诊断记录中。源码使用相对清单与内容 hash，实际加载绝对路径作为观测证据。已有 CUDA acceptance 若要求全环境精确匹配，在兼容层保留其语义，不能静默放宽。

每个角色分别与自己的 lock binding 匹配，不能直接要求 rollout/replay 的整个环境 JSON 相同。第一版优先要求两侧相同的受控数值计划；允许异构计划需另有明确的 plan 对及数值资格证据。

基础 checkpoint 与当前 `weight_snapshot_id` 分离。更新、export、reload 后验证实际权重及来源；本次比较还必须对齐逻辑 tokens、positions、masks、采样/logprob 语义以及状态边界。基础 profile 不自动证明任意更新后权重的数值 parity。

### 指纹和安全执行

规范化 JSON 使用稳定 key 顺序，拒绝 NaN/Infinity 等非规范值，schema 明确枚举、默认值和浮点表示。清单可以按逻辑名称排序；归约顺序及 rank tree 等具有执行语义的序列不得排序。hash 是身份编号，不是正确性证明。

native/JIT artifact 在构建或加载边界固定并记录，与 worker 实际加载关系绑定。仅对任意磁盘路径做 hash 不足以证明它就是已加载代码。采用隔离、不可变产物目录；更换产物或重新绑定 provider 要开启新的 worker/session 身份。完整 shard/artifact hash 不放到每 token 热路径。

## 实施里程碑

| 阶段 | 实现内容 | 完成条件 |
| --- | --- | --- |
| P0 基线与缺口盘点 | 核对最新 test-qwennext、CUDA 契约变化、实际 VIME/vLLM 版本；逐节点登记绑定和可观测字段 | 形成节点/阶段/角色矩阵，区分 supported、unobservable、missing；确定实际 AMD runner 与版本 |
| P1 共用契约与身份 | schema、规范化、lock ID、checkpoint/source/artifact 身份、字段差异报告 | GPU 无关测试通过；同版本文件修改、权重篡改、未知版本和缺失字段均被拒绝 |
| P2 Session 与 mock 两侧 | bind/authorize/completion/finish、call ID、scope 和覆盖；collect/seal/enforce 工具 | 两侧 mock 正向通过；变更 plan、绕过记录、复用旧报告、伪造 full-model scope 均失败 |
| P3 一个实际 ROCm provider | 优先评估已有 strict AITER/CK attention；读取真实实例、最终布局与 dispatch，固定 plan 与 launch | AMD 上两侧调用接通；错误设备、dtype、layout、workspace、split、graph 和 fallback 在声明边界拒绝 |
| P4 节点覆盖与集成验收 | 扩展 Qwen3-Next 必需节点的身份接口，接入 replay/rollout 报告和真实输入/权重对应关系 | 本行的实现与验证完整；未就绪节点 fail closed；不冒充其他 claim 的整链通过证据 |

P0 需核验 AITER attention 的 D=256/GQA、prefill/decode、KV layout 及梯度需求；普通 RMSNorm 不能直接作为 zero-centred/gated norm。若候选不满足条件，保留 manifest 基础设施并选择已有可核准路径，不在本 PR 顺带重写数值 kernel。

如需借用未合入 CUDA PR 的共用工具，先提炼最小共享依赖或明确依赖顺序，避免把整套 CUDA 模型栈连带引入。实际新增测试跟随 canonical 目录，旧 `rl_engine.kernels.*` 只保持兼容身份，不新增实现。

## 测试与证据矩阵

| 层次 | 正向检查 | 必需负向检查 | 运行条件 |
| --- | --- | --- | --- |
| Schema/identity | 等价结构得到稳定指纹、合法角色字段 | 占位符、缺字段、未知版本、非规范值、权重或源码内容变化 | CPU |
| Binding | 实际实例与角色绑定匹配 | 同显示名称不同实现类、错误artifact、未声明fallback | CPU mock；实际绑定补 AMD |
| Plan | 核准plan与真实launch共享配置 | split/tile/compile key/graph/workspace变化、内部动态再选择、不可观测分支 | mock + AMD |
| Lifecycle/coverage | 已授权调用成功完成后计入覆盖 | 重复/缺失call、只launch未完成、绕过observer、错误phase/rank、GPU异步失败 | CPU mock + AMD |
| Report | lock/run/attempt/snapshot及证据对应 | 旧attempt、不同lock、组件结果冒充全模型、未比较却标parity通过 | CPU |
| 数值 | 声明scope下的selected logprob或组件输出严格比较 | allclose通过但严格不相等、错误输入/位置/mask/状态边界 | AMD |

覆盖按 `(case_id, role, layer_id, phase, logical_call_id)` 及合法条件分支检查，不用一个节点名称集合代替所有层和调用。MoE 不要求每个 token 访问全部专家。数值相等策略遵循 RFC 的声明输出要求；若增强为原始位比较，应明确 signed zero 和 NaN 处理。

初次 AMD 验证至少保存设备/运行库身份、命令、lock、实际worker记录、plan清单、通过与故障注入报告。新硬件和 TP/EP/CP 拓扑分别验证；CUDA golden 不作为 ROCm 通过证据。完整 1 至 64 并发矩阵仍由 batch_invariance_ci 行拥有，本行先证明身份校验与计划固定机制。

## 待落实事项

1. 确定可用 AMD runner 的 GPU 架构、内存、ROCm/PyTorch/AITER/vLLM/VIME 版本和 checkpoint 存放位置。
2. 与 CUDA 维护者对齐共用 schema、数值规约引用和 acceptance 字段，避免两个平台形成不兼容格式。
3. P0 核对每个第三方 provider 能否在实际 dispatch 前取得 kernel/algorithm、split/tile、compile identity；记录需要的最小上游补充。
4. 分布式扩展沿用 rank 无关契约指纹预检，另行验证各 rank 的合法分片与实际实例；不能拿 rank 全字段相同作为一致性条件。

## 源码参考

- [RFC 428 任务和验收要求](https://github.com/RL-Align/RL-Kernel/issues/428)
- [基线 semantic registry](https://github.com/RL-Align/RL-Kernel/blob/a078298fc93854913588b79de5a0917344eb5fc1/rl_engine/runtime/semantic_registry.py)
- [基线 ROCm AITER attention](https://github.com/RL-Align/RL-Kernel/blob/a078298fc93854913588b79de5a0917344eb5fc1/rl_engine/backends/rocm/attention/flash_attn.py)
- [CUDA PR 543 immutable run](https://github.com/fusheng-ji/RL-Kernel/blob/ecafcc2768d32de9ac4415b93a80ca2546c14b60/tools/validation/models/qwen3_next/qwen3_next_run.py)
- [CUDA PR 543 acceptance](https://github.com/fusheng-ji/RL-Kernel/blob/ecafcc2768d32de9ac4415b93a80ca2546c14b60/tools/validation/models/qwen3_next/qwen3_next_contract.py)
- [CUDA PR 543 worker profile](https://github.com/fusheng-ji/RL-Kernel/blob/ecafcc2768d32de9ac4415b93a80ca2546c14b60/rl_engine/integrations/orchestrators/vime/experiments/qwen3_next/profile.py)
- [vLLM selector](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/attention/selector.py)
- [vLLM actual attention binding](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/model_executor/layers/attention/attention.py)
- [vLLM ROCm AITER implementation](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/attention/backends/rocm_aiter_fa.py)
- [vLLM configuration hash scope](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/config/vllm.py#L462)
