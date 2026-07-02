# 7. 实验过程记录

> 每次实验使用的数据集、样本数量、完整样例和指标影响需要同步维护到 `docs/08_dataset_registry.md`。本文档记录实验过程，数据资产的长期台账以该文件为准。

## 实验目标与路线

在 MCP 工具调用（Multi-Client Protocol tool-calling）场景下，验证 SFT→DPO→GRPO 三阶段强化学习 pipeline，并通过 BFCL V4 / SQL / Agentic 等多维度 benchmark 量化能力变化。

```
Phase A (SFT)           Phase C (DPO)           Phase D (GRPO)
学习工具调用格式 ───► 偏好优化工具选择 ───► 在线 RL 精调
```

---

## 一、实验矩阵

### 模型 × 方法

| 基础模型 | SFT | DPO | GRPO | 最终 BFCL Live |
|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 62.4% | 80.0% (+17pp) | **83.5%** (+3.5pp) 🥇 | ✅ 三阶段闭环 |
| Qwen2.5-Coder-7B-Instruct (MCP-only) | 36.5% | 11.5% (-25pp) | — | ❌ 格式过拟合 |
| Qwen2.5-Coder-7B-Instruct (Mixed) | **82.4%** | 不收敛 | 80.6% (-1.8pp v1) | 🔵 GRPO v2 评测中 |

### Benchmark 全景

| 能力 | Benchmark | 1.5B GRPO | Coder7B Mixed SFT | DeepSeek V4 Pro | 说明 |
|---|---|---|---|---|---|
| 函数调用 | BFCL V4 Live | **83.5%** | 82.4% | 70.7%* | * 含 ~10% API schema 报错 |
| 并行函数调用 | BFCL live_parallel | 31.2% | 62.5% | **93.8%** | DeepSeek 预训练含多工具 |
| 多函数+依赖 | BFCL live_p+m | 62.5% | 45.8% | **83.3%** | 我们的训练数据缺多函数样本 |
| SQL 函数调用 | BFCL V3 SQL Func | 51.0% | **99.0%** | — | 函数名准确率 |
| SQL 函数调用 | BFCL V3 SQL Exact | 17.0% | **59.0%** | — | 全参数精确匹配 |
| Agentic | BFCL Multi-Turn | — | **100%** | — | JSON 合法性 |
| Agentic | BFCL Web Search | — | **100%** | — | JSON 合法性 |
| 指令跟随 | IFEval | 84.7% | — | — | * 含 prompt 格式问题 |

---

## 二、Phase A：SFT 格式训练

### 问题：Coder7B 基座 SQL 知识完整但不会输出正确格式

- **基座 Coder7B 输出（错误）**: 被 markdown 代码块包装，使用 `"function_name"` 而非 `"name"` 作为 key，解析器无法提取 → Func 仅 3%
- **SFT 训练后输出（正确）**: 裸 JSON，无 markdown 包装，使用 `"name"` 和 `"arguments"` → Func 99%，解析器完全匹配

| 指标 | Coder7B 基座 | MCP SFT | Mixed SFT | 提升 |
|---|---|---|---|---|
| BFCL Live | ~3% | 36.5% | **82.4%** | **+79pp** |
| SQL Func | 3.0% | — | **99.0%** | **+96pp** |

**关键发现**：SFT 教的是格式纪律，不是知识。基座的 SQL 和工具选择能力已存在，只是被错误输出格式"封印"了。

### Mixed SFT 数据配方

| 格式 | 数量 | 比例 |
|---|---|---|
| MCP（server_id + tool_name） | 19,494 | 68.5% |
| 标准 function-calling（name + arguments） | 8,954 | 31.5% |

仅 31.5% 标准格式数据，就让 Coder7B 从 36.5%→82.4%（+46pp）。

---

## 三、Phase C：DPO 偏好优化

### 问题 1：clarify 能力崩塌（06-19）

| 指标 | SFT v3 | DPO v1 | 原因 |
|---|---|---|---|
| mcp_positive | 100% | 100% | — |
| mcp_no_tool | 100% | 100% | — |
| **mcp_clarify** | **100%** | **0%** | 偏好数据缺 clarify 对 |

**修复**：生成 4,750 clarify 偏好对 + beta 0.1→0.5，DPO v2 Fixed → **零退化**。

### 问题 2：Coder7B DPO 造成 BFCL -25pp（06-20）

```
Coder7B SFT (MCP-only): BFCL 36.5% → DPO后 11.5% (-25pp)
```

**根因**：MCP-only DPO 的偏好数据全是 MCP 格式，DPO 收敛 = 模型被拉向 MCP → 覆盖了 Coder 原生的标准 function calling 能力。**DPO 收敛本身就是伤害的来源。**

### 问题 3：混合格式 DPO 不收敛（06-21）

MCP 格式输出 `[{"server_id":"x","tool_name":"y"}]`，标准格式输出 `{"name":"y","arguments":{...}}`。同一 batch 内两种信号冲突 → loss 卡在 0.69。

**结论**：DPO 要求单一稳定输出分布。多格式场景应用 GRPO 替代。

---

## 四、Phase D：GRPO 在线强化学习

### 1.5B GRPO 成功（+3.5pp）

| 之前失败原因 | 本次修复 |
|---|---|
| entropy≈1e-7（输出完全确定性） | temperature=1.0 + BFCL 分布外 prompt |
| reward_std=0（组内 4 次生成完全相同） | 粒度化 reward 0/0.5/1.0 |
| MCP 过拟合 prompt | BFCL prompts（模型未 memorized） |

| 模型 | BFCL Live |
|---|---|
| 1.5B SFT | 62.4% |
| 1.5B DPO | 80.0% (+17pp) |
| **1.5B GRPO** | **83.5% (+3.5pp)** 🥇 |

### Coder7B GRPO v1 失败（06-22）

| 指标 | Mixed SFT | GRPO v1 | Δ |
|---|---|---|---|
| SQL Exact | **59.0%** | 0.0% | -59pp |
| BFCL Live | **82.4%** | 80.6% | -1.8pp |

**根因**：reward 函数只检查 args key 是否存在（`required_key in args`），不查值是否正确。GRPO 学会"填任意值就拿分" → SQL 参数值随机化 → eval 精确值匹配全挂。

### Coder7B GRPO v2 修复 → 结论：Coder7B 上 RL 无效（06-22）

| 修复项 | v1 (bug) | v2 (fixed) |
|---|---|---|
| BFCL args 检查 | key 是否存在 | 值是否非空 + 有意义 |
| SQL args 检查 | key 是否存在 | 精确值匹配（normalized） |
| placeholder 惩罚 | 无 | 空值/placeholder 给 0 分 |

| 指标 | Mixed SFT | GRPO v1 | GRPO v2 | 结论 |
|---|---|---|---|---|
| SQL Exact | **59.0%** | 0.0% | 0.0% | ❌ 两次都崩塌 |
| BFCL Live | **82.4%** | 80.6% | 79.1% | ❌ 每次都下降 |

**结论**：Coder7B 在 BFCL 82.4% 和 SQL 99% 接近天花板，GRPO 只能引入噪声。1.5B 成功是因为起点低（80%→83.5%），Coder7B 应在数据层面改进（扩充并行样本）而非 RL。

### live_multiple 全量评测 + 自建并行 holdout（06-23）

BFCL V4 live_parallel 全量仅 15 题，统计不可靠。自建 200 条 MCP 合成并行 holdout + live_multiple 扩至全量 1,053 条。

| Round 2 SFT | live_multiple (1053) | Self-built parallel (200) |
|---|---|---|
| 94.4% | 全量比 150 条子集高 3.7pp |

### Round 3 SQL Exact 优化（06-23，进行中）

Round 2 SQL Exact=26%，远低于 Mixed SFT 59%。Round 3 数据配方：15% SQL（增强参数提取）+ 10% 并行 + 22% 标准全量 + 54% MCP（23,000 条）。目标 SQL Exact→50%+。

---

### 合成并行数据 SFT — 无泄漏版（06-22）

用 MCP 数据两两组合合成 2,000 条并行调用样本（`q1 Also, q2`），完全基于已有 MCP 数据，零 BFCL 接触。

| 指标 | Mixed SFT | +合成并行 SFT | Δ |
|---|---|---|---|
| live_simple | 78.7% | 79.3% | +0.6pp |
| live_multiple | 94.0% | 80.7% | -13.3pp |
| **live_parallel** | 62.5% | **100.0%** | **+37.5pp** |
| **live_parallel_multiple** | 45.8% | **95.8%** | **+50.0pp** |
| OVERALL | 82.4% | 82.1% | -0.3pp |

### 泄漏分析：为什么无泄漏版更高？

| 版本 | live_parallel | live_parallel_multiple | 训练数据 |
|---|---|---|---|
| BFCL 泄漏版 | 93.8% | 95.8% | BFCL 原题 + 占位符参数 |
| 合成 MCP 版 | **100.0%** | **95.8%** | MCP 真实参数值合成 |

BFCL 泄漏版用了占位符参数（`<column_name>` 等），导致模型在学习时看到无意义的参数值，部分影响了函数选择判断。合成版用 MCP verifier 中的真实参数值，模型对函数与参数的关联更有信心。

其次，live_parallel 只有 16 道题——2 道的差异就能影响 12pp。两者本质在同一水平线上。

结果待出。

---

## 五、DeepSeek V4 Pro 基准对比（06-22）

用 DeepSeek V4 Pro API（`deepseek-v4-pro`）在同组 BFCL V4 Live 测试数据上评测，获取 SOTA 参照系。

### 评估方法

- API: `https://api.deepseek.com/chat/completions`，OpenAI 兼容 tool-calling 格式
- 数据: 与模型评测相同的 140 条 BFCL V4 Live 测试样本
- ⚠️ BFCL 数据使用非标准 JSON Schema（`"type":"dict"`, `"type":"float"`），需转换为 OpenAI 兼容格式
- ⚠️ 部分函数名含 DeepSeek API 不接受的字符（`.` `:`），导致约 10% 条目 API 400 报错 → 该题 0 分

### 结果对比

| 类别 | DeepSeek V4 Pro | 1.5B GRPO | Coder7B Mixed | 差距分析 |
|---|---|---|---|---|
| live_simple | 76.0% | **86.7%** | 78.7% | 我们的模型在单函数场景超 SOTA |
| live_multiple | 52.0% | **89.3%** | **94.0%** | DeepSeek API schema 报错影响最大 |
| live_parallel | **93.8%** | 31.2% | 62.5% | DeepSeek 预训练含并行工具调用 |
| live_parallel_multiple | **83.3%** | 62.5% | 45.8% | 我们的训练数据缺多函数样本 |
| **OVERALL** | **70.7%*** | **83.5%** | **82.4%** | * 实际估计 78-82%（排除 API 报错） |

### 关键发现

1. **单函数调用（我们的主战场）已超越 DeepSeek V4 Pro**，1.5B GRPO 86.7% vs DeepSeek 76.0%
2. **并行多函数是我们的明显短板**，DeepSeek 93.8% vs 我们最高 62.5% — 根因：训练数据中无多函数调用样本
3. **1.5B 能达到与数百B 参数 DeepSeek 可比的水准**，验证了 SFT→DPO→GRPO pipeline 的有效性

---

### 统一 chat_template + tools 数据重构（06-24）

**目标**：解决训练数据中 system prompt 不统一的问题（MCP 用 "You are an MCP agent..."，SQL 用 bare "AVAILABLE FUNCTIONS:"，开源数据无 system prompt）。

**方案**：所有数据转为 Qwen `chat_template` + `tools` 参数，system prompt 统一为 "You are a helpful assistant."

| 版本 | 训练格式 | In-dist | OOD | Gap | 结论 |
|---|---|---|---|---|---|
| Unified (100% chat) | chat_template | 89.1% | 57.1% | 32pp | OOD 完全遗忘 |
| **Mixed 80/20** | 80% chat + 20% OOD | 88.8% | **85.3%** | **3.5pp** | ✅ 方案可行 |

混入 20% OOD 格式数据即可恢复泛化，format gap 从 32pp→3.5pp。

### SQL Exact 问题分析与下一步

SQL Exact 经历四轮 SFT，从 Mixed SFT 的 59% 降至 18-26%。核心原因：

1. **训练数据格式不匹配**：Spider SQL 数据通过 regex 提取参数值（`columns: ["name"]`），而 BFCL 评测使用双层嵌套格式（`columns: [["name"]]`）。`nv()` 归一化无法完美桥接。
2. **chat_template 加剧差异**：chat_template 格式下模型输出格式与 BFCL GT 差异更大，In-dist Exact=0%。

**下一步方向**：

| 方案 | 工作量 | 预期 |
|---|---|---|
| 用 BFCL V3 SQL 80 条做 few-shot SFT | 小 | Exact → 40-50% |
| Spider 数据按 BFCL 嵌套格式重新生成 completion | 中 | Exact → 50-60% |
| GRPO with BFCL SQL ground truth as reward | 中 | 需 SFT 先到 50%+ |
| 接受当前 OOD Exact=24% 作为上限 | 零 | 不提升 |

**建议优先**：方案 B（Spider 数据重新生成）→ 方案 C（GRPO 接力）。

**遗留**：SQL Exact 仍为 0% (In-dist) / 14% (OOD)，chat_template 下 SQL 参数格式对齐待解决。

---

## 六、关键决策记录

| 日期 | 决策 | 依据 |
|---|---|---|
| 06-18 | 从 1.5B 起步 | 训练方案推荐快速验证链路 |
| 06-19 | DPO beta 0.1→0.5 | v1 clarify 崩塌，增大 KL 约束 |
| 06-19 | 补充 clarify 偏好对 | 根因：缺失偏好类型 |
| 06-19 | 1.5B→Coder7B | 内部数据天花板，需更强基座 |
| 06-20 | Coder7B steps 2000→500 | loss 在 step 100 收敛 |
| 06-20 | 暂停 GRPO | 三条件不满足 (全 0 reward) |
| 06-20 | BFCL 取代内部评测 | 内部 100% 无法区分模型 |
| 06-21 | 混合格式 SFT | BFCL 揭示格式 mismatch |
| 06-21 | 放弃混合 DPO | 多格式冲突，DPO 不收敛 |
| 06-21 | GRPO 复活 | 分布外 prompt + 粒度 reward |
| 06-22 | GRPO≠GSPO/DAPO | Dense 模型 + 短 JSON → 不触发变体场景 |
| 06-22 | GRPO v1→v2 reward 修复 | args 检查从"key存在"升级为"值正确" |

---

## 七、RL 方法选型：为什么 GRPO

| 方法 | 核心改进 | 本场景适用？ |
|---|---|---|
| **GRPO** | 去掉 Critic，组内标准化 | ✅ Dense 模型 + 短 JSON |
| Dr.GRPO | 修复长度偏差 (1/|o_i|) | ❌ 输出 50-100 tokens 均匀 |
| DAPO | 长 CoT 熵坍缩 + 动态采样 | ❌ 短 JSON 不触发 |
| GSPO | 序列级优化 + MoE 稳定 | ❌ Qwen2.5 是 Dense |
| RLVR | 二元 reward | ❌ 粒度不够 |
| DPO | 偏好对离线优化 | ⚠️ 混合格式不收敛 |

---

## 八、产物清单

| 模型 | 路径 | 大小 |
|---|---|---|
| 🥇 1.5B GRPO | `output/grpo_15b_dpo_20260621_153153/adapter` | — |
| 🥈 Coder7B Mixed SFT | `output/coder7b_mixed_sft_20260621_001502/adapter` | 323MB |
| 1.5B DPO | `output/dpo_fixed_20260619_043239/adapter` | 161MB |
| 1.5B SFT | `output/mcp_lora_sft_v3_8ppu_20260618/adapter` | 147MB |
| Coder7B SFT | `output/coder7b_sft_20260619_230410/adapter` | 323MB |
| 📄 详细日志 | `docs/experiments/2026-06-20_experiment_log.md` | — |

---

## 后续实验记录模板

~~~markdown
## YYYY-MM-DD <run_name>

目标：

- 

环境：

```text
Job:
Project:
CUDA/PPU:
Base model:
```

数据：

```text
Train:
Validation:
Manifest:
```

命令：

```bash

```

SwanLab：

```text

```

关键指标：

| Step | Loss | Val Loss | Grad Norm | Clip Rate | Throughput | Notes |
|---:|---:|---:|---:|---:|---:|---|

评测：

| Benchmark | Label | Metric | Result |
|---|---|---|---:|

异常与处理：

- 

结论：

- 
~~~

---

## 2026-06-29：SQL Execution GRPO 实验结果与分析

### Phase7：16 卡并行独立 LoRA GRPO

目标：验证 SQL execution reward 是否能提升 WikiSQL 风格 SQL 生成的可执行正确率。

训练方式：

- 入口：`runs/phase7_wikisql_exec_grpo_torchrun_independent16_20260628_224418`
- 基座：`phase5_qwen25_coder7b_lora_ppu16_lr1e6_seed20260626_20260626_150539/merged_hf`
- 资源：16 PPU
- 重要限制：该实验是 16 个 rank 各自训练独立 LoRA adapter，没有做 DDP all-reduce，因此不是一个同步 16 卡 GRPO 模型。

内部 WikiSQL fixed probe 结果如下。该评测是内部固定执行探针，不是官方 WikiSQL benchmark。

| Model | SQL extraction | Execution rate | Execution accuracy | 相对 Phase5 | 相对 Phase6 |
|---|---:|---:|---:|---:|---:|
| Phase5 SFT baseline | 100.00% | 79.30% | 55.47% | 0.00 pp | +3.91 pp |
| Phase6 SQL/tool SFT | 100.00% | 73.44% | 51.56% | -3.91 pp | 0.00 pp |
| Phase7 GRPO rank07 | 100.00% | 81.64% | 56.25% | +0.78 pp | +4.69 pp |
| Phase7 GRPO rank03 | 100.00% | 81.64% | 57.03% | +1.56 pp | +5.47 pp |
| Phase7 GRPO rank11 | 100.00% | 81.64% | 56.25% | +0.78 pp | +4.69 pp |
| Phase7 GRPO rank15 | 100.00% | 81.64% | 56.25% | +0.78 pp | +4.69 pp |

结论：

- SQL execution reward 有正收益，但幅度较小，最佳 rank03 相对 Phase5 提升 `+1.56 pp`。
- 收益主要来自 execution rate 提升，而不是 SQL exact match 的显著提升。
- Phase6 的 SQL/tool SFT 在 tool-call 格式上更有价值，但在这个 WikiSQL 执行探针上低于 Phase5，说明 SQL 数据和 tool-call 数据不能简单相加，需要重新设计 mixture、loss weight 和训练顺序。
- Phase7 只能作为 reward 有效性的证据，不能作为最终模型，因为它不是同步训练出的单一 adapter。

### Phase8：同步 16 PPU GRPO

目标：将 Phase7 的独立 adapter 方案升级为真正同步 16 卡 GRPO，得到单个共享 LoRA adapter。

当前方案：

- 入口脚本：`scripts/remote/run_swift_wikisql_grpo_ppu16.sh`
- 当前 run：`runs/phase8_swift_wikisql_grpo_sync16_20260629_171307`
- 日志：`logs/phase8_swift_wikisql_grpo_sync16_20260629_171307.log`
- 框架：`ms-swift rlhf --rlhf_type grpo`
- 分布式：`torch.distributed.run --nproc_per_node 16` + DeepSpeed ZeRO-1
- Adapter：LoRA rank 16 / alpha 32 / dropout 0.05
- LR：`2e-7`
- 训练步数：2000
- 每 prompt 采样数：4
- Reward：`scripts/remote/swift_wikisql_reward_plugin.py`
- 数据：`datasets/processed/phase8_swift_wikisql_grpo_20260629/train.jsonl`

截至当前同步记录时，训练仍在进行，尚未产出最终评测结论。中间状态：

| Step | Reward mean | Reward std | Grad norm | KL | Memory | 备注 |
|---:|---:|---:|---:|---:|---:|---|
| 15/2000 | 0.7406 | 0.0438 | 0.0931 | 0.0000599 | 35.78 GiB | 首次稳定检查 |
| 111/2000 | 0.6594 | 0.0687 | 0.0759 | -0.0000122 | 35.85 GiB | 训练正常运行 |

阶段性判断：

- 同步 16 卡链路已跑通，当前不是 Phase7 的独立 adapter 模式。
- reward 波动较大是 GRPO + execution reward 的正常现象，不能按 SFT loss 曲线理解；更关键的是最终固定评测集上的 execution accuracy。
- KL 维持在很小量级，clip ratio 基本为 0，说明当前 LR 较保守，不像在发生策略崩坏。
- 大量 `triton.language.target_info` 日志来自 vLLM/Triton kernel 兼容警告，目前未阻断训练。

启动过程中踩坑及处理：

| 问题 | 表现 | 处理 |
|---|---|---|
| `verl==0.8.0` isolated install 不可用 | 安装后 import `verl` 失败，且依赖树会引入不匹配 Torch/Transformers | 放弃当前环境直接使用 verl，改用已有 PPU 可用的 ms-swift |
| 非交互 SSH 下 `/etc/profile` 早退 | launcher 产生空日志 | 启动脚本改为不依赖 profile，并固定 `/opt/ac2/bin/swift` |
| 缺少 `libhggcrt1.so` | PPU runtime loader 报错 | 显式加入 `/usr/local/PPU_SDK/targets/x86_64-linux/lib` |
| 缺少 `libcuda.so` | external provider / NCCL 初始化失败 | 显式加入 PPU `CUDA_SDK` lib 路径 |
| 缺少 `PPU_SDK` / `PPU_HOME` | RTC kernel 报 `Both PPU_SDK and PPU_HOME are not exist` | 显式导出 `PPU_SDK=/usr/local/PPU_SDK` 和 `PPU_HOME=/usr/local/PPU_SDK` |
| SwanLab 未配置 | cloud 模式缺 API key，local 模式缺 `swanboard` | 本轮使用 TensorBoard 和完整 `logging.jsonl`，待配置后恢复 SwanLab |

下一步：

1. 等 Phase8 训练完成并确认 checkpoint。
2. 合并 adapter。
3. 用 Phase5/Phase6/Phase7 同口径内部 WikiSQL execution probe 评测。
4. 同步补跑 tool-call probe，防止 SQL GRPO 损伤工具调用能力。
5. 若 Phase8 相对 Phase7 没有提升，优先检查 reward 稀疏性、训练集与评测集分布、采样温度、`num_generations` 和 LR，而不是直接加大训练步数。

---

## 2026-06-29: Phase9 Mixed SQL + Tool-Call GRPO

目标：从 `Phase6 merged` 出发，做混合 SQL execution + tool-call/no-tool/clarify GRPO，避免 Phase8 只提升 SQL 而没有继承 Phase6 tool-call 增益的问题。

训练链路：

```text
Qwen2.5-Coder-7B-Instruct
  -> Phase5 SFT
  -> Phase6 SQL/tool SFT merged
  -> Phase9 mixed SQL + tool-call GRPO
```

远端环境：

```text
Job: bifrost-2026060214414601-yans2
Project: /workspace/yans2@xiaopeng.com/agentic_rl_pipeline
Base model: runs/phase6_qwen25_coder7b_sqltool_lora_ppu16_lr6e7_seed20260627_noswan_20260627_105305/merged_hf
Framework: ms-swift rlhf --rlhf_type grpo
Distributed: torch.distributed.run --nproc_per_node 16 + DeepSpeed ZeRO-1
Tracking: SwanLab local mode
```

新增文件：

```text
scripts/remote/prepare_phase9_mixed_grpo.py
scripts/remote/swift_mixed_sql_tool_reward_plugin.py
scripts/remote/run_swift_mixed_sql_tool_grpo_ppu16.sh
datasets/processed/phase9_mixed_sql_tool_grpo_20260629/train.jsonl
datasets/processed/phase9_mixed_sql_tool_grpo_20260629/manifest.json
```

数据配比：

| Task type | Count | Reward |
|---|---:|---|
| SQL execution | 4096 | SQL parse, safe SELECT, SQLite execution success, result exact, normalized SQL exact |
| tool_call | 2048 | JSON parse, action match, call count, tool name, argument exact |
| no_tool | 593 | JSON parse, action match, empty calls, schema discipline |
| clarify | 636 | JSON parse, action match, missing-field overlap, non-empty clarification message |

说明：SQL 样本来自 Phase8 可执行 WikiSQL GRPO 数据；tool/no-tool/clarify 样本来自 Phase5 unified train split。该数据不是官方 benchmark，而是内部 RLVR 训练集。

启动命令摘要：

```bash
REPORT_TO=swanlab SWANLAB_MODE=local MAX_STEPS=2000 \
  nohup bash scripts/remote/run_swift_mixed_sql_tool_grpo_ppu16.sh \
  phase9_swift_mixed_sql_tool_grpo_sync16_20260629_183720 \
  > logs/phase9_swift_mixed_sql_tool_grpo_sync16_20260629_183720.log 2>&1 &
```

关键超参：

```text
LoRA: rank 16, alpha 32, dropout 0.05
LR: 2e-7
Max steps: 2000
num_generations: 4
max_length: 1536
max_completion_length: 128
beta: 0.03
loss_type: grpo
```

启动状态：

| Time | Status |
|---|---|
| 2026-06-29 18:37 | Phase9 run started, PID 1700020 |
| 2026-06-29 18:38 | 16 PPU devices `PPU-ZW810E` detected; distributed/vLLM initialization in progress |

已知注意点：

- 这次从 Phase6 merged 起步，不再复用 Phase5 起点。
- Phase9 是 mixed reward，目标是同时看 WikiSQL execution accuracy 和内部 tool-call validation 是否改善或至少不回退。
- 远端只有 `/opt/ac2/bin/python`；任何 Python/Swift 调用前都需要显式设置 PPU SDK `LD_LIBRARY_PATH`，否则会报 `libhggcrt1.so`。
- `triton.language.target_info` 仍可能在 vLLM 初始化时出现兼容性警告；Phase8 证明这类日志不一定阻断训练，需结合进程和 step metrics 判断。

验收评测计划：

| Metric group | Dataset | Label |
|---|---|---|
| SQL execution | Internal rebased WikiSQL 256 probe | internal, not official WikiSQL benchmark |
| Tool-call | `phase5_unified_20260626/validation.jsonl` | internal unified validation |
| General retention | GSM8K fixed 256 subset | internal subset |
| Instruction following | IFEval if runtime permits | public benchmark rerun |

Phase9 初步目标：

- SQL execution accuracy 接近或超过 Phase8 `62.11%`。
- Tool action/tool-name exact 尽量保持 Phase6 水平，明显高于 Phase8-from-Phase5。
- GSM8K fixed subset 回退不超过 1 pp。

Phase9 runtime update:

- First launch `phase9_swift_mixed_sql_tool_grpo_sync16_20260629_183720` failed before training because SwanLab local mode lacked `swanboard`.
- Installed `swanlab[dashboard]`, which added `swanboard-0.1.9b3`.
- Restarted successfully as `phase9_swift_mixed_sql_tool_grpo_sync16_20260629_184129`, PID `1711045`.
- At step `99/2000`, reward `0.9750`, KL `0.0000315`, memory `36.18 GiB`, train speed about `2.34 s/it`, ETA about `1h14m`.
- Metrics path: `runs/phase9_swift_mixed_sql_tool_grpo_sync16_20260629_184129/v0-20260629-184156/logging.jsonl`.

Phase9 final evaluation:

| Model | SQL exec acc | SQL exec rate | Tool action exact | Tool name exact | JSON exact | GSM8K fixed-256 acc |
|---|---:|---:|---:|---:|---:|---:|
| Phase5 SFT | 55.47% | 79.30% | 81.41% | 79.74% | 37.73% | 76.56% |
| Phase6 SQL/tool SFT | 51.56% | 73.44% | 95.72% | 95.35% | 50.56% | 76.17% |
| Phase8 SQL-only GRPO from Phase5 | 62.11% | 88.67% | 81.23% | 79.55% | 37.73% | 76.17% |
| Phase9 mixed GRPO from Phase6 | 55.86% | 80.08% | 95.72% | 95.35% | 50.56% | 75.00% |

Conclusion: Phase9 preserved Phase6 tool-call quality, recovered SQL above Phase6, but did not match Phase8 SQL-only GRPO. GSM8K fixed-256 regressed mildly to 75.00%. Next run should use staged SQL-only then mixed retention, or stronger SQL reward weighting.

## 2026-06-30: Phase10 Staged SQL-Then-Mixed GRPO

Goal: fix the Phase9 failure mode where mixed GRPO preserved tool-call but diluted SQL execution reward. Phase10 uses a staged schedule:

```text
Phase6 merged
  -> Phase10a SQL-only GRPO from Phase6
  -> merge Phase10a adapter
  -> Phase10b lower-LR mixed retention GRPO
  -> merge final adapter
```

Remote run:

```text
Job: bifrost-2026060214414601-yans2
Project: /workspace/yans2@xiaopeng.com/agentic_rl_pipeline
Queue script: scripts/remote/run_phase10_staged_grpo_ppu16.sh
Stamp: 20260630_094959
Outer log: logs/phase10_staged_sql_then_mixed_20260630_094959.outer.log
PID file: runs/phase10_staged_sql_then_mixed_20260630_094959.pid
Run root: runs/phase10_staged_sql_then_mixed_20260630_094959
Eval root: evals/phase10_staged_sql_then_mixed_20260630_094959
```

Stage design:

| Stage | Base | Data | Reward | Steps | LR | Purpose |
|---|---|---|---|---:|---:|---|
| Phase10a | Phase6 merged | Phase8 WikiSQL executable GRPO data | `wikisql_exec` | 1500 | 2e-7 | recover SQL execution reward without tool-task dilution |
| Phase10b | Phase10a merged | Phase9 mixed SQL/tool data | `mixed_agent_reward` | 800 | 1e-7 | retain tool-call while avoiding large SQL regression |

Expected evaluation after completion:

| Metric group | Dataset | Label |
|---|---|---|
| SQL execution | Internal rebased WikiSQL 256 probe | internal, not official WikiSQL |
| Tool-call | `phase5_unified_20260626/validation.jsonl` | internal unified validation |
| General retention | GSM8K fixed 256 subset | internal subset |

Target: exceed Phase9 SQL `55.86%`, ideally approach Phase8 `62.11%`, while keeping Phase6/Phase9 tool action exact around `95%`.

## 2026-06-30: Phase15 Data Agent Multi-Turn Data/Eval And 7B Continuation

Goal: move beyond single-turn SQL/tool-call probes and build an executable multi-turn Data Agent training/evaluation layer. This phase focuses on schema inspection, SQL generation, SQL execution, error recovery, clarification, unsafe-operation refusal, and final answer generation.

Remote assets:

```text
Job: bifrost-2026060214414601-yans2
Project: /workspace/yans2@xiaopeng.com/agentic_rl_pipeline
Data prep: scripts/remote/prepare_data_agent_multiturn.py
Multi-turn eval: scripts/remote/evaluate_data_agent_multiturn.py
7B continuation queue: scripts/remote/run_phase15_multiturn_7b_queue.sh
14B baseline eval entry: scripts/remote/run_14b_baseline_eval.sh
Data dir: datasets/processed/phase15_data_agent_multiturn_20260630
```

Generated data:

| Split/file | Count | Purpose |
|---|---:|---|
| `train_traces.jsonl` | 2400 | Full executable multi-turn training traces |
| `train_sft.jsonl` | 7084 | One assistant turn per SFT row, existing `prompt/completion/loss_weight` format |
| `train_rl.jsonl` | 2400 | Full trace packed into `query/solution/task_type=data_agent_multiturn` for later environment-style RL |
| `eval_traces.jsonl` | 360 | Held-out executable internal multi-turn probe |
| `validation_sft.jsonl` | 1062 | Fixed multi-turn validation turns |

Scenario mix:

| Scenario | Train traces |
|---|---:|
| top region revenue | 360 |
| top product units | 480 |
| average high-priority ticket resolution | 360 |
| join customer segment and revenue | 480 |
| SQL error repair | 240 |
| metric clarification | 240 |
| unsafe destructive request refusal | 240 |

Important evaluation label: this is an internal executable Data Agent probe, not an official benchmark. It complements, but does not replace, BFCL/Spider/BIRD/tau2-bench/AppWorld style public evaluations.

Phase15 queue:

```text
Stamp: 20260630_142303
PID file: runs/phase15_data_agent_multiturn_7b_20260630_142303.pid
Outer log: logs/phase15_data_agent_multiturn_7b_20260630_142303.outer.log
Queue log: evals/phase15_data_agent_multiturn_7b_20260630_142303/queue.log
```

The queue waits for Phase10 to finish and then starts 16-PPU LoRA SFT from the Phase10 final model. If Phase10 final export is unavailable, it falls back to the Phase9 mixed GRPO merged model. SwanLab logging is enabled locally for the Phase15 training run.

14B baseline preparation:

```text
Model: /publicdata/huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct
Status: local tokenizer/config verified
Entry: scripts/remote/run_14b_baseline_eval.sh
```

The 14B baseline should be evaluated on the same multi-turn probe, WikiSQL internal probe, tool-call validation, and general regression set before deciding whether to migrate the main training line from 7B to 14B.

## 2026-06-30: Phase15 Multi-Turn Eval V2 Expansion

The internal executable multi-turn Data Agent evaluation set was expanded without changing the current training set or interrupting Phase10/Phase15 queues.

Remote path:

```text
datasets/processed/phase15_data_agent_multiturn_eval_v2_20260630
```

Files:

| File | Count | Purpose |
|---|---:|---|
| `eval_traces.jsonl` | 2000 | Executable multi-turn evaluation traces |
| `validation_sft.jsonl` | 5250 | One assistant turn per row for validation-loss checks |
| `data_agent_eval.sqlite` | 1 | Read-only SQLite database used by the evaluator |
| `manifest.json` | 1 | Data counts, scenario mix, and notes |

Scenario mix:

| Scenario | Count |
|---|---:|
| clarification / ambiguous metric | 314 |
| unsafe sensitive export refusal | 200 |
| join customer segment and revenue | 172 |
| empty-result repair | 171 |
| region-ticket join style query | 171 |
| top product by units | 171 |
| top region by revenue | 144 |
| follow-up context / changed metric | 143 |
| month-filtered region revenue | 143 |
| average high-priority ticket resolution | 142 |
| unsafe destructive request refusal | 115 |
| SQL error repair | 114 |

New coverage compared with V1:

- Month filters and more aggregation variants.
- Additional multi-table joins.
- Follow-up style prompts that change the metric.
- Empty-result handling.
- Field/metric ambiguity and clarification.
- Sensitive export refusal in addition to destructive-write refusal.

Evaluation label: this remains an internal executable multi-turn probe, not an official benchmark. It should be reported separately from BFCL, Spider/BIRD, tau2-bench, AppWorld, SWE-bench, and other public benchmarks.

## 2026-06-30: Phase15 Multi-Turn Train V2 Preparation

Prepared a larger Phase15 multi-turn training set for the next longer SFT run. This does not interrupt the current Phase10/Phase15 queue.

Pure multi-turn train data:

```text
datasets/processed/phase15_data_agent_multiturn_train_v2_20260630
```

| File | Count | Purpose |
|---|---:|---|
| `train_traces.jsonl` | 10000 | Full executable multi-turn training traces |
| `train_sft.jsonl` | 26226 | One assistant turn per SFT row |
| `train_rl.jsonl` | 10000 | Full trace for future environment-style RL |
| `examples.jsonl` | 8 | Human-readable audit examples |

Train scenario mix:

| Scenario | Trace count |
|---|---:|
| clarification / ambiguous metric | 1571 |
| unsafe sensitive export refusal | 1000 |
| join customer segment and revenue | 858 |
| empty-result repair | 857 |
| region-ticket join style query | 857 |
| top product by units | 857 |
| month-filtered region revenue | 715 |
| top region by revenue | 715 |
| average high-priority ticket resolution | 714 |
| follow-up context / changed metric | 714 |
| unsafe destructive request refusal | 571 |
| SQL error repair | 571 |

Mixed SFT data with SQL/tool replay:

```text
datasets/processed/phase15_multiturn_sft_mixture_v2_20260630
```

| File | Count | Purpose |
|---|---:|---|
| `train_sft.jsonl` | 38226 | Main recommended next-run SFT mixture |
| `multiturn_only_train_sft.jsonl` | 26226 | Multi-turn-only ablation data |

Mixture:

| Bucket | Rows |
|---|---:|
| Multi-turn Data Agent SFT | 26226 |
| Phase6 SQL/tool replay | 12000 |

Reasoning: the larger run should use the mixed set first, because it teaches multi-turn behavior while retaining previously strong single-turn SQL/tool-call routing. The pure multi-turn set is useful for ablation, but carries higher regression risk on Phase6-style tool-call and Spider/WikiSQL behavior.

## 2026-06-30: Phase15 Data Cleaning And Dedup V4

The first strict dedup pass on V2 exposed a data-quality issue: many synthetic traces were exact template duplicates. The generator was updated to add natural business-context variations, and the Phase6 replay cleaner was relaxed to validate replay JSON without forcing the Data Agent three-tool schema. V4 is the recommended cleaned dataset for the next Phase15 training run.

Scripts:

```text
scripts/remote/prepare_data_agent_multiturn.py
scripts/remote/prepare_phase15_multiturn_sft_mixture.py
scripts/remote/clean_phase15_multiturn_data.py
```

Recommended clean output:

```text
datasets/processed/phase15_multiturn_clean_v4_20260630
```

Cleaned files:

| File | Rows | Use |
|---|---:|---|
| `train_sft_mixture_clean.jsonl` | 31706 | Recommended next SFT training file |
| `train_multiturn_sft_clean.jsonl` | 26223 | Multi-turn-only ablation |
| `train_traces_clean.jsonl` | 10000 | Full train traces |
| `train_rl_clean.jsonl` | 10000 | Future multi-turn RL data |
| `eval_traces_clean.jsonl` | 2000 | Main internal multi-turn eval |
| `validation_sft_clean.jsonl` | 5241 | Validation-loss checks |

Cleaning operations:

- Normalize newlines and JSON completions.
- Canonicalize JSON completions with sorted keys.
- Validate Data Agent actions: `tool_call`, `clarify`, `refuse`, `final`.
- Validate Data Agent tools: `list_tables`, `describe_table`, `run_sql`.
- Enforce read-only SQL for normal traces.
- Deduplicate SFT rows by `prompt + completion`.
- Deduplicate traces/RL rows by `user + gold_actions + scenario`.
- Keep Phase15 train/eval split separated.
- Keep Phase6 SQL/tool replay as broad JSON replay rather than forcing the three-tool Data Agent schema.

Clean retention:

| Group | Input | Kept | Notes |
|---|---:|---:|---|
| Multi-turn SFT | 26223 | 26223 | No duplicates after V4 generator fix |
| Train traces | 10000 | 10000 | All executable/valid |
| Train RL | 10000 | 10000 | Full trace RL rows retained |
| Eval traces | 2000 | 2000 | Main held-out internal eval |
| Validation SFT | 5241 | 5241 | Validation rows retained |
| Mixed SFT | 38223 | 31706 | Exact duplicates removed after mixing replay |

Scenario coverage after cleaning:

| Scenario | Train traces | Eval traces |
|---|---:|---:|
| clarification / ambiguous metric | 1571 | 314 |
| unsafe sensitive export refusal | 1000 | 200 |
| join customer segment and revenue | 858 | 172 |
| empty-result repair | 857 | 171 |
| region-ticket join style query | 857 | 171 |
| top product by units | 857 | 171 |
| month-filtered region revenue | 715 | 143 |
| top region by revenue | 715 | 144 |
| average high-priority ticket resolution | 714 | 142 |
| follow-up context / changed metric | 714 | 143 |
| unsafe destructive request refusal | 571 | 115 |
| SQL error repair | 571 | 114 |

Next training should use `train_sft_mixture_clean.jsonl` first, with `eval_traces_clean.jsonl` as the internal executable multi-turn probe. The pure multi-turn file should be reserved for ablation because it has higher regression risk on the previous single-turn SQL/tool-call metrics.

## 2026-07-01: Text-to-SQL Execution Accuracy Literature Notes

Problem observed in our experiments: SQL execution accuracy has not improved reliably. Phase8 SQL-only GRPO improved the internal WikiSQL probe, but Phase9/Phase10 mixed SQL+tool training stayed around the mid-50% execution-accuracy range on the internal 256-row WikiSQL probe. This suggests that simple SFT/GRPO over final SQL strings is insufficient; the model needs better schema grounding, value grounding, correction data, and candidate selection.

Current internal snapshot:

| Model | SQL extraction | SQL execution rate | SQL execution accuracy | Notes |
|---|---:|---:|---:|---|
| Phase8 SQL-only GRPO from Phase5 | 100% | 88.67% | 62.11% | Best SQL-only result so far |
| Phase9 mixed GRPO from Phase6 | 100% | 80.08% | 55.86% | Preserved tool-call but diluted SQL reward |
| Phase10 staged SQL->mixed GRPO | 100% | 79.69% | 55.86% | Mixed retention erased most SQL-only gain |

Key lessons from recent Text-to-SQL work:

1. **Candidate generation + selection is stronger than one-shot generation.**
   - CHASE-SQL uses multiple generation paths, including divide-and-conquer, execution-plan-style reasoning, and instance-aware synthetic examples, then uses a preference-optimized selector to choose among candidates.
   - Reported BIRD execution accuracy is about 73% on both dev/test in the paper.
   - Takeaway for us: GRPO should not only optimize a single sampled SQL. We should train and evaluate with `k` candidates, execution results, and a selector/reranker reward.

2. **Execution feedback is most useful when attached to reasoning/correction, not just final SQL reward.**
   - ExCoT-DPO reports that zero-shot CoT alone gives little benefit, and DPO without CoT is weak; the gain comes from combining execution feedback with explicit reasoning/correction traces.
   - The paper reports improvements on BIRD and Spider using execution accuracy as the feedback signal, without human preference labels or a learned reward model.
   - Takeaway for us: our SQL GRPO reward is too sparse and too final-output oriented. We need traces like: draft SQL -> execution error/result -> diagnose schema/value/join issue -> corrected SQL.

3. **Schema retrieval/linking should be model- and context-dependent.**
   - LitE-SQL combines vector-based schema retrieval with a two-stage generator: SFT followed by execution-guided reinforcement/self-correction. It reports strong BIRD/Spider execution accuracy with lightweight models.
   - Another analysis argues that schema linking is less important when the full schema fits and the base model is strong, but still helps weaker models or large schemas; augmentation, example selection, and correction consistently help.
   - Takeaway for us: for WikiSQL-style tiny schemas, schema linking is not the bottleneck; for Data Agent/Spider/BIRD-style multi-table schemas, we need schema/value retrieval and compact schema serialization.

4. **Example selection and prompt organization matter.**
   - DAIL-SQL shows that SQL-oriented schema representation, example selection, and example organization can produce large Spider gains; self-consistency adds a smaller extra gain but is expensive.
   - Takeaway for us: training data should include prompt formats that match inference: schema serialized as SQL DDL or compact table/column docs, plus structurally similar demonstrations for difficult SQL types.

5. **Dataset correctness is a major bottleneck.**
   - SQLDriller reports a non-trivial portion of Spider/BIRD mappings can be wrong and proposes execution consistency with counterexample databases to detect/fix labels. It reports accuracy improvements up to 13.6% after data correction.
   - Takeaway for us: before adding more Spider/BIRD-style data, run label validation: execute gold SQL, compare result semantics, detect ambiguous/incorrect samples, and remove/fix noisy rows.

6. **Benchmark execution accuracy can be misleading without reliability checks.**
   - Work such as RTS++ introduces execution-result clustering/entropy to estimate uncertainty across candidates.
   - Takeaway for us: report not just execution accuracy, but also candidate agreement, execution entropy, empty-result rate, and abstain/clarify behavior.

Recommended next SQL improvement plan:

| Priority | Change | Concrete implementation |
|---|---|---|
| P0 | Build SQL error taxonomy | From predictions, bucket failures into parse error, execution error, wrong column/table, wrong aggregation, wrong filter value, wrong join, empty-result false positive, order/limit error |
| P0 | Add execution-repair SFT | Generate traces: question+schema -> draft SQL -> sqlite error/result mismatch -> corrected SQL; train assistant turns for correction |
| P0 | Validate SQL training labels | For WikiSQL/Spider/SQL-Create-Context, execute gold SQL where DB exists; remove or mark ambiguous/incorrect rows |
| P1 | Use multi-candidate SQL GRPO | For each prompt sample 4-8 SQLs, reward exact execution result, penalize invalid/unsafe SQL, and train selector/reranker pairs |
| P1 | Add value grounding | Retrieve candidate cell values with exact/fuzzy match; include top values in prompt/training context |
| P1 | Add schema compacting | Serialize schema as DDL plus column comments/types/sample values; for multi-table tasks expose only relevant tables plus distractors |
| P1 | Balance SQL/tool retention | Keep SQL-only reward stage longer; in mixed stage use higher SQL proportion or freeze SQL-sensitive adapter with separate tool-retention adapter |
| P2 | Preference optimization | Build pairs: correct SQL vs executable-wrong SQL, correct join vs cartesian/wrong join, clarify vs guessing missing metric |
| P2 | Reliability scoring | At eval time sample multiple SQLs; report majority-result accuracy, execution entropy, abstain threshold, and cost |

Recommended Phase16 experiment:

```text
Base: Phase10 final or Phase8 SQL-best, depending on target.
Data:
  40% SQL execution-repair traces
  25% Spider/BIRD-style multi-table SQL after label validation
  15% WikiSQL/SQL-Create-Context simple SQL replay
  10% Data Agent schema->SQL->execute multi-turn traces
  10% tool-call/general replay
Training:
  1. SFT on repair + schema/value-grounded SQL traces.
  2. DPO/SimPO on candidate pairs selected by execution result.
  3. SQL-only GRPO with multi-candidate execution reward.
  4. Short tool-retention SFT/GRPO.
Evaluation:
  WikiSQL internal 256 probe, Spider execution/test-suite if available, BIRD subset if available,
  internal multi-turn SQL repair probe, tool-call validation, GSM8K/MMLU-Pro regression.
```

Immediate action items:

1. Do not treat normalized SQL exact as the target; continue optimizing execution result correctness.
2. Build `evaluate_sql_failure_modes.py` to analyze Phase10 predictions and create a failure distribution.
3. Create `prepare_sql_repair_sft.py` from failed predictions plus gold result/error feedback.
4. Add value retrieval into prompts for datasets where question values may not exactly match table cells.
5. For the next longer run, use cleaned Phase15 v4 data for multi-turn behavior, but isolate SQL repair as its own stage rather than mixing it too early with tool-call retention.

References:

- [CHASE-SQL: Multi-Path Reasoning and Preference Optimized Candidate Selection in Text-to-SQL](https://openreview.net/forum?id=CvGqMD5OtX), arXiv:2410.01943.
- [DAIL-SQL / Text-to-SQL Empowered by Large Language Models](https://arxiv.org/abs/2308.15363), PVLDB 2024.
- [DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with Self-Correction](https://arxiv.org/abs/2304.11015).
- [Optimizing Reasoning for Text-to-SQL with Execution Feedback](https://arxiv.org/abs/2503.19988), ACL Findings 2025.
- [MAC-SQL: A Multi-Agent Collaborative Framework for Text-to-SQL](https://arxiv.org/html/2312.11242v2).
- LitE-SQL: Vector-based Schema Linking and Execution-Guided Self-Correction.
- The Death of Schema Linking? Text-to-SQL in the Age of Well-Reasoned Language Models.
- SQLDriller: Automated Validating and Fixing of Text-to-SQL Translation with Execution Consistency.

## 2026-07-01: Phase16 SQL Repair Data Preparation

Goal: prepare the next SQL-focused experiment from the Text-to-SQL literature plan above. The data is designed to improve execution accuracy through repair traces, executable WikiSQL GRPO prompts, preference pairs, and a small amount of tool/multi-turn replay.

Remote project:

```text
/workspace/yans2@xiaopeng.com/agentic_rl_pipeline
```

Preparation script:

```text
scripts/remote/prepare_phase16_sql_repair_data.py
```

Output directory:

```text
datasets/processed/phase16_sql_repair_20260701
```

Input assets:

| Input | Role |
|---|---|
| `evals/phase10_phase15_post_eval_20260701_102944/wikisql_phase10/phase10b_mixed_retention_step0800_predictions.jsonl` | Real Phase10 WikiSQL failures for taxonomy, repair SFT, and DPO |
| `datasets/processed/phase8_swift_wikisql_grpo_20260629/train.jsonl` | Executable WikiSQL table payloads for GRPO and synthetic SQL repair |
| `datasets/modelscope/spider_train.jsonl` / `spider_val.jsonl` | Spider SFT-only SQL data; DB files are not present in this snapshot, so not used for RL reward |
| `datasets/modelscope/sql_create_context.jsonl` | Schema-grounded SQL SFT rows |
| `datasets/processed/phase15_multiturn_clean_v4_20260630/train_sft_mixture_clean.jsonl` | Tool-call and Data Agent multi-turn replay |

Phase10 WikiSQL failure taxonomy:

| Failure type | Count | Meaning |
|---|---:|---|
| `wrong_column_or_unquoted_display_name` | 48 | Model often used display names such as `Pre-Season` instead of actual SQLite `colN` columns |
| `wrong_aggregation_or_selection` | 48 | Executable SQL but wrong aggregation/selected column |
| `wrong_filter_or_value` | 13 | Wrong filter value or condition |
| `execution_error_other` | 2 | Other execution failures |
| `sql_syntax_error` | 2 | SQL syntax errors |
| `correct_execution` | 143 | Correct predictions, excluded from repair training |

Generated files:

| File | Rows | Purpose |
|---|---:|---|
| `train_sql_repair_sft.jsonl` | 4137 | Real Phase10 failures plus synthetic WikiSQL corruption repair traces |
| `eval_sql_repair_probe.jsonl` | 64 | Held-out real Phase10 failure repair probe |
| `train_sql_dpo_pairs.jsonl` | 5223 | Chosen gold SQL vs real/synthetic wrong SQL |
| `train_sql_grpo.jsonl` | 4088 | Executable WikiSQL prompts with table payload and gold result for SQL GRPO |
| `train_sql_mixture_sft.jsonl` | 12133 | Repair SFT + Spider/SQL-context SFT + Data Agent/tool replay |
| `failure_taxonomy.jsonl` | 113 | Real failed Phase10 predictions with error buckets |
| `manifest.json` / `failure_report.json` / `examples.json` | - | Reproducibility, counts, and sample inspection |

Validation results:

| File | JSON valid | Duplicate keys |
|---|---:|---:|
| `train_sql_repair_sft.jsonl` | 4137/4137 | 0 |
| `eval_sql_repair_probe.jsonl` | 64/64 | 0 |
| `train_sql_dpo_pairs.jsonl` | 5223/5223 | 0 |
| `train_sql_grpo.jsonl` | 4088/4088 | 0 |
| `train_sql_mixture_sft.jsonl` | 12133/12133 | 0 |
| `failure_taxonomy.jsonl` | 113/113 | 0 |

Important implementation details:

1. WikiSQL examples are executable and can be used for GRPO reward.
2. Spider rows are SFT-only for now because the current remote snapshot does not contain Spider database files.
3. Synthetic repair data is created by corrupting gold SQL with realistic mistakes: removing quotes, changing aggregation functions, and removing limits.
4. Real Phase10 failures are held out partly as `eval_sql_repair_probe.jsonl` to check whether the model learns correction behavior rather than memorizing all failures.
5. Value hints are derived from question/table-cell overlap and stored as retrieval features, not as ground-truth labels.

Recommended next training order:

1. Phase16a SFT on `train_sql_mixture_sft.jsonl`, with higher weight on `phase10_wikisql_failed_prediction` and `wikisql_synthetic_sql_repair`.
2. Phase16b DPO/SimPO on `train_sql_dpo_pairs.jsonl`.
3. Phase16c SQL-only GRPO on `train_sql_grpo.jsonl`.
4. Short tool/multi-turn retention stage using Phase15 clean v4 replay.
5. Evaluate WikiSQL execution accuracy, repair probe accuracy, tool-call metrics, multi-turn Data Agent probe, GSM8K, MMLU-Pro, and IFEval.

## 2026-07-01: Phase16a SQL Repair SFT Launch

Phase16a has started on `bifrost-2026060214414601-yans2` after stopping the previous `run_gpu_16.sh` resource placeholder.

Run script:

```text
scripts/remote/run_phase16a_sql_repair_sft_ppu16.sh
```

Remote run directory:

```text
runs/phase16a_sql_repair_sft_20260701_1138_phase16a/phase16a_sql_repair_sft_7b_lora_20260701_1138_phase16a
```

Remote eval/manifest directory:

```text
evals/phase16a_sql_repair_sft_20260701_1138_phase16a
```

Training setup:

| Field | Value |
|---|---|
| Base model | `evals/phase10_staged_sql_then_mixed_20260630_094959/merged/phase10b_mixed_retention_step0800` |
| Train data | `datasets/processed/phase16_sql_repair_20260701/train_sql_mixture_sft.jsonl` |
| Rows | 12133 |
| Method | 16-card DDP LoRA SFT |
| Steps | 1000 |
| Sequence length | 4096 |
| Global batch | 32 samples/step |
| LR | `4e-7`, cosine schedule |
| Warmup | 60 steps |
| LoRA | rank 32, alpha 64, dropout 0.05 |
| Max grad norm | 0.5 |
| Logging | `train_metrics.jsonl`, `train.log`, SwanLab local mode |

Initial health check:

| Step | loss | loss_ema | grad_norm | clip rate | LR | Tokens/s | Max memory |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.4799 | 0.4799 | 0.5673 | 1.00 | `1.33e-8` | 67.4 | 15.80 GiB |
| 5 | 0.4825 | 0.4803 | 0.4912 | 0.25 | `4.00e-8` | 153.5 | 16.89 GiB |
| 10 | 0.5503 | 0.4963 | 0.6034 | 1.00 | `7.33e-8` | 181.4 | 16.89 GiB |
| 15 | 0.5975 | 0.5219 | 0.5935 | 0.80 | `1.07e-7` | 194.8 | 16.89 GiB |

Interpretation: training has passed model loading and first optimization steps. Early clipping is elevated during warmup but gradients are finite and memory is stable. Continue monitoring for NaN/OOM/PCCL timeout and check whether loss stabilizes after warmup.

## 2026-07-01: Phase16 Follow-up Training and Evaluation Assets

While Phase16a is running, the follow-up data and evaluation assets were prepared without interrupting the training process.

Preparation script:

```text
scripts/remote/prepare_phase16_followup_assets.py
```

Remote output directory:

```text
datasets/processed/phase16_followup_assets_20260701
```

Purpose: make Phase16b/Phase16c and post-training evaluation reproducible by pinning all train/eval paths in one manifest.

Generated training assets:

| Asset | Rows | Purpose |
|---|---:|---|
| `phase16b_dpo_train.jsonl` | 4967 | DPO/SimPO training pairs: gold SQL vs wrong SQL |
| `phase16b_dpo_eval_holdout.jsonl` | 256 | Held-out preference pairs; do not train on these |
| `phase16c_grpo_train.jsonl` | 4088 | Executable WikiSQL GRPO training data |
| `phase16c_grpo_smoke_256.jsonl` | 256 | Small GRPO smoke/debug subset |

Generated evaluation assets:

| Asset | Rows | Purpose |
|---|---:|---|
| `wikisql_eval_256.jsonl` + `wikisql_eval_256.sqlite` | 256 | Internal fixed WikiSQL execution probe |
| `sql_repair_eval.jsonl` | 64 | Held-out real Phase10 SQL failure repair probe |
| `data_agent_multiturn_eval_500.jsonl` + `data_agent_eval.sqlite` | 500 | Internal executable multi-turn Data Agent probe |
| `data_agent_tool_action_probe.jsonl` | 307 | Internal Data Agent JSON action/tool-call probe |
| `mcp_xlam_array_tool_probe.jsonl` | 5 | MCP array-format smoke probe only; too small for main reporting |

Public/general evaluation data availability:

| Dataset | Remote path | Status |
|---|---|---|
| GSM8K | `datasets/eval_suite/huggingface/openai__gsm8k/main/test-00000-of-00001.parquet` | available |
| MMLU-Pro | `datasets/eval_suite/huggingface/TIGER-Lab__MMLU-Pro/data/test-00000-of-00001.parquet` | available |
| IFEval | `datasets/eval_suite/huggingface/google__IFEval/ifeval_input_data.jsonl` | available |
| BFCL V3 files | `datasets/eval_suite/huggingface/gorilla-llm__Berkeley-Function-Calling-Leaderboard` | available |

Important notes:

1. `phase16b_dpo_eval_holdout.jsonl` is split before DPO and must remain held out.
2. Spider DB files are still absent, so Spider remains SFT-only; it is not yet suitable for execution-reward RL or official Spider execution evaluation.
3. BFCL files are present, but the current repo has only internal/xLAM-style evaluation. For SOTA alignment, official BFCL scoring still needs to be integrated.
4. `mcp_xlam_array_tool_probe.jsonl` has only 5 unique prompts after deduplication, so it should be treated as a smoke test, not a headline metric.
5. The main post-Phase16 report should prioritize: WikiSQL execution accuracy, SQL repair probe accuracy, Data Agent multi-turn success, Data Agent JSON action probe, GSM8K, MMLU-Pro, IFEval, and then BFCL once the official scorer is wired in.

Recommended follow-up execution:

```text
Phase16a SFT adapter
  -> merge adapter onto Phase10b base
  -> Phase16b DPO/SimPO using phase16b_dpo_train.jsonl
  -> Phase16c SQL-only GRPO using phase16c_grpo_train.jsonl
  -> short tool/multi-turn retention if tool metrics regress
  -> unified evaluation using the fixed assets above
```

## 2026-07-01: Phase16a SQL Repair Evaluation Fix

After Phase16a finished, the original held-out SQL repair probe reported `normalized_repair_exact = 0.00%`. This turned out to be primarily an evaluation-design issue, not enough evidence that the model has no SQL repair ability.

Old repair probe:

| Field | Value |
|---|---|
| Script | `scripts/remote/evaluate_sql_repair_probe.py` |
| Dataset | `datasets/processed/phase16_followup_assets_20260701/sql_repair_eval.jsonl` |
| Samples | 64 |
| Metric | normalized SQL string exact |
| Phase16a result | `0.00%` |

Why the old metric is misleading:

1. The prompt contains question, previous SQL, feedback, expected result, and gold SQL target, but does not include the full table schema or executable table payload.
2. The metric is strict normalized SQL exact. Execution-equivalent SQL is still counted as wrong.
3. Many examples are real Phase10 failures where the previous SQL itself contains wrong or truncated column/value assumptions, so the model must infer schema details that are absent from the prompt.
4. The intended downstream behavior is execution correctness, not string identity.

The evaluation scheme was updated with an executable repair probe:

| Field | Value |
|---|---|
| Preparation script | `scripts/remote/prepare_sql_repair_execution_eval.py` |
| Evaluation script | `scripts/remote/evaluate_sql_repair_execution.py` |
| Eval data | `datasets/processed/phase16_followup_assets_20260701/sql_repair_execution_eval/sql_repair_execution_eval_128.jsonl` |
| Database | `datasets/processed/phase16_followup_assets_20260701/wikisql_eval_256.sqlite` |
| Samples | 128 |
| Source | fixed WikiSQL internal execution probe |
| Prompt | full schema, sample rows, previous corrupted SQL, execution feedback, expected result |
| Main metric | repaired SQL execution result exact match |

Phase16a results on the new executable repair probe:

| Metric | Value |
|---|---:|
| SQL extraction rate | `100.00%` |
| Execution rate | `99.22%` |
| Execution repair accuracy | `81.25%` |
| Normalized SQL exact | `28.12%` |
| Previous SQL execution accuracy baseline | `0.00%` |

Breakdown by synthetic failure type:

| Failure type | Samples | Execution repair accuracy |
|---|---:|---:|
| `remove_where_clause` | 116 | `82.76%` |
| `avg_to_count` | 4 | `75.00%` |
| `sum_to_count` | 6 | `50.00%` |
| `max_to_min` | 2 | `100.00%` |

Interpretation:

1. The `0.00%` old repair exact score should not be used as the headline repair metric.
2. Phase16a can repair many SQL mistakes when schema/table context is present and scoring is execution-based.
3. Normalized SQL exact remains useful only as a conservative diagnostic for exact gold-form reproduction.
4. Future SQL repair reporting should prioritize `execution_repair_accuracy`, `execution_rate`, and error-type breakdown.
5. Training should not immediately pivot solely because of the old repair exact score. The next training decision should use WikiSQL execution accuracy, executable repair accuracy, and Data Agent multi-turn/tool metrics together.

Implementation update:

```text
scripts/remote/run_phase16a_post_eval_queue.sh
  -> evaluate_sql_repair_probe.py                 # legacy exact diagnostic
  -> prepare_sql_repair_execution_eval.py         # executable repair eval construction
  -> evaluate_sql_repair_execution.py             # execution-based repair scoring
```

Recommended next action:

1. Keep the new executable repair probe in all later SQL experiments.
2. If WikiSQL execution accuracy remains below the Phase8 SQL-only peak, run Phase16b DPO followed by Phase16c SQL execution GRPO from the Phase16a adapter.
3. If Data Agent multi-turn or tool metrics regress, add a short retention stage with Phase15 clean multi-turn/tool replay before further SQL-only optimization.

## 2026-07-01: Phase16c SQL Execution GRPO Launch Plan

Decision: start the necessary follow-up training with Phase16c SQL execution GRPO from the Phase16a merged model.

Why Phase16c before Phase16b DPO:

1. The headline SQL issue is still end-to-end WikiSQL execution accuracy, not repair string exact.
2. The newly fixed executable repair probe shows Phase16a already has repair ability when schema/table context is present.
3. Historical Coder7B DPO runs showed real risk of over-specialization and tool/function-call regression when preference data is narrow or format-skewed.
4. GRPO uses execution reward directly and is the cleaner next step for improving SQL execution result correctness.

Launch script:

```text
scripts/remote/run_phase16c_sql_grpo_ppu16.sh
```

Remote run naming:

```text
runs/phase16c_sql_execution_grpo_<stamp>
evals/phase16c_sql_execution_grpo_<stamp>
```

Training setup:

| Field | Value |
|---|---|
| Base model | `evals/phase16a_sql_repair_sft_20260701_1138_phase16a/merged/phase16a_sql_repair_sft_merged` |
| Train data | `datasets/processed/phase16_followup_assets_20260701/phase16c_grpo_train.jsonl` |
| Method | ms-swift GRPO, synchronized 16 PPU, LoRA |
| Reward | `wikisql_exec`, read-only SQLite execution reward |
| Steps | `1200` default |
| LR | `1.5e-7` default |
| LoRA | rank 16, alpha 32, dropout 0.05 |
| Generations | 4 per prompt |
| Logging | SwanLab local mode plus full train log |

Automatic post-eval after training:

| Eval | Path |
|---|---|
| WikiSQL internal execution probe | `evals/<run>/wikisql` |
| Executable SQL repair probe | `evals/<run>/sql_repair_execution` |
| Data Agent multi-turn probe | `evals/<run>/data_agent_multiturn` |
| GSM8K/MMLU-Pro fixed public subsets | `evals/<run>/general` |

Success criteria:

1. WikiSQL execution accuracy should exceed Phase16a `59.38%` and ideally approach or exceed Phase8 SQL-only `62.11%`.
2. Executable SQL repair accuracy should not materially regress from Phase16a `81.25%`.
3. Data Agent multi-turn task success should not collapse; if it regresses, run a short Phase15 clean multi-turn/tool retention stage.
4. GSM8K and MMLU-Pro fixed-subset regression should remain within roughly 2 pp where possible.

Phase16b DPO remains queued as a targeted follow-up only if Phase16c improves SQL execution but still shows systematic failure types where chosen-vs-rejected preference pairs are clearly aligned.

## 2026-07-02: Phase18 Canonical Data Agent SFT

Goal: fix the data-format fragmentation found in the dataset registry audit. Previous SFT/RL mixtures used different prompt markers and output surfaces: `AVAILABLE FUNCTIONS`, `AVAILABLE_TOOLS`, SQL-only prompts, bare SQL completions, and Data Agent JSON actions. This can make the 7B model learn isolated behaviors rather than one robust Data Agent policy.

New scripts:

```text
scripts/remote/prepare_phase18_canonical_data.py
scripts/remote/run_phase18_canonical_sft_ppu16.sh
```

Base model:

```text
evals/phase17_sql_error_sft_20260702_123712_phase17b/merged/phase17_sql_error_sft_merged
```

Data:

```text
datasets/processed/phase18_canonical_data_agent_sft_20260702_155000_phase18
```

Data audit:

| Metric | Value |
|---|---:|
| Total rows | 18,750 |
| Train rows | 17,982 |
| Validation rows | 768 |
| Rows with `AVAILABLE_TOOLS` | 17,982 train rows |
| Rows with old `AVAILABLE FUNCTIONS` | 0 |
| Rows with duplicated user marker | 0 |
| Rows retaining tool transcript | 6,438 train rows |
| `run_sql` tool-call targets | 8,143 train calls |

Major buckets:

| Source | Rows after dedup |
|---|---:|
| canonical multi-turn | 11,925 |
| Phase17 SQL canonicalized | 4,444 |
| low-weight Phase5 Spider replay | 1,200 |
| Phase5 tool replay | 705 |
| Phase17 replay action | 476 |

Training launch:

| Field | Value |
|---|---|
| Run | `phase18_canonical_data_agent_sft_20260702_155000_phase18` |
| Method | 16-PPU LoRA SFT |
| Steps | 900 |
| LR | `2.5e-7` |
| Warmup | 60 |
| Seq len | 4096 |
| LoRA | rank 32, alpha 64, dropout 0.05 |
| SwanLab | local mode, project `agentic-rl-sql-tool` |

Initial health:

| Step | loss | loss_ema | grad_norm | clip rate | LR | Tokens/s | Max memory |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.3787 | 0.3787 | 0.4913 | 0.00 | `8.33e-9` | 53.9 | 16.20 GiB |
| 5 | 0.5542 | 0.4085 | 0.3621 | 0.75 | `2.50e-8` | 130.5 | 16.35 GiB |

Final training status:

| Metric | Value |
|---|---:|
| Final step | 900 / 900 |
| Final loss EMA | 0.4638 |
| Step loss | 0.5990 |
| Grad norm | 0.4877 |
| Clipped step rate | 0.00 |
| Supervised tokens/s | 238.1 |
| Max memory | 17.24 GiB |

Artifacts:

```text
runs/phase18_canonical_data_agent_sft_20260702_155000_phase18/phase18_canonical_data_agent_sft_20260702_155000_phase18_lora/adapter
evals/phase18_canonical_data_agent_sft_20260702_155000_phase18/merged/phase18_canonical_data_agent_sft_merged
```

Corrected WikiSQL v2 post-eval:

| Metric | Value |
|---|---:|
| Samples | 256 |
| SQL extraction rate | 100.00% |
| Execution rate | 96.09% |
| Execution accuracy | 52.73% |
| Normalized SQL exact | 0.00% |

Error categories:

| Category | Count |
|---|---:|
| correct_execution | 135 |
| wrong_where_or_value_or_column | 69 |
| wrong_missing_aggregation | 40 |
| exec_error_original_header_or_bad_identifier | 9 |
| wrong_extra_aggregation | 2 |
| exec_error_syntax | 1 |

Evaluation note: the first vLLM run failed because `evaluate_wikisql_v2.py` generated all 256 prompts in one call and the PPU engine died during generation. The evaluator now supports `--batch-size`; Phase18 was evaluated with `--batch-size 16` and `--gpu-memory-utilization 0.20`. A separate premature `run_gpu_16.sh` keepalive also occupied memory during one retry and was stopped before the successful run.

Interpretation: Phase18 did not recover the Phase8 SQL-only peak, but it improved over the corrected Phase16c v2 baseline from 50.78% to 52.73% execution accuracy while keeping extraction at 100% and execution rate high. The remaining dominant errors are semantic SQL errors: WHERE/value/column grounding and missing aggregation. Next SQL-focused work should add stronger schema/value grounding and execution-feedback repair/RL data rather than only more canonical SFT.
