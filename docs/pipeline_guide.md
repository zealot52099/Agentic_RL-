# MCP Agent 训练评测流程

> 可复用的端到端 pipeline，适用于 Coder7B/Coder14B/其他模型迁移

## 一、环境准备

```bash
# 硬件: 1-16 PPU/GPU，单卡 7B LoRA 约需 20GB 显存
# 依赖: torch, transformers, peft, trl, swanlab, bfcl_eval, sqlparse

pip install peft trl swanlab bfcl-eval sqlparse
```

## 二、数据准备

### 2.1 数据格式标准

**所有训练数据统一为一种格式**。推荐 OOD native 格式（训练=评测对齐）：

```
AVAILABLE FUNCTIONS:
[{函数定义}]

USER:
{用户query}

ASSISTANT:
{completion}
```

### 2.2 必需数据源

| 数据源 | 数量 | 用途 | 来源 |
|---|---|---|---|
| MCP 工具调用 | 6K+ | 函数选择 | 自建/开源 |
| 并行调用 | 2K | 数组输出 | MCP合成或开源 |
| SQL 函数调用 | 3K | SQL 生成 | Spider + BFCL格式对齐 |
| Math COT | 500 | 推理能力 | GSM8K |
| Clarify/No-tool | 1.5K | 拒调/澄清 | MCP smoke |

### 2.3 格式规则（关键！）

```python
# ✅ 正确: 所有值用双层嵌套
columns: [["name"], ["age"]]
conditions: [["id = 1"]]

# ❌ 错误: 单层
columns: ["name", "age"]

# ✅ 正确: function calling 用 name + arguments
{"name": "query_records", "arguments": {"field": "project"}}

# ❌ 错误: MCP 格式(训练和评测不一致)
[{"server_id": "database", "tool_name": "query_records", "arguments": {...}}]
```

### 2.4 禁止事项

- ❌ 测试集数据**绝对不可**进入训练集
- ❌ 不要在数据中混用多种 prompt 格式
- ❌ 不要让 prompt 和 completion 格式矛盾（如在 prompt 说"输出 XML"但 completion 是 JSON）

## 三、训练

### 3.1 配置

```bash
# 单卡 (推荐，稳定)
CUDA_VISIBLE_DEVICES=0 python3 train_sft.py

# 多卡 DDP
torchrun --nproc_per_node=16 train_lora_sft.py
```

| 参数 | 值 |
|---|---|
| Model | Qwen2.5-Coder-7B-Instruct |
| LoRA | r=32, alpha=64, dropout=0.05 |
| LR | 2e-6 (SQL), 5e-6 (标准) |
| Steps | 300-500 |
| Warmup | 50 steps |
| BS | 2 (单卡) / 16 (16PPU) |
| Seq Len | 2048 |

### 3.2 训练脚本模板

```bash
#!/bin/bash
set -e
source /usr/local/PPU_SDK/envsetup.sh  # PPU 环境，普通 GPU 跳过
export CUDA_VISIBLE_DEVICES=0
export TRAIN_DATA=/path/to/train.jsonl
export OUTPUT_DIR=/path/to/output_$(date +%Y%m%d_%H%M%S)
export SWANLAB_RUN=experiment_name
mkdir -p $OUTPUT_DIR

python3 train_sft.py 2>&1 | tee -a $OUTPUT_DIR/train.log
```

## 四、评测

### 4.1 评测矩阵

| 评测 | 格式 | 指标 | 数据路径 |
|---|---|---|---|
| BFCL V4 Live | OOD native | Func name accuracy | `bfcl_eval/data/BFCL_v4_live_*.json` |
| BFCL V3 SQL | OOD native | Func + Exact | `eval_suite/.../BFCL_v3_sql.json` |
| BFCL Multi-Turn | OOD native | JSON validity | `bfcl_eval/data/BFCL_v4_multi_turn_base.json` |
| BFCL Web Search | OOD native | JSON validity | `bfcl_eval/data/BFCL_v4_web_search.json` |
| Self-built Parallel | OOD native | Parallel accuracy | 自建 |

### 4.2 Header→colN 映射（关键！WikiSQL +26~41pp）

模型训练时使用 Spider 等语义化列名（`name`, `age`），但 WikiSQL 评测使用物理列名（`col0`, `col1`）。必须在评测端做映射：

```python
def remap_headers(sql: str, headers: list) -> str:
    """Remap display header names to colN identifiers (quoted + unquoted)."""
    for i, h in enumerate(headers):
        col = f"col{i}"
        h_str = str(h).strip()
        if not h_str or h_str.startswith("col"):
            continue
        # Quoted: "DisplayName" -> "colN"
        sql = sql.replace(chr(34) + h_str + chr(34), chr(34) + col + chr(34))
        sql = sql.replace(chr(34) + h_str.lower() + chr(34), chr(34) + col + chr(34))
        sql = sql.replace(chr(34) + h_str.upper() + chr(34), chr(34) + col + chr(34))
        # Unquoted: bare DisplayName -> colN (word boundary)
        sql = re.sub(r'\b' + re.escape(h_str) + r'\b', col, sql, flags=re.IGNORECASE)
    return sql

# 在 extract_sql() 之后、执行之前调用
if sql is not None and row.get("header"):
    sql = remap_headers(sql, row["header"])
```

⚠️ **必须同时处理引号和无引号标识符**。仅处理引号 → +26pp，加上无引号 → +41pp（26.6%→67.6%）。

### 4.3 SQL Exact 评分关键（BFCL V3 旧格式）

```python
def nv(v):  # 归一化：递归展平嵌套列表
    if isinstance(v, list):
        flat = []
        def _f(x):
            if isinstance(x, list):
                for i in x: _f(i)
            else: flat.append(str(x).strip())
        _f(v)
        return ','.join(sorted(flat))
    return str(v).strip()

# 然后逐参数比较 nv(pred[pk]) == nv(gt[pk])
```

### 4.3 BFCL Live 评分

```python
# 提取 model output 中的函数名（支持裸JSON和数组）
obj = safe_parse(model_output)
if isinstance(obj, dict): obj = [obj]
pred_names = set(item.get('name','') for item in obj if isinstance(item, dict))
expected_names = set(f['name'] for f in function_definitions)
accuracy = 1 if pred_names & expected_names else 0
```

## 五、完整流程

```
1. 数据准备
   ├── 确认所有数据源格式统一
   ├── SQL 参数全部双层嵌套
   ├── 测试集不入训练集
   └── Prompt 和 completion 格式一致

2. 格式验证（训练前必做）
   ├── 每个数据源抽1个样例展示
   ├── 对比训练样例 vs 测试样例格式
   └── 检查 SQL 参数嵌套是否正确

3. 训练
   ├── 单卡优先保证稳定性
   ├── 观察 loss 收敛（7B 模型 step 50-100 内应收敛到 <1.0）
   └── 异常: loss 卡在 >2.0、step 1 后无进度

4. 评测
   ├── BFCL Live + SQL + Multi-Turn + Web Search
   ├── 结果写入 log 文件（nohup > logfile）
   └── 启动 GPU 占用脚本

5. 结果分析
   ├── 对比历史最佳
   ├── 检查 SQL Exact vs Func 差距
   └── 决定下步方向（GRPO/更多数据/新基座）
```

## 六、关键教训

| # | 教训 | 后果 |
|---|---|---|
| 1 | **测试集绝对不能进训练集** | 虚高 6-12pp |
| 2 | **prompt 和 completion 必须格式一致** | Exact=0% |
| 3 | **SQL 参数格式必须对齐评测** | 反复从 59% 降到 18% |
| 4 | **不要用 regex 提取 SQL 参数** | 基座知识被覆盖 |
| 5 | **SSH 输出必须 nohup > logfile** | 无数次结果丢失 |
| 6 | **单卡比 16-PPU DDP 更稳定** | 训练反复崩溃 |
| 7 | **训练前先验证数据格式** | 节省无效训练时间 |
| 8 | **Coder7B 的 SQL 知识来自预训练** | 外部 SQL 数据可能污染 |

## 七、详细路径

### 训练集

| 数据集 | 路径 | 数量 | 用途 |
|---|---|---|---|
| MCP Smoke 原始 | `/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/datasets/processed/mcp_lora_sft_v3_20260618/sft_v2_all/train_sft.jsonl` | 19,494 | 基础工具选择 |
| 合成并行 | `/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/datasets/processed/mcp_lora_sft_v3_20260618/sft_clean_parallel/train_clean_parallel.jsonl` | 2,000 | 数组输出 |
| Spider (SQL) | `/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/datasets/modelscope/spider_train.jsonl` | 7,000 | SQL 生成 |
| WikiSQL | `/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/datasets/modelscope/wikisql_test.parquet` | 7,500 | SQL 多样性 |
| HERMES FC | `/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/datasets/modelscope/hermes_fc_v1_train.jsonl` | 1,000 | 工具调用 |
| Glaive FC v2 | `/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/datasets/modelscope/glaive_fc_v2_train.jsonl` | 1,000 | 工具调用 |
| SQL Create Context | `/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/datasets/modelscope/sql_create_context.jsonl` | 500 | SQL |
| **Phase 3 (OOD 全量)** | `/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/datasets/processed/phase3_final/train_phase3.jsonl` | 18,000+ | 零泄漏 |
| **Phase 4v2 (In-dist)** | `/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/datasets/processed/phase4v2/train_phase4v2.jsonl` | 12,500 | chat_template |
| GRPO prompt 池 | `/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/datasets/processed/grpo_pool/grpo_prompts.jsonl` | 3,305 | RL 训练 |

### 测试集

| 数据集 | 路径 | 数量 | 指标 |
|---|---|---|---|
| BFCL V4 live_simple | `/opt/ac2/lib/python3.12/site-packages/bfcl_eval/data/BFCL_v4_live_simple.json` | 257 | Func name |
| BFCL V4 live_multiple | `.../BFCL_v4_live_multiple.json` | 1,052 | Func name |
| BFCL V4 live_parallel | `.../BFCL_v4_live_parallel.json` | 15 | Parallel |
| BFCL V4 live_parallel_multiple | `.../BFCL_v4_live_parallel_multiple.json` | 23 | Parallel+ |
| BFCL V3 SQL | `/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/datasets/eval_suite/huggingface/gorilla-llm__Berkeley-Function-Calling-Leaderboard/BFCL_v3_sql.json` | 100 | Func+Exact |
| BFCL V3 SQL GT | `/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/datasets/eval_suite/.../possible_answer/BFCL_v3_sql.json` | 100 | GT |
| BFCL V4 Multi-Turn | `/opt/ac2/.../BFCL_v4_multi_turn_base.json` | 200 | JSON |
| BFCL V4 Web Search | `/opt/ac2/.../BFCL_v4_web_search.json` | 100 | JSON |
| GSM8K | `/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/datasets/eval_suite/huggingface/openai__gsm8k/main/test-00000-of-00001.parquet` | 1,319 | Math |
| IFEval | `/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/datasets/eval_suite/huggingface/google__IFEval/ifeval_input_data.jsonl` | 540 | 指令跟随 |
| HumanEval+ | `/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/datasets/eval_suite/huggingface/openai__openai_humaneval/openai_humaneval/test-00000-of-00001.parquet` | — | 代码 |
| MBPP | `/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/datasets/eval_suite/huggingface/google-research-datasets__mbpp/` | — | 代码 |
| SWE-bench Lite | `/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/datasets/eval_suite/huggingface/SWE-bench__SWE-bench_Lite/` | — | 软件工程 |

### 模型产出

| 模型 | 路径 | BFCL | SQL Exact |
|---|---|---|---|
| Round 4 (最佳 BFCL) | `/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/output/coder7b_round4_sql_sft_20260623_220257/adapter` | 93.2% | 18% |
| Mixed SFT (最佳 SQL) | `/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/output/coder7b_mixed_sft_20260621_001502/adapter` | 82.4% | **59%** |
| Phase 3 (全量 OOD) | `/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/output/coder7b_phase3_1gpu_20260625_221844/adapter` | 92.6% | 24% |
| Phase 4v2 (In-dist) | `/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/output/coder7b_phase4_1gpu_*/adapter` | 🔵训练中 | — |
| 1.5B GRPO | `/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/output/grpo_15b_dpo_20260621_153153/adapter` | 83.5% | 17% |

### 评测脚本

| 脚本 | 绝对路径 |
|---|---|
| SQL 评测 (nv归一化) | `/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/scripts/eval_sql_fixed.py` |
| BFCL Live + SQL 联合 | `/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/scripts/eval_coder7b.py` |
| Phase 4 In-dist | `/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/scripts/eval_phase4.py` |
| DeepSeek API 评测 | 本地: `C:/Users/Xpeng/run_deepseek_eval.py` |
| 数据构建 Phase 3 | `/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/scripts/build_phase3_final.py` |
| 数据构建 Phase 4v2 | `/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/scripts/build_p4v2_final.py` |
| 训练启动单卡 | `/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/scripts/train_coder7b_sft.py` |
| 训练启动多卡 | `/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/scripts/remote/train_lora_sft.py` |
| GRPO 训练 | `/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/scripts/train_grpo_c7b_fixed.py` |
| GPU 占用 | `/code/run_gpu_16.sh` |

### 文档

| 文档 | 绝对路径 |
|---|---|
| 流程指南 | `/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/docs/pipeline_guide.md` |
| 实验记录 | `/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/docs/07_experiment_records.md` |
| 周报 | `/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/docs/weekly_report_2026-06-22.md` |
| 周报 | `docs/weekly_report_2026-06-22.md` |
| 训练计划 Round 2 | `docs/training_plan_round2.md` |

## 八、最新结果 (2026-07-05)

### Phase27: 轻量 SFT + Header→colN 映射

最佳 WikiSQL 执行准确率: **67.6%**

```
Phase27 轻量 SFT (3K 样本, 200步) → 53.9% baseline
  + 引号 header remap           → 61.3%
  + 无引号 header remap         → 67.6% (最终)
```

### 关键发现

1. **87% 的 SQL 错误是列名混淆**（显示名 vs 物理 colN）
2. **GRPO 在 SFT 收敛后无效**：三种 GRPO 配置（dense/binary/不同步数）都只变 <1pp
3. **3K 样本比 15K 更好**：过量 SFT 锁死模型，GRPO 无优化空间
4. **引号+无引号 remap 缺一不可**：仅引号 52.7%，加上无引号 70.7%（离线验证），正式版 67.6%
