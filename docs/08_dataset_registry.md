# 8. 数据集台账与指标影响

本文档维护项目中所有训练集、验证集和评测集的统一台账。目标是让每一次训练都能回答五个问题：

1. 这批数据从哪里来，为什么加入。
2. 样本数量是多少，创建日期是什么。
3. 进入训练或评测前后的格式是什么。
4. 用了哪个脚本生成，完整样例长什么样。
5. 加入之后，对 SQL、tool-call、多轮和通用指标有什么影响。

原则：公开 benchmark 的 test/eval 数据只作为评测资产，不得进入训练。内部 probe、官方 benchmark、公开论文数字必须分开标记。

## 快速索引

| ID | 阶段 | 数据集/资产 | 类型 | 创建日期 | 数量 | 远端路径 | 处理脚本 | 指标影响 |
|---|---|---|---|---|---:|---|---|---|
| `mcp_lora_sft_v3` | MCP SFT v3 | MCP smoke / trace SFT v2 all | 训练+验证 | 2026-06-18 | train 19,494; val 506 | `datasets/processed/mcp_lora_sft_v3_20260618/sft_v2_all` | `scripts/remote/prepare_mcp_sft_v2.py` | 修复 clarify/no-tool 同为 `[]` 的标签冲突；用于稳定 MCP/tool-call LoRA SFT。需用内部 MCP/xLAM 指标继续量化。 |
| `xlam_fc_60k` | Tool-call 基础 | xLAM Function Calling 60K | 训练+heldout eval | 2026-06-09 起 | 约 60K 原始样本，split 后以 manifest 为准 | `datasets/processed/xlam-function-calling-60k` | `scripts/prepare_remote_datasets.py`; `scripts/prepare_xlam_splits.py` | 提升单轮函数调用格式、tool name 和参数 JSON；不是 MCP 多轮环境数据，不能替代 BFCL 官方分数。 |
| `swe_gym_openhands` | Agent/代码轨迹 | SWE-Gym OpenHands SFT Trajectories | 训练 | 2026-06-09 起 | 以 processed manifest 为准 | `datasets/processed/swe-gym-openhands-sft/train.jsonl` | `scripts/prepare_remote_datasets.py`; `scripts/remote/prepare_sft_v4_agent_mixture.py` | 用于终端/代码 agent 轨迹回放；当前未作为 SQL/tool-call 主指标来源。 |
| `general_replay` | 能力保持 | Tulu/SmolTalk/No-Robots 等通用回放 | 训练 | 2026-06-18 起 | 按 mixture 抽样 | `datasets/sources/tulu-*`; `datasets/sources/smol-*`; `datasets/sources/no-robots` | `scripts/remote/prepare_sft_v3_mixture.py`; `scripts/remote/prepare_sft_v4_agent_mixture.py` | 用于控制 IFEval/GSM8K/MMLU-Pro 回退；影响需按每个训练阶段的 general eval 报告更新。 |
| `phase5_unified` | Data Agent SFT | Unified Data Agent action schema | 训练+验证 | 2026-06-26 | all 15,500; train 14,962; val 538 | `datasets/processed/phase5_unified_20260626` | `scripts/build_phase5_unified.py` | 建立统一 action schema；早期 1.5B smoke 主要验证训练稳定性，非最终效果结论。 |
| `phase15_multiturn_v1` | 多轮 Data Agent | Executable multi-turn traces v1 | 训练+评测 | 2026-06-30 | train traces 2,400; eval 360; validation turns 1,062 | `datasets/processed/phase15_data_agent_multiturn_20260630` | `scripts/remote/prepare_data_agent_multiturn.py` | 引入多轮 tool observation、状态推进和任务成功率评测；内部 executable probe，非官方 benchmark。 |
| `phase15_eval_v2` | 多轮 Data Agent | Multi-turn eval v2 | 评测 | 2026-06-30 | eval 2,000 | `datasets/processed/phase15_data_agent_multiturn_eval_v2_20260630` | `scripts/remote/prepare_data_agent_multiturn.py` | 扩大多轮内部评测覆盖；用于检查 multi-turn success、约束满足和执行稳定性。 |
| `phase15_clean_v4` | 多轮 Data Agent | Cleaned multi-turn SFT/RL/eval v4 | 训练+RL+评测 | 2026-06-30 | mixture train 38,223; multiturn-only 26,223; RL 10,000; eval 2,000 | `datasets/processed/phase15_multiturn_clean_v4_20260630` | `scripts/remote/prepare_phase15_multiturn_sft_mixture.py`; `scripts/remote/clean_phase15_multiturn_data.py` | 修复 synthetic trace 重复问题；推荐用于后续多轮 SFT/retention。保留 Phase6 SQL/tool replay，降低单轮 SQL/tool 指标回退风险。 |
| `phase16_sql_repair` | SQL SFT/GRPO | SQL repair + Spider + SQL-context + replay | 训练+评测 | 2026-07-01 | repair SFT 4,137; repair eval 64; GRPO 4,088; mixture SFT 12,133; taxonomy 113 | `datasets/processed/phase16_sql_repair_20260701` | `scripts/remote/prepare_phase16_sql_repair_data.py` | Phase16a executable repair accuracy 达 81.25%；旧 normalized exact 0% 被确认为评测设计问题。WikiSQL execution 目标仍需继续优化。 |
| `phase16_followup_assets` | SQL follow-up | Phase16b/16c train and eval assets | 训练+评测 | 2026-07-01 | DPO train 4,967; DPO holdout 256; GRPO train 4,088; smoke 256; WikiSQL eval 256; multi-turn eval 500; tool probe 307 | `datasets/processed/phase16_followup_assets_20260701` | `scripts/remote/prepare_phase16_followup_assets.py` | 固定 Phase16 后续训练/评测口径；Phase16c 使用 executable WikiSQL GRPO reward。 |
| `phase17_sql_error_sft` | SQL error SFT | Corrected WikiSQL v2 + Phase16c 错例 SFT | 训练+评测 | 2026-07-02 | Phase17 rows 380; repeated 3,040; replay 5,000; total 8,040; eval 256 | `datasets/processed/phase17_sql_error_sft_20260702_123712_phase17b` | `scripts/remote/prepare_phase17_sql_eval_and_sft.py`; `scripts/remote/evaluate_wikisql_v2.py`; `scripts/remote/run_phase17_sql_error_sft_ppu16.sh` | 修正 WikiSQL 大小写和 prompt 字段误导问题；corrected Phase16c baseline 为 execution accuracy 50.78%、execution rate 95.31%。Phase17 训练后影响待补。 |
| `phase18_canonical_data_agent_sft` | Canonical Data Agent SFT | SQL/tool/multi-turn 统一 action schema 重构数据 | 训练+验证 | 2026-07-02 | total 18,750; train 17,982; val 768 | `datasets/processed/phase18_canonical_data_agent_sft_20260702_155000_phase18` | `scripts/remote/prepare_phase18_canonical_data.py`; `scripts/remote/run_phase18_canonical_sft_ppu16.sh`; `scripts/remote/evaluate_wikisql_v2.py` | corrected WikiSQL v2: execution accuracy 52.73%、execution rate 96.09%、SQL extraction 100%。相对 Phase16c corrected baseline 50.78% 小幅提升，但仍低于 Phase8 SQL-only 62.11%。 |
| `phase19_sql_grounding_sft` | SQL grounding SFT | WikiSQL schema/value grounding + synthetic SQL repair + Phase18 replay | 训练+评测 | 2026-07-02 | 待远端 manifest 生成 | `datasets/processed/phase19_sql_grounding_sft_<stamp>` | `scripts/remote/prepare_phase19_sql_grounding_data.py`; `scripts/remote/run_phase19_sql_grounding_sft_ppu16.sh` | 目标修复 Phase18 的 `wrong_where_or_value_or_column` 与 `wrong_missing_aggregation`；fixed WikiSQL v2 256 probe 不进入训练，训练后补指标。 |
| `eval_suite_public` | 通用/官方评测 | IFEval/BFCL/SWE-bench/LiveCodeBench/HumanEval/MBPP/GSM8K/MMLU-Pro | 评测 | 2026-06-10 起 | 按各 benchmark 官方 split | `datasets/eval_suite` | `scripts/prepare_eval_suite.py` | 用于公开指标对齐和能力回归。不得训练；官方 scorer 未接入的指标只能标记为内部/待接入。 |
| `wikisql_internal_probe` | SQL 评测 | WikiSQL internal execution probe | 评测 | 2026-06-10 起；v2 2026-07-02 | 256 | `datasets/processed/phase16_followup_assets_20260701/wikisql_eval_256.*`; Phase17 normalized variant | `scripts/remote/evaluate_wikisql.py`; `scripts/remote/evaluate_wikisql_v2.py` | Phase8 SQL-only GRPO 曾达 62.11% execution accuracy；Phase9/10 mixed 约 55.86%；Phase16c corrected v2 baseline 50.78%。内部 probe，非官方 WikiSQL benchmark。 |
| `sql_repair_execution_eval` | SQL repair 评测 | Executable SQL repair probe | 评测 | 2026-07-01 | 128 | `datasets/processed/phase16_followup_assets_20260701/sql_repair_execution_eval/sql_repair_execution_eval_128.jsonl` | `scripts/remote/prepare_sql_repair_execution_eval.py`; `scripts/remote/evaluate_sql_repair_execution.py` | Phase16a execution repair accuracy 81.25%，normalized SQL exact 28.12%；替代旧 0% exact-only 误导指标。 |
| `data_agent_action_probe` | Tool/Data Agent 评测 | Data Agent JSON action/tool-call probe | 评测 | 2026-07-01 | 307 | `datasets/processed/phase16_followup_assets_20260701/data_agent_tool_action_probe.jsonl` | `scripts/remote/prepare_phase16_followup_assets.py` | 用于 JSON/action/tool name/args 回归，避免 SQL-only 训练损伤 tool-call。影响按 post-eval 更新。 |
| `mcp_xlam_array_smoke` | Tool smoke | MCP array-format smoke probe | 评测 | 2026-07-01 | 5 | `datasets/processed/phase16_followup_assets_20260701/mcp_xlam_array_tool_probe.jsonl` | `scripts/remote/prepare_phase16_followup_assets.py` | 仅 smoke test，样本太少，不能作为 headline metric。 |

## 2026-07-02 真实数据格式审计

本节记录对远端真实 processed JSONL 的抽样审计。结论是：上一版本文档中的部分“完整样例”是目标格式示意，不是远端真实样本逐字复制；其中 `phase5_unified` 样例尤其需要修正。后续台账中的样例必须优先来自真实数据，若是目标 schema 示例，必须显式标注。

| 数据集 | 真实消费字段 | 文档上一版是否完全一致 | 工具/schema 上下文实际位置 | 影响判断 |
|---|---|---|---|---|
| `phase5_unified` | `id,prompt,completion,mixture_source,source,loss_weight`，Spider 行额外有 `db_id` | 否。上一版写成 `messages/schema/completion/metadata`，与真实文件不一致 | 工具定义不在 top-level 字段，而是内嵌在 `prompt` 的 `AVAILABLE FUNCTIONS` 中 | 训练脚本可正常消费，但文档误导排查；SQL 行只有 `execute_sql` 工具和很弱 schema context，会限制 schema grounding。 |
| `phase15_clean_v4` | `id,prompt,completion,source,mixture_source,loss_weight,phase15_bucket,dedup_key` 等 | 部分一致 | 多轮工具定义在 `prompt` 的 `AVAILABLE_TOOLS` 中，非 top-level | 对训练可用；但与 Phase5 的 `AVAILABLE FUNCTIONS` 命名不一致，会增加模型学习格式负担。 |
| `phase16_sql_repair` | 统一为 SFT `prompt/completion`，额外带 `gold_sql/gold_result/value_hints/failure_type` 或 replay 字段 | 部分一致 | SQL repair 行通常无工具列表；multi-turn replay 行有 `AVAILABLE_TOOLS` | SQL 与 tool-call 数据形态割裂，容易造成 SQL-only 提升时 tool-call 回退。 |
| `phase16c_grpo_train` | `id,query,solution,gold_sql,question,source,value_hints` | 部分一致 | 无工具列表；`query` 中包含 table、columns、sample rows、value hints | 适合 SQL execution GRPO；不训练 Data Agent action schema 或 tool selection。 |
| `phase17_sql_error_sft` | 统一为 SFT `prompt/completion`，包含 Phase17 SQL 错例和 Phase16 replay | 部分一致 | SQL 错例无工具列表；multi-turn replay 有 `AVAILABLE_TOOLS` | 用于修复 SQL 失败模式；若比例过高，会进一步放大 SQL 与 tool-call 格式割裂。 |

字段不完全一致分两层看：

1. **训练器层面可以接受**：当前 SFT 训练器读取 `prompt/completion/loss_weight`，GRPO 读取 `query/solution`，所以 top-level 字段不同不会直接报错。
2. **模型学习层面有真实风险**：同一阶段内 prompt 模板、工具命名、SQL schema 表达、completion action schema 不统一，会让小模型把能力学成碎片。表现通常是 JSON/action 格式稳定但 tool 泛化弱，SQL 可以在单一 probe 上提升但 Data Agent tool-call 或 multi-turn 指标回退。

后续解决规则：

1. 每个训练阶段必须定义一个 **stage-level canonical schema**，例如 SFT 统一 `prompt/completion/source/mixture_source/loss_weight`，RL 统一 `query/solution/verifier/task_type`。
2. 每条样本增加 `render_template_version` 和 `action_schema_version`，区分 `AVAILABLE FUNCTIONS`、`AVAILABLE_TOOLS`、SQL-only prompt。
3. Data Agent SFT/RL 的 tool-call 与 SQL 样本都应显式暴露同一套工具：`inspect_schema/list_tables/run_sql/final_answer` 或 `execute_sql`，不要让 SQL 样本只训练裸 SQL、tool 样本只训练 action JSON。
4. 保留上游原始字段到 `metadata`，但训练消费字段必须统一。
5. 文档样例必须从真实数据抽样；如果出于可读性做脱敏/缩短，需要标注“结构等价、非逐字样本”。

## Phase18 Canonical Data Agent SFT

Phase18 是对上面审计问题的直接修复：不继续混用 `AVAILABLE FUNCTIONS`、`AVAILABLE_TOOLS`、SQL-only prompt 和裸 SQL completion，而是把 SQL、tool-call、多轮样本统一渲染为 Data Agent action schema。

远端数据目录：

```text
datasets/processed/phase18_canonical_data_agent_sft_20260702_155000_phase18
```

生成脚本：

```text
scripts/remote/prepare_phase18_canonical_data.py
```

训练脚本：

```text
scripts/remote/run_phase18_canonical_sft_ppu16.sh
```

Manifest 摘要：

```json
{
  "dataset_id": "phase18_canonical_data_agent_sft",
  "created_at": "2026-07-02",
  "render_template_version": "phase18_data_agent_action_v1",
  "action_schema_version": "data_agent_action_v1",
  "counts": {
    "total": 18750,
    "train": 17982,
    "validation": 768,
    "pre_dedup_buckets": {
      "phase5_tool": 4500,
      "phase5_spider_low_weight": 1200,
      "phase15_multiturn": 12000,
      "phase17_sql": 7276,
      "phase17_replay_action": 764
    },
    "source_counts": {
      "phase18_multiturn_canonical": 11925,
      "phase18_phase17_sql": 4444,
      "phase18_phase5_spider_low_weight": 1200,
      "phase18_phase5_tool": 705,
      "phase18_phase17_replay_action": 476
    }
  }
}
```

质量审计：

```json
{
  "train_rows": 17982,
  "available_tools_rows": 17982,
  "old_available_functions_rows": 0,
  "multi_user_marker_rows": 0,
  "transcript_with_tool_rows": 6438,
  "run_sql_calls": 8143
}
```

处理策略：

- 所有 SFT 行统一使用 `prompt/completion/source/mixture_source/loss_weight`。
- 所有 prompt 统一使用 `AVAILABLE_TOOLS`。
- SQL completion 统一表达为 `{"action":"tool_call","calls":[{"name":"run_sql","arguments":{"sql":"..."}}]}`。
- Spider 缺 schema 行只保留为低权重 SQL format replay，`loss_weight <= 0.35`。
- Multi-turn 样本保留中间 `ASSISTANT/TOOL` transcript，只去掉最后待生成的 assistant marker。

完整样例：

```json
{
  "id": "phase18-sql-phase5_spider_low_weight-spider-2296",
  "prompt": "You are the main model of a Data Agent. Use the provided tools when they are needed, ask for clarification when required information is missing, refuse unsafe requests, or provide a final answer when no tool is needed. Return exactly one JSON object and no extra text.\n\nAllowed actions:\n{\"action\":\"tool_call\",\"calls\":[{\"name\":\"tool_name\",\"arguments\":{...}}]}\n{\"action\":\"clarify\",\"missing\":[\"field\"],\"message\":\"question\"}\n{\"action\":\"refuse\",\"message\":\"reason\"}\n{\"action\":\"final\",\"answer\":\"answer\"}\n\nAVAILABLE_TOOLS:\n[{\"name\":\"list_tables\",\"description\":\"List available database tables.\",\"input_schema\":{\"type\":\"object\",\"properties\":{},\"additionalProperties\":false}},{\"name\":\"describe_table\",\"description\":\"Return columns and types for one table.\",\"input_schema\":{\"type\":\"object\",\"required\":[\"table\"],\"properties\":{\"table\":{\"type\":\"string\"}},\"additionalProperties\":false}},{\"name\":\"run_sql\",\"description\":\"Execute one read-only SQLite SELECT query.\",\"input_schema\":{\"type\":\"object\",\"required\":[\"sql\"],\"properties\":{\"sql\":{\"type\":\"string\"}},\"additionalProperties\":false}},{\"name\":\"final_answer\",\"description\":\"Return the final natural-language answer to the user.\",\"input_schema\":{\"type\":\"object\",\"required\":[\"answer\"],\"properties\":{\"answer\":{\"type\":\"string\"}},\"additionalProperties\":false}}]\n\nDATABASE CONTEXT:\ndatabase: entrepreneur\nschema: not provided in this local Spider JSONL snapshot. Use the question and database id; do not invent non-SQL text.\n\nUSER:\nWhat are the names of people who are not entrepreneurs?\n\nASSISTANT:\n",
  "completion": "{\"action\":\"tool_call\",\"calls\":[{\"arguments\":{\"sql\":\"SELECT Name FROM people WHERE People_ID NOT IN (SELECT People_ID FROM entrepreneur)\"},\"name\":\"run_sql\"}]}",
  "source": "phase18_phase5_spider_low_weight",
  "mixture_source": "sql_spider",
  "loss_weight": 0.35,
  "render_template_version": "phase18_data_agent_action_v1",
  "action_schema_version": "data_agent_action_v1",
  "metadata": {
    "db_id": "entrepreneur",
    "upstream_id": "spider-2296",
    "upstream_source": "spider_train",
    "has_value_hints": false
  }
}
```

预期影响：Phase18 主要目标不是单纯拉高 WikiSQL，而是减少 SQL/tool/multi-turn 之间的格式冲突。成功信号应同时看 Data Agent action/tool 指标、multi-turn probe、WikiSQL v2 和通用回归。

## 指标影响摘要

| 数据/阶段 | 加入前参考 | 加入后观察 | 结论 |
|---|---|---|---|
| Phase5 unified action schema | 各数据源输出格式不统一 | 统一为 `tool_call/no_tool/clarify` action；训练链路先跑通 | 格式统一是必要基础，但单靠 Phase5 不足以保证 SQL execution 提升。 |
| Phase8 SQL-only GRPO | Phase5 之后 SQL execution 仍偏低 | WikiSQL internal 256 probe: extraction 100%, execution rate 88.67%, execution accuracy 62.11% | SQL-only executable reward 对 WikiSQL 有收益，但容易牺牲 tool/general retention，需要后续混合保持。 |
| Phase9 mixed SQL+tool GRPO | Phase8 SQL-only 62.11% | WikiSQL execution accuracy 55.86%, execution rate 80.08% | 混合训练保留 tool-call，但 SQL reward 被稀释。 |
| Phase10 staged SQL->mixed GRPO | Phase9 55.86% | WikiSQL execution accuracy 55.86%, execution rate 79.69% | staged retention 没有恢复 SQL-only 峰值，说明需要更强 schema/value grounding 和错误修复数据。 |
| Phase16a SQL repair SFT | legacy repair normalized exact 0% | executable repair accuracy 81.25%; normalized exact 28.12% | 旧 repair exact 评测不合理；带 schema/table/error feedback 的 execution-based repair 能真实反映修复能力。 |
| Phase17 corrected WikiSQL v2 | old probe 存在大小写/字段名误导 | Phase16c corrected baseline: execution accuracy 50.78%, execution rate 95.31% | 评测口径修正后，当前主要错误集中在 where/value/column 与 missing aggregation。 |
| Phase18 canonical Data Agent SFT | Phase16c corrected WikiSQL v2 baseline 50.78% | Phase18 corrected WikiSQL v2: execution accuracy 52.73%, execution rate 96.09%, extraction 100% | 统一 action/schema 对 SQL 有小幅正收益，但主要瓶颈仍是 schema/value grounding 与聚合判断；后续应加入执行反馈修复轨迹和 SQL RL，而不是只扩大普通 SFT。 |
| Phase19 SQL grounding SFT | Phase18 corrected WikiSQL v2 52.73% | 训练中，待 post-eval | 通过显式 relevant columns、candidate values、required aggregations 和 repair feedback 针对性降低语义 SQL 错误。 |

## 完整数据样例

以下样例用于说明每个数据资产进入训练/评测时的消费格式。实际文件可能包含更多 metadata；新增数据必须在本节补一个完整单条样例。

### `mcp_lora_sft_v3` SFT 样例

```json
{
  "id": "mcp_smoke_calendar_000001",
  "mixture_source": "mcp_positive",
  "prompt": "SYSTEM: Return [] when no tool applies. When required information is missing, return one JSON object with action=\"clarify\".\nMCP_SERVER_CATALOG:\n[{\"server\":\"calendar\",\"tools\":[{\"name\":\"calendar.create_event\",\"description\":\"Create a calendar event\",\"parameters\":{\"type\":\"object\",\"required\":[\"title\",\"date\"],\"properties\":{\"title\":{\"type\":\"string\"},\"date\":{\"type\":\"string\"}}}}]}]\nUSER:\nSchedule project review for Friday.\nASSISTANT:",
  "completion": "[{\"action\":\"tool_call\",\"server\":\"calendar\",\"tool\":\"calendar.create_event\",\"arguments\":{\"title\":\"project review\",\"date\":\"Friday\"}}]",
  "loss_weight": 1.0,
  "metadata": {
    "source": "mcp_smoke",
    "schema_fingerprint": "calendar.create_event:v1",
    "split": "train"
  }
}
```

### `xlam_fc_60k` SFT 样例

```json
{
  "id": "xlam_train_000001",
  "prompt_template_version": "xlam_tool_json_v1",
  "prompt": "You are a tool-calling assistant. Select only tools from the provided definitions. Return only a JSON array.\n\nTOOLS:\n[{\"name\":\"get_weather\",\"description\":\"Get weather by city\",\"parameters\":{\"type\":\"object\",\"required\":[\"city\"],\"properties\":{\"city\":{\"type\":\"string\"}}}}]\n\nUSER:\nWhat is the weather in Paris?\n\nASSISTANT:\n",
  "completion": "[{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}]",
  "expected_calls": [
    {
      "name": "get_weather",
      "arguments": {
        "city": "Paris"
      }
    }
  ],
  "metadata": {
    "source": "xlam-function-calling-60k",
    "split_group_id": "get_weather",
    "split": "train"
  }
}
```

### `phase5_unified` Data Agent SFT 样例

```json
{
  "id": "mcp-smoke-20260618-000289",
  "prompt": "You are the main model of a data agent. Decide whether to call a tool, ask for clarification, or answer without tools. Return exactly one JSON object and no extra text. Use this schema: {\"action\":\"tool_call\",\"calls\":[{\"name\":\"tool_name\",\"arguments\":{...}}]} for tool calls; {\"action\":\"no_tool\",\"calls\":[]} when no tool is needed; {\"action\":\"clarify\",\"missing\":[\"field\"],\"message\":\"question\"} when required information is missing.\n\nAVAILABLE FUNCTIONS:\n[{\"server_id\":\"calendar\",\"tools\":[{\"name\":\"create_event\",\"description\":\"Create a calendar event.\",\"input_schema\":{\"type\":\"object\",\"required\":[\"title\",\"start\"],\"properties\":{\"title\":{\"type\":\"string\"},\"start\":{\"type\":\"string\"}}}}]},{\"server_id\":\"files\",\"tools\":[{\"name\":\"read_file\",\"description\":\"Read a text file.\",\"input_schema\":{\"type\":\"object\",\"required\":[\"path\"],\"properties\":{\"path\":{\"type\":\"string\"}}}},{\"name\":\"write_file\",\"description\":\"Write a text file.\",\"input_schema\":{\"type\":\"object\",\"required\":[\"path\",\"content\"],\"properties\":{\"path\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"}}}}]}]\n\nUSER:\nWrite 'release approved' to /notes/release.txt.\n\nASSISTANT:\n",
  "completion": "{\"action\":\"tool_call\",\"calls\":[{\"arguments\":{\"content\":\"release approved\",\"path\":\"/notes/release.txt\"},\"name\":\"write_file\"}]}",
  "mixture_source": "mcp_positive",
  "source": "mcp_sft_v2_all",
  "loss_weight": 1.0
}
```

说明：真实 `phase5_unified` 文件没有 top-level `tools/messages/metadata` 字段。工具定义被渲染进 `prompt` 的 `AVAILABLE FUNCTIONS` 块；SQL/Spider 行也是同样的 `prompt/completion` 消费格式，但通常只暴露 `execute_sql` 工具和弱 schema context。

### `phase5_unified` Spider/SQL 行真实样例

```json
{
  "id": "spider-5824",
  "prompt": "You are the main model of a data agent. Decide whether to call a tool, ask for clarification, or answer without tools. Return exactly one JSON object and no extra text. Use this schema: {\"action\":\"tool_call\",\"calls\":[{\"name\":\"tool_name\",\"arguments\":{...}}]} for tool calls; {\"action\":\"no_tool\",\"calls\":[]} when no tool is needed; {\"action\":\"clarify\",\"missing\":[\"field\"],\"message\":\"question\"} when required information is missing.\n\nAVAILABLE FUNCTIONS:\n[{\"description\":\"Execute a SQL query against the specified database.\",\"name\":\"execute_sql\",\"parameters\":{\"type\":\"object\",\"required\":[\"database\",\"sql\"],\"properties\":{\"database\":{\"type\":\"string\"},\"sql\":{\"type\":\"string\"}}}}]\n\nDATABASE CONTEXT:\ndatabase: workshop_paper\nschema: not provided in this local Spider JSONL snapshot. Use the question and database id; do not invent non-SQL text.\n\nUSER:\nWhat is the author of the submission with the highest score?\n\nASSISTANT:\n",
  "completion": "{\"action\":\"tool_call\",\"calls\":[{\"arguments\":{\"database\":\"workshop_paper\",\"sql\":\"SELECT Author FROM submission ORDER BY Scores DESC LIMIT 1\"},\"name\":\"execute_sql\"}]}",
  "mixture_source": "sql_spider",
  "source": "spider_train",
  "loss_weight": 1.3,
  "db_id": "workshop_paper"
}
```

影响：这一行能训练模型输出 Data Agent action 和调用 `execute_sql`，但不能充分训练 schema linking。因为 prompt 只有 database id，没有 Spider `tables.json` 中的表、列、外键、类型信息。后续 SQL 主训练必须补齐 schema/table/value context，否则模型会更像是在记 SQL 模式，而不是基于 schema 生成可执行 SQL。

### `phase15_clean_v4` Multi-turn SFT 样例

```json
{
  "id": "phase15_mt_clean_000001_turn03",
  "task_id": "phase15_mt_clean_000001",
  "prompt": "SYSTEM: You are a Data Agent. Use JSON actions only.\nTOOLS:\n- execute_sql(database, sql)\n- inspect_schema(database)\n- final_answer(answer)\n\nUSER:\nFind active enterprise customers with unpaid invoices and summarize the total amount.\nASSISTANT:{\"action\":\"tool_call\",\"calls\":[{\"name\":\"inspect_schema\",\"arguments\":{\"database\":\"sales_demo\"}}]}\nTOOL:\n{\"tables\":{\"customers\":[\"id\",\"name\",\"segment\",\"status\"],\"invoices\":[\"customer_id\",\"amount\",\"paid\"]}}\nASSISTANT:",
  "completion": "{\"action\":\"tool_call\",\"calls\":[{\"name\":\"execute_sql\",\"arguments\":{\"database\":\"sales_demo\",\"sql\":\"SELECT c.name, SUM(i.amount) AS unpaid_amount FROM customers c JOIN invoices i ON c.id = i.customer_id WHERE c.segment = 'enterprise' AND c.status = 'active' AND i.paid = 0 GROUP BY c.name\"}}]}",
  "metadata": {
    "source": "phase15_multiturn_clean_v4",
    "turn_index": 3,
    "split": "train",
    "verifier": "sqlite_state"
  }
}
```

### `phase16_sql_repair` SFT 样例

```json
{
  "id": "phase16_repair_real_failure_000001",
  "source": "phase10_wikisql_failed_prediction",
  "prompt": "You are a SQL repair assistant.\nQuestion: What is the number of players from Canada?\nTable schema: table(col0 TEXT, col1 TEXT, col2 TEXT)\nHeaders: col0=Player, col1=Country, col2=Score\nPrevious SQL: SELECT col0 FROM table WHERE col1 = 'Canada'\nExecution feedback: result does not answer the question; aggregation is missing.\nReturn only the corrected SQL.\n",
  "completion": "SELECT COUNT(col0) FROM table WHERE col1 = 'Canada'",
  "loss_weight": 1.5,
  "metadata": {
    "failure_type": "wrong_missing_aggregation",
    "split": "train"
  }
}
```

### `phase16c_grpo_train` RLVR 样例

```json
{
  "id": "phase16c_wikisql_grpo_000001",
  "query": "Generate SQLite SQL for the question. Use only physical columns col0, col1, ...\nQuestion: What is the total attendance where team is Boston?\nTable schema: table(col0 TEXT, col1 TEXT, col2 REAL)\nHeaders: col0=Team, col1=City, col2=Attendance\nSample rows: [[\"Boston\", \"Boston\", 12000], [\"Chicago\", \"Chicago\", 9000]]\nReturn only SQL.",
  "solution": "SELECT SUM(col2) FROM table WHERE col0 = 'Boston'",
  "task_type": "wikisql_exec",
  "verifier": {
    "kind": "sqlite_execution",
    "database": "datasets/processed/phase16_followup_assets_20260701/wikisql_eval_256.sqlite",
    "gold_result": [[12000]]
  },
  "metadata": {
    "source": "wikisql",
    "split": "train"
  }
}
```

### `phase17_sql_error_sft` 样例

```json
{
  "id": "phase17_wrong_where_or_value_000001",
  "source": "phase16c_corrected_wikisql_v2_error",
  "prompt": "Generate SQLite SQL. Use only physical SQL columns named col0, col1, ... Header names are descriptions only, not SQL identifiers.\nQuestion: What is the height of Hato Mayor?\nTable schema: table(col0 TEXT, col1 TEXT, col2 TEXT, col3 TEXT, col4 TEXT)\nHeaders: col0=Municipality, col1=Province, col2=Population, col3=Area, col4=Height\nSample rows: [[\"hato mayor\", \"hato mayor\", \"70000\", \"1200\", \"20\"]]\nPrevious model SQL: SELECT col4 FROM table WHERE col4 = 'hato mayor'\nError category: wrong_where_or_value_or_column\nReturn only the corrected SQL.\n",
  "completion": "SELECT col4 FROM table WHERE col0 = 'hato mayor'",
  "loss_weight": 1.5,
  "metadata": {
    "corrected_eval": "wikisql_v2_casefolded",
    "split": "train"
  }
}
```

### `wikisql_internal_probe` 评测样例

```json
{
  "id": "wikisql_eval_000001",
  "question": "What is the number of teams from Boston?",
  "table_id": "wikisql_000001",
  "sqlite_table": "table_000001",
  "header": ["Team", "City", "Wins"],
  "types": ["text", "text", "real"],
  "physical_columns": ["col0", "col1", "col2"],
  "sample_rows": [["boston", "boston", 10], ["chicago", "chicago", 7]],
  "gold_sql": "SELECT COUNT(col0) FROM table_000001 WHERE col1 = 'boston'",
  "gold_result": [[1]],
  "metadata": {
    "benchmark": "WikiSQL-derived internal probe",
    "scoring": "read-only SQLite execution; result rows compared order-insensitively",
    "split": "eval"
  }
}
```

### `sql_repair_execution_eval` 样例

```json
{
  "id": "sql_repair_exec_000001",
  "question": "What is the average score for Canada?",
  "schema": "table(col0 TEXT, col1 TEXT, col2 REAL)",
  "headers": {
    "col0": "Player",
    "col1": "Country",
    "col2": "Score"
  },
  "previous_sql": "SELECT COUNT(col2) FROM table WHERE col1 = 'Canada'",
  "execution_feedback": "SQL executed but result does not match expected result.",
  "expected_result": [[8.5]],
  "gold_sql": "SELECT AVG(col2) FROM table WHERE col1 = 'Canada'",
  "verifier": {
    "kind": "sqlite_execution",
    "database": "datasets/processed/phase16_followup_assets_20260701/wikisql_eval_256.sqlite"
  }
}
```

### `data_agent_action_probe` 评测样例

```json
{
  "id": "data_agent_action_probe_000001",
  "prompt": "SYSTEM: You are a Data Agent. Return exactly one JSON action.\nAVAILABLE_TOOLS:\n[{\"name\":\"inspect_schema\"},{\"name\":\"execute_sql\"},{\"name\":\"final_answer\"}]\nUSER:\nList unpaid invoice totals by customer. The schema is already known.\nASSISTANT:",
  "expected_action": {
    "action": "tool_call",
    "calls": [
      {
        "name": "execute_sql",
        "arguments": {
          "database": "data_agent_eval",
          "sql": "SELECT customer_id, SUM(amount) FROM invoices WHERE paid = 0 GROUP BY customer_id"
        }
      }
    ]
  },
  "metrics": [
    "json_valid",
    "action_exact",
    "tool_name_exact",
    "arguments_semantic"
  ]
}
```

### `eval_suite_public` IFEval 样例

```json
{
  "key": 1001,
  "prompt": "Write a 100-word paragraph about renewable energy. Include the word solar at least twice.",
  "instruction_id_list": [
    "length_constraints:number_words",
    "keywords:frequency"
  ],
  "kwargs": [
    {
      "num_words": 100
    },
    {
      "keyword": "solar",
      "frequency": 2
    }
  ]
}
```

### `eval_suite_public` BFCL 样例

```json
{
  "id": "BFCL_v3_simple_0001",
  "question": [
    [
      {
        "role": "user",
        "content": "Book a flight from SFO to JFK."
      }
    ]
  ],
  "function": [
    {
      "name": "book_flight",
      "parameters": {
        "type": "object",
        "required": ["from", "to"],
        "properties": {
          "from": {
            "type": "string"
          },
          "to": {
            "type": "string"
          }
        }
      }
    }
  ]
}
```

### `eval_suite_public` GSM8K 样例

```json
{
  "question": "Janet has 3 apples and buys 5 more. How many apples does she have?",
  "answer": "Janet has 3 + 5 = 8 apples. #### 8",
  "metadata": {
    "benchmark": "GSM8K",
    "split": "test",
    "train_use": false
  }
}
```

### `eval_suite_public` MMLU-Pro 样例

```json
{
  "question": "Which of the following best explains ...?",
  "options": ["A ...", "B ...", "C ...", "D ..."],
  "answer": "C",
  "category": "computer science",
  "metadata": {
    "benchmark": "MMLU-Pro",
    "scoring": "direct-logprob preferred; generation is auxiliary",
    "train_use": false
  }
}
```

## 维护规则

新增或修改任何数据集时，必须同步更新：

1. 快速索引表：ID、类型、创建日期、数量、路径、脚本、指标影响。
2. 完整数据样例：至少一条可读 JSON，字段不能只写省略号。
3. 指标影响摘要：如果尚未评测，写 `待补` 和预期评测路径；不要填猜测数字。
4. 污染边界：训练集和评测集必须标明是否可训练。
5. 关联实验记录：在 `docs/07_experiment_records.md` 中记录本批数据被哪个 Phase 使用。

建议每个 processed 数据目录都保留 `manifest.json`，至少包含：

```json
{
  "dataset_id": "phase17_sql_error_sft",
  "created_at": "2026-07-02",
  "source": ["wikisql", "phase16c_predictions", "base_replay"],
  "processing_script": "scripts/remote/prepare_phase17_sql_eval_and_sft.py",
  "counts": {
    "train": 8040,
    "eval": 256
  },
  "train_use": true,
  "eval_use": true,
  "contamination_policy": "public benchmark test answers excluded from training; internal probe split fixed",
  "known_effect": "baseline corrected WikiSQL v2 execution accuracy 50.78%; post-train result pending"
}
```
