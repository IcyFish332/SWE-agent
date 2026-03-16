# SWE-agent (Project SWE RL Fork)

这是当前项目中作为 `3rdparty` 依赖保留的一份 **SWE-agent fork**。它的目标不是继续承载整套训练脚本，而是作为一个尽量贴近上游、但已经补齐 **RL / rollout 训练主链路** 所需能力的运行时基座。

## 为什么要改造 SWE-agent

原始 SWE-agent 的主假设是：
- agent 以消息级 history 驱动；
- 模型输出主要按文本/工具调用消费；
- 重点在“解题运行”而不是“训练轨迹保真”。

但接入 `slime` 做 SWE 任务强化学习训练时，我们需要额外保证：
- **Token-In / Token-Out** 轨迹可用；
- rollout 中的 `loss_mask`、`logprobs`、`output_tokens` 可直接进入训练；
- tool-calling 的解析链路尽量严格，减少 heuristic repair；
- 多轮 agent/tool/env 交互中，token 轨迹不要因为本地重分词而漂移；
- `swerl/` 可以把它当作 library 使用，而不是继续维护一套重改版源码。

所以当前这份 fork 的改造重点，是把 SWE-agent 补成一个更适合 RL rollout 的 runtime。

## 当前 fork 的核心改造

### 1. RL token rollout agent

当前新增的核心 agent 是：
- `sweagent.agent.agents.RLTokenAgent`

它负责：
- 承接 SWE-agent 原本的 step loop / tool execution / trajectory 逻辑；
- 暴露训练需要的：
  - `input_ids`
  - `loss_mask`
  - `rollout_log_probs`
  - `rollout_routed_experts`
  - `error_logs`
  - `response_turn`
- 与 `SGLangModel` 配合，形成 token-level rollout 主链路。

### 2. SGLang model backend

当前新增的核心 model backend 是：
- `sweagent.agent.models.SGLangModelConfig`
- `sweagent.agent.models.SGLangModel`

它负责：
- 直接对接 SGLang `/generate`；
- 返回：
  - `output_tokens`
  - `rollout_log_probs`
  - `rollout_routed_experts`
- 使用：
  - `tool_call_parser`
  - `reasoning_parser`
  - `message_separator`
- 在 model 内部维护 **增量 tokenization 状态**，减少 retokenization drift。

### 3. 增量 tokenization 与 token trajectory

当前实现不是简单地让 agent 对每条 message 单独 `apply_chat_template` 后拼接，而是引入了：
- `sweagent.agent.token_manager.TokenManager`

并把更严格的增量 prompt token 计算放到 `SGLangModel` 里：
- 首轮：完整 prompt tokenize；
- 后续轮次：只对新增消息计算 prompt token 增量；
- 可选 debug invariant：
  - `debug_check_incremental_tokens`

这条链路的目的，是尽量向 `strands-sglang` 的 token 保真思路靠拢。

### 4. Response parsing

当前 response parsing 已收敛到 model 层，不再放在 `swerl/`：
- `sweagent.agent.response_parsing.parse_tool_calls_with_sglang(...)`

它负责：
- 用 SGLang 的 `ReasoningParser` 拆 reasoning；
- 用 SGLang 的 `FunctionCallParser` 解析 tool calls；
- 输出统一的：
  - `message`
  - `tool_calls`
  - `reasoning_content`

## 当前与 `swerl` / `slime` 的分工

### 放在 `3rdparty/SWE-agent` 的能力

这份 fork 里保留的是 **runtime 必需能力**：
- `RLTokenAgent`
- `SGLangModel`
- token rollout types / hooks 扩展
- response parsing
- 增量 tokenization / token manager

### 不放在 `3rdparty/SWE-agent` 的能力

这些能力已经放在项目自己的 `swerl/`：
- rollout 入口
- reward bridge
- SWE-bench / env glue
- slime config 对接
- sample 回填
- 训练脚本 / 实验脚本

也就是说：
- `3rdparty/SWE-agent` = runtime 基座
- `swerl/` = RL 项目主逻辑

## 当前主配置字段

当前 `SGLangModelConfig` 与根配置主线使用这些字段：
- `tool_call_parser`
- `reasoning_parser`
- `message_separator`
- `debug_check_incremental_tokens`

对应 YAML 位于：
- `configs/swe/default.yaml`

## 当前重点文件

如果要理解当前 fork 的关键改造，优先看：
- `sweagent/agent/agents.py`
- `sweagent/agent/models.py`
- `sweagent/agent/token_manager.py`
- `sweagent/agent/response_parsing.py`
- `sweagent/types.py`
- `sweagent/agent/hooks/abstract.py`

## 当前状态总结

目前这份 fork 已经不是”团队旧实验仓库的完整重放”，而是一次 **有选择的、围绕 RL 主链路的最小 runtime 回迁**。

就当前项目而言，它已经能够支撑：
- `slime -> swerl -> RLTokenAgent -> SGLangModel -> SWE env/reward`

如果后续继续精简，下一步通常会落在：
- 继续清理历史兼容字段；
- 继续加强增量 tokenization 的 invariant 与测试；
- 尽量把 fork 维持在最小 patch surface。

## 测试

本 fork 为 RL 改造部分新增了专项测试，覆盖 `TokenManager`、`SGLangModel`、`RLTokenAgent`、`response_parsing` 四个模块。

### 单元测试（无需外部服务）

```bash
cd 3rdparty/SWE-agent

# 运行全部 RL 相关单元测试
pytest tests/test_token_manager.py tests/test_response_parsing.py tests/test_sglang_model.py tests/test_rl_token_agent.py -v

# 运行全部非慢速、非 SGLang 服务的测试（含上游原有测试）
pytest tests/ -m “not slow and not sglang” -v
```

测试文件：

| 文件 | 覆盖范围 | 外部依赖 |
|------|---------|---------|
| `tests/test_token_manager.py` | TokenManager 全属性 / 多轮 segment / loss_mask | 无 |
| `tests/test_response_parsing.py` | `_detect_think_and_return_ori_think` + `parse_tool_calls_with_sglang`（真实 sglang parser） | sglang |
| `tests/test_sglang_model.py` | `_extract_logprobs` / `_build_payload` / `tokenize_prompt_messages` / `_single_query` 等 | unittest.mock |
| `tests/test_rl_token_agent.py` | property delegation / setup reset / forward / add_step_to_history | unittest.mock, DummyRuntime |

### 集成测试（需要运行中的 SGLang 服务）

```bash
# 需要先启动 SGLang 推理服务，然后：
pytest tests/test_sglang_integration.py --sglang-base-url http://<host>:<port> -v

# 或通过环境变量
export SGLANG_BASE_URL=http://localhost:30000
pytest tests/test_sglang_integration.py -v
```

集成测试内容：

| 测试类 | 验证目标 |
|--------|---------|
| `TestTITOConsistency` | token_ids / loss_mask / logprobs 长度一致性，loss_mask 二值性，logprobs 非占位符 |
| `TestIncrementalTokenization` | 增量分词 token 数 < 全量重分词；debug_check_incremental_tokens 无 drift |
| `TestRetokenizationDrift` | encode(decode(token_ids)) == token_ids 往返一致性 |
| `TestMultiTurnSegments` | 两轮对话后 segment 模式为 [P, R, P, R]；新增 segment 不改变历史 loss_mask |
