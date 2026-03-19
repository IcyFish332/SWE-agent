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

## Async 改造（`async` 分支）

### 动机

原始 SWE-agent 在所有涉及 `SWE-ReX` runtime 的地方都用 `asyncio.run()` 把异步调用包装成同步。这意味着每次 `communicate()`、`read_file()`、`execute()` 等操作都会阻塞整个线程，无法在同一 event loop 中并发运行多条 trajectory。

在 `slime` RL 框架中，rollout 阶段天然运行在一个 async event loop 里。如果底层 agent 全是同步调用，就只能退化为线程池并发（`ThreadPoolExecutor`），丧失了 step 级交错的可能性。

**Async 改造的目标**：移除所有 `asyncio.run()` 阻塞点，使整个调用链从 `swerl.generate()` 到 `SWE-ReX runtime` 全程 `async/await`，支持 step 级并发。

### 改造范围

共移除 **13 处 `asyncio.run()`**，涉及 6 个文件：

| 文件 | 改造内容 | 影响方法 |
|------|---------|---------|
| `sweagent/agent/models.py` | `requests.post` -> `httpx.AsyncClient`（持久连接）；`time.sleep` -> `asyncio.sleep`；`threading.Lock` -> `asyncio.Lock`；`Retrying` -> `AsyncRetrying`；线程级 API key 跟踪 -> `asyncio.current_task()` | `_single_query`, `_sleep`, `_update_stats`, `query` |
| `sweagent/environment/swe_env.py` | 8 处 `asyncio.run()` 移除，方法改为 `async def` | `communicate`, `close`, `start`, `reset`, `hard_reset`, `read_file`, `write_file`, `interrupt_session`, `execute_command`, `set_env_variables`, `_init_deployment` |
| `sweagent/agent/agents.py` | `setup` / `forward` / `step` / `run` 全链路 async；`PatchFormatter` 的 sync `read_method` 改为 pre-fetch 模式 | `DefaultAgent`, `RLTokenAgent`, `RetryAgent` 的主方法 |
| `sweagent/environment/repo.py` | 3 处 `asyncio.run()` 移除 | `LocalRepo.copy`, `GithubRepo.copy` |
| `sweagent/tools/tools.py` | 2 处 `asyncio.run()` 移除 | `ToolHandler.install_commands`, `ToolHandler.install`, `ToolHandler.reset`, `ToolHandler._get_state` |
| `pyproject.toml` | 新增 `httpx` 主依赖、`pytest-asyncio` 开发依赖 | — |

### 关键设计决策

#### httpx.AsyncClient 持久连接

`SGLangModel` 在 `__init__` 时创建一个持久的 `httpx.AsyncClient` 实例，在整个 agent 生命周期内复用同一 TCP 连接池。这在 RL rollout 场景下（一条 trajectory 几十个 step）避免了大量 TCP 握手开销：

```python
# __init__ 中
self._http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(args.completion_kwargs.get(“timeout”, 1800))
)

# _single_query 中直接使用
response = await self._http_client.post(url, json=payload, headers=headers)

# close() 时清理
await self._http_client.aclose()
```

#### PatchFormatter pre-fetch 模式

`agents.py` 中 `_get_edited_files_with_context` 原本给 `PatchFormatter` 传一个 sync lambda `read_method=lambda path: asyncio.run(env.read_file(path))`。改造后改为先批量 `await` 读取所有涉及文件，再把内容包装成同步 lambda：

```python
# 先异步批量预取
prefetched = {}
for fname in patched_files:
    prefetched[fname] = await self.env.read_file(fname)

# 再给 PatchFormatter 一个纯同步 lambda
read_method=lambda path, _cache=prefetched: _cache.get(path, “”)
```

#### asyncio.current_task 替代线程跟踪

原始代码用 `threading.current_thread()` 跟踪哪些线程使用了 API key。async 改造后改为 `asyncio.current_task()`，确保在同一线程的多个并发 coroutine 中也能正确区分：

```python
_TASKS_THAT_USED_API_KEYS: dict[str, set[asyncio.Task]] = {}
```

### 对接方式

#### 从 swerl 调用（推荐）

```python
# swerl/rollout/swe_agent.py
async def generate(samples, ...):
    results = await asyncio.gather(*[
        run_swe_agent(sample, ...) for sample in batch
    ])
    # step 级并发自动发生：多个 agent 在 await model.query() 处交错
```

不再需要 `ThreadPoolExecutor`，`generate()` 直接 `await` 所有 agent。

#### 直接使用 async agent

```python
import asyncio
from sweagent.agent.agents import RLTokenAgent

async def main():
    agent = RLTokenAgent(...)
    await agent.setup(instance)
    result = await agent.run()

asyncio.run(main())
```

注意：如果要从同步代码调用，仍需要在最外层用一次 `asyncio.run()`。区别在于 agent 内部不再有嵌套的 `asyncio.run()`，所以可以安全地放入已有 event loop。

### 并发模型

```
slime event loop
  |
  +-- swerl.generate(batch=[s1, s2, s3, ...])
        |
        +-- asyncio.gather(
        |     run_swe_agent(s1),  -- agent1.step() -> await model.query() --|
        |     run_swe_agent(s2),  -- agent2.step() -> await model.query() --|-- step 级交错
        |     run_swe_agent(s3),  -- agent3.step() -> await model.query() --|
        |   )
```

每个 agent 在 `await model.query()`（HTTP 请求）和 `await env.communicate()`（runtime 命令）处让出控制权，其他 agent 可以在此期间推进自己的 step。

## 当前状态总结

目前这份 fork 已经不是”团队旧实验仓库的完整重放”，而是一次 **有选择的、围绕 RL 主链路的最小 runtime 回迁**。

就当前项目而言，它已经能够支撑：
- `slime -> swerl -> RLTokenAgent -> SGLangModel -> SWE env/reward`

Async 改造后，调用链从 `swerl.generate()` 到 `SWE-ReX runtime` 全程 `async/await`，不再有嵌套 event loop 或线程池降级。

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

### Async 契约测试（无需外部服务）

```bash
cd 3rdparty/SWE-agent

# 运行全部 async 契约测试
pytest tests/test_async_sglang_model.py tests/test_async_swe_env.py tests/test_async_agents.py tests/test_async_repo.py tests/test_async_tools.py -v

# 运行 step 级并发交错测试
pytest tests/test_async_agents.py::TestStepLevelInterleaving -v
```

测试文件：

| 文件 | 覆盖范围 | 外部依赖 |
|------|---------|---------|
| `tests/test_async_sglang_model.py` | `_single_query` 使用 `httpx.AsyncClient`；`_sleep` 使用 `asyncio.sleep`；`query` 使用 `AsyncRetrying`；`GLOBAL_STATS_LOCK` 为 `asyncio.Lock`；`_TASKS_THAT_USED_API_KEYS` 基于 `asyncio.current_task`；无 `time.sleep` / `requests.post` / `asyncio.run` 残留 | unittest.mock |
| `tests/test_async_swe_env.py` | `communicate` / `close` / `start` / `reset` / `read_file` / `write_file` 等均为 coroutine；无 `asyncio.run()` 残留 | unittest.mock |
| `tests/test_async_agents.py` | `setup` / `forward` / `step` / `run` 全链路 async；`_get_edited_files_with_context` pre-fetch 模式；`attempt_autosubmission_after_error` 无 `asyncio.run`；step 级并发交错验证 | unittest.mock |
| `tests/test_async_repo.py` | `LocalRepo.copy` / `GithubRepo.copy` 为 coroutine 且 await runtime | unittest.mock |
| `tests/test_async_tools.py` | `install_commands` / `install` / `reset` / `_get_state` 为 coroutine；无 `asyncio.run()` 残留 | unittest.mock |

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
