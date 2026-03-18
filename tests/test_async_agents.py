"""T3: Async tests for DefaultAgent, RLTokenAgent, RetryAgent.

These tests verify the async target API. They will FAIL on the current sync
code and PASS after the async conversion of agents.py.

Covers:
- setup() / run() / step() / forward() / forward_with_handling()
- handle_action() / handle_submission()
- attempt_autosubmission_after_error()  (removes asyncio.run at L834)
- _get_edited_files_with_context()       (PatchFormatter pre-fetch pattern)
- RetryAgent.step() / run() / _next_attempt()
"""
from __future__ import annotations

import asyncio
from pathlib import PurePosixPath
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sweagent.agent.agents import DefaultAgent, RLTokenAgent, RetryAgent, TemplateConfig
from sweagent.agent.history_processors import DefaultHistoryProcessor
from sweagent.agent.models import InstanceStats
from sweagent.agent.token_manager import TokenManager
from sweagent.tools.parsing import Identity
from sweagent.tools.tools import ToolConfig, ToolHandler
from sweagent.types import StepOutput


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_async_model():
    """Model mock with async query()."""
    model = MagicMock()
    model.token_manager = TokenManager()
    model.stats = InstanceStats()
    model.instance_cost_limit = 0
    model.total_cost_limit = 0
    model.reset_rollout_state = MagicMock()
    model.query = AsyncMock(
        return_value={
            "message": "test output",
            "new_prompt_token_ids": [1, 2, 3],
            "new_prompt_logprobs": [0.0, 0.0, 0.0],
            "output_tokens": [10, 11],
            "rollout_log_probs": [-0.1, -0.2],
            "rollout_routed_experts": [],
        }
    )
    return model


@pytest.fixture
def mock_async_env():
    """Environment mock with all async methods."""
    env = MagicMock()
    env.communicate = AsyncMock(return_value="")
    env.read_file = AsyncMock(return_value="")
    env.write_file = AsyncMock()
    env.execute_command = AsyncMock()
    env.interrupt_session = AsyncMock()
    env.set_env_variables = AsyncMock()
    env.deployment = MagicMock()
    env.deployment.is_alive = AsyncMock(return_value=True)
    env.repo = None
    env.close = AsyncMock()
    env.start = AsyncMock()
    return env


@pytest.fixture
def tool_handler():
    return ToolHandler(ToolConfig(parse_function=Identity()))


@pytest.fixture
def rl_agent(mock_async_model, tool_handler):
    return RLTokenAgent(
        templates=TemplateConfig(),
        tools=tool_handler,
        history_processors=[DefaultHistoryProcessor()],
        model=mock_async_model,
    )


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------
class TestAsyncSetup:
    @pytest.mark.asyncio
    async def test_setup_is_coroutine(self, rl_agent):
        assert asyncio.iscoroutinefunction(rl_agent.setup)


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------
class TestAsyncForward:
    @pytest.mark.asyncio
    async def test_forward_is_coroutine(self, rl_agent):
        assert asyncio.iscoroutinefunction(rl_agent.forward)

    @pytest.mark.asyncio
    async def test_forward_awaits_model_query(self, rl_agent, mock_async_model, mock_async_env):
        """forward() must await model.query(), not call it synchronously."""
        from sweagent.agent.problem_statement import EmptyProblemStatement

        rl_agent._env = mock_async_env
        rl_agent._problem_statement = EmptyProblemStatement()
        rl_agent._history = []
        rl_agent._trajectory = []
        rl_agent._total_execution_time = 0

        with patch.object(rl_agent, "handle_action", new_callable=AsyncMock) as mock_handle:
            mock_handle.return_value = StepOutput()
            try:
                result = await rl_agent.forward(rl_agent.messages)
            except Exception:
                pass  # may fail due to incomplete mock setup

        mock_async_model.query.assert_awaited()


# ---------------------------------------------------------------------------
# handle_action
# ---------------------------------------------------------------------------
class TestAsyncHandleAction:
    @pytest.mark.asyncio
    async def test_handle_action_is_coroutine(self, rl_agent):
        assert asyncio.iscoroutinefunction(rl_agent.handle_action)

    @pytest.mark.asyncio
    async def test_handle_action_awaits_communicate(self, rl_agent, mock_async_env):
        rl_agent._env = mock_async_env
        rl_agent._total_execution_time = 0
        rl_agent._trajectory = []

        step = StepOutput()
        step.action = "ls"
        step.output = "ls"
        step.thought = "list files"
        mock_async_env.communicate.return_value = "file1.py"

        with patch.object(rl_agent.tools, "should_block_action", return_value=False):
            with patch.object(rl_agent.tools, "check_for_submission_cmd", return_value=False):
                with patch.object(rl_agent.tools, "get_state", new_callable=AsyncMock, return_value={}):
                    try:
                        result = await rl_agent.handle_action(step)
                    except Exception:
                        pass  # may fail due to incomplete mock setup

        mock_async_env.communicate.assert_awaited()


# ---------------------------------------------------------------------------
# handle_submission
# ---------------------------------------------------------------------------
class TestAsyncHandleSubmission:
    @pytest.mark.asyncio
    async def test_handle_submission_is_coroutine(self, rl_agent):
        assert asyncio.iscoroutinefunction(rl_agent.handle_submission)


# ---------------------------------------------------------------------------
# attempt_autosubmission_after_error
# ---------------------------------------------------------------------------
class TestAsyncAttemptAutosubmission:
    @pytest.mark.asyncio
    async def test_is_coroutine(self, rl_agent):
        assert asyncio.iscoroutinefunction(
            rl_agent.attempt_autosubmission_after_error
        )

    @pytest.mark.asyncio
    async def test_no_asyncio_run_in_method(self):
        """L834 asyncio.run(self._env.deployment.is_alive(...))
        must be replaced with await."""
        import inspect

        source = inspect.getsource(DefaultAgent.attempt_autosubmission_after_error)
        assert "asyncio.run(" not in source, (
            "attempt_autosubmission_after_error should use await, "
            "not asyncio.run()"
        )

    @pytest.mark.asyncio
    async def test_awaits_is_alive(self, rl_agent, mock_async_env):
        rl_agent._env = mock_async_env
        rl_agent._trajectory = []
        step = StepOutput()
        step.action = ""
        step.output = ""
        step.thought = ""

        try:
            await rl_agent.attempt_autosubmission_after_error(step)
        except Exception:
            pass  # may fail due to incomplete mock setup
        mock_async_env.deployment.is_alive.assert_awaited()


# ---------------------------------------------------------------------------
# forward_with_handling
# ---------------------------------------------------------------------------
class TestAsyncForwardWithHandling:
    @pytest.mark.asyncio
    async def test_is_coroutine(self, rl_agent):
        assert asyncio.iscoroutinefunction(rl_agent.forward_with_handling)


# ---------------------------------------------------------------------------
# step
# ---------------------------------------------------------------------------
class TestAsyncStep:
    @pytest.mark.asyncio
    async def test_step_is_coroutine(self, rl_agent):
        assert asyncio.iscoroutinefunction(rl_agent.step)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
class TestAsyncRun:
    @pytest.mark.asyncio
    async def test_run_is_coroutine(self, rl_agent):
        assert asyncio.iscoroutinefunction(rl_agent.run)


# ---------------------------------------------------------------------------
# _get_edited_files_with_context (PatchFormatter lambda)
# ---------------------------------------------------------------------------
class TestAsyncGetEditedFilesWithContext:
    @pytest.mark.asyncio
    async def test_is_coroutine(self, rl_agent):
        assert asyncio.iscoroutinefunction(rl_agent._get_edited_files_with_context)

    @pytest.mark.asyncio
    async def test_returns_dict_for_empty_patch(self, rl_agent, mock_async_env):
        """With empty patch, should still return dict with default values."""
        rl_agent._env = mock_async_env
        result = await rl_agent._get_edited_files_with_context("")
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_prefetches_files_for_patch_formatter(self, rl_agent, mock_async_env):
        """After async conversion, read_file must be awaited BEFORE PatchFormatter
        construction (pre-fetch), not passed as a sync lambda."""
        mock_async_env.repo = MagicMock()
        mock_async_env.repo.repo_name = "test-repo"
        mock_async_env.read_file.return_value = "line1\nline2\n"
        rl_agent._env = mock_async_env

        patch_str = (
            "diff --git a/file.py b/file.py\n"
            "--- a/file.py\n"
            "+++ b/file.py\n"
            "@@ -1,1 +1,2 @@\n"
            " line1\n"
            "+line2\n"
        )
        result = await rl_agent._get_edited_files_with_context(patch_str)
        # read_file must have been awaited (not called via sync lambda)
        mock_async_env.read_file.assert_awaited()
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# RetryAgent
# ---------------------------------------------------------------------------
class TestAsyncRetryAgent:
    @pytest.mark.asyncio
    async def test_step_is_coroutine(self):
        assert asyncio.iscoroutinefunction(RetryAgent.step)

    @pytest.mark.asyncio
    async def test_run_is_coroutine(self):
        assert asyncio.iscoroutinefunction(RetryAgent.run)

    @pytest.mark.asyncio
    async def test_next_attempt_is_coroutine(self):
        assert asyncio.iscoroutinefunction(RetryAgent._next_attempt)


# ---------------------------------------------------------------------------
# Source-level: no asyncio.run() in agents.py
# ---------------------------------------------------------------------------
class TestNoAsyncioRunInAgents:
    def test_no_asyncio_run_in_source(self):
        import inspect
        from sweagent.agent import agents

        source = inspect.getsource(agents)
        assert "asyncio.run(" not in source, (
            "agents.py should use await instead of asyncio.run()"
        )
