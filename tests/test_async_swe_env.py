"""T2: Async tests for SWEEnv.

These tests verify the async target API. They will FAIL on the current sync
code and PASS after the async conversion of swe_env.py.

Covers:
- communicate()      (hot path, every step)
- close()
- _init_deployment()
- start() / reset() / hard_reset()
- read_file() / write_file()
- interrupt_session()
- execute_command()
- set_env_variables()
- No asyncio.run() calls remain in source
"""
from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sweagent.environment.swe_env import SWEEnv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_deployment():
    dep = MagicMock()
    dep.start = AsyncMock()
    dep.stop = AsyncMock()
    dep.is_alive = AsyncMock(return_value=True)
    dep.runtime = MagicMock()
    dep.runtime.run_in_session = AsyncMock(
        return_value=MagicMock(output="", exit_code=0)
    )
    dep.runtime.create_session = AsyncMock()
    dep.runtime.read_file = AsyncMock(
        return_value=MagicMock(content="file content")
    )
    dep.runtime.write_file = AsyncMock()
    dep.runtime.execute = AsyncMock(
        return_value=MagicMock(exit_code=0, stdout="", stderr="")
    )
    dep.runtime.upload = AsyncMock()
    return dep


@pytest.fixture
def env(mock_deployment):
    """Construct SWEEnv with a mocked deployment, skipping start()."""
    e = SWEEnv.__new__(SWEEnv)
    e.deployment = mock_deployment
    e.repo = None
    e._post_startup_commands = []
    e.post_startup_command_timeout = 500
    e.name = "test-env"
    e.clean_multi_line_functions = lambda x: x
    e._chook = MagicMock()
    e.logger = MagicMock()
    return e


# ---------------------------------------------------------------------------
# communicate (HOT PATH)
# ---------------------------------------------------------------------------
class TestAsyncCommunicate:
    @pytest.mark.asyncio
    async def test_communicate_is_coroutine(self, env):
        assert asyncio.iscoroutinefunction(env.communicate)

    @pytest.mark.asyncio
    async def test_communicate_awaits_runtime(self, env, mock_deployment):
        result = await env.communicate("ls")
        mock_deployment.runtime.run_in_session.assert_awaited_once()
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_communicate_returns_output(self, env, mock_deployment):
        mock_deployment.runtime.run_in_session.return_value = MagicMock(
            output="hello world", exit_code=0
        )
        result = await env.communicate("echo hello world")
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_communicate_check_raise(self, env, mock_deployment):
        mock_deployment.runtime.run_in_session.return_value = MagicMock(
            output="error", exit_code=1
        )
        with pytest.raises(RuntimeError):
            await env.communicate("bad_cmd", check="raise")


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------
class TestAsyncClose:
    @pytest.mark.asyncio
    async def test_close_is_coroutine(self, env):
        assert asyncio.iscoroutinefunction(env.close)

    @pytest.mark.asyncio
    async def test_close_awaits_deployment_stop(self, env, mock_deployment):
        await env.close()
        mock_deployment.stop.assert_awaited_once()


# ---------------------------------------------------------------------------
# _init_deployment
# ---------------------------------------------------------------------------
class TestAsyncInitDeployment:
    @pytest.mark.asyncio
    async def test_init_deployment_is_coroutine(self, env):
        assert asyncio.iscoroutinefunction(env._init_deployment)

    @pytest.mark.asyncio
    async def test_init_deployment_awaits_start_and_create_session(
        self, env, mock_deployment
    ):
        await env._init_deployment()
        mock_deployment.start.assert_awaited_once()
        mock_deployment.runtime.create_session.assert_awaited_once()


# ---------------------------------------------------------------------------
# start
# ---------------------------------------------------------------------------
class TestAsyncStart:
    @pytest.mark.asyncio
    async def test_start_is_coroutine(self, env):
        assert asyncio.iscoroutinefunction(env.start)

    @pytest.mark.asyncio
    async def test_start_calls_init_and_reset(self, env):
        with (
            patch.object(env, "_init_deployment", new_callable=AsyncMock) as mock_init,
            patch.object(env, "reset", new_callable=AsyncMock) as mock_reset,
        ):
            await env.start()
            mock_init.assert_awaited_once()


# ---------------------------------------------------------------------------
# reset / hard_reset
# ---------------------------------------------------------------------------
class TestAsyncReset:
    @pytest.mark.asyncio
    async def test_reset_is_coroutine(self, env):
        assert asyncio.iscoroutinefunction(env.reset)


class TestAsyncHardReset:
    @pytest.mark.asyncio
    async def test_hard_reset_is_coroutine(self, env):
        assert asyncio.iscoroutinefunction(env.hard_reset)

    @pytest.mark.asyncio
    async def test_hard_reset_calls_close_then_start(self, env):
        with (
            patch.object(env, "close", new_callable=AsyncMock) as mock_close,
            patch.object(env, "start", new_callable=AsyncMock) as mock_start,
        ):
            await env.hard_reset()
            mock_close.assert_awaited_once()
            mock_start.assert_awaited_once()


# ---------------------------------------------------------------------------
# read_file / write_file
# ---------------------------------------------------------------------------
class TestAsyncReadFile:
    @pytest.mark.asyncio
    async def test_read_file_is_coroutine(self, env):
        assert asyncio.iscoroutinefunction(env.read_file)

    @pytest.mark.asyncio
    async def test_read_file_returns_content(self, env, mock_deployment):
        result = await env.read_file("/path/to/file")
        assert result == "file content"
        mock_deployment.runtime.read_file.assert_awaited_once()


class TestAsyncWriteFile:
    @pytest.mark.asyncio
    async def test_write_file_is_coroutine(self, env):
        assert asyncio.iscoroutinefunction(env.write_file)

    @pytest.mark.asyncio
    async def test_write_file_awaits_runtime(self, env, mock_deployment):
        await env.write_file("/path/to/file", "content")
        mock_deployment.runtime.write_file.assert_awaited_once()


# ---------------------------------------------------------------------------
# interrupt_session / execute_command / set_env_variables
# ---------------------------------------------------------------------------
class TestAsyncInterruptSession:
    @pytest.mark.asyncio
    async def test_interrupt_session_is_coroutine(self, env):
        assert asyncio.iscoroutinefunction(env.interrupt_session)

    @pytest.mark.asyncio
    async def test_interrupt_session_awaits_runtime(self, env, mock_deployment):
        await env.interrupt_session()
        mock_deployment.runtime.run_in_session.assert_awaited_once()


class TestAsyncExecuteCommand:
    @pytest.mark.asyncio
    async def test_execute_command_is_coroutine(self, env):
        assert asyncio.iscoroutinefunction(env.execute_command)

    @pytest.mark.asyncio
    async def test_execute_command_awaits_runtime(self, env, mock_deployment):
        await env.execute_command("ls -la")
        mock_deployment.runtime.execute.assert_awaited_once()


class TestAsyncSetEnvVariables:
    @pytest.mark.asyncio
    async def test_set_env_variables_is_coroutine(self, env):
        assert asyncio.iscoroutinefunction(env.set_env_variables)


# ---------------------------------------------------------------------------
# Source-level: no asyncio.run() in swe_env.py
# ---------------------------------------------------------------------------
class TestNoAsyncioRunInSweEnv:
    def test_no_asyncio_run_in_source(self):
        from sweagent.environment import swe_env

        source = inspect.getsource(swe_env)
        assert "asyncio.run(" not in source, (
            "SWEEnv should use await instead of asyncio.run()"
        )
