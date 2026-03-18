"""T5: Async tests for ToolHandler (startup path).

These tests verify the async target API. They will FAIL on the current sync
code and PASS after the async conversion of tools.py.

Covers:
- _install_commands() : 2 asyncio.run() -> await (_upload_bundles, _check_available_commands)
- install() / reset() / get_state() / _get_state() cascade
- No asyncio.run() calls remain in _install_commands source
"""
from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sweagent.tools.parsing import Identity
from sweagent.tools.tools import ToolConfig, ToolHandler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def handler():
    return ToolHandler(ToolConfig(parse_function=Identity()))


@pytest.fixture
def mock_async_env():
    env = MagicMock()
    env.communicate = AsyncMock(return_value="/root")
    env.set_env_variables = AsyncMock()
    env.read_file = AsyncMock(return_value="{}")
    env.write_file = AsyncMock()
    env.deployment = MagicMock()
    env.deployment.runtime.upload = AsyncMock()
    env.deployment.runtime.execute = AsyncMock(
        return_value=MagicMock(exit_code=0, stdout="", stderr="")
    )
    return env


# ---------------------------------------------------------------------------
# _install_commands
# ---------------------------------------------------------------------------
class TestAsyncInstallCommands:
    @pytest.mark.asyncio
    async def test_install_commands_is_coroutine(self, handler):
        assert asyncio.iscoroutinefunction(handler._install_commands)

    @pytest.mark.asyncio
    async def test_install_commands_awaits_upload_and_check(
        self, handler, mock_async_env
    ):
        with (
            patch.object(
                handler, "_upload_bundles", new_callable=AsyncMock
            ) as mock_upload,
            patch.object(
                handler, "_check_available_commands", new_callable=AsyncMock
            ) as mock_check,
        ):
            await handler._install_commands(mock_async_env)
            mock_upload.assert_awaited()
            mock_check.assert_awaited()


# ---------------------------------------------------------------------------
# install / reset
# ---------------------------------------------------------------------------
class TestAsyncInstall:
    @pytest.mark.asyncio
    async def test_install_is_coroutine(self, handler):
        assert asyncio.iscoroutinefunction(handler.install)

    @pytest.mark.asyncio
    async def test_reset_is_coroutine(self, handler):
        assert asyncio.iscoroutinefunction(handler.reset)


# ---------------------------------------------------------------------------
# get_state / _get_state
# ---------------------------------------------------------------------------
class TestAsyncGetState:
    @pytest.mark.asyncio
    async def test_get_state_is_coroutine(self, handler):
        assert asyncio.iscoroutinefunction(handler.get_state)

    @pytest.mark.asyncio
    async def test_get_state_is_coroutine_private(self, handler):
        assert asyncio.iscoroutinefunction(handler._get_state)

    @pytest.mark.asyncio
    async def test_get_state_returns_dict(self, handler, mock_async_env):
        mock_async_env.read_file.return_value = '{"open_file": "/test.py"}'
        result = await handler.get_state(mock_async_env)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Source-level
# ---------------------------------------------------------------------------
class TestNoAsyncioRunInTools:
    def test_no_asyncio_run_in_install_commands(self):
        source = inspect.getsource(ToolHandler._install_commands)
        assert "asyncio.run(" not in source, (
            "_install_commands should use await instead of asyncio.run()"
        )

    def test_no_asyncio_run_in_module(self):
        from sweagent.tools import tools

        source = inspect.getsource(tools)
        assert "asyncio.run(" not in source, (
            "tools.py should use await instead of asyncio.run()"
        )
