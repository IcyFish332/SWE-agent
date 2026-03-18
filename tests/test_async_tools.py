"""T5: Async tests for ToolHandler (startup path).

Covers:
- Async methods are coroutines (parametrized)
- _install_commands() awaits _upload_bundles + _check_available_commands
- get_state() returns dict
- No asyncio.run() calls remain in source
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
# Parametrized coroutine check
# ---------------------------------------------------------------------------
_ASYNC_METHODS = ["install", "reset", "_install_commands", "get_state", "_get_state"]


class TestMethodsAreCoroutines:
    @pytest.mark.parametrize("method", _ASYNC_METHODS)
    def test_is_coroutine(self, handler, method):
        assert asyncio.iscoroutinefunction(getattr(handler, method)), (
            f"ToolHandler.{method} should be async def"
        )


# ---------------------------------------------------------------------------
# Behavior tests
# ---------------------------------------------------------------------------
class TestAsyncInstallCommands:
    @pytest.mark.asyncio
    async def test_install_commands_awaits_upload_and_check(
        self, handler, mock_async_env
    ):
        with (
            patch.object(handler, "_upload_bundles", new_callable=AsyncMock) as mock_upload,
            patch.object(handler, "_check_available_commands", new_callable=AsyncMock) as mock_check,
        ):
            await handler._install_commands(mock_async_env)
            mock_upload.assert_awaited()
            mock_check.assert_awaited()


class TestAsyncGetState:
    @pytest.mark.asyncio
    async def test_get_state_returns_dict(self, handler, mock_async_env):
        mock_async_env.read_file.return_value = '{"open_file": "/test.py"}'
        result = await handler.get_state(mock_async_env)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Source-level
# ---------------------------------------------------------------------------
class TestNoAsyncioRunInTools:
    def test_no_asyncio_run_in_module(self):
        from sweagent.tools import tools

        source = inspect.getsource(tools)
        assert "asyncio.run(" not in source
