"""T1: Async tests for SGLangModel (hot path).

These tests verify the async target API. They will FAIL on the current sync
code and PASS after the async conversion of models.py.

Covers:
- _sleep()        : time.sleep -> asyncio.sleep
- _update_stats() : threading.Lock -> asyncio.Lock
- _single_query() : requests.post -> httpx.AsyncClient
- _query() / query() : sync -> async, Retrying -> AsyncRetrying
- GLOBAL_STATS_LOCK : threading.Lock -> asyncio.Lock
- _THREADS_THAT_USED_API_KEYS removal
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from sweagent.agent.models import (
    ContextWindowExceededError,
    SGLangModel,
    SGLangModelConfig,
)
from sweagent.tools.parsing import Identity
from sweagent.tools.tools import ToolConfig
from sweagent.types import History


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_sglang_response(
    text="response text",
    output_ids=None,
    output_logprobs=None,
    input_logprobs=None,
    routed_experts=None,
):
    if output_ids is None:
        output_ids = [101, 102, 103]
    if output_logprobs is None:
        output_logprobs = [[-0.1, 101], [-0.2, 102], [-0.3, 103]]
    data = {
        "text": text,
        "output_ids": output_ids,
        "meta_info": {
            "output_token_logprobs": output_logprobs,
        },
    }
    if input_logprobs is not None:
        data["meta_info"]["input_token_logprobs"] = input_logprobs
    if routed_experts is not None:
        data["meta_info"]["routed_experts"] = routed_experts
    return data


# ---------------------------------------------------------------------------
# Fixtures (mirrors test_sglang_model.py style)
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_tokenizer():
    tok = MagicMock()
    tok.apply_chat_template.return_value = [10, 20, 30, 40, 50]
    tok.encode.return_value = [99]
    return tok


@pytest.fixture
def sglang_config(mock_tokenizer):
    return SGLangModelConfig(
        name="test-model",
        api_base="http://localhost:30000",
        api_key=SecretStr("test-key"),
        tokenizer=mock_tokenizer,
        top_p=None,
        per_instance_cost_limit=0,
        total_cost_limit=0,
        per_instance_call_limit=0,
    )


@pytest.fixture
def model(sglang_config):
    return SGLangModel(sglang_config, ToolConfig(parse_function=Identity()))


# ---------------------------------------------------------------------------
# _sleep: time.sleep -> asyncio.sleep
# ---------------------------------------------------------------------------
class TestAsyncSleep:
    """_sleep() must use asyncio.sleep instead of time.sleep."""

    @pytest.mark.asyncio
    async def test_sleep_is_coroutine(self, model):
        assert asyncio.iscoroutinefunction(model._sleep)

    @pytest.mark.asyncio
    async def test_sleep_calls_asyncio_sleep(self, model):
        model.config.delay = 10.0  # large delay to guarantee sleep branch
        from sweagent.agent.models import GLOBAL_STATS

        GLOBAL_STATS.last_query_timestamp = time.time()  # just queried
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_async_sleep:
            await model._sleep()
            mock_async_sleep.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_sleep_skips_when_elapsed_sufficient(self, model):
        """No sleep when enough time has elapsed since last query."""
        model.config.delay = 0
        await model._sleep()  # should complete instantly


# ---------------------------------------------------------------------------
# _update_stats: threading.Lock -> asyncio.Lock
# ---------------------------------------------------------------------------
class TestAsyncUpdateStats:
    @pytest.mark.asyncio
    async def test_update_stats_is_coroutine(self, model):
        assert asyncio.iscoroutinefunction(model._update_stats)

    def test_global_stats_lock_is_asyncio_lock(self):
        from sweagent.agent.models import GLOBAL_STATS_LOCK

        assert isinstance(GLOBAL_STATS_LOCK, asyncio.Lock), (
            "GLOBAL_STATS_LOCK should be asyncio.Lock, not threading.Lock"
        )

    @pytest.mark.asyncio
    async def test_update_stats_increments_correctly(self, model):
        await model._update_stats(input_tokens=100, output_tokens=50, cost=0.01)
        assert model.stats.tokens_sent == 100
        assert model.stats.tokens_received == 50
        assert model.stats.api_calls == 1


# ---------------------------------------------------------------------------
# _single_query: requests.post -> httpx.AsyncClient
# ---------------------------------------------------------------------------
class TestAsyncSingleQuery:
    @pytest.mark.asyncio
    async def test_single_query_is_coroutine(self, model):
        assert asyncio.iscoroutinefunction(model._single_query)

    @pytest.mark.asyncio
    async def test_successful_query_with_httpx(self, model):
        """_single_query should use httpx, not requests."""
        mock_response = MagicMock()
        mock_response.json.return_value = _make_sglang_response(
            input_logprobs=[[0.0, i] for i in [10, 20, 30, 40, 50]],
        )
        mock_response.raise_for_status = MagicMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("sweagent.agent.models.httpx.AsyncClient", return_value=mock_client):
            with patch.object(model, "parse_response", return_value={"message": "response text"}):
                with patch.object(model, "_sleep", new_callable=AsyncMock):
                    with patch.object(model, "_update_stats", new_callable=AsyncMock):
                        history = History([{"role": "user", "content": "hello"}])
                        result = await model._single_query(history)

        assert len(result) == 1
        assert result[0]["message"] == "response text"
        assert result[0]["output_tokens"] == [101, 102, 103]
        mock_client.post.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_context_exceeded_before_http(self, model, mock_tokenizer):
        """Token limit check still works under async."""
        model.model_max_input_tokens = 3
        mock_tokenizer.apply_chat_template.return_value = [1, 2, 3, 4, 5]
        with patch.object(model, "_sleep", new_callable=AsyncMock):
            history = History([{"role": "user", "content": "hello"}])
            with pytest.raises(ContextWindowExceededError):
                await model._single_query(history)

    @pytest.mark.asyncio
    async def test_no_requests_import_in_single_query(self):
        """The hot-path _single_query should not use requests.post."""
        import inspect

        source = inspect.getsource(SGLangModel._single_query)
        assert "requests.post" not in source, (
            "_single_query should use httpx.AsyncClient, not requests.post"
        )


# ---------------------------------------------------------------------------
# _query / query: sync -> async
# ---------------------------------------------------------------------------
class TestAsyncQuery:
    @pytest.mark.asyncio
    async def test_query_is_coroutine(self, model):
        assert asyncio.iscoroutinefunction(model.query)

    @pytest.mark.asyncio
    async def test_query_method_is_coroutine(self, model):
        assert asyncio.iscoroutinefunction(model._query)

    @pytest.mark.asyncio
    async def test_query_n1_returns_dict(self, model):
        with patch.object(
            model,
            "_single_query",
            new_callable=AsyncMock,
            return_value=[{"message": "ok", "output_tokens": [1]}],
        ):
            result = await model.query(History([{"role": "user", "content": "hi"}]), n=1)
            assert isinstance(result, dict)
            assert result["message"] == "ok"

    @pytest.mark.asyncio
    async def test_query_uses_async_retrying(self, model):
        """query() must use tenacity.AsyncRetrying, not sync Retrying.

        If sync Retrying is used, the internal time.sleep will block the
        event loop, which defeats the purpose of async conversion.
        """
        call_count = 0

        async def mock_query(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("transient")
            return [{"message": "ok", "output_tokens": [1]}]

        model.config.retry.retries = 3
        with patch.object(model, "_query", side_effect=mock_query):
            result = await model.query(
                History([{"role": "user", "content": "hi"}]), n=1
            )
            assert result == {"message": "ok", "output_tokens": [1]}
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_query_n_returns_list(self, model):
        with patch.object(
            model,
            "_single_query",
            new_callable=AsyncMock,
            return_value=[{"message": "ok"}],
        ):
            result = await model.query(
                History([{"role": "user", "content": "hi"}]), n=3
            )
            assert isinstance(result, list)
            assert len(result) == 3


# ---------------------------------------------------------------------------
# _THREADS_THAT_USED_API_KEYS removal
# ---------------------------------------------------------------------------
class TestApiKeyByTask:
    def test_thread_tracking_list_removed(self):
        """_THREADS_THAT_USED_API_KEYS should be removed in favor of task-based tracking."""
        from sweagent.agent import models

        assert not hasattr(models, "_THREADS_THAT_USED_API_KEYS"), (
            "_THREADS_THAT_USED_API_KEYS should be replaced with "
            "task-based key selection for async compatibility"
        )


# ---------------------------------------------------------------------------
# Source-level checks: no sync blocking in async hot path
# ---------------------------------------------------------------------------
class TestNoSyncBlockingInSource:
    def test_no_time_sleep_in_sleep_method(self):
        import inspect

        source = inspect.getsource(SGLangModel._sleep)
        assert "time.sleep" not in source, (
            "_sleep should use asyncio.sleep, not time.sleep"
        )

    def test_no_threading_lock_in_module(self):
        import inspect
        from sweagent.agent import models

        source = inspect.getsource(models)
        # GLOBAL_STATS_LOCK should be asyncio.Lock, not threading.Lock
        assert "GLOBAL_STATS_LOCK = Lock()" not in source, (
            "GLOBAL_STATS_LOCK should be asyncio.Lock(), not threading.Lock()"
        )

    def test_no_sync_retrying_in_query(self):
        import inspect

        source = inspect.getsource(SGLangModel.query)
        assert "for attempt in Retrying(" not in source, (
            "query() should use 'async for attempt in AsyncRetrying(', "
            "not 'for attempt in Retrying('"
        )
