"""T1: Async tests for SGLangModel (hot path).

Covers:
- All async methods are coroutines (parametrized smoke test)
- _sleep() behavior with asyncio.sleep
- _update_stats() with asyncio.Lock
- _single_query() with persistent httpx client
- query() with AsyncRetrying
- _THREADS_THAT_USED_API_KEYS removal
- Source-level: no sync blocking patterns
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
# Fixtures
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
# Parametrized coroutine signature check
# ---------------------------------------------------------------------------
_ASYNC_METHODS = ["_sleep", "_update_stats", "_single_query", "_query", "query"]


class TestMethodsAreCoroutines:
    @pytest.mark.parametrize("method", _ASYNC_METHODS)
    def test_is_coroutine(self, model, method):
        assert asyncio.iscoroutinefunction(getattr(model, method)), (
            f"SGLangModel.{method} should be async def"
        )


# ---------------------------------------------------------------------------
# _sleep behavior
# ---------------------------------------------------------------------------
class TestAsyncSleep:
    @pytest.mark.asyncio
    async def test_sleep_calls_asyncio_sleep(self, model):
        model.config.delay = 10.0
        from sweagent.agent.models import GLOBAL_STATS

        GLOBAL_STATS.last_query_timestamp = time.time()
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_async_sleep:
            await model._sleep()
            mock_async_sleep.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_sleep_skips_when_elapsed_sufficient(self, model):
        model.config.delay = 0
        await model._sleep()


# ---------------------------------------------------------------------------
# _update_stats behavior
# ---------------------------------------------------------------------------
class TestAsyncUpdateStats:
    def test_global_stats_lock_is_asyncio_lock(self):
        from sweagent.agent.models import GLOBAL_STATS_LOCK

        assert isinstance(GLOBAL_STATS_LOCK, asyncio.Lock)

    @pytest.mark.asyncio
    async def test_update_stats_increments_correctly(self, model):
        await model._update_stats(input_tokens=100, output_tokens=50, cost=0.01)
        assert model.stats.tokens_sent == 100
        assert model.stats.tokens_received == 50
        assert model.stats.api_calls == 1


# ---------------------------------------------------------------------------
# _single_query behavior
# ---------------------------------------------------------------------------
class TestAsyncSingleQuery:
    @pytest.mark.asyncio
    async def test_successful_query_with_httpx(self, model):
        mock_response = MagicMock()
        mock_response.json.return_value = _make_sglang_response(
            input_logprobs=[[0.0, i] for i in [10, 20, 30, 40, 50]],
        )
        mock_response.raise_for_status = MagicMock()
        mock_response.status_code = 200

        model._http_client = AsyncMock()
        model._http_client.post.return_value = mock_response

        with patch.object(model, "parse_response", return_value={"message": "response text"}):
            with patch.object(model, "_sleep", new_callable=AsyncMock):
                with patch.object(model, "_update_stats", new_callable=AsyncMock):
                    history = History([{"role": "user", "content": "hello"}])
                    result = await model._single_query(history)

        assert len(result) == 1
        assert result[0]["message"] == "response text"
        assert result[0]["output_tokens"] == [101, 102, 103]
        model._http_client.post.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_context_exceeded_before_http(self, model, mock_tokenizer):
        model.model_max_input_tokens = 3
        mock_tokenizer.apply_chat_template.return_value = [1, 2, 3, 4, 5]
        with patch.object(model, "_sleep", new_callable=AsyncMock):
            history = History([{"role": "user", "content": "hello"}])
            with pytest.raises(ContextWindowExceededError):
                await model._single_query(history)


# ---------------------------------------------------------------------------
# query behavior
# ---------------------------------------------------------------------------
class TestAsyncQuery:
    @pytest.mark.asyncio
    async def test_query_n1_returns_dict(self, model):
        with patch.object(
            model, "_single_query", new_callable=AsyncMock,
            return_value=[{"message": "ok", "output_tokens": [1]}],
        ):
            result = await model.query(History([{"role": "user", "content": "hi"}]), n=1)
            assert isinstance(result, dict)
            assert result["message"] == "ok"

    @pytest.mark.asyncio
    async def test_query_uses_async_retrying(self, model):
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
            model, "_single_query", new_callable=AsyncMock,
            return_value=[{"message": "ok"}],
        ):
            result = await model.query(
                History([{"role": "user", "content": "hi"}]), n=3
            )
            assert isinstance(result, list)
            assert len(result) == 3


# ---------------------------------------------------------------------------
# API key tracking
# ---------------------------------------------------------------------------
class TestApiKeyByTask:
    def test_thread_tracking_list_removed(self):
        from sweagent.agent import models

        assert not hasattr(models, "_THREADS_THAT_USED_API_KEYS")


# ---------------------------------------------------------------------------
# Source-level checks
# ---------------------------------------------------------------------------
class TestNoSyncBlockingInSource:
    def test_no_time_sleep_in_sleep_method(self):
        import inspect

        source = inspect.getsource(SGLangModel._sleep)
        assert "time.sleep" not in source

    def test_no_threading_lock_in_module(self):
        import inspect
        from sweagent.agent import models

        source = inspect.getsource(models)
        assert "GLOBAL_STATS_LOCK = Lock()" not in source

    def test_no_sync_retrying_in_query(self):
        import inspect

        source = inspect.getsource(SGLangModel.query)
        assert "for attempt in Retrying(" not in source

    def test_no_requests_import_in_single_query(self):
        import inspect

        source = inspect.getsource(SGLangModel._single_query)
        assert "requests.post" not in source
