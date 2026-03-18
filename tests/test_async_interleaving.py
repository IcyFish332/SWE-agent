"""Test that async conversion enables step-level interleaving.

This is the core value proposition of the async refactoring: two agents
running in the same event loop can interleave at await points (model.query,
env.communicate), so one trajectory yields the loop while the other progresses.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sweagent.agent.models import SGLangModel, SGLangModelConfig
from sweagent.tools.parsing import Identity
from sweagent.tools.tools import ToolConfig
from sweagent.types import History


@pytest.fixture
def _make_model():
    """Factory that creates an SGLangModel with a mock tokenizer."""
    def factory():
        tok = MagicMock()
        tok.apply_chat_template.return_value = [10, 20, 30]
        tok.encode.return_value = [99]
        from pydantic import SecretStr
        config = SGLangModelConfig(
            name="test",
            api_base="http://localhost:30000",
            api_key=SecretStr("k"),
            tokenizer=tok,
            top_p=None,
            per_instance_cost_limit=0,
            total_cost_limit=0,
            per_instance_call_limit=0,
        )
        return SGLangModel(config, ToolConfig(parse_function=Identity()))
    return factory


class TestStepLevelInterleaving:
    """Prove that two concurrent model.query() calls interleave at await points."""

    @pytest.mark.asyncio
    async def test_two_queries_interleave(self, _make_model):
        """Two agents' model.query() calls should interleave within one event loop.

        Agent A has a slow inference (0.05s), Agent B has a fast one (0.01s).
        If run concurrently, B should finish between A's start and A's end.
        """
        call_order: list[str] = []

        def _make_sglang_response():
            return {
                "text": "ok",
                "output_ids": [101],
                "meta_info": {"output_token_logprobs": [[-0.1, 101]]},
            }

        async def slow_post(*args, **kwargs):
            call_order.append("A_start")
            await asyncio.sleep(0.05)
            call_order.append("A_end")
            resp = MagicMock()
            resp.json.return_value = _make_sglang_response()
            resp.raise_for_status = MagicMock()
            return resp

        async def fast_post(*args, **kwargs):
            call_order.append("B_start")
            await asyncio.sleep(0.01)
            call_order.append("B_end")
            resp = MagicMock()
            resp.json.return_value = _make_sglang_response()
            resp.raise_for_status = MagicMock()
            return resp

        model_a = _make_model()
        model_b = _make_model()

        model_a._http_client = AsyncMock()
        model_a._http_client.post = slow_post
        model_b._http_client = AsyncMock()
        model_b._http_client.post = fast_post

        with patch.object(SGLangModel, "_sleep", new_callable=AsyncMock):
            history = History([{"role": "user", "content": "hello"}])

            with patch.object(model_a, "parse_response", return_value={"message": "a"}):
                with patch.object(model_b, "parse_response", return_value={"message": "b"}):
                    results = await asyncio.gather(
                        model_a.query(history, n=1),
                        model_b.query(history, n=1),
                    )

        assert results[0]["message"] == "a"
        assert results[1]["message"] == "b"

        # Core assertion: B finishes before A because they interleave
        assert call_order.index("B_end") < call_order.index("A_end"), (
            f"Expected B to finish before A (interleaving), but got: {call_order}"
        )

    @pytest.mark.asyncio
    async def test_many_queries_share_event_loop(self, _make_model):
        """N concurrent queries should all run within the same event loop
        without blocking each other."""
        n_agents = 10
        completed: list[int] = []

        async def mock_post(agent_id, *args, **kwargs):
            await asyncio.sleep(0.01)
            completed.append(agent_id)
            resp = MagicMock()
            resp.json.return_value = {
                "text": "ok",
                "output_ids": [101],
                "meta_info": {"output_token_logprobs": [[-0.1, 101]]},
            }
            resp.raise_for_status = MagicMock()
            return resp

        models = []
        for i in range(n_agents):
            m = _make_model()
            m._http_client = AsyncMock()
            # Bind agent_id via default arg
            m._http_client.post = (lambda idx: (lambda *a, **kw: mock_post(idx, *a, **kw)))(i)
            models.append(m)

        with patch.object(SGLangModel, "_sleep", new_callable=AsyncMock):
            history = History([{"role": "user", "content": "hi"}])
            tasks = []
            for m in models:
                with patch.object(m, "parse_response", return_value={"message": "x"}):
                    pass
                # Need parse_response to persist, patch directly
                m.parse_response = MagicMock(return_value={"message": "x"})
                tasks.append(m.query(history, n=1))

            results = await asyncio.gather(*tasks)

        assert len(results) == n_agents
        assert len(completed) == n_agents
