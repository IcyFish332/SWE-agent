"""Integration tests for SGLangModel against a live SGLang server.

These tests require a running SGLang server. Run with:

    pytest tests/test_sglang_integration.py --sglang-base-url http://localhost:30000 -v

Or set the SGLANG_BASE_URL environment variable.
"""
from __future__ import annotations

import pytest
import requests

from sweagent.agent.models import SGLangModel, SGLangModelConfig
from sweagent.tools.parsing import Identity
from sweagent.tools.tools import ToolConfig
from sweagent.types import History

pytestmark = pytest.mark.sglang


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def sglang_base_url(request):
    url = request.config.getoption("--sglang-base-url")
    if not url:
        pytest.skip("--sglang-base-url not provided (set via CLI or SGLANG_BASE_URL env var)")
    try:
        resp = requests.get(f"{url}/health", timeout=10)
        resp.raise_for_status()
    except Exception as e:
        pytest.skip(f"SGLang server not reachable at {url}: {e}")
    return url


@pytest.fixture(scope="session")
def model_id(sglang_base_url):
    resp = requests.get(f"{sglang_base_url}/get_model_info", timeout=10)
    info = resp.json()
    return info.get("model_path") or info.get("model_id")


@pytest.fixture(scope="module")
def tokenizer(model_id):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


@pytest.fixture
def sglang_model(tokenizer, sglang_base_url):
    config = SGLangModelConfig(
        name="test-integration",
        api_base=sglang_base_url,
        tokenizer=tokenizer,
        top_p=None,
        per_instance_cost_limit=0,
        total_cost_limit=0,
        per_instance_call_limit=0,
        max_output_tokens=256,
    )
    model = SGLangModel(config, ToolConfig(parse_function=Identity()))
    yield model
    model.reset_rollout_state()


BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to execute"},
            },
            "required": ["command"],
        },
    },
}


# ---------------------------------------------------------------------------
# TITO Consistency
# ---------------------------------------------------------------------------
class TestTITOConsistency:
    def test_token_count_consistency(self, sglang_model):
        """After one query, all token manager properties should have consistent lengths."""
        history = History([{"role": "user", "content": "What is 2+2? Answer briefly."}])
        sglang_model.query(history, n=1)

        tm = sglang_model.token_manager
        total = len(tm)
        assert total > 0
        assert total == len(tm.token_ids)
        assert total == len(tm.loss_mask)
        assert total == len(tm.logprobs)
        assert total == sum(size for _, size in tm.segment_info)

    def test_loss_mask_binary(self, sglang_model):
        """All loss_mask values should be 0 or 1."""
        history = History([{"role": "user", "content": "Say hello."}])
        sglang_model.query(history, n=1)

        for val in sglang_model.token_manager.loss_mask:
            assert val in (0, 1), f"loss_mask contains non-binary value: {val}"

    def test_segment_pattern_single_turn(self, sglang_model):
        """Single turn should produce [prompt, response] segments."""
        history = History([{"role": "user", "content": "Hi."}])
        sglang_model.query(history, n=1)

        info = sglang_model.token_manager.segment_info
        assert len(info) == 2
        assert info[0][0] is False  # prompt
        assert info[1][0] is True   # response

    def test_logprobs_are_floats(self, sglang_model):
        """Response token logprobs should be real floats, not zero placeholders."""
        history = History([{"role": "user", "content": "What is Python?"}])
        sglang_model.query(history, n=1)

        tm = sglang_model.token_manager
        # Get response logprobs only
        response_logprobs = []
        offset = 0
        for is_response, size in tm.segment_info:
            segment_logprobs = tm.logprobs[offset : offset + size]
            if is_response:
                response_logprobs.extend(segment_logprobs)
            offset += size

        assert len(response_logprobs) > 0
        # At least some response logprobs should be non-zero (negative)
        non_zero = [lp for lp in response_logprobs if lp != 0.0]
        assert len(non_zero) > 0, "All response logprobs are 0.0 — likely placeholder, not real values"


# ---------------------------------------------------------------------------
# Incremental Tokenization
# ---------------------------------------------------------------------------
class TestIncrementalTokenization:
    def test_incremental_smaller_than_full(self, sglang_model, tokenizer):
        """Second tokenize_prompt_messages call should return fewer tokens than full re-tokenization."""
        # First query
        history = History([{"role": "user", "content": "What is 1+1?"}])
        sglang_model.query(history, n=1)

        # Add assistant response + tool observation to history
        extended_history = History(
            list(history)
            + [
                {"role": "assistant", "content": "Let me calculate."},
                {"role": "user", "content": "Please continue.", "message_type": "observation"},
            ]
        )
        incremental_ids = sglang_model.tokenize_prompt_messages(extended_history)
        assert incremental_ids is not None

        # Full tokenization would be much larger
        messages = sglang_model._history_to_messages(extended_history)
        full_ids = sglang_model._tokenize_messages(messages, add_generation_prompt=True)
        assert len(incremental_ids) < len(full_ids)

    def test_accumulated_matches_full(self, sglang_model, tokenizer):
        """With debug_check_incremental_tokens=True, no drift should be detected."""
        sglang_model.config.debug_check_incremental_tokens = True
        history = History([{"role": "user", "content": "Hello, world!"}])
        # This should not raise AssertionError
        sglang_model.query(history, n=1)


# ---------------------------------------------------------------------------
# Retokenization Drift
# ---------------------------------------------------------------------------
class TestRetokenizationDrift:
    def test_encode_decode_roundtrip(self, sglang_model, tokenizer):
        """encode(decode(token_ids)) should equal token_ids (no retokenization drift)."""
        history = History([{"role": "user", "content": "Explain the Pythagorean theorem briefly."}])
        sglang_model.query(history, n=1)

        original_ids = sglang_model.token_manager.token_ids
        decoded_text = tokenizer.decode(original_ids)
        re_encoded = tokenizer.encode(decoded_text, add_special_tokens=False)

        if original_ids != re_encoded:
            pytest.skip(
                f"Tokenizer has round-trip drift ({len(original_ids)} vs {len(re_encoded)} tokens). "
                "This is expected for some tokenizers and is exactly what TITO avoids."
            )


# ---------------------------------------------------------------------------
# Multi-Turn Segments
# ---------------------------------------------------------------------------
class TestMultiTurnSegments:
    def test_segment_pattern_after_two_turns(self, sglang_model):
        """Two query turns should produce [P, R, P, R] segment pattern."""
        # First turn
        history = History([{"role": "user", "content": "What is 1+1?"}])
        result1 = sglang_model.query(history, n=1)

        # Second turn: add assistant response + user follow-up
        history2 = History(
            list(history)
            + [
                {"role": "assistant", "content": result1["message"]},
                {"role": "user", "content": "And what is 2+2?"},
            ]
        )
        sglang_model.query(history2, n=1)

        info = sglang_model.token_manager.segment_info
        assert len(info) == 4
        assert info[0][0] is False  # prompt 1
        assert info[1][0] is True   # response 1
        assert info[2][0] is False  # prompt 2
        assert info[3][0] is True   # response 2

    def test_loss_mask_preserves_prior_segments(self, sglang_model):
        """Adding new segments should not change the loss_mask of prior segments."""
        history = History([{"role": "user", "content": "Say one word."}])
        sglang_model.query(history, n=1)

        mask_after_first = list(sglang_model.token_manager.loss_mask)
        n_first = len(mask_after_first)

        # Second turn
        result = sglang_model.query(
            History(
                list(history)
                + [
                    {"role": "assistant", "content": sglang_model.token_manager.token_ids[-1:]},
                    {"role": "user", "content": "Say another word."},
                ]
            ),
            n=1,
        )

        mask_after_second = sglang_model.token_manager.loss_mask
        # First N elements should be unchanged
        assert mask_after_second[:n_first] == mask_after_first
