from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests
from pydantic import SecretStr

from sweagent.agent.models import (
    ContentPolicyViolationError,
    ContextWindowExceededError,
    SGLangModel,
    SGLangModelConfig,
)
from sweagent.tools.parsing import Identity
from sweagent.tools.tools import ToolConfig
from sweagent.types import History


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
# SGLangModel Init
# ---------------------------------------------------------------------------
class TestSGLangModelInit:
    def test_basic_init(self, model):
        assert model.token_manager is not None
        assert model._processed_message_count == 0

    def test_defaults_max_tokens(self, mock_tokenizer):
        config = SGLangModelConfig(
            name="test",
            api_base="http://localhost:30000",
            tokenizer=mock_tokenizer,
            max_input_tokens=None,
            max_output_tokens=None,
        )
        m = SGLangModel(config, ToolConfig(parse_function=Identity()))
        assert m.model_max_input_tokens == 126_976
        assert m.model_max_output_tokens == 4096

    def test_custom_max_tokens(self, mock_tokenizer):
        config = SGLangModelConfig(
            name="test",
            api_base="http://localhost:30000",
            tokenizer=mock_tokenizer,
            max_input_tokens=1000,
            max_output_tokens=500,
        )
        m = SGLangModel(config, ToolConfig(parse_function=Identity()))
        assert m.model_max_input_tokens == 1000
        assert m.model_max_output_tokens == 500


# ---------------------------------------------------------------------------
# reset_rollout_state
# ---------------------------------------------------------------------------
class TestResetRolloutState:
    def test_clears_token_manager_and_count(self, model):
        model.token_manager.add_prompt([1, 2, 3])
        model._processed_message_count = 5
        model.reset_rollout_state()
        assert len(model.token_manager) == 0
        assert model._processed_message_count == 0


# ---------------------------------------------------------------------------
# _extract_logprobs
# ---------------------------------------------------------------------------
class TestExtractLogprobs:
    def test_pair_format(self, model):
        response = {"meta_info": {"output_token_logprobs": [[-0.1, 101], [-0.2, 102]]}}
        result = model._extract_logprobs(response, "output_token_logprobs")
        assert result == pytest.approx([-0.1, -0.2])

    def test_dict_format(self, model):
        response = {"meta_info": {"output_token_logprobs": [{"logprob": -0.5}, {"logprob": -0.6}]}}
        result = model._extract_logprobs(response, "output_token_logprobs")
        assert result == pytest.approx([-0.5, -0.6])

    def test_flat_format(self, model):
        response = {"meta_info": {"output_token_logprobs": [-0.1, -0.2, -0.3]}}
        result = model._extract_logprobs(response, "output_token_logprobs")
        assert result == pytest.approx([-0.1, -0.2, -0.3])

    def test_missing_key(self, model):
        response = {"meta_info": {}}
        result = model._extract_logprobs(response, "output_token_logprobs")
        assert result is None

    def test_empty_list(self, model):
        response = {"meta_info": {"output_token_logprobs": []}}
        result = model._extract_logprobs(response, "output_token_logprobs")
        assert result is None

    def test_none_value(self, model):
        response = {"meta_info": {"output_token_logprobs": None}}
        result = model._extract_logprobs(response, "output_token_logprobs")
        assert result is None

    def test_meta_info_priority(self, model):
        response = {
            "meta_info": {"output_token_logprobs": [-0.1]},
            "output_token_logprobs": [-0.9],
        }
        result = model._extract_logprobs(response, "output_token_logprobs")
        assert result == pytest.approx([-0.1])

    def test_fallback_to_top_level(self, model):
        response = {"meta_info": {}, "output_token_logprobs": [-0.9, -0.8]}
        result = model._extract_logprobs(response, "output_token_logprobs")
        assert result == pytest.approx([-0.9, -0.8])


# ---------------------------------------------------------------------------
# _build_payload
# ---------------------------------------------------------------------------
class TestBuildPayload:
    def test_minimal(self, model):
        payload = model._build_payload([1, 2, 3])
        assert payload["input_ids"] == [1, 2, 3]
        assert payload["return_logprob"] is True
        assert payload["stream"] is False
        assert payload["sampling_params"]["skip_special_tokens"] is False
        assert "max_new_tokens" in payload["sampling_params"]

    def test_with_logprob_start_len(self, model):
        payload = model._build_payload([1, 2], logprob_start_len=5)
        assert payload["logprob_start_len"] == 5

    def test_without_logprob_start_len(self, model):
        payload = model._build_payload([1, 2])
        assert "logprob_start_len" not in payload

    def test_with_top_p(self, mock_tokenizer):
        config = SGLangModelConfig(
            name="test",
            api_base="http://localhost:30000",
            tokenizer=mock_tokenizer,
            top_p=0.9,
        )
        m = SGLangModel(config, ToolConfig(parse_function=Identity()))
        payload = m._build_payload([1, 2])
        assert payload["sampling_params"]["top_p"] == 0.9

    def test_without_top_p(self, model):
        payload = model._build_payload([1, 2])
        assert "top_p" not in payload["sampling_params"]

    def test_with_return_routed_experts(self, mock_tokenizer):
        config = SGLangModelConfig(
            name="test",
            api_base="http://localhost:30000",
            tokenizer=mock_tokenizer,
            return_routed_experts=True,
        )
        m = SGLangModel(config, ToolConfig(parse_function=Identity()))
        payload = m._build_payload([1, 2])
        assert payload["return_routed_experts"] is True

    def test_none_tokens(self, model):
        payload = model._build_payload(None)
        assert payload["input_ids"] == []


# ---------------------------------------------------------------------------
# _raise_for_http_error
# ---------------------------------------------------------------------------
class TestRaiseForHttpError:
    def _make_http_error(self, message: str, status_code: int = 400) -> requests.HTTPError:
        response = MagicMock()
        response.json.return_value = {"error": message}
        response.status_code = status_code
        error = requests.HTTPError(response=response)
        return error

    def test_context_length_error(self, model):
        error = self._make_http_error("context length exceeded")
        with pytest.raises(ContextWindowExceededError):
            model._raise_for_http_error(error)

    def test_max_context_error(self, model):
        error = self._make_http_error("max context limit reached")
        with pytest.raises(ContextWindowExceededError):
            model._raise_for_http_error(error)

    def test_context_window_error(self, model):
        error = self._make_http_error("context window exceeded")
        with pytest.raises(ContextWindowExceededError):
            model._raise_for_http_error(error)

    def test_content_policy_error(self, model):
        error = self._make_http_error("content policy violation detected")
        with pytest.raises(ContentPolicyViolationError):
            model._raise_for_http_error(error)

    def test_safety_error(self, model):
        error = self._make_http_error("safety filter triggered")
        with pytest.raises(ContentPolicyViolationError):
            model._raise_for_http_error(error)

    def test_generic_error_reraises(self, model):
        error = self._make_http_error("some random server error")
        with pytest.raises(requests.HTTPError):
            model._raise_for_http_error(error)

    def test_no_response_reraises(self, model):
        error = requests.HTTPError()
        error.response = None
        with pytest.raises(requests.HTTPError):
            model._raise_for_http_error(error)


# ---------------------------------------------------------------------------
# _sort_incremental_history
# ---------------------------------------------------------------------------
class TestSortIncrementalHistory:
    def test_tool_messages_sorted_by_id(self, model):
        history = History(
            [
                {"role": "tool", "content": "result_b", "tool_call_ids": ["call_002"]},
                {"role": "tool", "content": "result_a", "tool_call_ids": ["call_001"]},
            ]
        )
        sorted_h = model._sort_incremental_history(history)
        assert sorted_h[0]["tool_call_ids"] == ["call_001"]
        assert sorted_h[1]["tool_call_ids"] == ["call_002"]

    def test_non_tool_preserved(self, model):
        history = History(
            [
                {"role": "user", "content": "hello"},
                {"role": "tool", "content": "r2", "tool_call_ids": ["call_002"]},
                {"role": "tool", "content": "r1", "tool_call_ids": ["call_001"]},
                {"role": "assistant", "content": "ok"},
            ]
        )
        sorted_h = model._sort_incremental_history(history)
        assert sorted_h[0]["role"] == "user"
        assert sorted_h[1]["tool_call_ids"] == ["call_001"]
        assert sorted_h[2]["tool_call_ids"] == ["call_002"]
        assert sorted_h[3]["role"] == "assistant"

    def test_empty(self, model):
        assert model._sort_incremental_history(History([])) == []


# ---------------------------------------------------------------------------
# tokenize_prompt_messages
# ---------------------------------------------------------------------------
class TestTokenizePromptMessages:
    def test_first_call_full_tokenization(self, model):
        history = History([{"role": "user", "content": "hello"}])
        result = model.tokenize_prompt_messages(history)
        assert result is not None
        assert result == [10, 20, 30, 40, 50]
        # config.tokenizer is a deep copy; check via model's own reference
        model.config.tokenizer.apply_chat_template.assert_called_once()

    def test_no_new_messages_returns_none(self, model):
        # Simulate: token_manager is non-empty (has prior data),
        # and processed_message_count == len(history)
        model.token_manager.add_prompt([1, 2, 3])
        model._processed_message_count = 2
        history = History([{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}])
        result = model.tokenize_prompt_messages(history)
        assert result is None

    def test_incremental_returns_new_tokens(self, model):
        # First call sets up state
        model.token_manager.add_prompt([1, 2, 3])
        model._processed_message_count = 1

        # Configure tokenizer ON THE MODEL's deep-copied config
        call_count = [0]
        def side_effect(*, conversation, tools=None, add_generation_prompt, tokenize, return_dict):
            call_count[0] += 1
            if call_count[0] == 1:
                # Full tokenization (fake prefix + new messages)
                return [50, 51, 52, 53, 54, 55]
            else:
                # Prefix-only tokenization
                return [50, 51, 52]

        model.config.tokenizer.apply_chat_template.side_effect = side_effect
        history = History([
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "new message"},
        ])
        result = model.tokenize_prompt_messages(history)
        assert result is not None
        # Result should be full_ids[len(prefix_ids):] = [53, 54, 55]
        assert result == [53, 54, 55]

    def test_incremental_with_separator(self, model):
        model.token_manager.add_prompt([1, 2, 3])
        model._processed_message_count = 1
        model.config.message_separator = "\n"

        call_count = [0]
        def side_effect(*, conversation, tools=None, add_generation_prompt, tokenize, return_dict):
            call_count[0] += 1
            if call_count[0] == 1:
                return [50, 51, 52, 53]
            else:
                return [50, 51]

        model.config.tokenizer.apply_chat_template.side_effect = side_effect
        model.config.tokenizer.encode.return_value = [99]  # separator token

        history = History([
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ])
        result = model.tokenize_prompt_messages(history)
        assert result is not None
        # separator_ids + new tokens = [99] + [52, 53]
        assert result == [99, 52, 53]


# ---------------------------------------------------------------------------
# _validate_incremental_tokens
# ---------------------------------------------------------------------------
class TestValidateIncrementalTokens:
    def test_disabled_is_noop(self, model, mock_tokenizer):
        model.config.debug_check_incremental_tokens = False
        model._validate_incremental_tokens(History([]), [1, 2, 3])
        mock_tokenizer.apply_chat_template.assert_not_called()

    def test_enabled_matching_passes(self, model, mock_tokenizer):
        model.config.debug_check_incremental_tokens = True
        model.token_manager.add_prompt([10, 20, 30])
        # _tokenize_messages will return the same thing via apply_chat_template
        mock_tokenizer.apply_chat_template.return_value = [10, 20, 30, 40, 50]
        # Should not raise
        model._validate_incremental_tokens(
            History([{"role": "user", "content": "hello"}]),
            [40, 50],
        )

    def test_enabled_mismatch_raises(self, model, mock_tokenizer):
        model.config.debug_check_incremental_tokens = True
        model.token_manager.add_prompt([10, 20, 30])
        mock_tokenizer.apply_chat_template.return_value = [10, 20, 30, 40, 50]
        with pytest.raises(AssertionError, match="Incremental tokenization drift"):
            model._validate_incremental_tokens(
                History([{"role": "user", "content": "hello"}]),
                [99, 99],  # wrong tokens
            )


# ---------------------------------------------------------------------------
# _single_query (requires mocking requests.post and parse_response)
# ---------------------------------------------------------------------------
class TestSingleQuery:
    @patch("sweagent.agent.models.time")
    @patch("sweagent.agent.models.requests.post")
    def test_successful_query(self, mock_post, mock_time, model):
        mock_time.time.return_value = 1000.0
        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_sglang_response(
            input_logprobs=[[0.0, i] for i in [10, 20, 30, 40, 50]]
        )
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        with patch.object(model, "parse_response", return_value={"message": "response text"}):
            history = History([{"role": "user", "content": "hello"}])
            result = model._single_query(history)

        assert len(result) == 1
        assert result[0]["message"] == "response text"
        assert result[0]["output_tokens"] == [101, 102, 103]
        assert len(result[0]["rollout_log_probs"]) == 3

    @patch("sweagent.agent.models.time")
    @patch("sweagent.agent.models.requests.post")
    def test_updates_token_manager(self, mock_post, mock_time, model):
        mock_time.time.return_value = 1000.0
        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_sglang_response(
            input_logprobs=[[0.0, i] for i in [10, 20, 30, 40, 50]]
        )
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        with patch.object(model, "parse_response", return_value={"message": "text"}):
            history = History([{"role": "user", "content": "hello"}])
            model._single_query(history)

        # Token manager should have prompt + response segments
        assert len(model.token_manager) > 0
        assert model.token_manager.segment_info[0][0] is False  # prompt
        assert model.token_manager.segment_info[1][0] is True   # response

    @patch("sweagent.agent.models.time")
    @patch("sweagent.agent.models.requests.post")
    def test_updates_processed_message_count(self, mock_post, mock_time, model):
        mock_time.time.return_value = 1000.0
        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_sglang_response()
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        with patch.object(model, "parse_response", return_value={"message": "text"}):
            history = History([{"role": "user", "content": "hello"}])
            model._single_query(history)

        # Should be len(history) + 1 = 2
        assert model._processed_message_count == 2

    def test_context_exceeded_before_http(self, model, mock_tokenizer):
        model.model_max_input_tokens = 3  # very small limit
        mock_tokenizer.apply_chat_template.return_value = [1, 2, 3, 4, 5]  # 5 tokens > 3
        history = History([{"role": "user", "content": "hello"}])
        with pytest.raises(ContextWindowExceededError):
            model._single_query(history)

    @patch("sweagent.agent.models.time")
    @patch("sweagent.agent.models.requests.post")
    def test_logprob_padding_when_shorter(self, mock_post, mock_time, model):
        mock_time.time.return_value = 1000.0
        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_sglang_response(
            output_ids=[101, 102, 103],
            output_logprobs=[[-0.1, 101]],  # only 1 logprob for 3 tokens
        )
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        with patch.object(model, "parse_response", return_value={"message": "text"}):
            result = model._single_query(History([{"role": "user", "content": "hello"}]))

        # Should be padded to 3
        assert len(result[0]["rollout_log_probs"]) == 3

    @patch("sweagent.agent.models.time")
    @patch("sweagent.agent.models.requests.post")
    def test_logprob_truncation_when_longer(self, mock_post, mock_time, model):
        mock_time.time.return_value = 1000.0
        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_sglang_response(
            output_ids=[101],
            output_logprobs=[[-0.1, 101], [-0.2, 102], [-0.3, 103]],  # 3 logprobs for 1 token
        )
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        with patch.object(model, "parse_response", return_value={"message": "text"}):
            result = model._single_query(History([{"role": "user", "content": "hello"}]))

        assert len(result[0]["rollout_log_probs"]) == 1

    @patch("sweagent.agent.models.time")
    @patch("sweagent.agent.models.requests.post")
    def test_non_list_output_ids_handled(self, mock_post, mock_time, model):
        mock_time.time.return_value = 1000.0
        mock_resp = MagicMock()
        data = _make_sglang_response()
        data["output_ids"] = "not a list"
        mock_resp.json.return_value = data
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        with patch.object(model, "parse_response", return_value={"message": "text"}):
            result = model._single_query(History([{"role": "user", "content": "hello"}]))

        assert result[0]["output_tokens"] == []


# ---------------------------------------------------------------------------
# query (public entry point)
# ---------------------------------------------------------------------------
class TestQuery:
    @patch("sweagent.agent.models.time")
    @patch("sweagent.agent.models.requests.post")
    def test_n1_returns_dict(self, mock_post, mock_time, model):
        mock_time.time.return_value = 1000.0
        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_sglang_response()
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        with patch.object(model, "parse_response", return_value={"message": "text"}):
            result = model.query(History([{"role": "user", "content": "hello"}]), n=1)

        assert isinstance(result, dict)

    def test_empty_history(self, model):
        # Empty history with token_manager empty -> should call _tokenize_messages with []
        # This tests the query dispatch, not the full HTTP flow
        with patch.object(model, "_single_query", return_value=[{"message": "ok"}]) as mock_sq:
            result = model.query(History([]), n=1)
            mock_sq.assert_called_once()
            assert result == {"message": "ok"}

    def test_invalid_history_type_raises(self, model):
        with pytest.raises(TypeError, match="message history"):
            model.query([1, 2, 3], n=1)
