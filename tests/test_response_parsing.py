from __future__ import annotations

import json

import pytest

from sweagent.agent.response_parsing import (
    _detect_think_and_return_ori_think,
    parse_tool_calls_with_sglang,
)


# ---------------------------------------------------------------------------
# _detect_think_and_return_ori_think — pure function, no mocks
# ---------------------------------------------------------------------------
class TestDetectThink:
    def test_no_thinking_tokens(self):
        reasoning, content = _detect_think_and_return_ori_think(
            "Hello world", "<think>", "</think>"
        )
        assert reasoning == ""
        assert content == "Hello world"

    def test_complete_block(self):
        text = "<think>I need to reason</think>The answer is 42"
        reasoning, content = _detect_think_and_return_ori_think(text, "<think>", "</think>")
        assert reasoning == "<think>I need to reason</think>"
        assert content == "The answer is 42"

    def test_unclosed_block(self):
        text = "<think>Still thinking..."
        reasoning, content = _detect_think_and_return_ori_think(text, "<think>", "</think>")
        assert reasoning == "<think>Still thinking..."
        assert content == ""

    def test_empty_text(self):
        reasoning, content = _detect_think_and_return_ori_think("", "<think>", "</think>")
        assert reasoning == ""
        assert content == ""

    def test_thinking_at_start_with_content_after(self):
        text = "<think>step 1, step 2</think>action: do something"
        reasoning, content = _detect_think_and_return_ori_think(text, "<think>", "</think>")
        assert "step 1, step 2" in reasoning
        assert content == "action: do something"

    def test_empty_reasoning(self):
        text = "<think></think>Just content"
        reasoning, content = _detect_think_and_return_ori_think(text, "<think>", "</think>")
        assert reasoning == "<think></think>"
        assert content == "Just content"


# ---------------------------------------------------------------------------
# parse_tool_calls_with_sglang — using real sglang parsers
# ---------------------------------------------------------------------------
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


class TestParseToolCallsEdgeCases:
    """Test edge cases that don't require sglang parsers."""

    def test_empty_text(self):
        result = parse_tool_calls_with_sglang("", [], "qwen25", "qwen25")
        assert result == {"message": "", "tool_calls": None, "reasoning_content": None}

    def test_none_text(self):
        result = parse_tool_calls_with_sglang(None, [], "qwen25", "qwen25")
        assert result == {"message": "", "tool_calls": None, "reasoning_content": None}

    def test_non_string_text(self):
        result = parse_tool_calls_with_sglang(123, [], "qwen25", "qwen25")
        assert result == {"message": "", "tool_calls": None, "reasoning_content": None}


class TestParseToolCallsWithRealSglang:
    """Tests using actual sglang FunctionCallParser and ReasoningParser."""

    @staticmethod
    def _get_reasoning_parser_name():
        """Discover a valid ReasoningParser model type from the installed sglang."""
        try:
            from sglang.srt.parser.reasoning_parser import ReasoningParser
            supported = list(ReasoningParser.DetectorMap.keys())
            if not supported:
                return None
            # Prefer deepseek-r1 or qwen3 if available; fallback to first
            for name in ("deepseek-r1", "qwen3"):
                if name in supported:
                    return name
            return supported[0]
        except Exception:
            return None

    @pytest.fixture(autouse=True)
    def _check_parsers(self):
        self.reasoning_parser = self._get_reasoning_parser_name()
        if self.reasoning_parser is None:
            pytest.skip("No supported ReasoningParser model type found in sglang")

    def test_plain_text_no_tools(self):
        result = parse_tool_calls_with_sglang(
            "Hello, this is a plain response.",
            [],
            "hermes",
            self.reasoning_parser,
        )
        assert result["message"] == "Hello, this is a plain response."
        assert result["tool_calls"] is None

    def test_plain_text_with_tools_but_no_call(self):
        result = parse_tool_calls_with_sglang(
            "I think we should proceed carefully.",
            [BASH_TOOL],
            "hermes",
            self.reasoning_parser,
        )
        assert "proceed carefully" in result["message"]
        assert result["tool_calls"] is None

    def test_reasoning_content_extracted(self):
        text = "<think>Let me think about this step by step.</think>The answer is 42."
        result = parse_tool_calls_with_sglang(text, [], "hermes", self.reasoning_parser)
        assert result["reasoning_content"] is not None
        assert "step by step" in result["reasoning_content"]
        assert result["message"] == "The answer is 42."

    def test_reasoning_only_no_content(self):
        text = "<think>Still reasoning about the problem...</think>"
        result = parse_tool_calls_with_sglang(text, [], "hermes", self.reasoning_parser)
        assert result["reasoning_content"] is not None
        assert result["message"] == ""

    def test_hermes_single_tool_call(self):
        text = '<tool_call>{"name": "bash", "arguments": {"command": "ls -la"}}</tool_call>'
        result = parse_tool_calls_with_sglang(text, [BASH_TOOL], "hermes", self.reasoning_parser)
        assert result["tool_calls"] is not None
        assert len(result["tool_calls"]) == 1
        call = result["tool_calls"][0]
        assert call["function"]["name"] == "bash"
        args = json.loads(call["function"]["arguments"])
        assert args["command"] == "ls -la"
        assert call["type"] == "function"
        assert "id" in call

    def test_hermes_multiple_tool_calls(self):
        text = (
            '<tool_call>{"name": "bash", "arguments": {"command": "pwd"}}</tool_call>'
            '<tool_call>{"name": "bash", "arguments": {"command": "ls"}}</tool_call>'
        )
        result = parse_tool_calls_with_sglang(text, [BASH_TOOL], "hermes", self.reasoning_parser)
        assert result["tool_calls"] is not None
        assert len(result["tool_calls"]) == 2

    def test_hermes_with_thinking(self):
        text = (
            "<think>I should list the files first.</think>"
            '<tool_call>{"name": "bash", "arguments": {"command": "ls"}}</tool_call>'
        )
        result = parse_tool_calls_with_sglang(text, [BASH_TOOL], "hermes", self.reasoning_parser)
        assert result["reasoning_content"] is not None
        assert "list the files" in result["reasoning_content"]
        assert result["tool_calls"] is not None
        assert len(result["tool_calls"]) == 1

    def test_tool_call_arguments_are_json_string(self):
        text = '<tool_call>{"name": "bash", "arguments": {"command": "echo hello"}}</tool_call>'
        result = parse_tool_calls_with_sglang(text, [BASH_TOOL], "hermes", self.reasoning_parser)
        assert result["tool_calls"] is not None
        # arguments should be a JSON string, not a dict
        assert isinstance(result["tool_calls"][0]["function"]["arguments"], str)
        parsed_args = json.loads(result["tool_calls"][0]["function"]["arguments"])
        assert parsed_args["command"] == "echo hello"

    def test_tool_call_has_id(self):
        text = '<tool_call>{"name": "bash", "arguments": {"command": "pwd"}}</tool_call>'
        result = parse_tool_calls_with_sglang(text, [BASH_TOOL], "hermes", self.reasoning_parser)
        assert result["tool_calls"] is not None
        call_id = result["tool_calls"][0]["id"]
        assert isinstance(call_id, str)
        assert call_id.startswith("call_")

    def test_malformed_tool_call_graceful_fallback(self):
        text = '<tool_call>this is not valid json</tool_call>'
        result = parse_tool_calls_with_sglang(text, [BASH_TOOL], "hermes", self.reasoning_parser)
        # Should not crash; may return with or without tool_calls depending on parser
        assert "message" in result
