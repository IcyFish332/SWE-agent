from __future__ import annotations

import json
import traceback
import uuid

from sweagent.utils.log import get_logger

logger = get_logger("swea-parser", emoji="🧩")


def _detect_think_and_return_ori_think(
    text: str,
    think_start_token: str,
    think_end_token: str,
) -> tuple[str, str]:
    in_reasoning = think_start_token in text

    if not in_reasoning:
        return "", text

    processed_text = text.replace(think_start_token, "")

    if think_end_token not in processed_text:
        return think_start_token + processed_text, ""

    reasoning_text, normal_text = processed_text.split(think_end_token, maxsplit=1)
    return think_start_token + reasoning_text + think_end_token, normal_text


def parse_tool_calls_with_sglang(
    text: str,
    tools: list[dict],
    tool_call_parser: str,
    reasoning_parser: str,
) -> dict:
    from sglang.srt.entrypoints.openai.protocol import Function as SglFunction
    from sglang.srt.entrypoints.openai.protocol import Tool as SglTool
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
    from sglang.srt.parser.reasoning_parser import ReasoningParser

    if not isinstance(text, str) or not text:
        return {"message": "", "tool_calls": None, "reasoning_content": None}

    reasoning_parser_p = ReasoningParser(reasoning_parser)
    think_start_token = reasoning_parser_p.detector.think_start_token
    think_end_token = reasoning_parser_p.detector.think_end_token

    reasoning_content, content_text = _detect_think_and_return_ori_think(
        text,
        think_start_token,
        think_end_token,
    )

    if reasoning_content:
        if think_start_token:
            reasoning_content = reasoning_content.replace(think_start_token, "", 1)
        if reasoning_content.endswith(think_end_token):
            reasoning_content = reasoning_content[: -len(think_end_token)]
        reasoning_content = reasoning_content.strip() or None
    else:
        reasoning_content = None

    if not tools or not content_text:
        return {
            "message": content_text,
            "tool_calls": None,
            "reasoning_content": reasoning_content,
        }

    sgl_tools = [
        SglTool(type=tool.get("type", "function"), function=SglFunction(**tool["function"]))
        for tool in tools
    ]
    parser = FunctionCallParser(sgl_tools, tool_call_parser)

    try:
        if parser.has_tool_call(content_text):
            message_text, call_info_list = parser.parse_non_stream(content_text)
            tool_calls = []
            for call in call_info_list:
                name = call.function.name if hasattr(call, "function") else call.name
                arguments = (
                    call.function.arguments
                    if hasattr(call, "function") and hasattr(call.function, "arguments")
                    else getattr(call, "arguments", getattr(call, "parameters", "{}"))
                )
                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                elif not isinstance(arguments, str):
                    arguments = str(arguments)
                tool_calls.append(
                    {
                        "type": "function",
                        "id": getattr(call, "id", None) or f"call_{uuid.uuid4().hex[:24]}",
                        "function": {
                            "name": name,
                            "arguments": arguments,
                        },
                    }
                )
            return {
                "message": message_text,
                "tool_calls": tool_calls or None,
                "reasoning_content": reasoning_content,
            }
    except Exception as exc:
        logger.error("Tool call parsing error: %s", exc)
        traceback.print_exc()

    return {
        "message": content_text,
        "tool_calls": None,
        "reasoning_content": reasoning_content,
    }
