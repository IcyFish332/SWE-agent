from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TokenSegment:
    token_ids: list[int]
    logprobs: list[float]
    is_response: bool


class TokenManager:
    def __init__(self) -> None:
        self._segments: list[TokenSegment] = []

    def reset(self) -> None:
        self._segments = []

    def add_prompt(self, token_ids: list[int], logprobs: list[float] | None = None) -> None:
        if not token_ids:
            return
        if logprobs is None:
            logprobs = [0.0] * len(token_ids)
        if len(logprobs) != len(token_ids):
            raise ValueError("logprobs length must match token_ids length")
        self._segments.append(TokenSegment(token_ids=list(token_ids), logprobs=list(logprobs), is_response=False))

    def add_response(self, token_ids: list[int], logprobs: list[float] | None = None) -> None:
        if not token_ids:
            return
        if logprobs is None:
            logprobs = [0.0] * len(token_ids)
        if len(logprobs) != len(token_ids):
            raise ValueError("logprobs length must match token_ids length")
        self._segments.append(TokenSegment(token_ids=list(token_ids), logprobs=list(logprobs), is_response=True))

    @property
    def token_ids(self) -> list[int]:
        return [token_id for segment in self._segments for token_id in segment.token_ids]

    @property
    def loss_mask(self) -> list[int]:
        return [int(segment.is_response) for segment in self._segments for _ in segment.token_ids]

    @property
    def logprobs(self) -> list[float]:
        return [logprob for segment in self._segments for logprob in segment.logprobs]

    @property
    def initial_prompt_token_ids(self) -> list[int]:
        if not self._segments:
            return []
        return list(self._segments[0].token_ids)

    @property
    def segment_info(self) -> list[tuple[bool, int]]:
        return [(segment.is_response, len(segment.token_ids)) for segment in self._segments]

    def __len__(self) -> int:
        return sum(len(segment.token_ids) for segment in self._segments)
