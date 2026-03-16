from __future__ import annotations

import pytest

from sweagent.agent.token_manager import TokenManager, TokenSegment


class TestTokenSegment:
    def test_dataclass_creation(self):
        seg = TokenSegment(token_ids=[1, 2, 3], logprobs=[-0.1, -0.2, -0.3], is_response=True)
        assert seg.token_ids == [1, 2, 3]
        assert seg.logprobs == [-0.1, -0.2, -0.3]
        assert seg.is_response is True

    def test_prompt_segment(self):
        seg = TokenSegment(token_ids=[10], logprobs=[0.0], is_response=False)
        assert seg.is_response is False


class TestTokenManagerEmpty:
    def test_init_empty(self):
        tm = TokenManager()
        assert len(tm) == 0
        assert tm.token_ids == []
        assert tm.loss_mask == []
        assert tm.logprobs == []

    def test_initial_prompt_token_ids_empty(self):
        tm = TokenManager()
        assert tm.initial_prompt_token_ids == []

    def test_segment_info_empty(self):
        tm = TokenManager()
        assert tm.segment_info == []


class TestTokenManagerAddPrompt:
    def test_basic(self):
        tm = TokenManager()
        tm.add_prompt([1, 2, 3])
        assert tm.token_ids == [1, 2, 3]
        assert tm.loss_mask == [0, 0, 0]
        assert len(tm) == 3

    def test_with_logprobs(self):
        tm = TokenManager()
        tm.add_prompt([1, 2], [-0.1, -0.2])
        assert tm.logprobs == [-0.1, -0.2]

    def test_none_logprobs_defaults_to_zeros(self):
        tm = TokenManager()
        tm.add_prompt([1, 2, 3], logprobs=None)
        assert tm.logprobs == [0.0, 0.0, 0.0]

    def test_empty_list_noop(self):
        tm = TokenManager()
        tm.add_prompt([])
        assert len(tm) == 0
        assert tm.segment_info == []

    def test_mismatched_logprobs_raises(self):
        tm = TokenManager()
        with pytest.raises(ValueError, match="logprobs length"):
            tm.add_prompt([1, 2, 3], [-0.1])


class TestTokenManagerAddResponse:
    def test_basic(self):
        tm = TokenManager()
        tm.add_response([10, 11, 12])
        assert tm.token_ids == [10, 11, 12]
        assert tm.loss_mask == [1, 1, 1]
        assert len(tm) == 3

    def test_with_logprobs(self):
        tm = TokenManager()
        tm.add_response([10, 11], [-0.5, -0.6])
        assert tm.logprobs == [-0.5, -0.6]

    def test_none_logprobs_defaults_to_zeros(self):
        tm = TokenManager()
        tm.add_response([10, 11], logprobs=None)
        assert tm.logprobs == [0.0, 0.0]

    def test_empty_list_noop(self):
        tm = TokenManager()
        tm.add_response([])
        assert len(tm) == 0

    def test_mismatched_logprobs_raises(self):
        tm = TokenManager()
        with pytest.raises(ValueError, match="logprobs length"):
            tm.add_response([10, 11], [-0.5, -0.6, -0.7])


class TestTokenManagerMultiTurn:
    def test_prompt_response_sequence(self):
        tm = TokenManager()
        tm.add_prompt([1, 2])
        tm.add_response([3, 4])
        assert tm.token_ids == [1, 2, 3, 4]
        assert tm.loss_mask == [0, 0, 1, 1]

    def test_multi_turn_four_segments(self):
        tm = TokenManager()
        tm.add_prompt([1, 2])
        tm.add_response([3, 4])
        tm.add_prompt([5, 6])
        tm.add_response([7, 8])
        assert tm.token_ids == [1, 2, 3, 4, 5, 6, 7, 8]
        assert tm.loss_mask == [0, 0, 1, 1, 0, 0, 1, 1]
        assert tm.segment_info == [(False, 2), (True, 2), (False, 2), (True, 2)]

    def test_logprobs_pattern(self):
        tm = TokenManager()
        tm.add_prompt([1, 2], [0.0, 0.0])
        tm.add_response([3, 4], [-0.1, -0.2])
        tm.add_prompt([5, 6], [0.0, 0.0])
        tm.add_response([7, 8], [-0.3, -0.4])
        assert tm.logprobs == [0.0, 0.0, -0.1, -0.2, 0.0, 0.0, -0.3, -0.4]

    def test_initial_prompt_token_ids_returns_first_segment_only(self):
        tm = TokenManager()
        tm.add_prompt([1, 2, 3])
        tm.add_response([10, 11])
        tm.add_prompt([20, 21])
        assert tm.initial_prompt_token_ids == [1, 2, 3]

    def test_len_counts_all_tokens(self):
        tm = TokenManager()
        tm.add_prompt([1, 2])
        tm.add_response([3])
        tm.add_prompt([4, 5, 6])
        assert len(tm) == 6


class TestTokenManagerReset:
    def test_reset_clears_all(self):
        tm = TokenManager()
        tm.add_prompt([1, 2])
        tm.add_response([3, 4])
        tm.reset()
        assert len(tm) == 0
        assert tm.token_ids == []
        assert tm.loss_mask == []
        assert tm.logprobs == []
        assert tm.segment_info == []
        assert tm.initial_prompt_token_ids == []

    def test_reset_allows_reuse(self):
        tm = TokenManager()
        tm.add_prompt([1, 2])
        tm.reset()
        tm.add_prompt([10, 11])
        tm.add_response([20, 21])
        assert tm.token_ids == [10, 11, 20, 21]
        assert tm.loss_mask == [0, 0, 1, 1]


class TestTokenManagerEdgeCases:
    def test_lists_are_copies(self):
        """Mutating returned lists should not affect internal state."""
        tm = TokenManager()
        tm.add_prompt([1, 2, 3])
        ids = tm.token_ids
        ids.append(999)
        assert tm.token_ids == [1, 2, 3]

        mask = tm.loss_mask
        mask.append(999)
        assert tm.loss_mask == [0, 0, 0]

    def test_single_token_segments(self):
        tm = TokenManager()
        tm.add_prompt([1])
        tm.add_response([2])
        assert tm.token_ids == [1, 2]
        assert tm.loss_mask == [0, 1]
        assert tm.segment_info == [(False, 1), (True, 1)]

    def test_initial_prompt_token_ids_is_copy(self):
        """Mutating the returned list should not affect internal state."""
        tm = TokenManager()
        tm.add_prompt([1, 2, 3])
        ids = tm.initial_prompt_token_ids
        ids.append(999)
        assert tm.initial_prompt_token_ids == [1, 2, 3]
