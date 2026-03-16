from __future__ import annotations

import copy
from unittest.mock import MagicMock, patch

import pytest

from sweagent.agent.agents import RLTokenAgent, TemplateConfig
from sweagent.agent.history_processors import DefaultHistoryProcessor
from sweagent.agent.models import InstanceStats
from sweagent.agent.token_manager import TokenManager
from sweagent.environment.swe_env import SWEEnv
from sweagent.tools.parsing import Identity
from sweagent.tools.tools import ToolConfig, ToolHandler
from sweagent.types import StepOutput


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_model():
    model = MagicMock()
    model.token_manager = TokenManager()
    model.stats = InstanceStats()
    model.instance_cost_limit = 0
    model.reset_rollout_state = MagicMock()
    model.query.return_value = {
        "message": "test output",
        "new_prompt_token_ids": [1, 2, 3],
        "output_tokens": [10, 11],
        "rollout_log_probs": [-0.1, -0.2],
        "rollout_routed_experts": [],
    }
    return model


@pytest.fixture
def tool_handler():
    return ToolHandler(ToolConfig(parse_function=Identity()))


@pytest.fixture
def rl_agent(mock_model, tool_handler):
    return RLTokenAgent(
        templates=TemplateConfig(),
        tools=tool_handler,
        history_processors=[DefaultHistoryProcessor()],
        model=mock_model,
    )


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------
class TestInit:
    def test_token_manager_bound_from_model(self, mock_model, tool_handler):
        agent = RLTokenAgent(
            templates=TemplateConfig(),
            tools=tool_handler,
            history_processors=[DefaultHistoryProcessor()],
            model=mock_model,
        )
        assert agent.token_manager is mock_model.token_manager

    def test_token_manager_fallback_when_model_lacks_it(self, tool_handler):
        model = MagicMock(spec=[])  # model with no attributes
        model.stats = InstanceStats()
        model.instance_cost_limit = 0
        agent = RLTokenAgent(
            templates=TemplateConfig(),
            tools=tool_handler,
            history_processors=[DefaultHistoryProcessor()],
            model=model,
        )
        assert isinstance(agent.token_manager, TokenManager)

    def test_init_input_ids_empty(self, rl_agent):
        assert rl_agent.init_input_ids == []

    def test_rollout_routed_experts_empty(self, rl_agent):
        assert rl_agent.rollout_routed_experts == []

    def test_error_logs_empty(self, rl_agent):
        assert rl_agent._error_logs == []


# ---------------------------------------------------------------------------
# Property delegation
# ---------------------------------------------------------------------------
class TestPropertyDelegation:
    def test_input_ids(self, rl_agent):
        rl_agent.token_manager.add_prompt([1, 2, 3])
        rl_agent.token_manager.add_response([4, 5])
        assert rl_agent.input_ids == [1, 2, 3, 4, 5]

    def test_loss_mask(self, rl_agent):
        rl_agent.token_manager.add_prompt([1, 2])
        rl_agent.token_manager.add_response([3, 4])
        assert rl_agent.loss_mask == [0, 0, 1, 1]

    def test_rollout_log_probs(self, rl_agent):
        rl_agent.token_manager.add_prompt([1], [0.0])
        rl_agent.token_manager.add_response([2], [-0.5])
        assert rl_agent.rollout_log_probs == [0.0, -0.5]


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
class TestSetup:
    def test_calls_reset_rollout_state(self, rl_agent, mock_model, dummy_env):
        from sweagent.agent.problem_statement import EmptyProblemStatement

        rl_agent.setup(dummy_env, EmptyProblemStatement())
        mock_model.reset_rollout_state.assert_called_once()

    def test_clears_state_lists(self, rl_agent, mock_model, dummy_env):
        from sweagent.agent.problem_statement import EmptyProblemStatement

        rl_agent.init_input_ids = [1, 2, 3]
        rl_agent.rollout_routed_experts = [[1], [2]]
        rl_agent._error_logs = [{"error": "test"}]

        rl_agent.setup(dummy_env, EmptyProblemStatement())

        assert rl_agent.init_input_ids == []
        assert rl_agent.rollout_routed_experts == []
        assert rl_agent._error_logs == []

    def test_rebinds_token_manager_from_model(self, mock_model, tool_handler, dummy_env):
        from sweagent.agent.problem_statement import EmptyProblemStatement

        agent = RLTokenAgent(
            templates=TemplateConfig(),
            tools=tool_handler,
            history_processors=[DefaultHistoryProcessor()],
            model=mock_model,
        )
        # Simulate model creating a new token_manager on reset
        new_tm = TokenManager()
        mock_model.token_manager = new_tm

        agent.setup(dummy_env, EmptyProblemStatement())
        assert agent.token_manager is new_tm


# ---------------------------------------------------------------------------
# _require_tokenizer
# ---------------------------------------------------------------------------
class TestRequireTokenizer:
    def test_raises_when_state_is_none(self, rl_agent):
        rl_agent.state = None
        with pytest.raises(ValueError, match="state.tokenizer"):
            rl_agent._require_tokenizer()

    def test_raises_when_state_lacks_tokenizer(self, rl_agent):
        rl_agent.state = MagicMock(spec=[])  # no tokenizer attribute
        with pytest.raises(ValueError, match="state.tokenizer"):
            rl_agent._require_tokenizer()

    def test_returns_tokenizer_when_present(self, rl_agent):
        mock_tok = MagicMock()
        rl_agent.state = MagicMock()
        rl_agent.state.tokenizer = mock_tok
        assert rl_agent._require_tokenizer() is mock_tok


# ---------------------------------------------------------------------------
# get_model_requery_history
# ---------------------------------------------------------------------------
class TestGetModelRequeryHistory:
    def test_returns_deepcopy_of_messages(self, rl_agent):
        from sweagent.agent.problem_statement import EmptyProblemStatement

        rl_agent._problem_statement = EmptyProblemStatement()
        rl_agent._env = MagicMock()
        rl_agent._env.repo = None
        rl_agent.history = [{"role": "user", "content": "hello"}]
        result = rl_agent.get_model_requery_history(
            error_template="Error: {{ output }}",
            output="bad output",
        )
        # Should be a deep copy, not the same object
        assert result is not rl_agent.messages
        assert result == rl_agent.messages


# ---------------------------------------------------------------------------
# add_step_to_history
# ---------------------------------------------------------------------------
class TestAddStepToHistory:
    @pytest.fixture(autouse=True)
    def _setup_agent_env(self, rl_agent):
        from sweagent.agent.problem_statement import EmptyProblemStatement

        rl_agent._problem_statement = EmptyProblemStatement()
        rl_agent._env = MagicMock()
        rl_agent._env.repo = None

    def test_assistant_message_includes_rollout_fields(self, rl_agent):
        step = StepOutput()
        step.output = "I will run ls"
        step.thought = "Let me list files"
        step.action = "ls -la"
        step.observation = "file1.py file2.py"
        step.rollout_log_probs = [-0.1, -0.2]
        step.rollout_routed_experts = [[1, 2]]
        step.output_tokens = [10, 11]
        step.state = {"open_file": "", "working_dir": "/root"}

        rl_agent.add_step_to_history(step)

        # Find the assistant message
        assistant_msgs = [m for m in rl_agent.history if m.get("role") == "assistant"]
        assert len(assistant_msgs) == 1
        msg = assistant_msgs[0]
        assert msg["rollout_log_probs"] == [-0.1, -0.2]
        assert msg["output_tokens"] == [10, 11]

    def test_tool_calls_included(self, rl_agent):
        step = StepOutput()
        step.output = "I need to run bash"
        step.thought = "Run ls"
        step.action = "ls"
        step.observation = "file1.py"
        step.tool_calls = [{"id": "call_001", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}]
        step.state = {"open_file": "", "working_dir": "/root"}

        rl_agent.add_step_to_history(step)

        assistant_msgs = [m for m in rl_agent.history if m.get("role") == "assistant"]
        assert assistant_msgs[0].get("tool_calls") is not None


# ---------------------------------------------------------------------------
# add_step_to_trajectory
# ---------------------------------------------------------------------------
class TestAddStepToTrajectory:
    def test_trajectory_step_includes_extended_fields(self, rl_agent):
        step = StepOutput()
        step.action = "ls"
        step.observation = "file1.py"
        step.output = "running ls"
        step.thought = "check files"
        step.execution_time = 0.5
        step.state = {}
        step.query = [1, 2, 3]
        step.extra_info = {}
        step.reasoning_content = "I should check the files"
        step.output_tokens = [10, 11]
        step.rollout_log_probs = [-0.1, -0.2]
        step.rollout_routed_experts = [[1]]

        rl_agent.add_step_to_trajectory(step)

        assert len(rl_agent.trajectory) == 1
        traj_step = rl_agent.trajectory[0]
        assert traj_step["output_tokens"] == [10, 11]
        assert traj_step["rollout_log_probs"] == [-0.1, -0.2]
        assert traj_step["rollout_routed_experts"] == [[1]]
        assert traj_step["reasoning_content"] == "I should check the files"
