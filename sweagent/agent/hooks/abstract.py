from typing import TYPE_CHECKING

from sweagent.types import AgentInfo, StepOutput, Trajectory

if TYPE_CHECKING:
    # avoid circular import
    from sweagent.agent.agents import DefaultAgent


class AbstractAgentHook:
    def on_init(self, *, agent: "DefaultAgent"):
        """Note: Depending on the internals of `Agent` should be done with care,
        it's best to use this as little as possible.
        """

    def on_run_start(
        self,
    ): ...

    def on_step_start(self): ...

    def on_actions_generated(self, *, step: StepOutput): ...

    def on_action_started(self, *, step: StepOutput): ...

    def on_action_executed(self, *, step: StepOutput): ...

    def on_step_done(self, *, step: StepOutput, info: AgentInfo): ...

    def on_run_done(self, *, trajectory: Trajectory, info: AgentInfo): ...

    def on_setup_attempt(self): ...

    def on_model_query(self, *, messages: list[dict[str, str]], agent: str):
        """Actually query the model with the complete history."""

    def on_query_message_added(
        self,
        *,
        agent: str,
        role: str,
        content: str,
        message_type: str,
        is_demo: bool = False,
        thought: str = "",
        action: str = "",
        tool_calls: list[dict[str, str]] | None = None,
        tool_call_ids: list[str] | None = None,
        name: str | None = None,
        tool_call_id: str | None = None,
        thinking_blocks: list[dict[str, str]] | None = None,
        reasoning_content: str = "",
        output_tokens: list[int] | None = None,
        rollout_log_probs: list[float] | None = None,
        rollout_routed_experts: list[list[int]] | None = None,
    ): ...

    def on_setup_done(self): ...

    def on_tools_installation_started(self): ...


class CombinedAgentHook(AbstractAgentHook):
    def __init__(self, hooks: list[AbstractAgentHook] | None = None):
        self._hooks = hooks or []

    def add_hook(self, hook: AbstractAgentHook):
        self._hooks.append(hook)

    @property
    def hooks(self) -> list[AbstractAgentHook]:
        return self._hooks

    def on_init(self, *, agent: "DefaultAgent"):
        for hook in self.hooks:
            hook.on_init(agent=agent)

    def on_run_start(self):
        for hook in self.hooks:
            hook.on_run_start()

    def on_step_start(self):
        for hook in self.hooks:
            hook.on_step_start()

    def on_actions_generated(self, *, step: StepOutput):
        for hook in self.hooks:
            hook.on_actions_generated(step=step)

    def on_action_started(self, *, step: StepOutput):
        for hook in self.hooks:
            hook.on_action_started(step=step)

    def on_action_executed(self, *, step: StepOutput):
        for hook in self.hooks:
            hook.on_action_executed(step=step)

    def on_step_done(self, *, step: StepOutput, info: AgentInfo):
        for hook in self.hooks:
            hook.on_step_done(step=step, info=info)

    def on_run_done(self, *, trajectory: Trajectory, info: AgentInfo):
        for hook in self.hooks:
            hook.on_run_done(trajectory=trajectory, info=info)

    def on_setup_attempt(self):
        for hook in self.hooks:
            hook.on_setup_attempt()

    def on_model_query(self, *, messages: list[dict[str, str]], agent: str):
        for hook in self.hooks:
            hook.on_model_query(messages=messages, agent=agent)

    def on_query_message_added(
        self,
        *,
        agent: str,
        role: str,
        content: str,
        message_type: str,
        is_demo: bool = False,
        thought: str = "",
        action: str = "",
        tool_calls: list[dict[str, str]] | None = None,
        tool_call_ids: list[str] | None = None,
        name: str | None = None,
        tool_call_id: str | None = None,
        thinking_blocks: list[dict[str, str]] | None = None,
        reasoning_content: str = "",
        output_tokens: list[int] | None = None,
        rollout_log_probs: list[float] | None = None,
        rollout_routed_experts: list[list[int]] | None = None,
    ):
        for hook in self.hooks:
            kwargs = dict(
                agent=agent,
                role=role,
                content=content,
                message_type=message_type,
                is_demo=is_demo,
                thought=thought,
                action=action,
                tool_calls=tool_calls,
                tool_call_ids=tool_call_ids,
                thinking_blocks=thinking_blocks,
                name=name,
                tool_call_id=tool_call_id,
                reasoning_content=reasoning_content,
                output_tokens=output_tokens,
                rollout_log_probs=rollout_log_probs,
                rollout_routed_experts=rollout_routed_experts,
            )
            fallback_keys = [
                'rollout_routed_experts',
                'rollout_log_probs',
                'output_tokens',
                'reasoning_content',
                'tool_call_id',
                'name',
                'thinking_blocks',
            ]
            while True:
                try:
                    hook.on_query_message_added(**kwargs)
                    break
                except TypeError as e:
                    if 'unexpected keyword argument' not in str(e):
                        raise
                    removed = False
                    for key in fallback_keys:
                        if f"'{key}'" in str(e) and key in kwargs:
                            kwargs.pop(key, None)
                            removed = True
                            break
                    if not removed:
                        raise

    def on_setup_done(self):
        return super().on_setup_done()

    def on_tools_installation_started(self):
        for hook in self.hooks:
            hook.on_tools_installation_started()
