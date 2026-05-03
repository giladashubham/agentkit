from __future__ import annotations

from agentkit.agent import AgentState, AgentTool
from agentkit.llm import Message, Model


def test_tools_setter_copies_list(mock_model: Model, echo_agent_tool: AgentTool) -> None:
    tools = [echo_agent_tool]
    state = AgentState(mock_model)
    state.tools = tools
    tools.clear()

    assert state.tools == [echo_agent_tool]


def test_messages_setter_copies_list(mock_model: Model) -> None:
    message = Message.user("hello")
    messages = [message]
    state = AgentState(mock_model)
    state.messages = messages
    messages.clear()

    assert state.messages == [message]


def test_getters_return_mutable_state_lists(mock_model: Model, echo_agent_tool: AgentTool) -> None:
    state = AgentState(mock_model, _tools=[echo_agent_tool], _messages=[Message.user("hello")])

    state.tools.clear()
    state.messages.clear()

    assert state.tools == []
    assert state.messages == []


def test_tool_lookup_and_llm_tools(mock_model: Model, echo_agent_tool: AgentTool) -> None:
    state = AgentState(mock_model, _tools=[echo_agent_tool])

    assert state.tool_lookup() == {"echo": echo_agent_tool}
    assert state.llm_tools() == [echo_agent_tool.tool]
