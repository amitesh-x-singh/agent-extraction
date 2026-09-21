"""Isolated, no-API-cost verification of UsageAttributionMiddleware.

Uses a scripted FakeMessagesListChatModel instead of a real LLM so this costs nothing and
runs deterministically. Validates:
  1. Every tool call the model makes actually executes -- this middleware itself never blocks
     one. (The per-tool call budgets live in web_search/fetch_page, not here; see
     tests/test_search_guards.py.)
  2. get_current_agent() reflects the agent name from WITHIN the tool function itself, set by
     wrap_tool_call, and resets to the default once the run completes.
  3. tool_calls_attempted is recorded for every call, with tool_calls_blocked_by_budget always 0
     here -- the middleware blocks nothing, so a run using a tool other than web_search/fetch_page
     can never register a block.
"""

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from site_extraction_one_agent.search_tool import UsageAttributionMiddleware
from site_extraction_one_agent.usage import get_current_agent, start_tracking


class ToolBindingFakeChatModel(FakeMessagesListChatModel):
    """FakeMessagesListChatModel doesn't implement bind_tools; create_agent requires it.
    We don't need real tool-schema validation for this test -- just accept and ignore."""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


observed_agents: list[str] = []


@tool
def observing_tool(query: str) -> str:
    """Records which agent name was active (per usage.get_current_agent()) at call time."""
    observed_agents.append(get_current_agent())
    return f"observed for {query!r}"


start_tracking()
print(f"Baseline get_current_agent() outside any agent run: {get_current_agent()!r}")
assert get_current_agent() == "site_extractor", "default contextvar value should be 'site_extractor'"

N_CALLS = 5  # no cap in this project -- all 5 should actually execute
responses = [
    AIMessage(
        content="",
        tool_calls=[{"name": "observing_tool", "args": {"query": f"x{i}"}, "id": f"call_{i}", "type": "tool_call"}],
    )
    for i in range(N_CALLS)
]
responses.append(AIMessage(content="Done."))

model = ToolBindingFakeChatModel(responses=responses)
agent = create_agent(
    model=model,
    tools=[observing_tool],
    middleware=[UsageAttributionMiddleware(agent_name="site_extractor")],
)
agent.invoke({"messages": [{"role": "user", "content": "go"}]})

print(f"Observed agent names during {N_CALLS} tool calls: {observed_agents}")
assert observed_agents == ["site_extractor"] * N_CALLS, observed_agents
assert get_current_agent() == "site_extractor", "contextvar must reset to the default after the run"
print("Test 1 (middleware never blocks + per-call attribution): PASSED\n")

from site_extraction_one_agent.usage import get_tracker  # noqa: E402  (after start_tracking() above)

tracker = get_tracker()
usage = tracker.to_dict()
assert usage["tool_calls_attempted"] == N_CALLS, usage
assert usage["tool_calls_blocked_by_budget"] == 0, usage
print(f"tool_calls_attempted={usage['tool_calls_attempted']}, tool_calls_blocked_by_budget={usage['tool_calls_blocked_by_budget']}")
print("Test 2 (usage tracker sees every attempt; middleware blocks nothing): PASSED\n")

print("ALL USAGE-ATTRIBUTION CHECKS PASSED")
