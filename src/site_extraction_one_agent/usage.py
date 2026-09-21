"""Token-usage tracking for the agent's chat calls, the raw web_search Responses-API calls and
the ScrapingBee fetches fetch_page falls back to, for cost analysis. Persisted as append-only
JSONL so cost can be tracked across runs/companies over time.

Per-agent attribution uses a contextvar (`_current_agent`) that UsageAttributionMiddleware
(search_tool.py) sets immediately before synchronously calling into the model/tool handler
(via `wrap_model_call`/`wrap_tool_call`) -- i.e. set-then-immediately-use within one call
stack, never across a thread/node boundary, which is what makes this reliable regardless of
how LangGraph schedules nodes internally. There's only one agent in this project, so the
per-agent breakdown will show a single bucket -- this module is kept generic rather than
hard-coding that assumption, in case more agents are added later.
"""

from __future__ import annotations

import contextvars
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from .config import (
    EST_INPUT_COST_PER_1M_TOKENS,
    EST_OUTPUT_COST_PER_1M_TOKENS,
    EST_SCRAPINGBEE_COST_PER_CREDIT,
    EST_WEB_SEARCH_COST_PER_CALL,
    QUERIES_LOG_PATH,
    SCRAPINGBEE_LOG_PATH,
    USAGE_LOG_PATH,
)

_current_agent: contextvars.ContextVar[str] = contextvars.ContextVar("current_agent", default="site_extractor")


def get_current_agent() -> str:
    return _current_agent.get()


def set_current_agent(name: str) -> contextvars.Token:
    """Set the current-agent contextvar; returns a token to pass to reset_current_agent().

    Must be paired with reset_current_agent() in a try/finally around a SYNCHRONOUS,
    same-call-stack region (e.g. immediately before calling a handler()) -- see
    SearchBudgetMiddleware in search_tool.py for the pattern this is designed for.
    """
    return _current_agent.set(name)


def reset_current_agent(token: contextvars.Token) -> None:
    _current_agent.reset(token)


def _empty_agent_bucket() -> dict[str, int]:
    return {
        "chat_calls": 0,
        "chat_input_tokens": 0,
        "chat_output_tokens": 0,
        "chat_total_tokens": 0,
        "web_search_calls": 0,
        "web_search_input_tokens": 0,
        "web_search_output_tokens": 0,
        "scrapingbee_calls": 0,
        "scrapingbee_credits": 0,
        "tool_calls_attempted": 0,
        "tool_calls_blocked_by_budget": 0,
    }


@dataclass
class UsageTracker:
    """Thread-safe accumulator for one CLI run's total token/tool usage, broken down by
    agent and by model."""

    chat_calls: int = 0
    chat_input_tokens: int = 0
    chat_output_tokens: int = 0
    chat_total_tokens: int = 0
    web_search_calls: int = 0
    web_search_input_tokens: int = 0
    web_search_output_tokens: int = 0
    scrapingbee_calls: int = 0
    scrapingbee_credits: int = 0
    scrapingbee_failures: int = 0
    scrapingbee_skipped_no_key: int = 0
    tool_calls_attempted: int = 0
    tool_calls_blocked_by_budget: int = 0
    web_searches_redirected_to_fetch: int = 0
    per_model: dict[str, dict[str, int]] = field(default_factory=dict)
    per_agent: dict[str, dict[str, int]] = field(default_factory=dict)
    queries: list[dict[str, str]] = field(default_factory=list)
    tool_call_counts: dict[str, int] = field(default_factory=dict)
    scrapingbee_details: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_query(self, agent: str, tool: str, query: str) -> None:
        """Records every query/fetch actually issued (only the ones the budget check in
        search_tool.py let through -- a call refused for being over budget, or bounced back
        because it should have been a fetch_page, never reaches the network and so is not a
        real query). `tool_call_counts` is what the budget check itself reads."""
        with self._lock:
            self.queries.append({"agent": agent, "tool": tool, "query": query})
            self.tool_call_counts[tool] = self.tool_call_counts.get(tool, 0) + 1

    def has_issued(self, tool: str, query: str) -> bool:
        """Whether this exact (tool, query) pair already went out this run. Used by fetch_page
        to short-circuit a repeat fetch: now that fetching is uncapped, re-reading a page is the
        one fetch pattern that genuinely costs real money, since a second copy of the text is
        re-billed on every remaining turn."""
        with self._lock:
            return any(q["tool"] == tool and q["query"] == query for q in self.queries)

    def calls_used(self, tool: str) -> int:
        """How many calls of `tool` have actually been issued this run -- the input to
        search_tool.py's budget check."""
        with self._lock:
            return self.tool_call_counts.get(tool, 0)

    def record_budget_block(self, agent: str) -> None:
        """A tool call refused because its per-tool budget was already spent."""
        with self._lock:
            self.tool_calls_blocked_by_budget += 1
            agent_bucket = self.per_agent.setdefault(agent, _empty_agent_bucket())
            agent_bucket["tool_calls_blocked_by_budget"] += 1

    def record_search_redirected_to_fetch(self) -> None:
        """A `site:<specific page>` web_search bounced back with an instruction to call
        fetch_page on that URL instead -- costs nothing, so it's tracked separately from a
        budget block to show how much paid search this guard is displacing."""
        with self._lock:
            self.web_searches_redirected_to_fetch += 1

    def record_chat_usage(self, agent: str, model: str, input_tokens: int, output_tokens: int, total_tokens: int) -> None:
        with self._lock:
            self.chat_calls += 1
            self.chat_input_tokens += input_tokens
            self.chat_output_tokens += output_tokens
            self.chat_total_tokens += total_tokens

            model_bucket = self.per_model.setdefault(
                model, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            )
            model_bucket["calls"] += 1
            model_bucket["input_tokens"] += input_tokens
            model_bucket["output_tokens"] += output_tokens
            model_bucket["total_tokens"] += total_tokens

            agent_bucket = self.per_agent.setdefault(agent, _empty_agent_bucket())
            agent_bucket["chat_calls"] += 1
            agent_bucket["chat_input_tokens"] += input_tokens
            agent_bucket["chat_output_tokens"] += output_tokens
            agent_bucket["chat_total_tokens"] += total_tokens

    def record_web_search_usage(self, agent: str, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self.web_search_calls += 1
            self.web_search_input_tokens += input_tokens
            self.web_search_output_tokens += output_tokens

            agent_bucket = self.per_agent.setdefault(agent, _empty_agent_bucket())
            agent_bucket["web_search_calls"] += 1
            agent_bucket["web_search_input_tokens"] += input_tokens
            agent_bucket["web_search_output_tokens"] += output_tokens

    def record_scrapingbee_usage(
        self,
        agent: str,
        credits: int,
        succeeded: bool,
        *,
        url: str | None = None,
        status_code: int | None = None,
        initial_status_code: int | None = None,
        reason: str | None = None,
    ) -> None:
        """One ScrapingBee API call made by fetch_page's middle tier. `credits` is what
        ScrapingBee itself reported charging (0 for a failure -- they only bill 200/404/410),
        so the cost estimate tracks real spend rather than an assumed per-call price, and stays
        correct however `mode=auto` escalates between their 1/5/10/25/75-credit tiers.

        `status_code` is ScrapingBee's OWN gateway response code (what `succeeded` is based on);
        `initial_status_code` is the TARGET site's status as ScrapingBee saw it (their
        `Spb-initial-status-code` header) -- these can differ, e.g. a 200 gateway response
        wrapping a target that itself 403'd before ScrapingBee's retry ladder kicked in.
        `reason` explains why this call didn't yield usable page content: on a non-200 gateway
        response it's ScrapingBee's own reported error (parsed from their JSON error body, e.g.
        "mode=auto tried tier(s) ... without success (Server responded with 613)"); on a 200
        that still didn't produce usable text it's the caller's own explanation (e.g.
        "non_text_content", "still_blocked_after_render"); `None` when the call worked.
        `url`/`status_code`/`initial_status_code`/`reason` are recorded per-call in
        `scrapingbee_details` (and from there into scrapingbee_log.jsonl) specifically so a
        batch's ScrapingBee failures can be diagnosed afterwards instead of re-guessed from a
        manual sample re-test."""
        with self._lock:
            self.scrapingbee_calls += 1
            self.scrapingbee_credits += credits
            if not succeeded:
                self.scrapingbee_failures += 1
            self.scrapingbee_details.append({
                "agent": agent,
                "url": url,
                "succeeded": succeeded,
                "credits": credits,
                "status_code": status_code,
                "initial_status_code": initial_status_code,
                "reason": reason,
            })

            agent_bucket = self.per_agent.setdefault(agent, _empty_agent_bucket())
            agent_bucket["scrapingbee_calls"] += 1
            agent_bucket["scrapingbee_credits"] += credits

    def record_scrapingbee_unavailable(self) -> None:
        """A fetch that WOULD have used the ScrapingBee tier but went straight to the paid
        search fallback because SCRAPINGBEE_API_KEY isn't set. Run-global (like
        web_searches_redirected_to_fetch) and the first thing to check when a run costs more
        than a comparable one in usage_log.jsonl."""
        with self._lock:
            self.scrapingbee_skipped_no_key += 1

    def record_tool_attempt(self, agent: str, allowed: bool) -> None:
        """Records every attempted tool call from the middleware, which wraps the tool
        regardless of whether the tool then refuses the call -- so `tool_calls_attempted`
        reflects what the model TRIED to do. The refusals are counted separately by
        record_budget_block()/record_search_redirected_to_fetch(), which is where
        `tool_calls_blocked_by_budget` comes from; `allowed` stays here for callers that
        gate before the tool body runs."""
        with self._lock:
            self.tool_calls_attempted += 1
            agent_bucket = self.per_agent.setdefault(agent, _empty_agent_bucket())
            agent_bucket["tool_calls_attempted"] += 1
            if not allowed:
                self.tool_calls_blocked_by_budget += 1
                agent_bucket["tool_calls_blocked_by_budget"] += 1

    def estimated_cost_usd(self) -> float:
        chat_in_cost = (self.chat_input_tokens / 1_000_000) * EST_INPUT_COST_PER_1M_TOKENS
        chat_out_cost = (self.chat_output_tokens / 1_000_000) * EST_OUTPUT_COST_PER_1M_TOKENS
        search_token_in_cost = (self.web_search_input_tokens / 1_000_000) * EST_INPUT_COST_PER_1M_TOKENS
        search_token_out_cost = (self.web_search_output_tokens / 1_000_000) * EST_OUTPUT_COST_PER_1M_TOKENS
        search_call_cost = self.web_search_calls * EST_WEB_SEARCH_COST_PER_CALL
        # Credits x price, not calls x price: ScrapingBee's mode=auto charges anywhere from 1 to
        # 75 credits depending on which tier got the page, so only the reported credits are real.
        scrapingbee_cost = self.scrapingbee_credits * EST_SCRAPINGBEE_COST_PER_CREDIT
        return (
            chat_in_cost
            + chat_out_cost
            + search_token_in_cost
            + search_token_out_cost
            + search_call_cost
            + scrapingbee_cost
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "chat_calls": self.chat_calls,
            "chat_input_tokens": self.chat_input_tokens,
            "chat_output_tokens": self.chat_output_tokens,
            "chat_total_tokens": self.chat_total_tokens,
            "web_search_calls": self.web_search_calls,
            "web_search_input_tokens": self.web_search_input_tokens,
            "web_search_output_tokens": self.web_search_output_tokens,
            "scrapingbee_calls": self.scrapingbee_calls,
            "scrapingbee_credits": self.scrapingbee_credits,
            "scrapingbee_failures": self.scrapingbee_failures,
            "scrapingbee_skipped_no_key": self.scrapingbee_skipped_no_key,
            "tool_calls_attempted": self.tool_calls_attempted,
            "tool_calls_blocked_by_budget": self.tool_calls_blocked_by_budget,
            "web_searches_redirected_to_fetch": self.web_searches_redirected_to_fetch,
            "tool_call_counts": self.tool_call_counts,
            "per_model": self.per_model,
            "per_agent": self.per_agent,
            "queries": self.queries,
            "scrapingbee_details": self.scrapingbee_details,
            "estimated_cost_usd": round(self.estimated_cost_usd(), 4),
        }


class TokenUsageCallbackHandler(BaseCallbackHandler):
    """LangChain callback that records chat-model token usage into a UsageTracker.

    Passed via `config={"callbacks": [...]}"` on the agent's `.invoke()` call. Per-agent
    attribution comes from `get_current_agent()`, which UsageAttributionMiddleware sets
    immediately before invoking the model (see search_tool.py).
    """

    def __init__(self, tracker: UsageTracker) -> None:
        super().__init__()
        self.tracker = tracker

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        llm_output_model = (response.llm_output or {}).get("model_name")
        agent = get_current_agent()
        for generation_list in response.generations:
            for generation in generation_list:
                message = getattr(generation, "message", None)
                usage = getattr(message, "usage_metadata", None) if message else None
                if not usage:
                    continue
                response_metadata = getattr(message, "response_metadata", None) or {}
                model = (
                    llm_output_model
                    or response_metadata.get("model_name")
                    or response_metadata.get("model")
                    or "unknown"
                )
                self.tracker.record_chat_usage(
                    agent=agent,
                    model=model,
                    input_tokens=usage.get("input_tokens", 0) or 0,
                    output_tokens=usage.get("output_tokens", 0) or 0,
                    total_tokens=usage.get("total_tokens", 0) or 0,
                )


_active_tracker: UsageTracker | None = None
_active_tracker_lock = threading.Lock()


def start_tracking() -> UsageTracker:
    """Reset and return the module-level tracker for a new CLI run."""
    global _active_tracker
    with _active_tracker_lock:
        _active_tracker = UsageTracker()
        return _active_tracker


def get_tracker() -> UsageTracker | None:
    return _active_tracker


def append_usage_log(
    company: str, tracker: UsageTracker, repo_root: Path, run_config: dict[str, Any] | None = None
) -> Path:
    """Append one JSON line with this run's usage/cost to USAGE_LOG_PATH for cost analysis.

    `run_config` records the knobs that don't otherwise show up in token counts (e.g.
    reasoning_effort, which tools were available) so later runs can be told apart in the log.
    """
    log_path = repo_root / USAGE_LOG_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": time.time(), "company": company, **(run_config or {}), **tracker.to_dict()}
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return log_path


def append_queries_log(company: str, tracker: UsageTracker, repo_root: Path) -> Path:
    """Append one JSON line PER query/fetch actually issued this run (agent, tool, query
    text, company, timestamp) to QUERIES_LOG_PATH, so search behavior can be reviewed or
    analyzed across runs."""
    log_path = repo_root / QUERIES_LOG_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.time()
    with log_path.open("a", encoding="utf-8") as f:
        for entry in tracker.queries:
            record = {"timestamp": timestamp, "company": company, **entry}
            f.write(json.dumps(record) + "\n")
    return log_path


def append_scrapingbee_log(company: str, tracker: UsageTracker, repo_root: Path) -> Path:
    """Append one JSON line PER ScrapingBee call actually made this run (url, status codes,
    credits, succeeded, reason) to SCRAPINGBEE_LOG_PATH -- see record_scrapingbee_usage's
    docstring for what each field means. Mirrors append_queries_log's shape/mechanics."""
    log_path = repo_root / SCRAPINGBEE_LOG_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.time()
    with log_path.open("a", encoding="utf-8") as f:
        for entry in tracker.scrapingbee_details:
            record = {"timestamp": timestamp, "company": company, **entry}
            f.write(json.dumps(record) + "\n")
    return log_path
