"""Two research tools -- web_search and fetch_page -- plus a shared usage-attribution middleware.

web_search wraps OpenAI's hosted Responses-API web_search tool ourselves (rather than handing
the agent the bare `{"type": "web_search"}` dict tool) so that results come back as a clean
structured list of {title, url, snippet} hits the LLM can read and the code-based tier-scoring
rubric can classify by domain.

web_search alone is a poor way to enumerate a company's many sites: OpenAI's hosted tool answers
a narrow question in a short synthesized response and cites a couple of sources for it -- we only
ever get the small text window around each citation, never the actual page content, so one query
rarely surfaces more than a couple of addresses even if the underlying page it drew from lists
many more.

fetch_page(url) closes that gap: given a specific URL (e.g. one found via web_search), it pulls
the page's FULL text, so a single call can surface every address listed on that page instead of
burning many paid web_search rounds to slowly dribble them out one at a time. It walks three
tiers, stopping at the first that returns real content:

  1. A free plain HTTP GET. Handles the large majority of pages.

  2. A ScrapingBee fetch (headless browser + rotating proxy), reached when the plain GET is
     blocked (403, bot-detection interstitial, etc.) or the page's real content is only rendered
     client-side by JS -- common for interactive "office locations" maps, where the static HTML
     has nothing in it. This returns the page's REAL rendered text, which the condenser can mine
     for every address on it, and costs ~$0.001 -- roughly a tenth of the tier below it. Skipped
     for PDFs/binaries, which no amount of browser rendering turns into text.

  3. Asking the hosted web_search tool to retrieve the page, explicitly instructed to list
     everything it contains rather than answer a narrow question. Costs a full web_search call
     and returns an LLM's summary of the page rather than the page, so it is last: only reached
     when both cheaper tiers failed, or the URL is a PDF.

Both tools were briefly reduced to web_search-only, with an added adversarial verification pass,
in an earlier version of this project. Measured cost/quality data showed that combination was
worse on both axes -- without fetch_page, the agent burned 3x more web_search calls (and chat
turns) trying to dribble out addresses one at a time instead of pulling a whole locations page in
one shot, and verification's catch rate (~5% of candidates refuted) didn't justify its share of
that cost. fetch_page is back; verification stays removed.

Two guards keep the agent from re-inventing the expensive pattern the query logs showed it
falling into (measured on ELLSWORTH: 349 web_search calls and 2.9M chat input tokens in one run,
for the same ~33 sites a 67-call run found):

  * `site:`-with-a-path queries are refused. 43% of that run's searches were
    `site:<one specific page> "<one term>"` -- 354 paid searches aimed at just 65 distinct
    pages (48 of them at a single Fisnar contact page), i.e. the search engine used as a page
    reader. Those queries already contain the URL, so web_search bounces them back with an
    instruction to call fetch_page on it instead. The refusal costs nothing.

  * Per-tool call budgets (config.WEB_SEARCH_CALL_BUDGET / FETCH_PAGE_CALL_BUDGET). Prose
    budgets in the system prompt were ignored by roughly 3x, so the count is now reported back
    inside every tool result and the tool refuses calls past the cap. UsageAttributionMiddleware
    still never blocks anything -- it only tags calls for usage.py's accounting; the budget is
    enforced in the tool bodies, where the tracker's per-tool counts live.

Uncapping fetch_page was tried and reverted -- see the comment in config.py. Two things from
that experiment were kept because they are cheap and correct either way: a 404/410 short-circuit
(_FETCH_NOT_FOUND) so a guessed URL that does not exist never escalates to a render or a paid
retrieval, and a separate `fetch_page_search_fallback` row in queries_log.jsonl for every tier-3
escalation, which is what makes it possible to see which URLs actually cost money.
"""

from html.parser import HTMLParser
from typing import Any, NamedTuple

import os
import re
import requests
import threading
import time
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.tools import tool
from openai import APIConnectionError, APIStatusError, OpenAI

from .config import (
    BUDGET_WARN_FRACTION,
    FETCH_PAGE_CALL_BUDGET,
    SEARCH_MODEL,
    WEB_SEARCH_CALL_BUDGET,
    search_user_agent,
)
from .usage import get_current_agent, get_tracker, reset_current_agent, set_current_agent

_client: OpenAI | None = None

_MAX_HITS_PER_SEARCH = 15
# Cap on plain-HTTP page text. This used to be 8000, which was itself a major driver of wasted
# searches: a global "our locations" page or a 20-country contact page runs well past 8k chars,
# so the agent would fetch one, see it cut off before the countries it wanted, and then go back
# to per-country `site:<that same URL>` searches to read the rest -- 38 follow-up searches
# against one already-fetched Ellsworth locator page. These are input tokens on a free GET, and
# _condense_page_text() below strips the page down to its address-bearing lines before the cap
# ever bites, so the cap now matches the (paid) fallback's.
_MAX_PAGE_CHARS = 30000
_MAX_FALLBACK_CHARS = 30000  # cap the web_search fallback's text (dense, already-paid-for signal)
_CONDENSE_CONTEXT_LINES = 2  # lines kept either side of an address-looking line
_FETCH_TIMEOUT_SECONDS = 10
_MAX_API_RETRIES = 3
_API_RETRY_BACKOFF_SECONDS = 2.0

_SCRAPINGBEE_ENDPOINT = "https://app.scrapingbee.com/api/v1/"
# mode=auto walks ScrapingBee's own tiers in sequence (plain proxy -> JS render -> premium ->
# stealth), so one call legitimately takes far longer than the 10s a plain GET gets. The
# client-side timeout MUST stay above the server-side one: if we abort first we throw away --
# and are still billed for -- a render ScrapingBee has already completed.
_SCRAPINGBEE_SERVER_TIMEOUT_MS = 90000
_SCRAPINGBEE_TIMEOUT_SECONDS = 100
_SCRAPINGBEE_ASSUMED_CREDITS = 5  # only if a billed response arrives with no cost header

# The agent is prompted to batch several tool calls into one turn, which LangGraph's ToolNode
# runs concurrently in separate threads -- so a single turn's fetch_page calls can otherwise hit
# ScrapingBee with unbounded concurrency. Capped at 2 simultaneous requests process-wide.
_SCRAPINGBEE_CONCURRENCY_LIMIT = 2
_scrapingbee_semaphore = threading.Semaphore(_SCRAPINGBEE_CONCURRENCY_LIMIT)

# Genuine blocks: the server refused us, and no amount of rendering changes that.
_BLOCKED_MARKERS = (
    "access denied",
    "are you a robot",
    "captcha",
    "verify you are a human",
    "just a moment",  # Cloudflare interstitial
    "403 forbidden",
)
# Not blocks: the real content (e.g. an interactive office-locations map) is rendered
# client-side by JS and never appears in the static HTML a plain GET receives -- functionally
# the same failure for tier 1, so it escalates the same way. But these must NOT count against a
# BROWSER-rendered fetch, which has already run that JS: a single leftover "Loading..." node in
# an otherwise complete page would push a good result on to the paid tier for nothing.
_JS_PENDING_MARKERS = (
    "enable javascript",
    "loading map",
    "loading...",
    "please wait",
)


# A `site:` operator whose target has a non-empty path is aimed at ONE specific page, not at a
# domain -- i.e. the model already knows the URL and is paying for a search to read part of a
# page fetch_page could return whole. `site:example.com locations` (no path) is a real search and
# is left alone; `site:example.com/contact-us/ "Germany"` is not.
_DEEP_SITE_QUERY_RE = re.compile(r"site:(?P<host>[^\s/\"']+\.[^\s/\"']+)/(?P<path>[^\s\"']+)", re.IGNORECASE)

# Lines worth keeping when a fetched page is too long to pass through whole: street/building
# numbers and postal codes, thoroughfare words in the languages this pipeline actually meets in
# supplier addresses, and the phone/fax labels that almost always sit next to an address block.
# The supplier list spans Europe, Asia and the Americas, so this deliberately covers more
# languages than any single company needs -- a missing thoroughfare word costs a dropped address
# on a long page, while a spurious one costs a few kept lines, so it errs toward keeping.
_ADDRESS_HINT_RE = re.compile(
    r"\d{3,}"
    r"|\b\d+\s*[-/]\s*\d+\b"
    # English
    r"|\b(?:street|st|road|rd|avenue|ave|drive|dr|lane|ln|boulevard|blvd|highway|hwy|way|"
    r"suite|ste|unit|floor|fl|building|bldg|block|plaza|tower|centre|center|park|estate|"
    r"industrial|business|zone|postal|zip)\b"
    r"|\bp\.?o\.? box\b"
    # German / Dutch / Nordic
    r"|\b(?:strasse|straße|str|weg|allee|platz|gasse|straat|laan|plein|gatan|"
    r"vägen|vagen|vei|katu|gade|vej)\b"
    # French / Italian / Spanish / Portuguese
    r"|\b(?:rue|chauss[eé]e|impasse|chemin|via|viale|corso|piazza|strada|calle|"
    r"carretera|ctra|avenida|paseo|pol[ií]gono|nave|rua|estrada|rodovia)\b"
    # Central / Eastern Europe / Turkey
    r"|\b(?:ulica|aleja|náměstí|namesti|třída|utca|sokak|cadde|"
    r"mahalle|bulvar)\b"
    # South / South-East / East Asia
    r"|\b(?:jalan|jln|lorong|taman|persiaran|nagar|marg|sector|phase|chowk|soi|thanon|"
    r"chome|banchi|dori|dong|gu|ku)\b"
    r"|\b(?:district|province|prefecture|industrial park|business park|free zone)\b"
    r"|\b(?:tel|telephone|phone|fax|t:|p:|f:)\b",
    re.IGNORECASE,
)


def _redirect_to_fetch_message(query: str, url: str) -> str:
    return (
        f"REFUSED (this cost nothing): {query!r} uses `site:` against one specific page, so you "
        f"already know the URL -- searching it back out is the single most wasteful pattern in "
        f"this pipeline. Call fetch_page(\"{url}\") instead: it returns that page's ENTIRE text "
        "in one call, including whatever term you were filtering for and every other address on "
        "it. If the page turns out not to contain what you wanted, search for it WITHOUT a "
        "`site:` path (a bare `site:<domain>` or a plain query is fine)."
    )


def _calls_used(tool: str) -> int:
    tracker = get_tracker()
    return tracker.calls_used(tool) if tracker is not None else 0


def _budget_note(tool: str, used: int, budget: int, guidance: str) -> str:
    """Status line appended to every successful tool result, so the model sees its own spend
    instead of having to trust a number in the system prompt."""
    if used >= budget * BUDGET_WARN_FRACTION:
        return f"\n\n[{tool} budget: {used}/{budget} used -- nearly spent. {guidance}]"
    return f"\n\n[{tool} budget: {used}/{budget} used]"


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()  # reads OPENAI_API_KEY from the environment
    return _client


def _create_response_with_retries(**kwargs: Any) -> Any:
    """Wraps client.responses.create() with a few retries on transient network/server
    errors (DNS blips, connection resets, 5xx) -- a single flaky call shouldn't crash the
    whole run. Client-side errors (bad request, auth, etc.) are NOT retried and propagate
    immediately."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_API_RETRIES):
        try:
            return _get_client().responses.create(**kwargs)
        except APIConnectionError as exc:
            last_exc = exc
        except APIStatusError as exc:
            if exc.status_code < 500:  # client error (4xx) -- retrying won't help
                raise
            last_exc = exc
        if attempt < _MAX_API_RETRIES - 1:
            time.sleep(_API_RETRY_BACKOFF_SECONDS * (attempt + 1))
    assert last_exc is not None
    raise last_exc


def _extract_hits(resp: Any, max_hits: int = _MAX_HITS_PER_SEARCH) -> list[dict[str, str]]:
    """Pull {title, url, snippet} hits out of a Responses API result's url_citation annotations."""
    hits: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for item in getattr(resp, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", None) != "output_text":
                continue
            text = getattr(content, "text", "") or ""
            for annotation in getattr(content, "annotations", []) or []:
                if getattr(annotation, "type", None) != "url_citation":
                    continue
                url = getattr(annotation, "url", "") or ""
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                start = max(0, getattr(annotation, "start_index", 0) - 120)
                end = min(len(text), getattr(annotation, "end_index", 0) + 120)
                hits.append(
                    {
                        "title": getattr(annotation, "title", "") or url,
                        "url": url,
                        "snippet": text[start:end].strip(),
                    }
                )
                if len(hits) >= max_hits:
                    return hits
    return hits


def _format_hits(query: str, hits: list[dict[str, str]]) -> str:
    if not hits:
        return f"No web results found for: {query!r}"
    lines = [f"Web search results for: {query!r}"]
    for i, hit in enumerate(hits, start=1):
        lines.append(f"{i}. {hit['title']} -- {hit['url']}\n   {hit['snippet']}")
    return "\n".join(lines)


def _record_search_usage(resp: Any) -> None:
    # This is a raw openai SDK call, outside LangChain's callback tracing, so its usage has
    # to be recorded separately from the LangChain chat-model usage callback.
    tracker = get_tracker()
    if tracker is not None and getattr(resp, "usage", None) is not None:
        tracker.record_web_search_usage(
            agent=get_current_agent(),
            input_tokens=resp.usage.input_tokens or 0,
            output_tokens=resp.usage.output_tokens or 0,
        )


@tool
def web_search(query: str) -> str:
    """Search the web for a query and return titled results with source URLs and snippets.
    PAID per call and budgeted -- see the budget line at the end of every result.

    Use specific, targeted queries (e.g. company name + "headquarters address", or company
    name + "SEC 10-K filing address"). Vary queries across site-type keywords (plant, factory,
    warehouse, distribution center, office, R&D, facility) and source categories rather than
    repeating near-identical queries -- a company usually has more than one kind of site.

    Two things this tool will REFUSE, so don't spend turns on them:
      * A `site:` query pointing at a specific page (`site:acme.com/contact-us "Germany"`).
        You already know that URL -- call fetch_page on it and read the whole page at once.
      * Anything past the call budget.
    Use it to DISCOVER pages and facts, not to read a page you can already name, and not to
    re-confirm an address you have already seen.
    """
    tracker = get_tracker()
    agent = get_current_agent()

    deep_site = _DEEP_SITE_QUERY_RE.search(query)
    if deep_site:
        host, path = deep_site.group("host"), deep_site.group("path").rstrip('"\'')
        url = f"https://{host}/{path}"
        if tracker is not None:
            tracker.record_search_redirected_to_fetch()
        return _redirect_to_fetch_message(query, url)

    used = _calls_used("web_search")
    if used >= WEB_SEARCH_CALL_BUDGET:
        if tracker is not None:
            tracker.record_budget_block(agent)
        return (
            f"REFUSED: web_search budget exhausted ({used}/{WEB_SEARCH_CALL_BUDGET} used). No "
            "further searches will run. Do not retry -- either call fetch_page on a URL you "
            "already know (including sibling country-site URLs you can construct by pattern), "
            "or produce your final answer NOW with every address you have already seen, "
            "including the weakly-sourced ones."
        )

    if tracker is not None:
        tracker.record_query(agent, "web_search", query)

    try:
        resp = _create_response_with_retries(
            model=SEARCH_MODEL,
            input=query,
            tools=[{"type": "web_search"}],
        )
    except Exception as exc:  # a single flaky call must not crash the whole run
        return f"web_search failed for {query!r} after retries ({exc}). Try a different query or proceed without it."

    _record_search_usage(resp)
    hits = _extract_hits(resp)
    note = _budget_note(
        "web_search",
        used + 1,
        WEB_SEARCH_CALL_BUDGET,
        "Stop broadening: fetch the pages you already know about and start assembling your "
        "final answer.",
    )
    return _format_hits(query, hits) + note


class _TextExtractingHTMLParser(HTMLParser):
    """Minimal stdlib HTML-to-text extractor -- no extra dependency (bs4/lxml) needed for
    the address-extraction use case this feeds into."""

    _SKIP_TAGS = {"script", "style", "noscript", "svg", "template"}
    _BLOCK_TAGS = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        elif tag in self._BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in self._BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self._chunks.append(data.strip())

    def text(self) -> str:
        joined = " ".join(self._chunks)
        return re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n\n", joined)).strip()


def _html_to_text(html: str) -> str:
    parser = _TextExtractingHTMLParser()
    try:
        parser.feed(html)
    except Exception:  # malformed HTML shouldn't crash the tool
        pass
    return parser.text()


def _condense_page_text(text: str, limit: int = _MAX_PAGE_CHARS) -> str:
    """Fit a fetched page into `limit` chars while keeping its ADDRESSES, not just its top.

    Head-truncating a long locations page is exactly how the agent used to lose the second
    half of a global office directory and then buy it back one paid `site:` search per country.
    So when a page is over the limit, drop the lines that can't be part of an address block
    (nav, legal boilerplate, product copy) and keep the address-looking ones plus a couple of
    lines of context either side -- the city/country/entity-name lines that carry no digits sit
    right next to the street line. Only if that is still too long does it get truncated, and
    then it says so explicitly, because a silent cut is what makes the model go searching.
    """
    if len(text) <= limit:
        return text

    lines = text.split("\n")
    keep: set[int] = set()
    for i, line in enumerate(lines):
        if _ADDRESS_HINT_RE.search(line):
            keep.update(range(max(0, i - _CONDENSE_CONTEXT_LINES), min(len(lines), i + _CONDENSE_CONTEXT_LINES + 1)))

    if keep:
        condensed = "\n".join(lines[i] for i in sorted(keep))
        prefix = (
            f"[NOTE: this page is {len(text)} chars; non-address lines (nav/boilerplate) were "
            "stripped and only address-bearing lines kept. Nothing address-like was dropped.]\n\n"
        )
    else:
        condensed = text
        prefix = ""

    if len(prefix) + len(condensed) <= limit:
        return prefix + condensed

    kept = condensed[: limit - 400]
    return (
        f"[NOTE: this page is {len(text)} chars and had to be cut to fit; you are seeing the "
        f"first {len(kept)} chars of its address-bearing content. Do NOT run a web_search "
        "against this URL to recover the rest -- that is refused. Instead fetch a deeper, more "
        "specific URL from this same site (e.g. a single country's or region's contact page).]\n\n"
        + kept
    )


def _looks_blocked(status_code: int, text: str, *, js_rendered: bool = False) -> bool:
    """Whether a fetched page should be escalated to the next tier. `js_rendered=True` says the
    page came back from a headless browser, which makes the JS-pending markers meaningless --
    the JS has already run, so what's in the text is what a human would see."""
    if status_code in (401, 403, 429, 503):
        return True
    markers = _BLOCKED_MARKERS if js_rendered else _BLOCKED_MARKERS + _JS_PENDING_MARKERS
    lowered = text.lower()
    return any(marker in lowered for marker in markers) or len(text.strip()) < 200


# Why the plain tier reports HOW it failed rather than just None: fetch_page uses it to decide
# whether the ScrapingBee tier is worth paying for. Everything here can be fixed by a headless
# browser on a residential IP -- except _FETCH_NON_TEXT, which cannot be fixed by anything.
_FETCH_OK = "ok"
_FETCH_BLOCKED = "blocked"  # 403/captcha/interstitial, or a JS-only shell with no content in it
_FETCH_NON_TEXT = "non_text"  # PDF or binary asset: no render turns it into text, so skip tier 2
_FETCH_ERROR = "error"  # DNS/TLS/reset/timeout -- often IP-level blocking, worth a proxy retry
# The page is simply not there. Distinct from _FETCH_BLOCKED because nothing downstream can help:
# a headless browser renders the same 404, and buying a search retrieval for a URL that does not
# exist is pure waste. This matters far more now that the prompt tells the agent to fetch
# speculatively -- a measured run guessed five sibling URLs that 404'd (three spellings of one
# brand's contact page) and, before this existed, paid ~$0.012 to "retrieve" each of them.
_FETCH_NOT_FOUND = "not_found"


class _PlainFetch(NamedTuple):
    text: str | None
    reason: str


def _is_text_content_type(content_type: str) -> bool:
    return "text/html" in content_type or "text/plain" in content_type


def _decode_response(resp: "requests.Response") -> str:
    """requests' `.text` falls back to guessing latin-1 when a server doesn't declare a
    charset, which mangles UTF-8 pages (mojibake on curly quotes/apostrophes etc.) -- most
    modern pages ARE UTF-8 regardless of what the header says, so decode explicitly."""
    try:
        return resp.content.decode("utf-8")
    except UnicodeDecodeError:
        return resp.content.decode(resp.apparent_encoding or "utf-8", errors="replace")


def _fetch_via_plain_http(url: str) -> _PlainFetch:
    """Tier 1. Returns extracted page text, or the reason it could not be had."""
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": search_user_agent()},
            timeout=_FETCH_TIMEOUT_SECONDS,
        )
    except requests.RequestException:
        return _PlainFetch(None, _FETCH_ERROR)

    # Checked before anything else: a 404/410 body is usually a short error page, which the
    # length heuristic in _looks_blocked would otherwise misread as a bot block and escalate.
    if resp.status_code in (404, 410):
        return _PlainFetch(None, _FETCH_NOT_FOUND)

    content_type = resp.headers.get("Content-Type", "")
    if not _is_text_content_type(content_type):
        return _PlainFetch(None, _FETCH_NON_TEXT)

    raw_text = _decode_response(resp)
    text = _html_to_text(raw_text) if "html" in content_type else raw_text
    if _looks_blocked(resp.status_code, text):
        return _PlainFetch(None, _FETCH_BLOCKED)
    return _PlainFetch(text, _FETCH_OK)


def _scrapingbee_api_key() -> str | None:
    """Read at CALL time, never at import time: cli.py's load_dotenv() runs long after this
    module is first imported, and the test scripts import it with no key set at all."""
    return (os.environ.get("SCRAPINGBEE_API_KEY") or "").strip() or None


def _scrapingbee_params(url: str) -> dict[str, str]:
    """Pure and key-free (auth is a Bearer header), so the request shape can be asserted in a
    network-free test and the secret never enters a param dict that might get logged.

    Deliberately nothing beyond these three: mode=auto REJECTS render_js/premium_proxy/
    stealth_proxy/transparent_status_code with a 400, because it picks those itself -- trying
    the cheapest configuration first and billing only for the one that worked (1 credit plain,
    5 rendered, 10/25 premium, 75 stealth; 0 if every tier fails). Two known follow-ups, both
    needing a check that mode=auto accepts them: `wait_browser=networkidle2` for locator maps
    that fetch their pins by XHR after DOMContentLoaded, and a `max_cost` ceiling to keep auto
    off the 75-credit stealth tier."""
    return {
        "url": url,  # requests percent-encodes this, including any ?a=b&c=d of its own
        "mode": "auto",
        "timeout": str(_SCRAPINGBEE_SERVER_TIMEOUT_MS),
    }


def _scrapingbee_credits(resp: "requests.Response") -> int:
    """Credits ScrapingBee actually charged: Spb-auto-cost under mode=auto, Spb-cost otherwise.
    Always read from the header rather than assumed from the outcome -- their docs say failed
    requests are free, but a measured non-200 still reported 25 credits spent."""
    raw = resp.headers.get("Spb-auto-cost") or resp.headers.get("Spb-cost") or ""
    try:
        return int(raw)
    except ValueError:
        return _SCRAPINGBEE_ASSUMED_CREDITS if resp.status_code in (200, 404, 410) else 0


def _scrapingbee_initial_status_code(resp: "requests.Response") -> int | None:
    """The TARGET site's status as ScrapingBee saw it (their Spb-initial-status-code header) --
    distinct from resp.status_code, which is ScrapingBee's OWN gateway response and is what
    `succeeded` is based on. None if the header is missing/unparseable."""
    raw = resp.headers.get("Spb-initial-status-code", "")
    try:
        return int(raw)
    except ValueError:
        return None


_SCRAPINGBEE_ERROR_BODY_MAX_CHARS = 500


def _scrapingbee_error_reason(resp: "requests.Response") -> str:
    """Human-readable explanation for a non-200 ScrapingBee gateway response, for
    scrapingbee_log.jsonl. Their error envelope is JSON with `error`/`reason`/`help` fields
    (e.g. {"error": "mode=auto tried tier(s) html, js, premium, premium_js, stealth without
    success", "reason": "Server responded with 613", "help": "..."} -- confirmed by a live
    re-test of a failing URL). Falls back to the raw (truncated) body if it isn't JSON, so a
    differently-shaped failure still logs SOMETHING instead of nothing."""
    try:
        body = resp.json()
        parts = [str(body[k]) for k in ("error", "reason") if body.get(k)]
        if parts:
            return " -- ".join(parts)
    except ValueError:
        pass
    return resp.text[:_SCRAPINGBEE_ERROR_BODY_MAX_CHARS]


def _fetch_via_scrapingbee(url: str) -> str | None:
    """Tier 2: re-fetch through ScrapingBee's headless browser and rotating proxy, which fixes
    the two things that defeat a plain GET -- bot blocking, and content that only exists after
    the page's JS runs. Returns the extracted page text, or None if the key is absent, the API
    errored, or the rendered page STILL looks blocked, in which case the caller escalates to
    the paid web_search fallback."""
    api_key = _scrapingbee_api_key()
    tracker = get_tracker()
    if api_key is None:
        if tracker is not None:
            tracker.record_scrapingbee_unavailable()
        return None

    try:
        with _scrapingbee_semaphore:
            resp = requests.get(
                _SCRAPINGBEE_ENDPOINT,
                params=_scrapingbee_params(url),
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=_SCRAPINGBEE_TIMEOUT_SECONDS,
            )
    except requests.RequestException as exc:
        if tracker is not None:
            tracker.record_scrapingbee_usage(
                get_current_agent(), credits=0, succeeded=False, url=url,
                reason=f"request exception: {exc}",
            )
        return None

    succeeded = resp.status_code == 200
    initial_status_code = _scrapingbee_initial_status_code(resp)
    credits = _scrapingbee_credits(resp)

    def _record(reason: str | None) -> None:
        # Called on every exit from here on, successful or not -- this used to be one
        # unconditional call right after the response came back ("recorded before any other
        # branching, so no early return can silently skip the accounting"), but `reason` now
        # depends on branches decided further down, so each return path calls this instead.
        if tracker is not None:
            tracker.record_scrapingbee_usage(
                get_current_agent(), credits=credits, succeeded=succeeded, url=url,
                status_code=resp.status_code, initial_status_code=initial_status_code,
                reason=reason,
            )

    if not succeeded:
        # transparent_status_code isn't available under mode=auto, so a non-200 is always
        # ScrapingBee's own error envelope (400 bad params, 401 bad key, 402 out of credits,
        # 429 concurrency, 500 target/render failure) -- a short JSON body, never page content.
        # No retry: auto mode has already exhausted its escalation ladder, and the tier behind
        # this one is a better use of those seconds than an identical second render.
        _record(_scrapingbee_error_reason(resp))
        return None

    if not _is_text_content_type(resp.headers.get("Content-Type", "")):
        _record("non_text_content")
        return None

    # ScrapingBee proxies the target's bytes, so it inherits the same latin-1 mojibake trap.
    text = _html_to_text(_decode_response(resp))
    if _looks_blocked(initial_status_code or 200, text, js_rendered=True):
        _record("still_blocked_after_render")
        return None

    _record(None)
    return text


def _fetch_via_web_search_fallback(url: str) -> str:
    """Tier 3, the last resort: ask the hosted web_search tool to retrieve the page's
    underlying data. Only reached when BOTH cheaper tiers failed -- the plain GET and the
    ScrapingBee browser fetch -- or when the URL is a PDF, which neither of them can read.
    Costs the same as a normal web_search call, but is much higher-value per dollar than a
    plain query: it can return the page's ENTIRE address list in one call instead of one or
    two addresses per query."""
    try:
        resp = _create_response_with_retries(
            model=SEARCH_MODEL,
            input=(
                f"Retrieve this exact web page and list EVERY physical address or office "
                f"location it contains: {url}\n\n"
                "Output ONLY a plain list, one entry per line, formatted as "
                "'Business/organisation name | City, Country | full address'. The business name "
                "is REQUIRED on every line, copied exactly as the page gives it: directory and "
                "business-listing pages mix many different companies together, and without the "
                "name each address is unattributable. Use '(unnamed)' only if the page truly "
                "gives no name for that entry. Do not add any commentary, explanation, caveats, "
                "or summary text before, after, or between entries -- if the page has hundreds of "
                "entries, list all of them anyway in this compact one-line-each format; do not "
                "stop early or summarize instead of listing."
            ),
            tools=[{"type": "web_search"}],
            max_output_tokens=16000,
        )
    except Exception as exc:  # a single flaky call must not crash the whole run
        return f"Could not retrieve any content for {url} after retries ({exc})."

    _record_search_usage(resp)
    text = getattr(resp, "output_text", "") or ""
    return text or f"Could not retrieve any content for {url}."


@tool
def fetch_page(url: str) -> str:
    """Fetch the full text content of a specific URL -- typically a promising result from
    web_search (e.g. a company's "our locations"/"offices"/"contact" page). One call can
    surface every address listed on that page, which is far more efficient than searching
    for each one individually. If the page blocks direct fetching or its content is only
    rendered client-side by JavaScript (common for interactive "office locations" maps), it
    automatically retries through a rendered browser, and only then falls back to a
    search-based retrieval.

    This is the CHEAP tool -- the plain-HTTP path is a free GET -- and it is the right way to
    read any page whose URL you already know, including sibling URLs you construct yourself:
    country sites in one family almost always share a path (acme.com.my/company/contact.html,
    acme.in/company/contact.html), so fetch them directly rather than searching per country.
    A URL that does not exist is reported as NOT FOUND and costs nothing, but a PDF or a page
    that defeats both free routes is retrieved via a PAID fallback, so prefer pages you have real
    reason to think list several addresses at once.
    """
    tracker = get_tracker()
    agent = get_current_agent()

    used = _calls_used("fetch_page")
    if used >= FETCH_PAGE_CALL_BUDGET:
        if tracker is not None:
            tracker.record_budget_block(agent)
        return (
            f"REFUSED: fetch_page budget exhausted ({used}/{FETCH_PAGE_CALL_BUDGET} used). "
            "Produce your final answer NOW with every address you have already seen, including "
            "the weakly-sourced ones."
        )

    already_fetched = tracker is not None and tracker.has_issued("fetch_page", url)
    if tracker is not None:
        tracker.record_query(agent, "fetch_page", url)

    note = _budget_note(
        "fetch_page",
        used + 1,
        FETCH_PAGE_CALL_BUDGET,
        "Fetch only pages you have real reason to think list addresses, and start assembling "
        "your final answer.",
    )

    # Re-fetching a URL wastes a budgeted call and pastes a second copy of the page into a
    # context that is re-sent every turn. Behaviour-neutral in practice -- the last 12 logged
    # runs fetched zero repeat URLs -- so this only ever catches a genuine mistake.
    if already_fetched:
        return (
            f"ALREADY FETCHED: you fetched {url} earlier in this conversation and its full text "
            "is still above you. Read it there rather than pulling in a second copy. If you need "
            f"something that page did not contain, fetch a DIFFERENT, deeper URL.{note}"
        )

    plain = _fetch_via_plain_http(url)
    if plain.text is not None:
        return f"Content of {url}:\n\n{_condense_page_text(plain.text)}{note}"

    # A guessed URL that does not exist stops here, costing nothing. Escalating a 404 would
    # spend a browser render and then a paid retrieval on a page no tier can produce -- which is
    # exactly what makes guessing sibling URLs cheap enough to be worth encouraging.
    if plain.reason == _FETCH_NOT_FOUND:
        return (
            f"NOT FOUND (this cost nothing): {url} returned 404/410 -- that page does not exist. "
            "If you were guessing a URL pattern, guess a different one, or fetch the site's "
            f"locator page and read the real link off it.{note}"
        )

    # A PDF/binary is the one failure a headless browser cannot fix, so it skips the middle
    # tier rather than spending credits rendering something with no HTML in it.
    if plain.reason != _FETCH_NON_TEXT:
        rendered = _fetch_via_scrapingbee(url)
        if rendered is not None:
            # Raw page text like tier 1's, so it gets tier 1's treatment: the condenser, not
            # the fallback's larger cap below.
            return f"Content of {url} (retrieved via rendered browser):\n\n{_condense_page_text(rendered)}{note}"

    if tracker is not None:
        # Logged under its own key so it both gates this budget and shows up in
        # queries_log.jsonl as a distinct row -- that is how we can tell afterwards which URLs
        # actually needed paying for, i.e. whether the ScrapingBee tier is earning its place.
        tracker.record_query(agent, "fetch_page_search_fallback", url)

    # The fallback's output is already a dense, LLM-condensed address list (not noisy raw
    # HTML), and it was paid for in full -- truncating it back down to _MAX_PAGE_CHARS would
    # throw away most of what that call just retrieved, so it gets a much larger budget.
    text = _fetch_via_web_search_fallback(url)
    return f"Content of {url} (retrieved via search fallback):\n\n{text[:_MAX_FALLBACK_CHARS]}{note}"


class UsageAttributionMiddleware(AgentMiddleware):
    """Tags every model call and tool call made by this agent with `agent_name` for
    usage.py's cost/query tracker. This middleware itself never blocks a call -- it only exists
    so usage.py's per-agent breakdown and queries_log.jsonl have a label to attribute calls to.
    The per-tool call budgets are enforced inside web_search/fetch_page above (where the
    tracker's per-tool counts are), so `tool_calls_attempted` here counts what the model tried
    and `tool_calls_blocked_by_budget` counts what those tools refused.

    Per-agent attribution (`usage.get_current_agent()`) is set via a contextvar immediately
    before synchronously calling into `handler(request)` from both `wrap_model_call` and
    `wrap_tool_call` -- never across a thread/node boundary, which is what makes the
    attribution reliable regardless of how LangGraph schedules nodes internally.
    """

    def __init__(self, agent_name: str) -> None:
        super().__init__()
        self.agent_name = agent_name

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        agent_token = set_current_agent(self.agent_name)
        try:
            return handler(request)
        finally:
            reset_current_agent(agent_token)

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        tracker = get_tracker()
        if tracker is not None:
            tracker.record_tool_attempt(self.agent_name, allowed=True)

        agent_token = set_current_agent(self.agent_name)
        try:
            return handler(request)
        finally:
            reset_current_agent(agent_token)
