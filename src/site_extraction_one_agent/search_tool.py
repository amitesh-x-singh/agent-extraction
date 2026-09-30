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

  1. A free plain HTTP GET. Handles the large majority of pages -- and PDFs: a PDF body is
     recognised by its magic bytes (never by Content-Type or a .pdf suffix, both of which lie in
     both directions) and its text layer extracted locally with pypdf. Certification documents
     are the densest address source a company publishes -- one measured 24-page ISO 14001 bundle
     lists ~20 exact plant addresses -- and every one of them used to be bought from tier 3.

  2. A ScrapingBee fetch (headless browser + rotating proxy), reached when the plain GET is
     blocked (403, bot-detection interstitial, etc.) or the page's real content is only rendered
     client-side by JS -- common for interactive "office locations" maps, where the static HTML
     has nothing in it. This returns the page's REAL rendered text, which the condenser can mine
     for every address on it, and costs ~$0.001 -- roughly a tenth of the tier below it. If what
     it renders turns out to be PDF bytes (common when a .pdf URL is really a JS redirect shell),
     those go through the same local parser as tier 1. Skipped only for non-PDF binaries, and for
     a PDF tier 1 already found has no text layer -- no amount of rendering OCRs a scan.

  3. Asking the hosted web_search tool to retrieve the page, explicitly instructed to list
     everything it contains rather than answer a narrow question. Costs a full web_search call
     and returns an LLM's summary of the page rather than the page, so it is last: only reached
     when both cheaper tiers failed, or a PDF turned out to be a scan with no text layer. This is
     the only tier with a call budget, because it is the only one that costs money per call.

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

  * Call budgets on the PAID paths (config.WEB_SEARCH_CALL_BUDGET /
    FETCH_PAGE_PAID_CALL_BUDGET). Prose budgets in the system prompt were ignored by roughly 3x,
    so the count is now reported back inside every tool result and the tool refuses calls past
    the cap. UsageAttributionMiddleware
    still never blocks anything -- it only tags calls for usage.py's accounting; the budget is
    enforced in the tool bodies, where the tracker's per-tool counts live.

fetch_page's two free tiers are UNCAPPED; only its paid tier is budgeted -- see the comment in
config.py. An earlier experiment that uncapped all three was reverted, and two things from it
were kept because they are cheap and correct either way: a 404/410 short-circuit
(_FETCH_NOT_FOUND) so a guessed URL that does not exist never escalates to a render or a paid
retrieval, and a separate `fetch_page_search_fallback` row in queries_log.jsonl for every tier-3
escalation, which is what makes it possible to see which URLs actually cost money.
"""

from html.parser import HTMLParser
from typing import Any, NamedTuple
from urllib.parse import unquote, urljoin, urlsplit

import io
import logging
import os
import re
import requests
import urllib3
import warnings
import threading
import time
import unicodedata
from pypdf import PdfReader
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.tools import tool
from openai import APIConnectionError, APIStatusError, OpenAI

from .config import (
    BUDGET_WARN_FRACTION,
    FETCH_PAGE_PAID_CALL_BUDGET,
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
# requests' timeout is a BETWEEN-CHUNKS read timeout, not a transfer budget, so a server that
# trickles bytes forever never trips it. Streaming the body makes a total deadline possible.
_MAX_FETCH_WALL_SECONDS = 30
_MAX_PAID_FETCHES_PER_HOST = 2  # see fetch_page: a site no free tier can read is a dead end
# PDF guards. The largest certificate bundle measured in queries_log.jsonl is 3.7MB / 24 pages,
# so these are multiples of real documents, not guesses: the size cap keeps a hostile body out of
# memory, and the page cap keeps pypdf -- pure Python and CPU-bound -- from stalling a batched
# turn on a 400-page report.
_MAX_PDF_BYTES = 12 * 1024 * 1024
_MAX_PDF_PAGES = 120
# Below this, a PDF has no usable text layer: a scan, or nothing but page furniture. Same floor
# _looks_blocked uses for a page, for the same reason -- under it there is nothing to mine.
_MIN_PDF_TEXT_CHARS = 200
_MAX_API_RETRIES = 3
_API_RETRY_BACKOFF_SECONDS = 2.0

# pypdf logs a warning per recovered object on a damaged file ("Ignoring wrong pointing object",
# xref rebuilds). We hand it whatever the web serves, so those are expected and would otherwise
# flood a batch run's console.
logging.getLogger("pypdf").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)

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
        # The Responses API names these input_tokens_details.cached_tokens /.cache_write_tokens
        # (the Chat Completions spelling is prompt_tokens_details.*). Both are parts of
        # input_tokens and each is billed at its own rate, so recording only the total would
        # price every retrieved page as uncached -- and search content is the larger half of
        # this pipeline's token spend.
        details = getattr(resp.usage, "input_tokens_details", None)
        tracker.record_web_search_usage(
            agent=get_current_agent(),
            input_tokens=resp.usage.input_tokens or 0,
            output_tokens=resp.usage.output_tokens or 0,
            cached_input_tokens=getattr(details, "cached_tokens", 0) or 0,
            cache_write_tokens=getattr(details, "cache_write_tokens", 0) or 0,
        )


# --- Near-duplicate guard ---------------------------------------------------------------------
# 15% of the 6,795 logged searches share 80%+ of their words with an earlier search for the same
# company -- the same question reworded, which returns the same pages. Words that carry no
# meaning for a search engine are ignored so "X office address" and "X offices addresses" match.
_DUP_FILLER_WORDS = frozenset(
    "the a an of and or for in on at to by with from address addresses official office offices "
    "location locations site sites".split()
)
_DUP_JACCARD = 0.8


def _query_words(query: str) -> frozenset[str]:
    return frozenset(w for w in _GUARD_WORD_RE.findall(_fold(query)) if w not in _DUP_FILLER_WORDS)


def _near_duplicate_of(query: str, earlier: list[str]) -> str | None:
    words = _query_words(query)
    if len(words) < 2:
        return None
    for prior in earlier:
        prior_words = _query_words(prior)
        union = words | prior_words
        if union and len(words & prior_words) / len(union) >= _DUP_JACCARD:
            return prior
    return None


# --- Certificate gate --------------------------------------------------------------------------
# The 23 logged companies that searched for certificates did so 12 times each on average, found
# or not. One probe decides: follow-ups run only once something shows this supplier publishes
# certificates -- a certificate-looking search hit, or a certificate link on a fetched page.
_CERT_QUERY_RE = re.compile(
    r"\biso\s?-?\d{4,5}|\biatf\b|\bas\s?9100|\b16949\b|\b14001\b|\b9001\b|\b45001\b|\b13485\b"
    r"|certif|zertifi",
    re.I,
)
_CERT_BODY_HOSTS = (
    "iafcertsearch.org", "tuvsud.com", "tuv.com", "dnv.com", "bureauveritas", "sgs.com", "ul.com",
    "bsigroup.com", "lrqa.com", "nqa.com", "intertek.com", "dekra", "iatfglobaloversight.org",
    "certipedia", "ukas.com", "qmscert", "apave",
)
_MAX_CERT_SEARCHES = 3


def _is_cert_evidence(url: str, text: str = "") -> bool:
    host = urlsplit(url).netloc.lower()
    if any(body in host for body in _CERT_BODY_HOSTS):
        return True
    path = unquote(urlsplit(url).path)
    return bool(_CERT_QUERY_RE.search(path) or (".pdf" in path.lower() and _CERT_QUERY_RE.search(text)))


def _no_certificates_message(query: str, reason: str) -> str:
    return (
        f"REFUSED (this cost nothing): {query!r} is another certificate search, but {reason}. "
        "Do not search for certificates again for this supplier. Spend searches on other source "
        "types, and fetch any certificate you are shown on a page directly."
    )


@tool
def web_search(query: str) -> str:
    """Search the web for a query and return titled results with source URLs and snippets.
    PAID per call and budgeted -- see the budget line at the end of every result.

    Use specific, targeted queries (e.g. company name + "headquarters address", or company
    name + "SEC 10-K filing address"). Vary queries across site-type keywords (plant, factory,
    warehouse, distribution center, office, R&D, facility) and source categories rather than
    repeating near-identical queries -- a company usually has more than one kind of site.

    Things this tool will REFUSE, so don't spend turns on them:
      * A `site:` query pointing at a specific page (`site:acme.com/contact-us "Germany"`).
        You already know that URL -- call fetch_page on it and read the whole page at once.
      * A search for a place named by a document link you were already shown and have not yet
        fetched (e.g. a per-plant certificate PDF) -- fetch that document instead.
      * Anything past the call budget.
    Use it to DISCOVER pages and facts, not to read a page you can already name, and not to
    re-confirm an address you have already seen.
    """
    tracker = get_tracker()
    agent = get_current_agent()

    if tracker is not None:
        fetched = {q["query"] for q in tracker.queries if q["tool"] == "fetch_page"}
        known = _known_doc_links_for_query(query, tracker.doc_links(), fetched)
        if known:
            tracker.record_search_redirected_to_doc_link()
            return _redirect_to_doc_links_message(query, known)

    deep_site = _DEEP_SITE_QUERY_RE.search(query)
    if deep_site:
        host, path = deep_site.group("host"), deep_site.group("path").rstrip('"\'')
        url = f"https://{host}/{path}"
        if tracker is not None:
            tracker.record_search_redirected_to_fetch()
        return _redirect_to_fetch_message(query, url)

    if tracker is not None:
        prior = _near_duplicate_of(query, [q["query"] for q in tracker.queries if q["tool"] == "web_search"])
        if prior is not None:
            tracker.record_search_refused("duplicate")
            return (
                f"REFUSED (this cost nothing): {query!r} asks almost the same thing as your earlier "
                f"search {prior!r}, whose results are already above. A reworded query returns the "
                "same pages. Fetch the promising results you already have, or search a different "
                "place, site type or source category."
            )

    is_cert_query = bool(_CERT_QUERY_RE.search(query))
    if is_cert_query and tracker is not None and tracker.cert_searches >= 1:
        if tracker.cert_searches >= _MAX_CERT_SEARCHES:
            tracker.record_search_refused("no_certificates")
            return _no_certificates_message(query, f"the {_MAX_CERT_SEARCHES}-search limit for certificates is reached")
        if not tracker.cert_evidence_seen:
            tracker.record_search_refused("no_certificates")
            return _no_certificates_message(query, "your first certificate search found no certificate for this supplier")

    used = _calls_used("web_search")
    reserved = (
        tracker.reserve_call(agent, "web_search", query, WEB_SEARCH_CALL_BUDGET)
        if tracker is not None
        else used + 1
    )
    if reserved is None:
        used = _calls_used("web_search")
        tracker.record_budget_block(agent)
        return (
            f"REFUSED: web_search budget exhausted ({used}/{WEB_SEARCH_CALL_BUDGET} used). No "
            "further searches will run. Do not retry -- either call fetch_page on a URL you "
            "already know (including sibling country-site URLs you can construct by pattern), "
            "or produce your final answer NOW with every address you have already seen, "
            "including the weakly-sourced ones."
        )
    used = reserved - 1

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
    if tracker is not None:
        # Every domain a result points at or names is now "seen" and may be fetched.
        tracker.record_hosts(
            {_norm_host(h["url"]) for h in hits}
            | {host for h in hits for host in _hosts_in_text(h.get("title", "") + " " + h.get("snippet", ""))}
        )
        found = any(_is_cert_evidence(h["url"], h.get("title", "") + " " + h.get("snippet", "")) for h in hits)
        if is_cert_query:
            tracker.record_cert_search(found)
        elif found:
            tracker.record_cert_evidence()
    note = _budget_note(
        "web_search",
        used + 1,
        WEB_SEARCH_CALL_BUDGET,
        "Stop broadening: fetch the pages you already know about and start assembling your "
        "final answer.",
    )
    return _format_hits(query, hits) + note


# A certificate library page reads "ISO 9001 Betzdorf / ISO 14001 Bzenec / ..." as text, but the
# PDF URLs behind those labels live only in href attributes, which text extraction drops. Without
# them the agent spends a web_search to rediscover each certificate's URL, or skips the library.
# So links that look like documents or certificates are kept -- capped -- and appended as a compact
# block that the condenser never strips.
# Standard numbers must stand alone: a Companies House URL /company/11590013 contains "9001".
_DOC_LINK_RE = re.compile(
    r"\.pdf\b|certif|zertifi|iso[\s_-]?\d{4,5}|iatf|as[\s_-]?9100"
    r"|(?<!\d)(?:16949|14001|9001|45001|13485)(?!\d)",
    re.I,
)
# Large enough for a whole certificate library (the measured AVX page links 119 PDFs). An earlier
# cap of 20 showed only the first region's certificates, and the agent then spent ~20 paid
# web_searches looking up the other plants one by one. Tokens are cheap next to searches.
_MAX_DOC_LINKS = 200
_MAX_DOC_LINK_URL_CHARS = 220
_DOC_LINKS_HEADER = "[LINKS ON THIS PAGE -- fetch_page these directly (free); do not search for them]"
# Other websites a page links to (country sites, brand sites, parent group), shown by domain so
# the agent opens a sister site it was SHOWN instead of guessing one. Social/CDN noise dropped.
_MAX_LINKED_HOSTS = 40
_NOISE_HOSTS = (
    "facebook.com", "linkedin.com", "twitter.com", "x.com", "youtube.com", "instagram.com",
    "google.com", "googleapis.com", "gstatic.com", "apple.com", "tiktok.com", "pinterest.com",
    "xing.com", "wa.me", "whatsapp.com", "cloudflare.com", "jsdelivr.net", "doubleclick.net",
)


def _norm_host(url_or_host: str) -> str:
    """'https://WWW.Acme.co.uk:443/x' -> 'acme.co.uk'. Hosts are compared in this form everywhere."""
    text = (url_or_host or "").strip().lower()
    host = urlsplit(text).netloc if "://" in text else text.split("/")[0]
    return host.split("@")[-1].split(":")[0].removeprefix("www.").strip(".")


# Domains written in page or snippet text: URLs, bare "acme.de", and e-mail addresses (info@acme.de
# names the company's domain just as well). A false match only widens what may be fetched.
_TEXT_HOST_RE = re.compile(r"(?<![\w.-])(?:https?://)?(?:[\w.+-]+@)?((?:[a-z0-9-]+\.)+[a-z]{2,24})(?![\w-])", re.I)
# The fixed registry lookups written into the prompt: known in advance, never guessed.
_REGISTRY_HOSTS = frozenset({
    "api.gleif.org", "find-and-update.company-information.service.gov.uk", "sec.gov",
    "en.wikipedia.org", "wikipedia.org",
})


def _hosts_in_text(text: str) -> set[str]:
    return {_norm_host(m) for m in _TEXT_HOST_RE.findall(text or "")}


def _host_is_known(host: str, known: set[str]) -> bool:
    """Seen, a registry, or a subdomain of either (sc.kyocera-avx.com once kyocera-avx.com is known)."""
    return any(host == k or host.endswith("." + k) for k in known | _REGISTRY_HOSTS)


class _TextExtractingHTMLParser(HTMLParser):
    """Minimal stdlib HTML-to-text extractor -- no extra dependency (bs4/lxml) needed for
    the address-extraction use case this feeds into."""

    _SKIP_TAGS = {"script", "style", "noscript", "svg", "template"}
    _BLOCK_TAGS = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article"}

    def __init__(self, base_url: str = "") -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0
        self._base_url = base_url
        self._href: str | None = None
        self._anchor_chunks: list[str] = []
        self.doc_links: list[tuple[str, str]] = []  # (anchor text, absolute URL), first-seen order
        self._seen_links: set[str] = set()
        self.link_hosts: list[str] = []  # every linked domain, first-seen order

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        elif tag in self._BLOCK_TAGS:
            self._chunks.append("\n")
        if tag == "a":
            self._href = dict(attrs).get("href")
            self._anchor_chunks = []

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in self._BLOCK_TAGS:
            self._chunks.append("\n")
        if tag == "a" and self._href is not None:
            self._record_link(self._href, " ".join(self._anchor_chunks))
            self._href = None

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self._chunks.append(data.strip())
            if self._href is not None:
                self._anchor_chunks.append(data.strip())

    def _record_link(self, href: str, label: str) -> None:
        href = href.strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            return
        url = urljoin(self._base_url, href) if self._base_url else href
        if urlsplit(url).scheme not in ("http", "https"):
            return
        host = _norm_host(url)
        if host and host not in self.link_hosts:
            self.link_hosts.append(host)
        if len(url) > _MAX_DOC_LINK_URL_CHARS:
            return
        # Match the URL path (not the query string, where tracking junk lives) or the anchor text.
        if not (_DOC_LINK_RE.search(urlsplit(url).path) or _DOC_LINK_RE.search(label)):
            return
        if url in self._seen_links or len(self.doc_links) >= _MAX_DOC_LINKS:
            return
        self._seen_links.add(url)
        self.doc_links.append((_normalize_whitespace(label)[:80], url))

    def text(self) -> str:
        return _normalize_whitespace(" ".join(self._chunks))


def _normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs and blank lines, KEEPING single newlines -- _condense_page_text
    works line by line, so flattening newlines here would blind it to every address."""
    return re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n\n", text)).strip()


def _html_to_text(html: str, base_url: str = "") -> str:
    """Page text, followed by a capped block of the page's document/certificate links (see
    _DOC_LINK_RE). Judge blocking on _page_body() of the result, so a bot wall with a few PDF
    links in its footer is not mistaken for real content."""
    parser = _TextExtractingHTMLParser(base_url)
    try:
        parser.feed(html)
    except Exception:  # malformed HTML shouldn't crash the tool
        pass
    text = parser.text()
    own = _norm_host(base_url)
    other_sites = [
        h for h in parser.link_hosts
        if h != own and not any(h == n or h.endswith("." + n) for n in _NOISE_HOSTS)
    ][:_MAX_LINKED_HOSTS]
    if not parser.doc_links and not other_sites:
        return text
    lines = [f"- {label} -> {url}" if label else f"- {url}" for label, url in parser.doc_links]
    if other_sites:
        # Not a "- " line, so _parse_doc_links never mistakes it for a document link.
        lines.append("Other websites this page links to: " + ", ".join(other_sites))
    return f"{text}\n\n{_DOC_LINKS_HEADER}\n" + "\n".join(lines)


def _split_doc_links(text: str) -> tuple[str, str]:
    """(page body, document-links block including its leading blank line); block is '' if absent."""
    idx = text.rfind(f"\n\n{_DOC_LINKS_HEADER}\n")
    if idx < 0:
        return text, ""
    return text[:idx], text[idx:]


def _page_body(text: str) -> str:
    return _split_doc_links(text)[0]


def _parse_doc_links(block: str) -> list[tuple[str, str]]:
    links = []
    for line in block.split("\n"):
        if not line.startswith("- "):
            continue
        label, sep, url = line[2:].rpartition(" -> ")
        links.append((label, url) if sep else ("", line[2:]))
    return links


# The known-document guard. Words that describe WHAT is wanted rather than WHERE: a query made
# only of these plus the company name has nothing to match against a certificate's place name.
_GUARD_GENERIC_WORDS = frozenset(
    "address addresses official factory factories plant plants office offices site sites "
    "facility facilities location locations manufacturing production headquarters contact "
    "company corporation group limited gmbh sro directory google maps business profile "
    "yellowpages street city country warehouse center centre distribution service sales "
    "certificate certificates certification certified quality pdf docs document documents "
    "north south east west america europe asia central".split()
)
_GUARD_WORD_RE = re.compile(r"[a-z0-9]+")
# A word that appears in this share of the labels is the company name or a standard's name, not
# a place.
_GUARD_COMMON_WORD_SHARE = 0.3
_GUARD_MIN_LINKS_FOR_SHARE = 5


def _fold(text: str) -> str:
    """Lower-case and strip accents, so 'Uherské Hradiště' matches 'Uherske_Hradiste.pdf'."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


def _link_words(label: str, url: str) -> set[str]:
    return set(_GUARD_WORD_RE.findall(_fold(f"{label} {unquote(urlsplit(url).path)}")))


def _known_doc_links_for_query(query: str, doc_links: dict[str, str], fetched: set[str]) -> list[str]:
    """Unfetched document links whose label or path names a place this query is searching for.

    Only DISTINCTIVE words count: at least 4 letters, not a digit run, not generic, and not
    common to most of the collected labels (that filters out the company name and "ISO 9001"
    without needing to know the company). And EVERY distinctive word must be covered by the
    unfetched documents: one shared word ("Carolina", "United") is not enough, or a directory
    sweep over "North Carolina Georgia Tennessee" would be refused because of one South Carolina
    certificate. A wrongly refused search loses sites; a wrongly allowed one only costs a call."""
    if not doc_links:
        return []
    words_per_link = {url: _link_words(label, url) for url, label in doc_links.items()}
    share: dict[str, int] = {}
    for words in words_per_link.values():
        for w in words:
            share[w] = share.get(w, 0) + 1
    common = (
        {w for w, n in share.items() if n / len(words_per_link) > _GUARD_COMMON_WORD_SHARE}
        if len(words_per_link) >= _GUARD_MIN_LINKS_FOR_SHARE
        else set()
    )
    query_words = {
        w for w in _GUARD_WORD_RE.findall(_fold(query))
        if len(w) >= 4 and not w.isdigit() and w not in _GUARD_GENERIC_WORDS and w not in common
    }
    unfetched = {url: words for url, words in words_per_link.items() if url not in fetched}
    if not query_words or not query_words <= set().union(*unfetched.values()):
        return []
    # Most specific first: "Gunpo Korea" should list the Gunpo certificate before other Korean ones.
    hits = [(len(query_words & words), url) for url, words in unfetched.items() if query_words & words]
    return [url for _, url in sorted(hits, key=lambda h: -h[0])]


def _redirect_to_doc_links_message(query: str, urls: list[str]) -> str:
    listed = "\n".join(f"  fetch_page(\"{u}\")" for u in urls[:10])
    return (
        f"REFUSED (this cost nothing): {query!r} searches for a place that a document you were "
        "already shown covers -- a page you fetched links to it, and you have not opened it yet. "
        "Certificates and site documents carry the site's full address, and fetching them is free. "
        f"Fetch these first (in one batched turn):\n{listed}\n"
        "Search for this place only if those documents turn out not to give its address."
    )


def _condense_page_text(text: str, limit: int = _MAX_PAGE_CHARS, *, is_pdf: bool = False) -> str:
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

    # The links block is already capped and is exactly what a certificate index page is for, so
    # it is carried through whole, on top of the body's limit, and never condensed away.
    text, links_block = _split_doc_links(text)
    if links_block:
        return _condense_page_text(text, limit, is_pdf=is_pdf) + links_block

    lines = text.split("\n")
    keep: set[int] = set()
    for i, line in enumerate(lines):
        if _ADDRESS_HINT_RE.search(line):
            keep.update(range(max(0, i - _CONDENSE_CONTEXT_LINES), min(len(lines), i + _CONDENSE_CONTEXT_LINES + 1)))

    if keep:
        condensed = "\n".join(lines[i] for i in sorted(keep))
        what = "PDF" if is_pdf else "page"
        prefix = (
            f"[NOTE: this {what} is {len(text)} chars; non-address lines (nav/boilerplate) were "
            "stripped and only address-bearing lines kept. Nothing address-like was dropped.]\n\n"
        )
    else:
        condensed = text
        prefix = ""

    if len(prefix) + len(condensed) <= limit:
        return prefix + condensed

    kept = condensed[: limit - 400]
    # The recovery advice has to differ by kind: "fetch a deeper URL on this site" is right for a
    # website and meaningless for a document -- there is nothing deeper inside a PDF, and that
    # advice would send the agent back to guessing URLs or, worse, to a paid search.
    recovery = (
        "There is no deeper URL inside a document, so do NOT fetch this one again and do NOT "
        "search against it -- that is refused. If this certificate has per-region or per-site "
        "appendices published as separate PDFs on the same site, fetch those; otherwise report "
        "the addresses you can see here."
        if is_pdf
        else "Do NOT run a web_search against this URL to recover the rest -- that is refused. "
        "Instead fetch a deeper, more specific URL from this same site (e.g. a single country's "
        "or region's contact page)."
    )
    what = "PDF" if is_pdf else "page"
    return (
        f"[NOTE: this {what} is {len(text)} chars and had to be cut to fit; you are seeing the "
        f"first {len(kept)} chars of its address-bearing content. {recovery}]\n\n"
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
# A PDF whose text layer we extracted locally, for free. Tier-1 SUCCESS, not a failure reason:
# before this existed every PDF was bought from tier 3, which on one measured 24-page certificate
# returned 6 of its ~20 addresses and attributed 2 to it that are not in the document at all.
_FETCH_OK_PDF = "ok_pdf"
_FETCH_BLOCKED = "blocked"  # 403/captcha/interstitial, or a JS-only shell with no content in it
_FETCH_NON_TEXT = "non_text"  # a NON-PDF binary (image, archive, office doc). PDFs have their own
# reasons above and below; this is the residue no tier can read, so it skips tier 2.
# It IS a PDF, but it yielded no usable text: a scanned image, an encrypted file, or a malformed
# one. No render fixes any of those -- nothing OCRs a scan -- so it skips tier 2 exactly like a
# binary and goes to the paid tier, which is where ALL PDFs went before local parsing existed.
# That makes this path a strict no-regression.
_FETCH_PDF_NO_TEXT = "pdf_no_text"
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
    # Only set for _FETCH_PDF_NO_TEXT: "scanned" | "encrypted" | "parse_error". Defaulted so
    # every existing two-argument construction (including the tests') is unchanged.
    detail: str = ""


_SITEMAP_RE = re.compile(r"<(?:urlset|sitemapindex)\b", re.I)
_SITEMAP_LOC_RE = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>", re.I)
# Sitemap URLs worth a fetch for site research; everything else (products, blog posts) is noise.
_SITEMAP_USEFUL_RE = re.compile(
    r"locat|contact|kontakt|standort|office|plant|factory|facilit|site|imprint|impressum|about|"
    r"worldwide|global|region|countr|cert|quality|iso|sitemap|\.pdf",
    re.I,
)
_MAX_SITEMAP_URLS = 300


def _gleif_to_text(data: bytes) -> str | None:
    """GLEIF's LEI API (free, no key) as one line per legal entity: its registered and
    headquarters addresses. Raw, it is minified JSON a line-based condenser cannot work with."""
    import json

    try:
        records = json.loads(data.decode("utf-8")).get("data") or []
    except Exception:
        return None

    def fmt(addr: dict | None) -> str:
        addr = addr or {}
        parts = [*(addr.get("addressLines") or []), addr.get("city"), addr.get("region"),
                 addr.get("postalCode"), addr.get("country")]
        return ", ".join(p for p in parts if p)

    lines = [f"[GLEIF LEI REGISTER: {len(records)} legal entities -- check each name really belongs to the company]"]
    for rec in records:
        entity = (rec.get("attributes") or {}).get("entity") or {}
        name = (entity.get("legalName") or {}).get("name") or "?"
        status = entity.get("status") or ""
        lines.append(
            f"{name} ({status}) | registered: {fmt(entity.get('legalAddress'))} | "
            f"headquarters: {fmt(entity.get('headquartersAddress'))}"
        )
    return "\n".join(lines)


def _sitemap_to_text(xml: str) -> str:
    """A sitemap reduced to its location/contact/certificate URLs (and child sitemaps), one per
    line: a raw sitemap is often thousands of product URLs, which would bury the few that matter."""
    urls = _SITEMAP_LOC_RE.findall(xml)
    useful = [u for u in urls if _SITEMAP_USEFUL_RE.search(unquote(urlsplit(u).path))][:_MAX_SITEMAP_URLS]
    header = (
        f"[SITEMAP: {len(urls)} URLs, {len(useful)} of them look like locations, contact, about, "
        "certificate or child-sitemap pages -- fetch_page the relevant ones directly (free)]"
    )
    return header + "\n" + "\n".join(useful)


def _is_text_content_type(content_type: str) -> bool:
    # XML so sitemap.xml can be read: it lists a site's locations/contact pages without a search.
    return "text/html" in content_type or "text/plain" in content_type or "xml" in content_type


def _decode_bytes(data: bytes, declared_encoding: str | None) -> str:
    """requests' `.text` falls back to guessing latin-1 when a server doesn't declare a
    charset, which mangles UTF-8 pages (mojibake on curly quotes/apostrophes etc.) -- most
    modern pages ARE UTF-8 regardless of what the header says, so decode explicitly.

    Takes bytes rather than a response because the plain tier streams its body: under
    `stream=True` there is no `.content` to hand to `resp.apparent_encoding`.
    """
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode(declared_encoding or "utf-8", errors="replace")


def _decode_response(resp: "requests.Response") -> str:
    """Non-streamed convenience wrapper, used by the ScrapingBee tier, which proxies the
    target's bytes and so inherits the same latin-1 mojibake trap."""
    return _decode_bytes(resp.content, resp.apparent_encoding)


def _looks_like_pdf(data: bytes) -> bool:
    """Whether a response body IS a PDF -- decided by its bytes, never by Content-Type or a
    .pdf URL suffix, both of which lie in both directions.

    Content-Type: servers serve certificate PDFs as application/octet-stream,
    binary/octet-stream and application/force-download as readily as application/pdf.

    URL suffix: plenty of certificate PDFs sit behind extension-less CMS routes, and plenty of
    `.pdf` URLs return an HTML login wall or a JS redirect shell instead (a measured one returns
    269 bytes of text/html). Sniffing the body is what tells those apart.

    The header is supposed to sit at byte 0, but every real reader tolerates leading junk, so
    this scans the first KB the way pypdf's own recovery does.
    """
    return b"%PDF-" in data[:1024]


# OCR for scanned PDFs (7 of the 12 certificate PDFs AVX's run had to pay for were scans).
# RapidOCR + pypdfium2 are pip-only -- no system binary -- and run on CPU at ~6 s a page, so only
# the first few pages are read and the whole document gets a wall-clock ceiling. Certificates put
# the address on page 1; a long scanned bundle past the page cap still falls through to the paid
# tier rather than stalling a batch.
_OCR_MAX_PAGES = 3
_OCR_MAX_SECONDS = 20
_OCR_RENDER_DPI = 200
# Lower than _MIN_PDF_TEXT_CHARS: OCR output is text actually recognised on the page, not page
# furniture, and a one-page certificate can carry its whole address in ~150 characters.
_OCR_MIN_TEXT_CHARS = 80
_ocr_engine: Any = None
_ocr_lock = threading.Lock()


def _ocr_pdf(data: bytes) -> str | None:
    """Text of a scanned PDF's first pages via local OCR, or None if OCR is unavailable, finds
    too little text, or the document is unreadable. Never raises."""
    global _ocr_engine
    try:
        import pypdfium2 as pdfium
        from rapidocr_onnxruntime import RapidOCR
    except Exception:  # OCR is an optional speed-up; without it scans take the paid tier as before
        return None
    deadline = time.monotonic() + _OCR_MAX_SECONDS
    chunks: list[str] = []
    # One engine for the process, and one document at a time: the model load is ~1 s, and the
    # agent's batched fetches run in parallel threads that must not share an engine concurrently.
    with _ocr_lock:
        try:
            if _ocr_engine is None:
                _ocr_engine = RapidOCR()
            doc = pdfium.PdfDocument(data)
            for i in range(min(len(doc), _OCR_MAX_PAGES)):
                if time.monotonic() > deadline:
                    break
                image = doc[i].render(scale=_OCR_RENDER_DPI / 72).to_numpy()
                result, _ = _ocr_engine(image)
                lines = [r[1] for r in (result or []) if r and len(r) > 1 and r[1].strip()]
                if lines:
                    chunks.append(f"[page {i + 1}, OCR]\n" + "\n".join(lines))
        except Exception:
            return None
    text = _normalize_whitespace("\n\n".join(chunks))
    return text if len(text) >= _OCR_MIN_TEXT_CHARS else None


def _pdf_bytes_to_text(data: bytes) -> tuple[str | None, str]:
    """Extract a PDF's text layer locally, for free. Returns (text, detail); text is None when
    the document yielded nothing usable and `detail` says which kind of nothing.

    A document with no text layer (a scan) is OCR'd locally (see _ocr_pdf) and returned with
    detail "ocr"; only if that also fails does it fall through to the paid tier as before.
    """
    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted:
            # Many "protected" certificates carry only owner restrictions (no-print/no-copy)
            # with an EMPTY user password, which opens fine -- AES ones included, now that the
            # `cryptography` package is installed (5 of AVX's 12 paid PDFs were AES). A real user
            # password still returns 0 here.
            if not reader.decrypt(""):
                return None, "encrypted"
        pages = reader.pages[:_MAX_PDF_PAGES]
        total_pages = len(reader.pages)
    except Exception:
        # pypdf raises PdfReadError/EmptyFileError, but a corrupt xref also surfaces as
        # KeyError/struct.error/RecursionError. Same precedent as _html_to_text on malformed
        # HTML: bad input from the open web must not crash the tool.
        return None, "parse_error"

    chunks: list[str] = []
    for i, page in enumerate(pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception:
            continue  # one unparseable page must not cost the other 23
        if page_text.strip():
            # Explicit separator, never "".join: pypdf does not reliably newline-terminate a
            # page, so concatenating runs page N's last line into page N+1's first -- which
            # silently corrupts an address block straddling a page break, and those are common
            # in the multi-site certificate appendices this tier exists for.
            chunks.append(f"[page {i}]\n{page_text.strip()}")

    if total_pages > _MAX_PDF_PAGES:
        chunks.append(f"[NOTE: this PDF has {total_pages} pages; only the first "
                      f"{_MAX_PDF_PAGES} were read.]")

    text = _normalize_whitespace("\n\n".join(chunks))
    if len(text.strip()) < _MIN_PDF_TEXT_CHARS:
        ocr_text = _ocr_pdf(data)
        if ocr_text is not None:
            return ocr_text, "ocr"
        return None, "scanned"
    return text, ""


def _fetch_via_plain_http(url: str) -> _PlainFetch:
    """Tier 1. Returns extracted page text -- from HTML, or from a PDF's text layer -- or the
    reason it could not be had.

    The body is STREAMED rather than pulled in whole: `resp.content` would buffer a hostile
    200MB "PDF" in full before any size check could run, and streaming is also what makes a
    total wall-clock deadline possible (requests' own timeout only measures gaps between
    chunks). The `with` block is load-bearing for the same reason -- a streamed response
    abandoned mid-body by the size cap otherwise returns a poisoned connection to the pool.
    """
    deadline = time.monotonic() + _MAX_FETCH_WALL_SECONDS
    # A broken HTTPS setup (expired or mis-chained certificate, TLS alert) is common on small
    # suppliers' sites and defeats ScrapingBee too, so it used to go straight to a paid
    # retrieval. These are public pages read for their text, so one retry without certificate
    # verification is safe -- and free (redmangroup.cn: TLS error, readable without the check).
    for verify in (True, False):
        try:
            with requests.get(
                url,
                headers={"User-Agent": search_user_agent()},
                timeout=_FETCH_TIMEOUT_SECONDS,
                stream=True,
                verify=verify,
            ) as resp:
                # Checked before anything else: a 404/410 body is usually a short error page, which
                # the length heuristic in _looks_blocked would otherwise misread as a bot block and
                # escalate -- spending a render and a paid retrieval on a page that does not exist.
                if resp.status_code in (404, 410):
                    return _PlainFetch(None, _FETCH_NOT_FOUND)

                declared_length = resp.headers.get("Content-Length", "")
                if declared_length.isdigit() and int(declared_length) > _MAX_PDF_BYTES:
                    return _PlainFetch(None, _FETCH_NON_TEXT)

                body = bytearray()
                for chunk in resp.iter_content(chunk_size=64 * 1024):
                    body += chunk
                    if len(body) > _MAX_PDF_BYTES:
                        return _PlainFetch(None, _FETCH_NON_TEXT)
                    if time.monotonic() > deadline:
                        # A trickle-feeding origin. _FETCH_ERROR rather than _FETCH_NON_TEXT,
                        # because a proxy sometimes does fix a slow origin.
                        return _PlainFetch(None, _FETCH_ERROR)

                data = bytes(body)
                status_code = resp.status_code
                content_type = resp.headers.get("Content-Type", "")
                declared_encoding = resp.encoding
            break
        except requests.exceptions.SSLError:
            if verify:
                continue
            return _PlainFetch(None, _FETCH_ERROR)
        except requests.RequestException:
            return _PlainFetch(None, _FETCH_ERROR)

    # BYTES BEFORE HEADERS. A Content-Type lies in both directions -- a real PDF served as
    # text/html, an HTML login wall served as application/pdf -- and the magic number cannot.
    # This must run before the text branch so a mislabelled PDF never reaches _html_to_text,
    # which would hand the agent a page of binary noise.
    if _looks_like_pdf(data):
        text, detail = _pdf_bytes_to_text(data)
        if text is not None:
            return _PlainFetch(text, _FETCH_OK_PDF, detail)
        return _PlainFetch(None, _FETCH_PDF_NO_TEXT, detail)

    if "json" in content_type and urlsplit(url).netloc.lower().endswith("api.gleif.org"):
        gleif = _gleif_to_text(data)
        if gleif is not None:
            return _PlainFetch(gleif, _FETCH_OK)

    if not _is_text_content_type(content_type):
        return _PlainFetch(None, _FETCH_NON_TEXT)

    raw_text = _decode_bytes(data, declared_encoding)
    if "xml" in content_type and _SITEMAP_RE.search(raw_text[:2000]):
        return _PlainFetch(_sitemap_to_text(raw_text), _FETCH_OK)
    text = _html_to_text(raw_text, url) if "html" in content_type else raw_text
    if _looks_blocked(status_code, _page_body(text)):
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


def _fetch_via_scrapingbee(url: str) -> _PlainFetch:
    """Tier 2: re-fetch through ScrapingBee's headless browser and rotating proxy, which fixes
    the two things that defeat a plain GET -- bot blocking, and content that only exists after
    the page's JS runs.

    Returns the same _PlainFetch shape tier 1 does, so the caller can treat both free tiers
    alike: text with _FETCH_OK (a rendered page) or _FETCH_OK_PDF (the render turned out to be
    PDF bytes), or None with the reason it failed, in which case the caller escalates to the
    paid web_search fallback.
    """
    api_key = _scrapingbee_api_key()
    tracker = get_tracker()
    if api_key is None:
        if tracker is not None:
            tracker.record_scrapingbee_unavailable()
        return _PlainFetch(None, _FETCH_ERROR)

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
        return _PlainFetch(None, _FETCH_ERROR)

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
        return _PlainFetch(None, _FETCH_BLOCKED)

    # A render that comes back as PDF bytes is not a failure -- it is the document itself, and
    # the most common way to reach here is a `.pdf` URL that serves a JS redirect shell, so tier
    # 1 never saw the PDF at all. Parse it with the same local extractor tier 1 uses rather than
    # throwing away a render we have already paid credits for.
    if _looks_like_pdf(resp.content):
        pdf_text, detail = _pdf_bytes_to_text(resp.content)
        if pdf_text is not None:
            _record(None)
            return _PlainFetch(pdf_text, _FETCH_OK_PDF, detail)
        _record(f"pdf_no_text: {detail}")
        return _PlainFetch(None, _FETCH_PDF_NO_TEXT, detail)

    if not _is_text_content_type(resp.headers.get("Content-Type", "")):
        _record("non_text_content")
        return _PlainFetch(None, _FETCH_NON_TEXT)

    # ScrapingBee proxies the target's bytes, so it inherits the same latin-1 mojibake trap.
    text = _html_to_text(_decode_response(resp), url)
    if _looks_blocked(initial_status_code or 200, _page_body(text), js_rendered=True):
        _record("still_blocked_after_render")
        return _PlainFetch(None, _FETCH_BLOCKED)

    _record(None)
    return _PlainFetch(text, _FETCH_OK)


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


def _free_tier_result(
    url: str,
    fetched: _PlainFetch,
    note: str,
    tracker: Any,
    agent: str,
    *,
    rendered: bool,
) -> str:
    """Format a result from either FREE tier, and log a locally-parsed PDF under its own key.

    The synthetic `fetch_page_pdf_local` row mirrors `fetch_page_search_fallback`: it gives
    queries_log.jsonl a distinct row per PDF read for free, which is the direct measure of what
    this path displaces (194 of 412 logged paid retrievals were PDFs). It must never be recorded
    under the literal key "fetch_page" -- both the duplicate-fetch guard and the free-fetch count
    match on that name.
    """
    is_pdf = fetched.reason == _FETCH_OK_PDF
    is_ocr = is_pdf and fetched.detail == "ocr"
    if tracker is not None:
        # Linked and mentioned domains (the links block lists them) become fetchable.
        tracker.record_hosts(_hosts_in_text(fetched.text or ""))
    if tracker is not None and not is_pdf:
        links = _parse_doc_links(_split_doc_links(fetched.text or "")[1])
        tracker.record_doc_links(links)
        if any(_is_cert_evidence(u, label) for label, u in links):
            tracker.record_cert_evidence()
    if is_pdf and tracker is not None:
        tracker.record_query(agent, "fetch_page_pdf_ocr" if is_ocr else "fetch_page_pdf_local", url)
        tracker.record_pdf_extracted()

    if is_pdf:
        if is_ocr:
            how = (
                "scanned PDF, text read locally by OCR -- spellings and accents may be slightly "
                "off, so copy the address as printed and correct only obvious OCR slips"
            )
        elif rendered:
            how = "PDF, text extracted locally from a rendered fetch"
        else:
            how = "PDF, text extracted locally"
        header = f"Content of {url} ({how} -- this cost nothing)"
    elif rendered:
        header = f"Content of {url} (retrieved via rendered browser)"
    else:
        header = f"Content of {url}"
    return f"{header}:\n\n{_condense_page_text(fetched.text or '', is_pdf=is_pdf)}{note}"


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
    There is NO limit on how many pages you may fetch this way.

    PDFs are free too -- their text is extracted locally -- which makes a certification
    (ISO 9001/14001, IATF 16949, AS9100), permit or sustainability PDF one of the best-value
    fetches available: a single certificate appendix routinely lists every certified plant with
    its exact street address, dozens at a time. A URL that does not exist is reported as NOT
    FOUND and costs nothing. Only fetch domains you have SEEN -- in the input, a search result, or
    a fetched page (its links block lists the other websites it links to); a guessed domain is
    refused. Building a path on a domain you have seen is fine. Only a page that defeats both free routes, or a PDF that is a
    scanned image with no text layer, falls back to a PAID retrieval -- that fallback is the one
    budgeted part of this tool.
    """
    tracker = get_tracker()
    agent = get_current_agent()

    # Never a guessed domain: only one the input, a search result or a fetched page showed. Paths
    # on a known domain may still be built -- a wrong one is a free 404 and invents nothing.
    host = _norm_host(url)
    if tracker is not None and not _host_is_known(host, tracker.hosts()):
        tracker.record_fetch_refused_unknown_domain()
        return (
            f"NOT FETCHED (this cost nothing): the domain {host} has not appeared in the input, in "
            "any search result, or on any page you fetched, so it would be a guess. Never guess a "
            "company's domain from its name or a country pattern. If you need this company's site, "
            "find it with web_search, or use a domain a fetched page links to."
        )

    # No up-front budget gate: at this point the tier -- and therefore the cost -- is unknown,
    # and the large majority of fetches are free. Only the paid tier is budgeted, and it is
    # checked immediately before it runs, further down.
    already_fetched = tracker is not None and tracker.has_issued("fetch_page", url)
    if tracker is not None:
        tracker.record_query(agent, "fetch_page", url)

    # Reports the PAID budget, not a fetch count: what the agent needs to steer by is how many
    # retrievals it has left for pages no free tier can read, while free fetching is unlimited.
    paid_used = _calls_used("fetch_page_search_fallback")
    note = _budget_note(
        "paid fetch retrievals",
        paid_used,
        FETCH_PAGE_PAID_CALL_BUDGET,
        "Free fetches remain unlimited -- keep fetching pages and PDFs, but expect no more "
        "retrievals of pages that block us.",
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
        return _free_tier_result(url, plain, note, tracker, agent, rendered=False)

    # A guessed URL that does not exist stops here, costing nothing. Escalating a 404 would
    # spend a browser render and then a paid retrieval on a page no tier can produce -- which is
    # exactly what makes guessing sibling URLs cheap enough to be worth encouraging.
    if plain.reason == _FETCH_NOT_FOUND:
        return (
            f"NOT FOUND (this cost nothing): {url} returned 404/410 -- that page does not exist. "
            "If you were guessing a URL pattern, guess a different one, or fetch the site's "
            f"locator page and read the real link off it.{note}"
        )

    # A non-PDF binary, or a PDF tier 1 found has no text layer, is the one failure a headless
    # browser cannot fix -- nothing renders an archive into prose and nothing OCRs a scan -- so
    # those skip the middle tier rather than spending credits on it.
    no_text_pdf = plain if plain.reason == _FETCH_PDF_NO_TEXT else None
    if plain.reason not in (_FETCH_NON_TEXT, _FETCH_PDF_NO_TEXT):
        rendered = _fetch_via_scrapingbee(url)
        if rendered.text is not None:
            # Raw page text (or PDF text) like tier 1's, so it gets tier 1's treatment: the
            # condenser, not the fallback's larger cap below.
            return _free_tier_result(url, rendered, note, tracker, agent, rendered=True)
        # A render can be the first tier to SEE the PDF at all, when a .pdf URL serves a JS
        # shell that tier 1 read as a blocked HTML page. If that document turns out to have no
        # text layer, the warning below has to fire on its reason, not tier 1's.
        if rendered.reason == _FETCH_PDF_NO_TEXT:
            no_text_pdf = rendered

    # THE one budget in this tool, enforced here because here is the first point at which this
    # call's cost is known: both free tiers have now failed.
    # A site that defeats both free tiers on two URLs is unreadable, not unlucky: REDMAN's run
    # spent 7 of its 10 paid retrievals on URLs of one dead guessed domain. Past that, refuse free.
    if tracker is not None:
        host_paid = sum(
            1 for q in tracker.queries
            if q["tool"] == "fetch_page_search_fallback" and _norm_host(q["query"]) == host
        )
        if host_paid >= _MAX_PAID_FETCHES_PER_HOST:
            return (
                f"NOT RETRIEVED (this cost nothing): {host} could not be read by any free route on "
                f"{host_paid} earlier URLs, so no further paid retrievals are spent on it. Treat this "
                "site as unreadable: use other sources for its addresses (search results you already "
                f"have, registries, directories).{note}"
            )

    # Reserve atomically, re-reading the live count: `paid_used` above was read before the free
    # tiers ran, and parallel fetches in one turn all saw the same stale value (12 of 10 on AVX).
    reserved = (
        tracker.reserve_call(agent, "fetch_page_search_fallback", url, FETCH_PAGE_PAID_CALL_BUDGET)
        if tracker is not None
        else paid_used + 1
    )
    if reserved is None:
        paid_used = _calls_used("fetch_page_search_fallback")
        tracker.record_budget_block(agent)
        return (
            f"NOT RETRIEVED: {url} defeated both free routes, and the paid-retrieval budget is "
            f"spent ({paid_used}/{FETCH_PAGE_PAID_CALL_BUDGET} used). Free fetching is still "
            "unlimited, so fetch a different URL on this site (a country or region page, a "
            "certificate or permit PDF) instead, or answer with what you already have."
        )

    # reserve_call logged it under its own key, so it both gates this budget and shows up in
    # queries_log.jsonl as a distinct row -- that is how we can tell afterwards which URLs
    # actually needed paying for, i.e. whether the ScrapingBee tier is earning its place.

    # The fallback's output is already a dense, LLM-condensed address list (not noisy raw
    # HTML), and it was paid for in full -- truncating it back down to _MAX_PAGE_CHARS would
    # throw away most of what that call just retrieved, so it gets a much larger budget.
    text = _fetch_via_web_search_fallback(url)
    if tracker is not None:
        tracker.record_hosts(_hosts_in_text(text))
    pdf_note = ""
    if no_text_pdf is not None:
        kind = {
            "scanned": " (it is a scanned image)",
            "encrypted": " (it is password-protected)",
        }.get(no_text_pdf.detail, "")
        pdf_note = (
            f"\n\n[NOTE: this PDF has no machine-readable text layer{kind} -- what you see above "
            "is a search retrieval's reading of the document, not the document itself. Do NOT "
            "fetch this URL again, and treat any address here as weakly sourced.]"
        )
    # `note` was computed before this call; on the paid path the call itself has now been
    # spent, so the agent must be shown the post-spend count.
    paid_note = _budget_note(
        "paid fetch retrievals",
        paid_used + 1,
        FETCH_PAGE_PAID_CALL_BUDGET,
        "Free fetches remain unlimited -- keep fetching pages and PDFs, but expect no more "
        "retrievals of pages that block us.",
    )
    return (
        f"Content of {url} (retrieved via search fallback):\n\n"
        f"{text[:_MAX_FALLBACK_CHARS]}{pdf_note}{paid_note}"
    )


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
