"""Isolated, no-API-cost verification of the two cost guards in search_tool.py plus the page
condenser. Nothing here touches the network: every case either is refused before the API call
or operates on a literal string.

Each check corresponds to a measured failure mode from the ELLSWORTH runs in queries_log.jsonl:
  1. `site:<specific page> "<term>"` searches -- 43% of that company's 816 searches, aimed at
     only 65 distinct pages -- must be refused and pointed at fetch_page, for free.
  2. Calls past the per-tool budget must be refused (the prose budget was overrun ~3x).
  3. A long fetched page must keep its ADDRESS lines rather than its first N chars, which is
     what used to send the agent back to search for the half of a locator page it lost.
  4. fetch_page's three fetch tiers must escalate in the right order and, above all, must not
     spend ScrapingBee credits rendering a PDF -- the one failure a browser cannot fix.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used-no-network-calls-here")
# Popped, not defaulted: section 4 asserts the no-key path, which has to behave the same way on
# a developer machine that exports a real key as it does in a bare checkout.
os.environ.pop("SCRAPINGBEE_API_KEY", None)

from site_extraction_one_agent import search_tool as st  # noqa: E402
from site_extraction_one_agent.config import (  # noqa: E402
    EST_SCRAPINGBEE_COST_PER_CREDIT,
    FETCH_PAGE_CALL_BUDGET,
    WEB_SEARCH_CALL_BUDGET,
)
from site_extraction_one_agent.usage import start_tracking  # noqa: E402

tracker = start_tracking()

# --- 1. site:-with-a-path is refused; a real search is not ------------------------------------
DEEP_SITE_QUERIES = [
    'site:ellsworth.com/contact-us/global-locator "United Kingdom" address',
    'site:fisnar.com/contact-us-europe "Sweden" "Vastberga"',
    'site:ellsworth.com.my/Company/Contact.html "No. 35G"',
    "site:careers.ellsworth.com/divisions/",
    'site:ellsworthadhesives.co.uk/wp-content/uploads/2025/01/Cert.pdf "France"',
]
GENUINE_QUERIES = [
    "site:ellsworth.com locations Ellsworth Adhesives address",  # bare domain: a real search
    "Ellsworth Adhesives distribution center warehouse address",
    "site:ellsworth.com.cn Ellsworth address",
    '"Ellsworth Adhesives" "Carrollton" address',
]

for query in DEEP_SITE_QUERIES:
    out = st.web_search.invoke({"query": query})
    assert out.startswith("REFUSED"), out
    assert 'fetch_page("https://' in out, out
for query in GENUINE_QUERIES:
    assert st._DEEP_SITE_QUERY_RE.search(query) is None, f"real search wrongly refused: {query}"

assert tracker.web_searches_redirected_to_fetch == len(DEEP_SITE_QUERIES)
# A refusal never hits the network, so it must not spend budget or land in queries_log.jsonl --
# that log is the input to the cost analysis and has to stay a record of real traffic.
assert tracker.calls_used("web_search") == 0, tracker.tool_call_counts
assert tracker.queries == []
print(f"site:-with-path refused ({len(DEEP_SITE_QUERIES)}), genuine searches untouched: OK")

# --- 2. web_search is capped; fetch_page is NOT ------------------------------------------------
for i in range(WEB_SEARCH_CALL_BUDGET):
    tracker.record_query("site_extractor", "web_search", f"query {i}")
out = st.web_search.invoke({"query": "one search too many"})
assert out.startswith("REFUSED: web_search budget exhausted"), out
assert f"({WEB_SEARCH_CALL_BUDGET}/{WEB_SEARCH_CALL_BUDGET} used)" in out, out

for i in range(FETCH_PAGE_CALL_BUDGET):
    tracker.record_query("site_extractor", "fetch_page", f"https://example.com/{i}")
out = st.fetch_page.invoke({"url": "https://example.com/one-page-too-many"})
assert out.startswith("REFUSED: fetch_page budget exhausted"), out

assert tracker.tool_calls_blocked_by_budget == 2, tracker.tool_calls_blocked_by_budget
assert st._budget_note("web_search", 5, 35, "wrap up") == "\n\n[web_search budget: 5/35 used]"
assert "nearly spent" in st._budget_note("web_search", 28, 35, "wrap up")
print(f"budget refusals at {WEB_SEARCH_CALL_BUDGET} searches / {FETCH_PAGE_CALL_BUDGET} fetches: OK")

# --- 3. The condenser keeps addresses head-truncation would have thrown away -------------------
BOILERPLATE = "\n".join(["Navigation Home Products About Careers Legal Privacy Cookies"] * 900)
ADDRESS_BLOCKS = """
Ellsworth Adhesives Malaysia
No. 35G, Jalan Aman Tiara 3, Taman Perindustrian Aman
81300 Skudai, Johor, Malaysia
Tel: +60 7 559 1888

Ellsworth Adhesives Vietnam
5B, 86 Duy Tan Street, Cau Giay District
Hanoi, Vietnam
"""
page = BOILERPLATE + ADDRESS_BLOCKS
assert len(page) > st._MAX_PAGE_CHARS, "fixture must exceed the cap for this to mean anything"
assert "Jalan Aman Tiara 3" not in page[: st._MAX_PAGE_CHARS], "head-truncation would lose it"

condensed = st._condense_page_text(page)
assert len(condensed) <= st._MAX_PAGE_CHARS
for expected in ("Jalan Aman Tiara 3", "86 Duy Tan Street", "81300 Skudai, Johor, Malaysia", "Hanoi, Vietnam"):
    assert expected in condensed, expected  # incl. the digit-free country lines next to a street
print(f"condenser: {len(page)} chars -> {len(condensed)}, every address block kept: OK")

short = "Acme Ltd\n1 Main Street\nLondon"
assert st._condense_page_text(short) == short, "a page under the cap must pass through untouched"
print("short pages pass through untouched: OK")

# --- 4. fetch_page's three fetch tiers ---------------------------------------------------------
# Section 2 deliberately exhausted both call budgets, so anything invoking fetch_page from here
# on would just be refused -- this section needs its own tracker.
tracker = start_tracking()

assert st._is_text_content_type("text/html; charset=utf-8")
assert st._is_text_content_type("text/plain")
for non_text in ("application/pdf", "application/octet-stream", "image/png", ""):
    assert not st._is_text_content_type(non_text), non_text

# The JS-pending markers mean "a plain GET can't see this content" -- after a browser has run
# the page's JS they mean nothing, and treating them as a block would push good rendered pages
# on to the paid tier. Genuine blocks and empty pages still count in both modes.
JS_SHELL = "Loading map...\n" + ("Site navigation footer legal privacy cookies " * 20)
assert len(JS_SHELL) > 200, "fixture must clear the short-page check to test the markers"
assert st._looks_blocked(200, JS_SHELL)
assert not st._looks_blocked(200, JS_SHELL, js_rendered=True)
CAPTCHA = "Access denied. Please complete the captcha. " + ("filler text " * 40)
assert st._looks_blocked(200, CAPTCHA) and st._looks_blocked(200, CAPTCHA, js_rendered=True)
assert st._looks_blocked(403, JS_SHELL) and st._looks_blocked(403, JS_SHELL, js_rendered=True)
assert st._looks_blocked(200, "tiny") and st._looks_blocked(200, "tiny", js_rendered=True)
print("blocked-marker split (plain vs js_rendered): OK")

# Request shape. The key must never travel in a param dict, and mode=auto 400s if any of the
# explicit render/proxy flags are sent alongside it.
params = st._scrapingbee_params("https://example.com/locations?region=emea")
assert params["url"] == "https://example.com/locations?region=emea"
assert params["mode"] == "auto"
for forbidden in ("api_key", "render_js", "premium_proxy", "stealth_proxy", "transparent_status_code"):
    assert forbidden not in params, forbidden
# If the client gives up first we discard -- and still pay for -- a render already in flight.
assert st._SCRAPINGBEE_TIMEOUT_SECONDS * 1000 > st._SCRAPINGBEE_SERVER_TIMEOUT_MS


class _StubResponse:
    def __init__(self, headers, status_code=200):
        self.headers = headers
        self.status_code = status_code


assert st._scrapingbee_credits(_StubResponse({"Spb-auto-cost": "25"})) == 25
assert st._scrapingbee_credits(_StubResponse({"Spb-cost": "5"})) == 5
assert st._scrapingbee_credits(_StubResponse({"Spb-auto-cost": "1", "Spb-cost": "5"})) == 1
assert st._scrapingbee_credits(_StubResponse({}, status_code=500)) == 0  # unbilled status
assert st._scrapingbee_credits(_StubResponse({})) == st._SCRAPINGBEE_ASSUMED_CREDITS
assert st._scrapingbee_credits(_StubResponse({"Spb-cost": "n/a"})) == st._SCRAPINGBEE_ASSUMED_CREDITS
print("scrapingbee request shape + credit-header parsing: OK")

# Graceful degradation: with no key the tier returns without touching the network at all.
assert st._scrapingbee_api_key() is None
assert st._fetch_via_scrapingbee("https://example.com/locations") is None
assert tracker.scrapingbee_skipped_no_key == 1

# Tier routing, with all three tiers stubbed out so nothing reaches the network.
calls: list[str] = []
originals = (st._fetch_via_plain_http, st._fetch_via_scrapingbee, st._fetch_via_web_search_fallback)


def _run_tiers(plain_result, url=None):
    """Invoke fetch_page with the three tiers stubbed; returns (output, tiers reached).

    Each case gets a DISTINCT url by default -- fetch_page short-circuits a repeat fetch before
    any tier runs, so reusing one url across cases would silently test the duplicate guard
    instead of the tier order.
    """
    calls.clear()
    _run_tiers.n += 1
    url = url or f"https://example.com/locations/{_run_tiers.n}"

    def _plain(url):
        calls.append("plain")
        return plain_result

    def _bee(url):
        calls.append("scrapingbee")
        return _run_tiers.bee_text

    def _search(url):
        calls.append("web_search")
        return "Acme Ltd | Berlin, Germany | Musterstrasse 1"

    st._fetch_via_plain_http, st._fetch_via_scrapingbee, st._fetch_via_web_search_fallback = (
        _plain,
        _bee,
        _search,
    )
    try:
        return st.fetch_page.invoke({"url": url}), list(calls)
    finally:
        st._fetch_via_plain_http, st._fetch_via_scrapingbee, st._fetch_via_web_search_fallback = originals


_run_tiers.n = 0  # distinct url per case, so the duplicate guard never masks a tier-order check
PAGE = "Acme GmbH\nMusterstrasse 1\n10115 Berlin, Germany\nTel: +49 30 1234567"

_run_tiers.bee_text = None
out, reached = _run_tiers(st._PlainFetch(PAGE, st._FETCH_OK))
assert "Musterstrasse 1" in out and "retrieved via" not in out, out
assert reached == ["plain"], reached

# The credit-waste guard, stated as behaviour: a PDF must never reach ScrapingBee.
out, reached = _run_tiers(st._PlainFetch(None, st._FETCH_NON_TEXT))
assert "(retrieved via search fallback)" in out, out
assert reached == ["plain", "web_search"], reached

_run_tiers.bee_text = PAGE
out, reached = _run_tiers(st._PlainFetch(None, st._FETCH_BLOCKED))
assert "(retrieved via rendered browser)" in out and "Musterstrasse 1" in out, out
assert reached == ["plain", "scrapingbee"], reached

_run_tiers.bee_text = None
out, reached = _run_tiers(st._PlainFetch(None, st._FETCH_ERROR))
assert "(retrieved via search fallback)" in out, out
assert reached == ["plain", "scrapingbee", "web_search"], reached
print("tier order (plain -> scrapingbee -> web_search), PDFs skipping scrapingbee: OK")

# Re-fetching is the one fetch pattern still worth blocking: a second copy of a page is re-sent
# on every remaining turn. The guard must fire before any tier runs, so it costs nothing.
tracker = start_tracking()
_run_tiers.bee_text = None
DUP = "https://example.com/the-same-page"
out, reached = _run_tiers(st._PlainFetch(PAGE, st._FETCH_OK), DUP)
assert reached == ["plain"], reached
out, reached = _run_tiers(st._PlainFetch(PAGE, st._FETCH_OK), DUP)  # same URL a second time
assert out.startswith("ALREADY FETCHED"), out
assert reached == [], reached
print("duplicate-URL fetch short-circuits before any tier: OK")

# A guessed URL that 404s must cost nothing. Before this, the short 404 body tripped the
# length heuristic in _looks_blocked, so a wrong guess burned a browser render AND a paid
# retrieval -- which made the speculative fetching the prompt now encourages expensive.
tracker = start_tracking()
_run_tiers.bee_text = PAGE  # would succeed if reached; it must not be reached
out, reached = _run_tiers(st._PlainFetch(None, st._FETCH_NOT_FOUND))
assert out.startswith("NOT FOUND"), out
assert reached == ["plain"], reached
print("404 stops at tier 1, no render and no paid retrieval: OK")

# Cost accounting: credits x price, and a free failure adds nothing but the failure count.
tracker = start_tracking()
tracker.record_scrapingbee_usage("site_extractor", credits=5, succeeded=True)
assert round(tracker.estimated_cost_usd(), 6) == round(5 * EST_SCRAPINGBEE_COST_PER_CREDIT, 6)
tracker.record_scrapingbee_usage("site_extractor", credits=0, succeeded=False)
assert round(tracker.estimated_cost_usd(), 6) == round(5 * EST_SCRAPINGBEE_COST_PER_CREDIT, 6)
assert tracker.scrapingbee_calls == 2 and tracker.scrapingbee_failures == 1
usage = tracker.to_dict()
for key in ("scrapingbee_calls", "scrapingbee_credits", "scrapingbee_failures", "scrapingbee_skipped_no_key"):
    assert key in usage, key
assert usage["per_agent"]["site_extractor"]["scrapingbee_credits"] == 5
print("scrapingbee cost accounting (credits, failures, per-agent): OK")

print("\nALL SEARCH-GUARD CHECKS PASSED")
