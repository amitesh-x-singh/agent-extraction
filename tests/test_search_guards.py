"""Isolated, no-API-cost verification of the two cost guards in search_tool.py plus the page
condenser. Nothing here touches the network: every case either is refused before the API call
or operates on a literal string.

Each check corresponds to a measured failure mode from the ELLSWORTH runs in queries_log.jsonl:
  1. `site:<specific page> "<term>"` searches -- 43% of that company's 816 searches, aimed at
     only 65 distinct pages -- must be refused and pointed at fetch_page, for free.
  2. Calls past a PAID budget must be refused (the prose budget was overrun ~3x) -- while
     fetch_page's two FREE tiers stay uncapped, since fetching is the behaviour we want more of.
  3. A long fetched page must keep its ADDRESS lines rather than its first N chars, which is
     what used to send the agent back to search for the half of a locator page it lost.
  4. fetch_page's three fetch tiers must escalate in the right order; a readable PDF must be
     parsed locally in the free tier and never escalate at all; and a non-PDF binary, or a PDF
     with no text layer, must skip ScrapingBee -- no render turns an archive into prose or OCRs
     a scan.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used-no-network-calls-here")
# Popped, not defaulted: section 4 asserts the no-key path, which has to behave the same way on
# a developer machine that exports a real key as it does in a bare checkout.
os.environ.pop("SCRAPINGBEE_API_KEY", None)

from site_extraction_one_agent import search_tool as st  # noqa: E402
from site_extraction_one_agent.config import (  # noqa: E402
    SCRAPINGBEE_COST_PER_CREDIT,
    FETCH_PAGE_PAID_CALL_BUDGET,
    WEB_SEARCH_CALL_BUDGET,
)
from site_extraction_one_agent.usage import start_tracking as _start_tracking  # noqa: E402

# fetch_page refuses domains the agent has not seen. These cases test other behaviour on made-up
# domains, so they are registered as seen up front; the domain guard has its own test below.
_TEST_HOSTS = {"example.com", "dead-site.example", "other-site.example", "acme.example"} | {
    f"example{i}.com" for i in range(500)
}


def start_tracking():
    tracker = _start_tracking()
    tracker.record_hosts(_TEST_HOSTS)
    return tracker


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

# --- 2. web_search is capped, and so is fetch_page's PAID tier -- but not its free ones --------
for i in range(WEB_SEARCH_CALL_BUDGET):
    tracker.record_query("site_extractor", "web_search", f"query {i}")
out = st.web_search.invoke({"query": "one search too many"})
assert out.startswith("REFUSED: web_search budget exhausted"), out
assert f"({WEB_SEARCH_CALL_BUDGET}/{WEB_SEARCH_CALL_BUDGET} used)" in out, out

# Free fetching is UNLIMITED: what costs money is the paid retrieval tier, not the call. Well
# past any previous cap, a fetch must still be attempted rather than refused up front -- proven
# here by the duplicate guard answering instead, which only runs AFTER the budget check it
# replaced. (_run_tiers below proves the same thing without the network.)
for i in range(200):
    tracker.record_query("site_extractor", "fetch_page", f"https://example.com/{i}")
out = st.fetch_page.invoke({"url": "https://example.com/7"})
assert not out.startswith("REFUSED"), out
assert out.startswith("ALREADY FETCHED"), out

assert tracker.tool_calls_blocked_by_budget == 1, tracker.tool_calls_blocked_by_budget
assert st._budget_note("web_search", 5, 35, "wrap up") == "\n\n[web_search budget: 5/35 used]"
assert "nearly spent" in st._budget_note("web_search", 28, 35, "wrap up")
print(f"budget refusals at {WEB_SEARCH_CALL_BUDGET} searches; free fetching uncapped: OK")

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

# --- 3b. Local PDF text extraction: the free tier's new job -----------------------------------
# A PDF is identified by BYTES, never by Content-Type or URL suffix -- servers serve certificate
# PDFs as octet-stream/force-download, and plenty of `.pdf` URLs are JS redirect shells.
assert st._looks_like_pdf(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj")
assert st._looks_like_pdf(b"\n\n   %PDF-1.4\n")  # leading junk, as every real reader tolerates
assert not st._looks_like_pdf(b"<!DOCTYPE html><html><head><meta http-equiv=refresh")
# The measured micron.com shape: a 269-byte text/html JS wrapper served at a .pdf URL. It is NOT
# a PDF and must not be treated as one -- it stays on the blocked -> render -> fallback path.
assert not st._looks_like_pdf(b"<html><script>location.href='/real.pdf'</script></html>")


def _make_pdf(pages):
    """Smallest legal PDF with one uncompressed text stream per page, with REAL xref offsets --
    so these checks exercise our extraction, not pypdf's damaged-file recovery."""
    objs = []
    n = len(pages)
    page_ids = [4 + 2 * i for i in range(n)]
    content_ids = [5 + 2 * i for i in range(n)]
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(b"<< /Type /Pages /Kids [" + b" ".join(b"%d 0 R" % p for p in page_ids)
                + b"] /Count %d >>" % n)
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i, body_text in enumerate(pages):
        objs.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources "
                    b"<< /Font << /F1 3 0 R >> >> /Contents %d 0 R >>" % content_ids[i])
        # No literal ( or ) in fixture text -- they would need PDF string escaping.
        shown = b"".join(b"(%s) Tj T*\n" % ln.encode("latin-1") for ln in body_text.split("\n"))
        stream = b"BT /F1 12 Tf 14 TL 50 700 Td\n" + shown + b"ET"
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")

    out_bytes, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out_bytes))
        out_bytes += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref_at = len(out_bytes)
    out_bytes += b"xref\n0 %d\n" % (len(objs) + 1) + b"0000000000 65535 f \n"
    for off in offsets:
        out_bytes += b"%010d 00000 n \n" % off
    out_bytes += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objs) + 1, xref_at)
    return bytes(out_bytes)


# A two-page certificate appendix: the shape this whole tier exists for, one site per page,
# padded past _MIN_PDF_TEXT_CHARS the way a real certificate's boilerplate pads it.
FILLER = "\n".join(["ISO 14001 2015 certificate of registration scope of supply"] * 4)
CERT_PDF = _make_pdf([
    "Certificate No. 12345\nAcme GmbH Werk Nord\nMusterstrasse 1\n10115 Berlin Germany\n" + FILLER,
    "Acme Malaysia Sdn Bhd\nNo. 35G Jalan Aman Tiara 3\n81300 Skudai Johor Malaysia\n" + FILLER,
])
assert st._looks_like_pdf(CERT_PDF)
pdf_text, detail = st._pdf_bytes_to_text(CERT_PDF)
assert detail == "", detail
assert "Musterstrasse 1" in pdf_text and "Jalan Aman Tiara 3" in pdf_text, pdf_text
# Pages are joined with an EXPLICIT separator: pypdf does not reliably newline-terminate a page,
# so a naive concat runs page N's last line into page N+1's first -- which silently corrupts an
# address block straddling a page break, common in multi-site certificate appendices.
assert "[page 1]" in pdf_text and "[page 2]" in pdf_text, pdf_text
assert "Germany" in pdf_text.split("[page 2]")[0], "page 1's last line leaked into page 2"
# It stays line-based, because _condense_page_text works line by line.
assert "10115 Berlin Germany" in pdf_text.split("\n"), pdf_text

# A scan: a valid PDF with pages but no text layer. No OCR here (a system binary and seconds per
# page), so it reports itself and falls through to the paid tier -- where every PDF went before
# this existed, which makes that path a strict no-regression.
assert st._pdf_bytes_to_text(_make_pdf(["", ""])) == (None, "scanned")
# Page furniture alone is below the same 200-char floor _looks_blocked uses for a short page.
assert st._pdf_bytes_to_text(_make_pdf(["Page 1 of 2", "Page 2 of 2"]))[0] is None

# Malformed input must not crash the tool -- the same rule _html_to_text follows for bad HTML.
assert st._pdf_bytes_to_text(b"%PDF-1.4\nnot actually a pdf at all") == (None, "parse_error")
assert st._pdf_bytes_to_text(b"") == (None, "parse_error")

# Guards sized against the real corpus: the largest measured certificate bundle is 3.7MB/24 pages.
assert st._MAX_PDF_BYTES > 4 * 1024 * 1024
assert st._MAX_PDF_PAGES > 24
assert st._MIN_PDF_TEXT_CHARS == 200  # same floor as _looks_blocked's short-page check

# The truncation note must not tell the agent to "fetch a deeper URL from this same site" --
# there is nothing deeper inside a document, and that advice sends it back to guessing URLs.
LONG_PDF_TEXT = "\n".join(["Acme GmbH", "Musterstrasse 1", "10115 Berlin Germany"] * 4000)
cut = st._condense_page_text(LONG_PDF_TEXT, is_pdf=True)
assert "fetch a deeper, more specific URL" not in cut, cut[:400]
assert "single country" not in cut, cut[:400]
assert "no deeper URL inside a document" in cut, cut[:400]
assert "this PDF is" in cut, cut[:200]
print("local PDF extraction (magic bytes, page joins, scans, malformed, caps): OK")

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

# Graceful degradation: with no key the tier returns without touching the network at all. It
# reports its failure in the same _PlainFetch shape tier 1 uses, so the caller escalates.
assert st._scrapingbee_api_key() is None
assert st._fetch_via_scrapingbee("https://example.com/locations").text is None
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
    # A distinct HOST too: fetch_page caps paid retrievals per host, which is tested on its own.
    url = url or f"https://example{_run_tiers.n}.com/locations"

    def _plain(url):
        calls.append("plain")
        return plain_result

    def _bee(url):
        calls.append("scrapingbee")
        # Tier 2 now returns the same _PlainFetch shape tier 1 does, so it can report PDF bytes
        # a render turned up rather than only page text.
        if isinstance(_run_tiers.bee_text, st._PlainFetch):
            return _run_tiers.bee_text
        if _run_tiers.bee_text is None:
            return st._PlainFetch(None, st._FETCH_BLOCKED)
        return st._PlainFetch(_run_tiers.bee_text, st._FETCH_OK)

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

# The credit-waste guard, stated as behaviour: a NON-PDF binary must never reach ScrapingBee.
out, reached = _run_tiers(st._PlainFetch(None, st._FETCH_NON_TEXT))
assert "(retrieved via search fallback)" in out, out
assert reached == ["plain", "web_search"], reached

# A readable PDF now stops at tier 1 for free -- no render, and above all no paid retrieval.
# That is the whole point: 194 of 412 logged paid retrievals were PDFs.
_run_tiers.bee_text = PAGE  # would succeed if reached; it must not be reached
out, reached = _run_tiers(st._PlainFetch("[page 1]\nAcme GmbH\nMusterstrasse 1", st._FETCH_OK_PDF))
assert reached == ["plain"], reached
assert "Musterstrasse 1" in out and "cost nothing" in out, out
assert "retrieved via" not in out, out

# A PDF with no text layer still skips ScrapingBee -- no render OCRs a scan -- and still reaches
# the paid tier, exactly as every PDF did before. It must say WHY, so the agent does not spend
# another fetch on it.
out, reached = _run_tiers(st._PlainFetch(None, st._FETCH_PDF_NO_TEXT, "scanned"))
assert reached == ["plain", "web_search"], reached
assert "no machine-readable text layer" in out and "scanned image" in out, out

# A render that comes back as PDF bytes (a .pdf URL that was really a JS redirect shell) is
# parsed by the same local extractor rather than thrown away for a paid retrieval.
_run_tiers.bee_text = st._PlainFetch("[page 1]\nAcme GmbH\nMusterstrasse 1", st._FETCH_OK_PDF)
out, reached = _run_tiers(st._PlainFetch(None, st._FETCH_BLOCKED))
assert reached == ["plain", "scrapingbee"], reached
assert "Musterstrasse 1" in out and "cost nothing" in out, out
_run_tiers.bee_text = None

_run_tiers.bee_text = PAGE
out, reached = _run_tiers(st._PlainFetch(None, st._FETCH_BLOCKED))
assert "(retrieved via rendered browser)" in out and "Musterstrasse 1" in out, out
assert reached == ["plain", "scrapingbee"], reached

_run_tiers.bee_text = None
out, reached = _run_tiers(st._PlainFetch(None, st._FETCH_ERROR))
assert "(retrieved via search fallback)" in out, out
assert reached == ["plain", "scrapingbee", "web_search"], reached
print("tier order (plain -> scrapingbee -> web_search), PDFs parsed free, scans skipping "
      "scrapingbee: OK")

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

# The ONE budget in fetch_page is on the paid tier, and it is checked only after both free
# tiers have failed -- so a spent paid budget must still leave free fetching working.
tracker = start_tracking()
_run_tiers.bee_text = None
for i in range(FETCH_PAGE_PAID_CALL_BUDGET):
    tracker.record_query("site_extractor", "fetch_page_search_fallback", f"https://example.com/p{i}")
out, reached = _run_tiers(st._PlainFetch(None, st._FETCH_BLOCKED))
assert out.startswith("NOT RETRIEVED"), out
assert f"({FETCH_PAGE_PAID_CALL_BUDGET}/{FETCH_PAGE_PAID_CALL_BUDGET} used)" in out, out
assert reached == ["plain", "scrapingbee"], reached  # both FREE tiers still ran
assert tracker.tool_calls_blocked_by_budget == 1, tracker.tool_calls_blocked_by_budget
_run_tiers.bee_text = PAGE
out, reached = _run_tiers(st._PlainFetch(PAGE, st._FETCH_OK))
assert "Musterstrasse 1" in out and reached == ["plain"], out
print("paid-retrieval budget capped, free tiers unaffected: OK")

# A host that defeated both free tiers twice gets no third paid retrieval (REDMAN spent 7 on one).
tracker = start_tracking()
_run_tiers.bee_text = None
for path in ("a", "b"):
    tracker.record_query("site_extractor", "fetch_page_search_fallback", f"https://www.dead-site.example/{path}")
out, reached = _run_tiers(st._PlainFetch(None, st._FETCH_BLOCKED), "https://dead-site.example/contact")
assert out.startswith("NOT RETRIEVED (this cost nothing)") and "web_search" not in reached, (out, reached)
assert tracker.calls_used("fetch_page_search_fallback") == 2
out, reached = _run_tiers(st._PlainFetch(None, st._FETCH_BLOCKED), "https://other-site.example/contact")
assert reached == ["plain", "scrapingbee", "web_search"], reached  # other hosts unaffected
print("per-host paid cap: dead site stops at 2 paid retrievals, others unaffected: OK")

# The free PDF path logs its own synthetic row, so queries_log.jsonl shows how much paid
# retrieval it displaced -- without confusing the duplicate guard, which keys on "fetch_page".
tracker = start_tracking()
_run_tiers.bee_text = None
PDF_URL = "https://example.com/iso-14001-certificate.pdf"
out, reached = _run_tiers(
    st._PlainFetch("[page 1]\nAcme GmbH\nMusterstrasse 1", st._FETCH_OK_PDF), PDF_URL
)
assert reached == ["plain"], reached
assert tracker.calls_used("fetch_page") == 1, tracker.tool_call_counts
assert tracker.calls_used("fetch_page_pdf_local") == 1, tracker.tool_call_counts
assert tracker.calls_used("fetch_page_search_fallback") == 0, tracker.tool_call_counts
assert tracker.pdfs_extracted_locally == 1
assert tracker.has_issued("fetch_page", PDF_URL)  # duplicate guard still armed
out, reached = _run_tiers(st._PlainFetch(None, st._FETCH_OK_PDF), PDF_URL)
assert out.startswith("ALREADY FETCHED") and reached == [], out
assert "pdfs_extracted_locally" in tracker.to_dict()
print("free-PDF accounting: own log row, duplicate guard intact: OK")

# Cost accounting: credits x price, and a free failure adds nothing but the failure count.
tracker = start_tracking()
tracker.record_scrapingbee_usage("site_extractor", credits=5, succeeded=True)
assert round(tracker.cost_usd(), 6) == round(5 * SCRAPINGBEE_COST_PER_CREDIT, 6)
tracker.record_scrapingbee_usage("site_extractor", credits=0, succeeded=False)
assert round(tracker.cost_usd(), 6) == round(5 * SCRAPINGBEE_COST_PER_CREDIT, 6)
assert tracker.scrapingbee_calls == 2 and tracker.scrapingbee_failures == 1
usage = tracker.to_dict()
for key in ("scrapingbee_calls", "scrapingbee_credits", "scrapingbee_failures", "scrapingbee_skipped_no_key"):
    assert key in usage, key
assert usage["per_agent"]["site_extractor"]["scrapingbee_credits"] == 5
print("scrapingbee cost accounting (credits, failures, per-agent): OK")

# --- 5. certificate/document links survive HTML-to-text --------------------------------------
# A certificate library lists its PDFs only as link targets; dropping hrefs hid every one of them.
CERT_INDEX = (
    "<html><body><h1>Certificates</h1><ul>"
    '<li><a href="/docs/certificates/Europe/Germany/ISO_14001_Betzdorf.pdf">ISO 14001 Betzdorf</a></li>'
    '<li><a href="/quality/iatf">IATF 16949 Suzhou</a></li>'
    '<li><a href="/contact">Contact us</a></li>'
    '<li><a href="mailto:q@example.com">ISO 9001 questions</a></li>'
    '<li><a href="https://x.example/track?iso9001=1">Newsletter</a></li>'
    "</ul></body></html>"
)
linked = st._html_to_text(CERT_INDEX, "https://www.example.com/about/certificates/")
body, block = st._split_doc_links(linked)
assert st._DOC_LINKS_HEADER in block, linked
assert "https://www.example.com/docs/certificates/Europe/Germany/ISO_14001_Betzdorf.pdf" in block
assert "https://www.example.com/quality/iatf" in block  # matched on anchor text
assert "/contact" not in block and "mailto:" not in block
assert "track?iso9001" not in block  # query strings don't qualify a link
assert "ISO 14001 Betzdorf" in body and st._DOC_LINKS_HEADER not in body

# A whole certificate library fits: a cap of 20 once hid 99 of AVX's 119 certificates.
assert st._MAX_DOC_LINKS >= 150
many = "".join(f'<a href="/c/cert-{i}.pdf">c{i}</a>' for i in range(st._MAX_DOC_LINKS + 40))
assert st._html_to_text(many, "https://e.com/").count(".pdf") == st._MAX_DOC_LINKS

# The links block survives condensing whole, on top of the body's own cap.
LONG_PAGE = CERT_INDEX.replace("</ul>", "</ul>" + "<p>marketing copy without any address</p>" * 3000)
long_text = st._html_to_text(LONG_PAGE, "https://www.example.com/")
assert len(long_text) > st._MAX_PAGE_CHARS
cond = st._condense_page_text(long_text)
assert cond.endswith(st._split_doc_links(long_text)[1])
assert len(st._split_doc_links(cond)[0]) <= st._MAX_PAGE_CHARS

# A blocked page is judged on its body alone: footer PDF links must not make it look real.
assert st._looks_blocked(200, st._page_body(st._html_to_text(
    '<p>Access denied</p><a href="/a.pdf">a</a><a href="/b.pdf">b</a>' * 1, "https://e.com/")))
print("document links: kept, resolved, filtered, capped, survive condensing: OK")

# --- 6. known-document guard: no paid search for a place a shown document already covers ------
CERT_LIBRARY = {
    f"https://acme.example/docs/certificates/{region}/{name}.pdf": f"ACME | {city}: ISO 9001:2015"
    for region, name, city in [
        ("Europe/Czech_Republic", "ISO_9001_Lanskroun", "Lanskroun"),
        ("Europe/Czech_Republic", "ISO-TS_22163_Uherske", "Uherské Hradiště"),
        ("Asia/China", "ISO-9001-Tianjin", "Tianjin"),
        ("Asia/China", "ISO-9001-Chengdu", "Chengdu"),
        ("North_America/Maine", "ISO_9001_biddeford", "Biddeford, ME"),
        ("Europe/France", "ISO9001-St-Apollinaire", "Saint-Apollinaire"),
    ]
}
LANSKROUN = "https://acme.example/docs/certificates/Europe/Czech_Republic/ISO_9001_Lanskroun.pdf"
hit = st._known_doc_links_for_query("ACME Lanskroun factory address", CERT_LIBRARY, set())
assert hit == [LANSKROUN], hit
# Accents fold, and the city matches via the label as well as the path.
assert st._known_doc_links_for_query("ACME Uherske Hradiste plant", CERT_LIBRARY, set())
# Once the document is fetched it no longer blocks a follow-up search.
assert st._known_doc_links_for_query("ACME Lanskroun factory address", CERT_LIBRARY, {LANSKROUN}) == []
# The company name and standard names appear on every label, so they never trigger it...
assert st._known_doc_links_for_query("ACME ISO 9001 certificate", CERT_LIBRARY, set()) == []
assert st._known_doc_links_for_query("ACME official global locations", CERT_LIBRARY, set()) == []
# ...and a place no document covers is searched normally.
assert st._known_doc_links_for_query("ACME Penang factory address", CERT_LIBRARY, set()) == []
assert st._known_doc_links_for_query("ACME Lanskroun", {}, set()) == []
# One shared word is not enough: a multi-region sweep naming places no document covers still runs.
assert st._known_doc_links_for_query("ACME plant Lanskroun Penang Georgia", CERT_LIBRARY, set()) == []

# End to end through the tool: refused for free, counted, and no API call attempted.
tracker = start_tracking()
tracker.record_doc_links([(label, url) for url, label in CERT_LIBRARY.items()])
out = st.web_search.invoke({"query": "ACME Tianjin factory address"})
assert out.startswith("REFUSED (this cost nothing)") and "ISO-9001-Tianjin.pdf" in out, out
assert tracker.calls_used("web_search") == 0
assert tracker.to_dict()["web_searches_redirected_to_doc_link"] == 1
print("known-document guard: refuses free, folds accents, ignores company/standard words: OK")

# --- 7. no guessed domains: fetch_page opens only domains the agent has seen --------------------
tracker = _start_tracking()  # nothing seen yet
_run_tiers.bee_text = None
out, reached = _run_tiers(st._PlainFetch(PAGE, st._FETCH_OK), "https://www.redmancorp.com/contact")
assert out.startswith("NOT FETCHED (this cost nothing)") and reached == [], (out, reached)
assert tracker.to_dict()["web_fetches_refused_unknown_domain"] == 1
# Registries in the prompt's templates are known in advance.
out, reached = _run_tiers(st._PlainFetch(PAGE, st._FETCH_OK), "https://api.gleif.org/api/v1/lei-records?filter[fulltext]=x")
assert reached == ["plain"], out
# The input domain (seeded by the CLI/API) and its subdomains and paths are allowed.
tracker.record_hosts([st._norm_host("https://www.Acme.co.uk/")])
for url in ("https://acme.co.uk/de/kontakt", "https://careers.acme.co.uk/jobs"):
    out, reached = _run_tiers(st._PlainFetch(PAGE, st._FETCH_OK), url)
    assert reached == ["plain"], (url, out)
# A page's links block and e-mail addresses make the domains they name fetchable.
linked = st._html_to_text(
    '<p>Contact info@acme-asia.com.sg</p><a href="https://www.acme.de/kontakt">Germany</a>'
    '<a href="https://facebook.com/acme">fb</a>',
    "https://acme.co.uk/worldwide",
)
assert "Other websites this page links to: acme.de" in linked and "facebook" not in linked, linked
tracker.record_hosts(st._hosts_in_text(linked))
for url in ("https://www.acme.de/impressum", "https://acme-asia.com.sg/contact"):
    out, reached = _run_tiers(st._PlainFetch(PAGE, st._FETCH_OK), url)
    assert reached == ["plain"], (url, out)
# A country TLD the agent was never shown is still a guess.
out, reached = _run_tiers(st._PlainFetch(PAGE, st._FETCH_OK), "https://acme.fr/contact")
assert out.startswith("NOT FETCHED") and reached == [], out
assert st._norm_host("HTTPS://WWW.Acme.co.uk:443/x?y=1") == "acme.co.uk"
print("no guessed domains: unseen refused free; input, links, e-mails, registries allowed: OK")

print("\nALL SEARCH-GUARD CHECKS PASSED")
