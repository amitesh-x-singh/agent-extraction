"""No-API-cost checks for the cost cuts: local OCR and AES decryption for certificate PDFs, the
near-duplicate and certificate-gate search guards, the atomic paid-budget reservation, sitemap and
GLEIF readers, and the 3+-sources confidence promotion. Nothing here touches the network: every
guarded search is refused before the API call, and every document is generated in memory.
"""

import io
import os
import threading

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used-no-network-calls-here")

from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from pypdf import PdfReader, PdfWriter  # noqa: E402

from site_extraction_one_agent import search_tool as st  # noqa: E402
from site_extraction_one_agent.results_csv import confidence_for  # noqa: E402
from site_extraction_one_agent.usage import start_tracking  # noqa: E402

# --- 1. scanned certificate: OCR reads an image-only PDF locally ------------------------------
ADDRESS_LINES = [
    "CERTIFICATE ISO 9001:2015",
    "ACME Components GmbH",
    "Musterstrasse 12",
    "10115 Berlin, Germany",
    "Scope: design and manufacture of capacitors",
    "Certificate number 12345-QMS valid until 2027",
]
image = Image.new("RGB", (1700, 1100), "white")
draw = ImageDraw.Draw(image)
try:
    font = ImageFont.truetype("arial.ttf", 48)
except OSError:
    font = ImageFont.load_default(size=48)
for i, line in enumerate(ADDRESS_LINES):
    draw.text((80, 80 + i * 140), line, fill="black", font=font)
scan = io.BytesIO()
image.save(scan, format="PDF", resolution=150)
text, detail = st._pdf_bytes_to_text(scan.getvalue())
assert detail == "ocr", (detail, text)
assert "Berlin" in text and "Musterstrasse" in text, text
# And through the tool's result formatter, which is where the first live OCR'd PDF crashed.
tracker = start_tracking()
result = st._free_tier_result(
    "https://acme.com/cert.pdf", st._PlainFetch(text, st._FETCH_OK_PDF, "ocr"), "", tracker, "site_extractor", rendered=False
)
assert "read locally by OCR" in result and "Berlin" in result, result[:300]
assert tracker.calls_used("fetch_page_pdf_ocr") == 1
for rendered in (False, True):  # the plain and rendered PDF headers still work
    st._free_tier_result("https://acme.com/a.pdf", st._PlainFetch(text, st._FETCH_OK_PDF, ""), "", None, "x", rendered=rendered)
print("scanned PDF read by local OCR, formatted into the tool result: OK")

# --- 2. AES-encrypted certificate with an empty user password opens locally --------------------
def _text_pdf(line: bytes, repeat: int) -> bytes:
    """A minimal valid one-page PDF with a real text layer (objects plus a correct xref table)."""
    stream = b"BT /F1 12 Tf 72 720 Td " + b" ".join(b"(" + line + b") Tj 0 -16 Td" for _ in range(repeat)) + b" ET"
    objects = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>",
        b"<</Length %d>>stream\n" % len(stream) + stream + b"\nendstream",
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj" % number + body + b"endobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer<</Size %d/Root 1 0 R>>\nstartxref\n%d\n%%%%EOF" % (len(objects) + 1, xref)
    return bytes(out)


PLAIN_PDF = _text_pdf(b"ACME Components GmbH, Musterstrasse 12, 10115 Berlin, Germany - ISO 9001", 6)
writer = PdfWriter(clone_from=PdfReader(io.BytesIO(PLAIN_PDF)))
writer.encrypt(user_password="", owner_password="owner-secret", algorithm="AES-256")
encrypted = io.BytesIO()
writer.write(encrypted)
assert PdfReader(io.BytesIO(encrypted.getvalue())).is_encrypted
text, detail = st._pdf_bytes_to_text(encrypted.getvalue())
assert text is not None and "Berlin" in text, (detail, text)
print("AES-encrypted PDF (empty user password) read locally: OK")

# --- 3. near-duplicate guard -----------------------------------------------------------------
earlier = ["ACME Ohio office address", "ACME official global locations offices"]
assert st._near_duplicate_of("ACME offices Ohio addresses", earlier) == "ACME Ohio office address"
assert st._near_duplicate_of("ACME global locations official", earlier) is not None
assert st._near_duplicate_of("ACME Texas office address", earlier) is None  # different place
assert st._near_duplicate_of("ACME warehouse Ohio", earlier) is None  # different site type
tracker = start_tracking()
tracker.record_query("site_extractor", "web_search", "ACME Ohio office address")
out = st.web_search.invoke({"query": "ACME offices in Ohio address"})
assert out.startswith("REFUSED (this cost nothing)") and "almost the same" in out, out
assert tracker.calls_used("web_search") == 1 and tracker.to_dict()["web_searches_refused_duplicate"] == 1
print("near-duplicate searches refused for free, other places allowed: OK")

# --- 4. certificate gate ---------------------------------------------------------------------
assert st._CERT_QUERY_RE.search('"ACME" ISO 9001 OR IATF 16949 certificate')
assert not st._CERT_QUERY_RE.search("ACME plant Ohio address")
assert st._is_cert_evidence("https://acme.com/docs/certificates/ISO_9001_Berlin.pdf")
assert st._is_cert_evidence("https://www.tuvsud.com/en/certificates/acme")
assert not st._is_cert_evidence("https://acme.com/contact")
tracker = start_tracking()
tracker.record_cert_search(found_evidence=False)  # the one probe ran and found nothing
out = st.web_search.invoke({"query": "ACME IATF 16949 certificate Germany plant"})
assert out.startswith("REFUSED (this cost nothing)") and "no certificate" in out, out
assert tracker.calls_used("web_search") == 0
assert tracker.to_dict()["web_searches_refused_no_certificates"] == 1
# Non-certificate searches are unaffected by a closed certificate gate.
assert not st._CERT_QUERY_RE.search("ACME warehouse Texas")
# Evidence opens the gate, and the hard cap still closes it.
tracker.record_cert_evidence()
assert tracker.cert_evidence_seen
tracker.cert_searches = st._MAX_CERT_SEARCHES
out = st.web_search.invoke({"query": "ACME ISO 14001 certificate"})
assert out.startswith("REFUSED") and "limit" in out, out
print("certificate gate: one probe, follow-ups only with evidence, hard cap: OK")

# --- 5. paid budget reservation is atomic under parallel calls --------------------------------
tracker = start_tracking()
granted: list[int] = []
lock = threading.Lock()


def _try() -> None:
    got = tracker.reserve_call("site_extractor", "fetch_page_search_fallback", "u", 10)
    if got is not None:
        with lock:
            granted.append(got)


threads = [threading.Thread(target=_try) for _ in range(40)]
for t in threads:
    t.start()
for t in threads:
    t.join()
assert sorted(granted) == list(range(1, 11)), granted
assert tracker.calls_used("fetch_page_search_fallback") == 10
print("paid budget: 40 parallel reservations, exactly 10 granted: OK")

# --- 6. sitemap, GLEIF, and document-link precision --------------------------------------------
SITEMAP = (
    '<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
    "<url><loc>https://acme.com/products/widget-1</loc></url>"
    "<url><loc>https://acme.com/about/locations</loc></url>"
    "<url><loc>https://acme.com/de/kontakt</loc></url>"
    "<url><loc>https://acme.com/quality/ISO9001.pdf</loc></url>"
    "<url><loc>https://acme.com/blog/post-17</loc></url></urlset>"
)
assert st._SITEMAP_RE.search(SITEMAP)
sitemap_text = st._sitemap_to_text(SITEMAP)
assert "5 URLs, 3 of them" in sitemap_text, sitemap_text
assert "about/locations" in sitemap_text and "kontakt" in sitemap_text and "ISO9001.pdf" in sitemap_text
assert "widget-1" not in sitemap_text and "post-17" not in sitemap_text
assert st._is_text_content_type("application/xml; charset=utf-8")

GLEIF = (
    b'{"data":[{"attributes":{"entity":{"legalName":{"name":"ACME GMBH"},"status":"ACTIVE",'
    b'"legalAddress":{"addressLines":["Musterstrasse 12"],"city":"Berlin","region":"DE-BE",'
    b'"postalCode":"10115","country":"DE"},"headquartersAddress":{"addressLines":["Werkstr. 1"],'
    b'"city":"Hamburg","postalCode":"20095","country":"DE"}}}}]}'
)
gleif = st._gleif_to_text(GLEIF)
assert "ACME GMBH (ACTIVE)" in gleif and "Musterstrasse 12, Berlin" in gleif and "Werkstr. 1, Hamburg" in gleif, gleif
assert st._gleif_to_text(b"not json") is None

# A registry URL whose company number contains "9001" is not a certificate link.
page = '<a href="/company/11590013">ACME TECH UK LTD</a><a href="/docs/iso-9001.pdf">ISO 9001</a>'
links = st._html_to_text(page, "https://find-and-update.company-information.service.gov.uk/search")
assert "11590013" not in links and "iso-9001.pdf" in links, links
print("sitemap filtering, GLEIF formatting, doc-link precision: OK")

# --- 7. confidence promotion ------------------------------------------------------------------
assert confidence_for("A", 1) == "High"
assert confidence_for("B", 2) == "Medium"
assert confidence_for("D", 2) == "Low"
assert confidence_for("D", 3) == "High"  # three sources list the same address
assert confidence_for("C", 5) == "High"
print("confidence: tier-based, promoted to High with 3+ sources: OK")

print("\nALL COST-GUARD CHECKS PASSED")
