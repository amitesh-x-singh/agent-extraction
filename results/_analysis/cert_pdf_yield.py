"""Count sites whose evidence is an ISO/IATF/AS9100-style certificate PDF, per run and supplier.

Free: reads result files already on disk, makes no API calls. A site counts as cert-PDF-sourced
when its source_url is a PDF and either the URL or the evidence quote looks like a certificate.
Each site keeps one source_url, so these counts are lower bounds.

    uv run python results/_analysis/cert_pdf_yield.py [result files...]
"""

import collections
import csv
import glob
import io
import re
import sys

csv.field_size_limit(10**9)
sys.stdout.reconfigure(encoding="utf-8")

CERT_URL = re.compile(r"cert|iso[-_ ]?\d|iatf|as9100|16949|14001|9001|45001|13485|22163|zertifik", re.I)
CERT_QUOTE = re.compile(r"certif|iso\s?\d{4,5}|iatf|as\s?9100|registrar|zertifik", re.I)

DEFAULT_RUNS = {
    "pdf_tier_20260922": ["results/pdf_tier_20260922/combined_5_companies.csv"],
    "combined_51": ["results/combined_51_companies.csv"],
    "sites_sample_20260923": sorted(glob.glob("results/sites_sample_20260923/shard_0*_combined.csv")),
    "cert_pdf_test_20260924": sorted(glob.glob("results/cert_pdf_test_20260924/*.csv")),
}


def load(path: str) -> list[dict[str, str]]:
    text = open(path, encoding="utf-8-sig").read()
    delim = "\t" if text.split("\n", 1)[0].count("\t") > 2 else ","
    return list(csv.DictReader(io.StringIO(text), delimiter=delim))


def is_cert_pdf(row: dict[str, str]) -> bool:
    url = row.get("source_url") or ""
    return "pdf" in url.lower() and bool(CERT_URL.search(url) or CERT_QUOTE.search(row.get("evidence_quote") or ""))


def has_any_cert_pdf(row: dict[str, str]) -> bool:
    """Any of the site's sources (the `all_sources` column, newer runs only) is a cert PDF."""
    urls = re.findall(r'"([^"]+)"', row.get("all_sources") or "")
    return is_cert_pdf(row) or any("pdf" in u.lower() and CERT_URL.search(u) for u in urls)


def report(name: str, paths: list[str]) -> None:
    rows = [r for p in paths for r in load(p)]
    if not rows:
        return
    total = collections.Counter(r["supplier_name"] for r in rows)
    cert = collections.Counter(r["supplier_name"] for r in rows if is_cert_pdf(r))
    n_cert = sum(cert.values())
    n_any = sum(1 for r in rows if has_any_cert_pdf(r))
    print(f"== {name}: suppliers={len(total)} sites={len(rows)} cert_pdf_sites={n_cert} "
          f"({100 * n_cert / len(rows):.1f}%) suppliers_with_any={len(cert)} "
          f"sites_with_any_cert_pdf_source={n_any}")
    for supplier, n in cert.most_common():
        print(f"   {supplier:45s} {n:4d} of {total[supplier]}")


if __name__ == "__main__":
    runs = {p: [p] for p in sys.argv[1:]} if len(sys.argv) > 1 else DEFAULT_RUNS
    for run_name, run_paths in runs.items():
        report(run_name, run_paths)
