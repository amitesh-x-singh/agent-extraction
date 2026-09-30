"""CLI entrypoint: run the single agent for one company -- or for a whole supplier list with
--input -- score its findings deterministically, optionally compare the top candidates against a
ground-truth address file (text-only city/country match, no geocoding), write a results CSV, and
record full token/tool usage for cost analysis.

Output paths are resolved against the CURRENT WORKING DIRECTORY (see _output_root), not the
source checkout, so a pip-installed copy writes results where it was run rather than into
site-packages.

The per-tool call budgets are fixed constants in config.py (WEB_SEARCH_CALL_BUDGET /
FETCH_PAGE_PAID_CALL_BUDGET) rather than CLI flags, so every run in usage_log.jsonl is comparable
against every other; change them there to run a cost experiment.

A --suppliers batch runs companies SERIALLY and never in parallel: usage.py's tracker is a
module-level singleton reset per company by start_tracking(), so concurrent companies in one
process would cross-contaminate each other's cost accounting, and concurrent processes would
interleave their appends to usage_log.jsonl / queries_log.jsonl."""

import argparse
import csv
import os
import re
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv

# Windows' console defaults to a codepage (e.g. cp1252) that can't encode a lot of Unicode
# (accented/non-Latin characters are common in real addresses -- Czech, Vietnamese, Chinese,
# etc.). Without this, a single such character in a print() crashes the whole process.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from .address_ground_truth import compare_by_city_country, load_address_ground_truth
from .address_ground_truth import rows_for_company as address_rows_for_company
from .config import (
    SCRAPINGBEE_COST_PER_CREDIT,
    FETCH_PAGE_PAID_CALL_BUDGET,
    GROUND_TRUTH_ADDRESS_CSV,
    RESULTS_DIR,
    WEB_SEARCH_CALL_BUDGET,
)
from .results_csv import write_combined_results_csv, write_results_csv
from .schemas import CandidateAddress, SiteExtractionResult
from .scoring import dedupe_ranked_by_address, rank_candidates
from .usage import (
    TokenUsageCallbackHandler,
    UsageTracker,
    append_queries_log,
    append_scrapingbee_log,
    append_usage_log,
    start_tracking,
)

_DOMAIN_RE = re.compile(r"\b([a-z0-9-]+\.(?:com|net|org|io|co|de|fr|uk|jp|cn|in|biz))\b", re.IGNORECASE)


def _output_root() -> Path:
    """Where results/ and the JSONL logs are written: the current working directory.

    This used to be the source checkout (Path(__file__).parents[2]), which put outputs next to
    the code and wrote into site-packages when the project was pip-installed as a wheel. Running
    from the repo root -- the normal case -- resolves to the same directory as before, so
    existing results/ files keep their location."""
    return Path.cwd()


def _slugify(company: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", company.strip()).strip("_") or "company"


def _dedupe_candidates(candidates: list[CandidateAddress]) -> list[CandidateAddress]:
    """The agent can sometimes report the exact same (address, source_url) pair more than
    once across a long research pass -- collapse those."""
    seen: set[tuple[str, str]] = set()
    deduped: list[CandidateAddress] = []
    for c in candidates:
        key = (c.full_address().strip().lower(), c.source_url.strip())
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    return deduped


def _find_official_domain(notes: str) -> str | None:
    """Best-effort: pull a domain-looking token out of the agent's `notes` field (it's
    prompted to state the company's official domain there)."""
    match = _DOMAIN_RE.search(notes or "")
    return match.group(1).lower() if match else None


def _normalize_domains(domains: list[str]) -> list[str]:
    """The agent reports `company_domains` as free-form strings -- accept the shapes it
    plausibly emits ('https://www.acme.co.uk/contact', 'www.acme.co.uk', 'acme.co.uk') and
    reduce each to a bare host, since that is what scoring matches against."""
    cleaned: list[str] = []
    for raw in domains or []:
        host = (raw or "").strip().lower()
        host = re.sub(r"^[a-z]+://", "", host).split("/")[0].split("?")[0]
        host = host.split("@")[-1].split(":")[0].removeprefix("www.").strip(".")
        if host and "." in host and host not in cleaned:
            cleaned.append(host)
    return cleaned


class PipelineResult:
    def __init__(
        self,
        candidates: list[CandidateAddress],
        official_domain: str | None,
        tracker: UsageTracker,
        company_domains: list[str] | None = None,
    ) -> None:
        self.candidates = candidates
        self.official_domain = official_domain
        self.company_domains = company_domains or []
        self.tracker = tracker


def user_message(company: str, supplier_url: str | None = None) -> str:
    """The agent's opening instruction. With a website from the input, the agent starts there;
    without one it must search for it -- it is never allowed to guess a domain."""
    message = f"Research physical site locations for: {company}"
    if supplier_url and supplier_url.strip():
        message += (
            f"\n\nThe input gives this company's official website: {supplier_url.strip()}. Start by "
            "fetching it and its /sitemap.xml instead of searching for the domain."
        )
    else:
        message += (
            "\n\nNo website was given for this company. Do not guess one: find its official "
            "website with web_search first."
        )
    return message


def _run_pipeline(
    company: str, *, reasoning_effort: str | None = None, supplier_url: str | None = None
) -> PipelineResult:
    from .agent import build_agent  # deferred: needs OPENAI_API_KEY at import time
    from .search_tool import _norm_host

    tracker = start_tracking()
    if supplier_url:
        tracker.record_hosts([_norm_host(supplier_url)])  # the one domain fetch_page trusts up front
    usage_callback = TokenUsageCallbackHandler(tracker)

    agent = build_agent(reasoning_effort=reasoning_effort)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_message(company, supplier_url)}]},
        config={"callbacks": [usage_callback]},
    )

    extraction: SiteExtractionResult | None = result.get("structured_response")
    if extraction is None:
        print("ERROR: agent did not return a structured response.", file=sys.stderr)
        print(result.get("messages", [])[-1] if result.get("messages") else result, file=sys.stderr)
        return PipelineResult([], None, tracker)

    candidates = _dedupe_candidates(extraction.candidates)
    # An input website is the company's own, so pages on it score as first-party (as in the API).
    company_domains = _normalize_domains(([supplier_url] if supplier_url else []) + extraction.company_domains)
    # The primary domain can come from either place; notes is the older path and stays as the
    # fallback for a run where the agent fills in only one of the two.
    official_domain = _find_official_domain(extraction.notes) or (company_domains[0] if company_domains else None)
    return PipelineResult(candidates, official_domain, tracker, company_domains)


def _compute_ranked(company: str, all_candidates, ground_truth_path: Path | None, official_domain: str, company_domains=None):
    ranked = rank_candidates(all_candidates, official_domain, company_domains)
    # rank_candidates sorts best-score-first, so this keeps the most credible source in each
    # near-duplicate cluster (e.g. the same office reported by two sources with its address
    # components in a different order -- see dedupe_ranked_by_address's docstring).
    ranked = dedupe_ranked_by_address(ranked)

    # The ground-truth comparison is optional: it exists to benchmark a run against a known
    # answer key, and most users have no such file. A missing one is skipped silently rather
    # than warned about once per company -- the gt_* columns simply stay empty.
    address_gt_rows: list[dict] = []
    if ground_truth_path is not None and ground_truth_path.exists():
        address_gt_rows_all = load_address_ground_truth(ground_truth_path)
        address_gt_rows = address_rows_for_company(address_gt_rows_all, company)
    ranked = compare_by_city_country(ranked, address_gt_rows)

    return ranked, address_gt_rows


class OutputOptions:
    """Everything about where a run's files go and what shape they take, resolved once in main()
    so run_one() behaves identically whether it is called for a single company or from inside a
    batch."""

    def __init__(
        self,
        *,
        results_dir: Path | None = None,
        delimiter: str = "\t",
        ground_truth: Path | None = None,
        log_dir: Path | None = None,
    ) -> None:
        root = _output_root()
        self.results_dir = results_dir or (root / RESULTS_DIR)
        self.delimiter = delimiter
        self.ground_truth = ground_truth
        self.log_dir = log_dir or root


def _require_api_keys() -> None:
    """Checked once per process, before any company runs -- a batch must not discover a missing
    key three hours in."""
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set (check your .env file).", file=sys.stderr)
        sys.exit(1)
    # fetch_page degrades gracefully without this (it just skips its middle tier and escalates
    # straight to the paid search fallback), but a run that does so silently costs materially
    # more for no reason, so a missing key is a misconfiguration worth stopping on.
    if not os.environ.get("SCRAPINGBEE_API_KEY"):
        print("ERROR: SCRAPINGBEE_API_KEY is not set (check your .env file).", file=sys.stderr)
        sys.exit(1)


def run_one(
    company: str,
    options: OutputOptions,
    *,
    reasoning_effort: str | None = None,
    results_path: Path | None = None,
    supplier_url: str | None = None,
) -> tuple[list[dict], UsageTracker] | None:
    """Research one company end to end and write its results CSV. Returns (ranked, tracker), or
    None if the agent produced no candidates.

    Returning None rather than calling sys.exit() is what makes this reusable from the batch
    driver: one company finding nothing must not take the other nine down with it.
    """
    print(f"Researching site locations for: {company}" + (f" ({supplier_url})" if supplier_url else "") + "\n")
    pipeline = _run_pipeline(company, reasoning_effort=reasoning_effort, supplier_url=supplier_url)
    if not pipeline.candidates:
        return None

    print(f"Collected {len(pipeline.candidates)} distinct candidate address(es).\n")

    repo_root = options.log_dir
    ranked, address_gt_rows = _compute_ranked(
        company, pipeline.candidates, options.ground_truth, pipeline.official_domain, pipeline.company_domains
    )
    print(f"Company domains treated as first-party: {', '.join(pipeline.company_domains) or '(none reported)'}\n")

    # Suffix the filename with the reasoning effort when it's explicitly set, so a non-default
    # run (e.g. --reasoning-effort max) never clobbers the default-effort results for the same
    # company -- both stay on disk side by side for comparison.
    filename_suffix = f"__effort-{reasoning_effort}" if reasoning_effort else ""
    # Write to disk BEFORE printing -- a printing failure (e.g. a Unicode character the
    # terminal's codepage can't render) must never risk losing an already-computed result.
    if results_path is None:
        results_path = options.results_dir / f"{_slugify(company)}{filename_suffix}.csv"
    write_results_csv(results_path, company, ranked, delimiter=options.delimiter)
    run_config = {
        "reasoning_effort": reasoning_effort or "medium",  # API default when unset
        "web_search_call_budget": WEB_SEARCH_CALL_BUDGET,
        "fetch_page_paid_call_budget": FETCH_PAGE_PAID_CALL_BUDGET,
        # Records whether fetch_page's middle tier was available, so rows in usage_log.jsonl
        # from before/without it stay comparable against rows with it.
        "scrapingbee_enabled": bool(os.environ.get("SCRAPINGBEE_API_KEY")),
    }
    usage_log_path = append_usage_log(company, pipeline.tracker, repo_root, run_config)
    queries_log_path = append_queries_log(company, pipeline.tracker, repo_root)
    scrapingbee_log_path = append_scrapingbee_log(company, pipeline.tracker, repo_root)

    _print_report(company, ranked, address_gt_rows)
    print(f"\nResults written to: {results_path}")
    _print_usage_summary(pipeline.tracker)
    print(f"\nUsage record appended to: {usage_log_path}")
    print(f"Queries appended to: {queries_log_path}")
    if pipeline.tracker.scrapingbee_details:
        print(f"ScrapingBee call details appended to: {scrapingbee_log_path}")
    return ranked, pipeline.tracker


def run(
    company: str,
    options: OutputOptions,
    *,
    reasoning_effort: str | None = None,
    output: Path | None = None,
    supplier_url: str | None = None,
) -> None:
    _require_api_keys()
    if run_one(company, options, reasoning_effort=reasoning_effort, results_path=output, supplier_url=supplier_url) is None:
        sys.exit(1)


_SUPPLIER_COLUMN_NAMES = {"supplier", "supplier_name"}
_URL_COLUMN_NAMES = {"url", "website", "supplier_url", "website_url", "official_url", "domain"}


def _read_suppliers(path: Path) -> list[tuple[str, str | None]]:
    """(company, website or None) from a suppliers CSV. Comma-delimited with a `Supplier,URL` (or
    `supplier_name,website`) header, unlike the tab-delimited results files. A website given here
    is what the agent starts from; without one it has to search for it, never guess it."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    header = list(rows[0]) if rows else []
    column = next((col for col in header if col.strip().lower() in _SUPPLIER_COLUMN_NAMES), None)
    url_column = next((col for col in header if col.strip().lower() in _URL_COLUMN_NAMES), None)
    if not rows or column is None:
        print(f"ERROR: {path} has no 'Supplier' or 'supplier_name' column.", file=sys.stderr)
        sys.exit(1)
    return [
        (name, ((row.get(url_column) or "").strip() or None) if url_column else None)
        for row in rows
        if (name := (row.get(column) or "").strip())
    ]


def run_batch(
    suppliers_path: Path, combined_out: Path, options: OutputOptions, *, reasoning_effort: str | None = None
) -> None:
    _require_api_keys()
    suppliers = _read_suppliers(suppliers_path)
    websites = dict(suppliers)
    companies = [name for name, _ in suppliers]
    print(f"Batch: {len(companies)} companies from {suppliers_path}\n{', '.join(companies)}\n")

    completed: list[tuple[str, list[dict]]] = []
    summaries: list[tuple[str, int, dict]] = []
    failures: list[tuple[str, str]] = []

    for i, company in enumerate(companies, start=1):
        print(f"\n{'=' * 78}\n[{i}/{len(companies)}] {company}\n{'=' * 78}")
        try:
            result = run_one(company, options, reasoning_effort=reasoning_effort, supplier_url=websites.get(company))
        except Exception as exc:  # one company's failure must not end the batch
            traceback.print_exc()
            failures.append((company, f"{type(exc).__name__}: {exc}"))
            continue
        if result is None:
            failures.append((company, "no candidates returned"))
            continue

        ranked, tracker = result
        completed.append((company, ranked))
        summaries.append((company, len(ranked), tracker.to_dict()))
        # Rewritten after every company rather than once at the end, so killing the run partway
        # (or a crash on company 7) still leaves a complete, valid combined file on disk.
        write_combined_results_csv(combined_out, completed, delimiter=options.delimiter)
        print(f"\nCombined CSV now holds {len(completed)} companies: {combined_out}")

    _print_batch_summary(summaries, failures, combined_out)


def _print_batch_summary(
    summaries: list[tuple[str, int, dict]], failures: list[tuple[str, str]], combined_out: Path
) -> None:
    print(f"\n\n{'=' * 78}\nBATCH SUMMARY\n{'=' * 78}")
    header = f"{'company':<20}{'rows':>6}{'search':>8}{'fetch':>7}{'paid_fb':>9}{'bee':>6}{'credits':>9}{'cost':>9}"
    print(header)
    totals = {"rows": 0, "search": 0, "fetch": 0, "paid_fb": 0, "bee": 0, "credits": 0, "cost": 0.0}
    for company, rows, usage in summaries:
        counts = usage.get("tool_call_counts", {})
        fetch = counts.get("fetch_page", 0)
        paid_fb = counts.get("fetch_page_search_fallback", 0)
        cost = usage["cost_usd"]
        print(
            f"{company:<20}{rows:>6}{usage['web_search_calls']:>8}{fetch:>7}{paid_fb:>9}"
            f"{usage['scrapingbee_calls']:>6}{usage['scrapingbee_credits']:>9}{cost:>9.3f}"
        )
        totals["rows"] += rows
        totals["search"] += usage["web_search_calls"]
        totals["fetch"] += fetch
        totals["paid_fb"] += paid_fb
        totals["bee"] += usage["scrapingbee_calls"]
        totals["credits"] += usage["scrapingbee_credits"]
        totals["cost"] += cost
    print(
        f"{'TOTAL':<20}{totals['rows']:>6}{totals['search']:>8}{totals['fetch']:>7}"
        f"{totals['paid_fb']:>9}{totals['bee']:>6}{totals['credits']:>9}{totals['cost']:>9.3f}"
    )
    print(f"\nCombined results: {combined_out}")
    if failures:
        print(f"\nFAILED ({len(failures)}):")
        for company, reason in failures:
            print(f"  {company}: {reason}")


def _print_usage_summary(tracker: UsageTracker) -> None:
    usage = tracker.to_dict()
    print("\n=== Token usage / estimated cost ===")
    print(
        f"TOTAL chat: {usage['chat_calls']} calls, {usage['chat_input_tokens']} in / "
        f"{usage['chat_output_tokens']} out tokens"
    )
    print(
        f"TOTAL web_search: {usage['web_search_calls']} calls, {usage['web_search_input_tokens']} in / "
        f"{usage['web_search_output_tokens']} out tokens"
    )
    # Printed unconditionally, like the lines above it, so run summaries stay diffable; the
    # "not set" clause only appears when that is the explanation for a zero.
    print(
        f"TOTAL scrapingbee: {usage['scrapingbee_calls']} calls, {usage['scrapingbee_credits']} credits "
        f"(~${usage['scrapingbee_credits'] * SCRAPINGBEE_COST_PER_CREDIT:.4f}), "
        f"{usage['scrapingbee_failures']} failed -> escalated to paid search"
        + (
            f"; {usage['scrapingbee_skipped_no_key']} fetches skipped this tier "
            "(SCRAPINGBEE_API_KEY not set)"
            if usage["scrapingbee_skipped_no_key"]
            else ""
        )
    )
    counts = usage.get("tool_call_counts", {})
    print(
        f"TOTAL tool calls attempted: {usage['tool_calls_attempted']} "
        f"(issued: web_search {counts.get('web_search', 0)}/{WEB_SEARCH_CALL_BUDGET}, "
        f"fetch_page {counts.get('fetch_page', 0)} (uncapped), of which "
        f"{counts.get('fetch_page_pdf_local', 0)} were PDFs parsed locally for free "
        f"({counts.get('fetch_page_pdf_ocr', 0)} more read by local OCR) and "
        f"{counts.get('fetch_page_search_fallback', 0)}/{FETCH_PAGE_PAID_CALL_BUDGET} needed the "
        f"paid search fallback; "
        f"refused over budget: {usage['tool_calls_blocked_by_budget']}; "
        f"site:-queries redirected to fetch_page: {usage['web_searches_redirected_to_fetch']}; "
        f"searches redirected to known document links: {usage.get('web_searches_redirected_to_doc_link', 0)}; "
        f"near-duplicate searches refused: {usage.get('web_searches_refused_duplicate', 0)}; "
        f"certificate searches refused: {usage.get('web_searches_refused_no_certificates', 0)})"
    )
    b = usage["cost_breakdown_usd"]
    print(
        f"TOTAL cost: ${usage['cost_usd']:.4f}  "
        f"(chat ${b['chat_uncached_input'] + b['chat_cached_input'] + b['chat_cache_write'] + b['chat_output']:.4f}, "
        f"search tokens ${b['web_search_uncached_input'] + b['web_search_cached_input'] + b['web_search_cache_write'] + b['web_search_output']:.4f}, "
        f"search call fee ${b['web_search_call_fee']:.4f}, "
        f"scrapingbee ${b['scrapingbee']:.4f})"
    )
    print(
        f"  input tokens priced as: {usage['chat_input_tokens'] + usage['web_search_input_tokens'] - usage['chat_cached_input_tokens'] - usage['web_search_cached_input_tokens'] - usage['chat_cache_write_tokens'] - usage['web_search_cache_write_tokens']} uncached, "
        f"{usage['chat_cached_input_tokens'] + usage['web_search_cached_input_tokens']} cached read, "
        f"{usage['chat_cache_write_tokens'] + usage['web_search_cache_write_tokens']} cache write"
    )

    if usage["per_model"]:
        print("\n--- Per-model breakdown (chat only) ---")
        for model_name, bucket in sorted(usage["per_model"].items()):
            print(
                f"{model_name:<20} {bucket['calls']:>2} calls, "
                f"{bucket['input_tokens']} in / {bucket['output_tokens']} out tokens"
            )

    if usage["queries"]:
        print(f"\n--- Queries issued ({len(usage['queries'])}) ---")
        for q in usage["queries"]:
            print(f"[{q['agent']}] {q['tool']}: {q['query']}")


def _print_report(company: str, ranked: list[dict], address_gt_rows: list[dict]) -> None:
    print(f"=== Ranked candidates for {company} ===")
    if not ranked:
        print("(no candidates found)")
        return

    for entry in ranked:
        c = entry["candidate"]
        city_match = "YES" if entry.get("gt_address_city_match") else "no"
        country_match = "YES" if entry.get("gt_address_country_match") else "no"
        print(
            f"- [{entry['tier']}] score={entry['score']:.2f} "
            f"address_city_matched={city_match:<4} address_country_matched={country_match}\n"
            f"    address: {c.full_address()}\n"
            f"    site_type: {c.site_type or ''}\n"
            f"    source:  {c.source_url}\n"
            f"    quote:   {c.evidence_quote}"
        )
        print()

    print(f"Ground-truth address row(s) available for {company!r}: {len(address_gt_rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="site-extraction-one-agent",
        description="Research supplier site locations with a single web-search research agent. "
        "Takes a CSV of company names and writes a CSV of evidence-backed site addresses.",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--company", help="Single company name to research, e.g. 'TENNECO'.")
    parser.add_argument(
        "--website",
        default=None,
        help="With --company: the company's official website, e.g. https://www.tenneco.com. The "
        "agent starts from it; without it the agent finds the site with one web search (it never "
        "guesses a domain). With --input, give websites in a 'URL' or 'website' column instead.",
    )
    target.add_argument(
        "-i",
        "--input",
        "--suppliers",
        dest="input",
        type=Path,
        help="Input CSV with a 'Supplier' or 'supplier_name' column (e.g. suppliers.csv). Runs "
        "every company in it, serially, and writes one combined results CSV.",
    )
    parser.add_argument(
        "-o",
        "--output",
        "--combined-out",
        dest="output",
        type=Path,
        default=None,
        help="Output CSV path. With --input this is the combined file for every company "
        f"(default: {RESULTS_DIR}/combined_<n>_companies.csv); with --company it is that "
        f"company's file (default: {RESULTS_DIR}/<COMPANY>.csv).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=f"Directory for the per-company result files (default: ./{RESULTS_DIR}).",
    )
    parser.add_argument(
        "--delimiter",
        choices=["tab", "comma"],
        default="tab",
        help="Field delimiter for the output files. Defaults to tab, which is what every "
        "existing file in results/ uses despite the .csv extension.",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help="Optional tab-delimited, headerless ground-truth address file used to fill the "
        "gt_address_city_match / gt_address_country_match columns. Skipped when absent.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh", "max"],
        default=None,
        help="gpt-5.6-luna reasoning effort. Omit to use the API default (medium).",
    )
    args = parser.parse_args()

    # An explicit --ground-truth is required to exist: the user asked for that comparison, so
    # silently producing empty gt_* columns would misreport the run. The implicit default is
    # allowed to be missing -- that is the ordinary case for anyone without an answer key.
    ground_truth = args.ground_truth
    if ground_truth is not None and not ground_truth.exists():
        print(f"ERROR: ground truth file not found: {ground_truth}", file=sys.stderr)
        sys.exit(1)
    if ground_truth is None:
        default_gt = _output_root() / GROUND_TRUTH_ADDRESS_CSV
        ground_truth = default_gt if default_gt.exists() else None

    options = OutputOptions(
        results_dir=args.results_dir,
        delimiter="\t" if args.delimiter == "tab" else ",",
        ground_truth=ground_truth,
    )

    if args.input:
        combined_out = args.output or (
            options.results_dir / f"combined_{len(_read_suppliers(args.input))}_companies.csv"
        )
        run_batch(args.input, combined_out, options, reasoning_effort=args.reasoning_effort)
    else:
        run(args.company, options, reasoning_effort=args.reasoning_effort, output=args.output, supplier_url=args.website)


if __name__ == "__main__":
    main()
