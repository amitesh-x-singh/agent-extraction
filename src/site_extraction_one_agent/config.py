"""Central constants: model choice, tier weights, per-tool call budgets. No geocoding settings
(coordinate-based ground truth is not used in this single-agent pipeline)."""

import os

MODEL = "openai:gpt-5.6-luna"
SEARCH_MODEL = "gpt-5.6-luna"  # model used inside the raw Responses API web_search call

# Per-tool call budgets, enforced in search_tool.py (they were prose-only guidance in the system
# prompt before, which the model routinely blew past: the median run in queries_log.jsonl issued
# ~88 tool calls against a stated "25-40" budget, and the worst issued 804). The two tools get
# separate budgets because they cost very differently: every web_search is billed per call
# ($0.01) plus its tokens, while fetch_page starts with a free GET, escalates only a blocked or
# JS-rendered page to a ~$0.001 ScrapingBee browser fetch, and reaches the paid search call only
# when that also fails or the URL is a PDF. So searching is what needs to be scarce; fetching is
# what we want the agent to do instead.
#
# Uncapping fetch_page was tried and reverted. The theory was that the cap starved the
# ScrapingBee tier; measured on ELLSWORTH it did not -- uncapped, ScrapingBee got 1 call against
# 2 capped, because that company's pages are static HTML a plain GET already handles. Coverage
# did rise (33 -> 46 candidates) but so did cost (~1.5x) and, contrary to the intent, web_search
# rose with it (26 -> 36 calls) as extra fetching produced extra leads to search on.
WEB_SEARCH_CALL_BUDGET = 45
FETCH_PAGE_CALL_BUDGET = 40
# How much of WEB_SEARCH_CALL_BUDGET the prompt tells the agent to hold back for the mandatory
# home-market maps/directory sweep. Small company-operated sites (branch sales offices, service
# centers, satellite warehouses, locations inherited with an acquisition) are usually published
# nowhere except a maps/business-directory listing, so this is the one part of the budget that
# cannot be spent on first-party pages: a run that skipped it reached 6 of one company's 29
# known home-country cities, while a reference system that swept directories reached 12.
DIRECTORY_SWEEP_RESERVE = 14
# Fraction of a budget past which the tool starts telling the model to wrap up rather than
# just reporting the count.
BUDGET_WARN_FRACTION = 0.8

TIER_WEIGHTS = {"A": 1.0, "B": 0.6, "C": 0.25, "D": 0.05}

# Bag-of-words containment ratio (see scoring._containment) above which two candidate addresses
# are treated as the same physical site, e.g. "..., London, EC4V 4HN, United Kingdom" vs
# "..., London, United Kingdom, EC4V 4HN" (reordered) or a company-name prefix added on one
# copy -- both score 1.0. Picked by hand against real duplicates in results/*.csv: high enough
# that two addresses differing by street number/city/postal code (which drags the ratio well
# below this since those tokens are unique, not shared) don't fall inside it.
ADDRESS_DEDUPE_JACCARD_THRESHOLD = 0.9

# Sent on every outbound page fetch so site operators can identify (and reach) whoever is
# crawling them. The contact comes from the SEARCH_CONTACT_EMAIL environment variable rather
# than being hardcoded: whoever runs this should be the one answering for its traffic, and a
# personal address does not belong in a shared repository.
SEARCH_USER_AGENT = "site-extraction-one-agent/0.1 (+https://github.com/amitesh-x-singh/agent-extraction)"


def search_user_agent() -> str:
    """Read at call time, not import time: config is imported before load_dotenv() runs, so a
    SEARCH_CONTACT_EMAIL set in .env is not yet visible when this module executes."""
    contact = os.environ.get("SEARCH_CONTACT_EMAIL", "").strip()
    return f"site-extraction-one-agent/0.1 (contact: {contact})" if contact else SEARCH_USER_AGENT

GROUND_TRUTH_ADDRESS_CSV = "groundtruth.csv"  # tab-delimited, structured street/city/state/postal/country
RESULTS_DIR = "results"  # one tab-delimited CSV per company, results/<company>.csv

# Real gpt-5.6-luna standard-tier billing rates (as of the July 30, 2026 OpenAI price update;
# verify against https://openai.com/api/pricing/ before relying on these for exact invoicing --
# rates change over time and vary by service tier: Batch/Flex 0.5x, Standard 1x, Fast mode 2x).
EST_INPUT_COST_PER_1M_TOKENS = 0.20
EST_OUTPUT_COST_PER_1M_TOKENS = 1.20
EST_WEB_SEARCH_COST_PER_CALL = 0.01  # $10.00 / 1K calls

# ScrapingBee bills in CREDITS, not calls, and the per-credit price depends on the plan:
# $19/75k (Hobby) = $0.000253, $49/250k (Freelance) = $0.000196, $99/1M (Startup) = $0.000099.
# This is set for Freelance, rounded up; verify against https://www.scrapingbee.com/#pricing.
# A JS-rendered fetch costs 5 credits (~$0.001), roughly a TENTH of the web_search fallback it
# displaces ($0.01 call fee plus 1-3k output tokens) -- that ratio is why the tier exists at all.
# Their docs say failed requests aren't charged; measured, that is optimistic -- a zoominfo.com
# fetch that came back non-200 still reported 25 credits spent. So cost is tracked from the
# credits ScrapingBee reports on each response, never assumed per call, and failures are counted
# separately (usage.scrapingbee_failures) precisely because they are not free.
EST_SCRAPINGBEE_COST_PER_CREDIT = 0.0002

# Append-only JSONL log of per-run token usage/cost, relative to repo root.
USAGE_LOG_PATH = "usage_log.jsonl"
# Append-only JSONL log of every web_search query actually issued.
QUERIES_LOG_PATH = "queries_log.jsonl"
# Append-only JSONL log of every ScrapingBee call fetch_page's middle tier actually made --
# status code, credits, and (when it didn't yield usable page content) ScrapingBee's own
# reported reason. usage_log.jsonl only ever recorded a pass/fail COUNT per company; this is
# what lets a bad run be diagnosed afterwards instead of re-guessed from a sample re-test.
SCRAPINGBEE_LOG_PATH = "scrapingbee_log.jsonl"
