"""Central constants: model choice, tier weights, per-tool call budgets. No geocoding settings
(coordinate-based ground truth is not used in this single-agent pipeline)."""

import os

MODEL = "openai:gpt-5.6-luna"
SEARCH_MODEL = "gpt-5.6-luna"  # model used inside the raw Responses API web_search call

# Call budgets, enforced in search_tool.py (they were prose-only guidance in the system prompt
# before, which the model routinely blew past: the median run in queries_log.jsonl issued ~88
# tool calls against a stated "25-40" budget, and the worst issued 804).
#
# What is budgeted is what COSTS money, not what the agent does. Every web_search is billed per
# call ($0.01) plus its tokens. fetch_page is not one cost at all but three: a free plain GET,
# then a ~$0.001 ScrapingBee browser fetch for a blocked or JS-rendered page, and only then a
# ~$0.01 paid search retrieval. So the first two tiers are UNCAPPED -- fetching is exactly the
# behaviour we want more of, and a certificate library (one index page plus a dozen per-site PDFs,
# all of them now parsed locally for free) would have burned a third of the old flat 40-call
# budget for zero spend -- and only the paid tier is capped, below.
#
# An earlier experiment that uncapped fetch_page ENTIRELY was reverted: coverage rose on ELLSWORTH
# (33 -> 46 candidates) but so did cost (~1.5x) and, contrary to the intent, web_search rose with
# it (26 -> 36 calls) as extra fetching produced extra leads to search on. That loop is bounded
# here at both ends -- web_search is still capped, and so is the paid fetch tier -- which is what
# makes this split different from that experiment.
WEB_SEARCH_CALL_BUDGET = 45
# Cap on fetch_page's PAID tier only: the hosted-search retrieval reached when both free tiers
# fail (a bot-walled page, or a PDF that is a scanned image with no text layer). Measured basis:
# paid fallbacks per run in queries_log.jsonl are median 2, mean 3.9, max 30 -- and ~47% of them
# (194 of 412) were PDFs, which are now read locally for free. 10 is generous headroom against
# that median while still stopping the max-30 shape.
FETCH_PAGE_PAID_CALL_BUDGET = 10
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

# gpt-5.6-luna standard-tier billing rates, verified against OpenAI's published pricing on
# 2026-09-23. These are the real rates, not placeholders: a run's reported cost is computed from
# them and the token counts the API itself returns, so it is the billed amount rather than an
# approximation of it. Re-verify against https://developers.openai.com/api/docs/pricing before
# relying on a figure for invoicing -- rates change, and they vary by service tier (Batch/Flex
# 0.5x, Standard 1x, Fast mode 2x).
#
# INPUT TOKENS ARE NOT ONE PRICE. Every prompt_tokens count the API returns splits three ways,
# and pricing all of it at the uncached rate -- which this module used to do -- is wrong in both
# directions at once. The split is reported per call as prompt_tokens_details.cached_tokens and
# prompt_tokens_details.cache_write_tokens (LangChain surfaces the same two as
# usage_metadata.input_token_details.cache_read / .cache_creation), so the exact figure is
# available and there is no reason to approximate it:
#   * cached read  -- a prefix served from the prompt cache, 10% of the uncached rate. This
#     agent re-sends a growing conversation on every turn, so most of its input after the first
#     call is cache reads, and charging them at full rate overstated the model cost ~10x.
#   * cache write  -- the first send of a cacheable prefix, billed at 1.25x uncached.
#   * uncached     -- everything else.
INPUT_COST_PER_1M_TOKENS = 0.20
CACHED_INPUT_COST_PER_1M_TOKENS = 0.02   # 0.1x uncached
CACHE_WRITE_COST_PER_1M_TOKENS = 0.25    # 1.25x uncached
OUTPUT_COST_PER_1M_TOKENS = 1.20         # reasoning tokens are part of the output count
# The hosted web_search tool bills a flat fee per call ($10.00 / 1K calls) ON TOP OF the search
# content tokens it feeds the model, which are billed separately at the rates above. Both are
# counted here: the fee from the call count, the tokens from the Responses API usage object.
WEB_SEARCH_COST_PER_CALL = 0.01

# ScrapingBee bills in CREDITS, not calls, and the per-credit price depends on the plan:
# $19/75k (Hobby) = $0.000253, $49/250k (Freelance) = $0.000196, $99/1M (Startup) = $0.000099.
# This is set for Freelance, rounded up; verify against https://www.scrapingbee.com/#pricing.
# It is the one rate here that depends on which plan the account is on, so it is the one to
# check first if a reported cost has to reconcile exactly against an invoice.
# A JS-rendered fetch costs 5 credits (~$0.001), roughly a TENTH of the web_search fallback it
# displaces ($0.01 call fee plus 1-3k output tokens) -- that ratio is why the tier exists at all.
# Their docs say failed requests aren't charged; measured, that is optimistic -- a zoominfo.com
# fetch that came back non-200 still reported 25 credits spent. So cost is tracked from the
# credits ScrapingBee reports on each response, never assumed per call, and failures are counted
# separately (usage.scrapingbee_failures) precisely because they are not free.
SCRAPINGBEE_COST_PER_CREDIT = 0.0002

# Append-only JSONL log of per-run token usage/cost, relative to repo root.
USAGE_LOG_PATH = "usage_log.jsonl"
# Append-only JSONL log of every web_search query actually issued.
QUERIES_LOG_PATH = "queries_log.jsonl"
# Append-only JSONL log of every ScrapingBee call fetch_page's middle tier actually made --
# status code, credits, and (when it didn't yield usable page content) ScrapingBee's own
# reported reason. usage_log.jsonl only ever recorded a pass/fail COUNT per company; this is
# what lets a bad run be diagnosed afterwards instead of re-guessed from a sample re-test.
SCRAPINGBEE_LOG_PATH = "scrapingbee_log.jsonl"
