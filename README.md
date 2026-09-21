# site-extraction-one-agent

Give it a CSV of company names. It researches each company's real physical sites on the web and
gives you back a CSV of addresses, each with the source URL and a verbatim quote that supports it.

One LLM agent, two tools, deterministic scoring afterwards. No orchestration, no subagents, no
geocoding.

```
suppliers.csv  ->  [ agent: web_search + fetch_page ]  ->  scoring & dedupe  ->  results.csv
```

---

## How it works

**The agent** ([`agent.py`](src/site_extraction_one_agent/agent.py)) runs one company at a time
through four prompted phases:

| Phase | What it does |
| --- | --- |
| ORIENT | Finds the company's official domain and any country/brand domains it owns. |
| HARVEST | Pulls addresses from first-party pages — locators, contact, imprint, investor filings. |
| GAP-FILL | Fills geographic holes, then runs a **mandatory** home-market maps/directory sweep. |
| REPORT | Returns structured candidates, each with a `source_url` and an `evidence_quote`. |

The directory sweep is mandatory and holds a reserved slice of the search budget. Small branch
sites — sales offices, service centers, satellite warehouses — are typically published nowhere
except a business directory, and first-party fetching cannot substitute for it.

**Two tools** ([`search_tool.py`](src/site_extraction_one_agent/search_tool.py)):

- `web_search` — OpenAI's hosted web search. Billed per call, so it is the scarce resource.
  Budget: 45 calls. `site:`-with-a-path queries are refused and redirected to `fetch_page`.
- `fetch_page` — three tiers, cheapest first. Budget: 40 calls.
  1. Plain `requests.get` — free.
  2. ScrapingBee headless browser — ~$0.001, for bot-blocked or JS-rendered pages.
  3. Paid `web_search` retrieval — ~$0.01, last resort and the only path for PDFs.

**Scoring** ([`scoring.py`](src/site_extraction_one_agent/scoring.py)) is pure Python — no LLM, no
network. Each candidate's source URL gets a tier, and near-duplicate addresses collapse to the
highest-scoring copy.

| Tier | Weight | Sources |
| --- | --- | --- |
| A | 1.00 | Government registries and filings (`sec.gov`, Companies House, `*.gov`); the company's own contact/locations/imprint pages. |
| B | 0.60 | Other pages on the company's own domains; Google, LinkedIn. |
| C | 0.25 | Directory aggregators — ZoomInfo, D&B, Manta, Yelp, Crunchbase, job boards. |
| D | 0.05 | Everything else, including lookalike domains. |

A company's country and brand sites (`acme.co.uk`, `acmebrand.com`) count as first-party, matched
against the domain list the agent reports rather than by brand-name substring.

---

## Install

Requires Python 3.12+. [uv](https://docs.astral.sh/uv/) is the supported path.

```bash
git clone https://github.com/amitesh-x-singh/agent-extraction.git
cd agent-extraction
uv sync
```

Without uv:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -e .
```

## Configure

Copy `.env.example` to `.env` and fill in both keys. The CLI exits with an error if either is
missing, before any company runs — a batch must not discover a missing key three hours in.

```ini
OPENAI_API_KEY=sk-...
SCRAPINGBEE_API_KEY=...
SEARCH_CONTACT_EMAIL=you@example.com   # optional, advertised in the User-Agent
```

`SCRAPINGBEE_API_KEY` is required rather than optional. The pipeline runs without it, but every
blocked page then escalates to the ~10x more expensive paid search fallback, so a run that
silently lacked it would cost several times what it should.

## Commands

```bash
# CSV in, CSV out -- the main path
uv run site-extraction-one-agent --input suppliers.csv --output results/my_run.csv

# a single company
uv run site-extraction-one-agent --company "TENNECO"

# comma-delimited output instead of the default tab
uv run site-extraction-one-agent -i suppliers.csv -o results/my_run.csv --delimiter comma

# benchmark against an answer key, at a higher reasoning effort
uv run site-extraction-one-agent -i suppliers.csv -o results/my_run.csv \
    --ground-truth groundtruth.csv --reasoning-effort high

# without uv
python -m site_extraction_one_agent.cli --input suppliers.csv --output results/my_run.csv
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--company NAME` | — | Research one company. Mutually exclusive with `--input`. |
| `-i`, `--input PATH` | — | Input CSV of company names. Also accepted as `--suppliers`. |
| `-o`, `--output PATH` | see below | Output CSV. Also accepted as `--combined-out`. |
| `--results-dir PATH` | `./results` | Directory for the per-company files. |
| `--delimiter tab\|comma` | `tab` | Output field delimiter. |
| `--ground-truth PATH` | none | Answer key for the `gt_*` columns. Skipped when absent. |
| `--reasoning-effort ...` | API default | `none`, `low`, `medium`, `high`, `xhigh`, `max`. |

Default output paths: `results/<COMPANY>.csv` per company, and
`results/combined_<n>_companies.csv` for a batch. Paths resolve against the current working
directory, so run from wherever you want the output.

A batch runs **serially, never in parallel**. The usage tracker is a module-level singleton reset
per company, so concurrent companies would cross-contaminate the cost accounting and interleave
their appends to the JSONL logs.

## Input format

Comma-delimited, `utf-8-sig`, with a `Supplier` or `supplier_name` column (case-insensitive). Any
other column is ignored — the pipeline discovers each company's domain itself.

```csv
supplier_name,url
ACXIOM,
ADIENT LTD,
AIRGAS,
```

## Output format

> **The default delimiter is TAB, not comma**, despite the `.csv` extension. Every existing file
> in `results/` is tab-delimited and the comparison tooling reads them that way. Pass
> `--delimiter comma` for a conventional CSV.

21 columns. The first nine mirror the ground-truth layout so rows line up against it; the rest are
this project's own.

| Column | |
| --- | --- |
| `supplier_name` | Company as given in the input. |
| `street_address` | Building number, street, industrial park. |
| `address_line_2` | Always blank; present for layout parity. |
| `city`, `state`, `postal_code`, `country` | Address components, kept separate. |
| `confidence` | `High` / `Medium` / `Low`, derived from `tier`. |
| `category` | Always blank; present for layout parity. |
| `legal_entity` | Operating subsidiary that runs the site, if different from the parent. |
| `site_name` | Official facility name, if published. |
| `site_type` | Every function the site serves, e.g. `Manufacturing Plant, R&D, Office`. |
| `ownership_type` | Wholly owned, leased, joint venture, etc. |
| `operational_status` | Active, under construction, planned, idle, closed. Blank if unconfirmed. |
| `status_as_of` | Date a source last confirmed that status. |
| `tier` | Source tier `A`–`D`. |
| `score` | Tier weight, two decimals. |
| `source_url` | Page that supports this address. |
| `evidence_quote` | Verbatim snippet from that page. |
| `gt_address_city_match` | City matched the ground truth. Empty without `--ground-truth`. |
| `gt_address_country_match` | Country matched the ground truth. |

Three append-only JSONL logs are written beside the results:

- `usage_log.jsonl` — per-run tokens, tool calls, estimated cost, and the run's configuration.
- `queries_log.jsonl` — every search query actually issued.
- `scrapingbee_log.jsonl` — every ScrapingBee call: URL, status, credits, failure reason.

## Cost

A run prints its own token/tool/cost summary, and a batch prints a per-company table at the end.

Budgets live in [`config.py`](src/site_extraction_one_agent/config.py) as constants, deliberately
**not** as CLI flags, so every row in `usage_log.jsonl` stays comparable against every other.
Change them there to run a cost experiment.

```python
WEB_SEARCH_CALL_BUDGET = 45
FETCH_PAGE_CALL_BUDGET = 40
DIRECTORY_SWEEP_RESERVE = 14   # of the search budget, held back for the directory sweep
```

Cost figures in the summary are estimates from the rates in `config.py`, not billed amounts.
Verify against current OpenAI and ScrapingBee pricing before relying on them.

## Tests

Standalone assert scripts, not pytest. Run each directly:

```bash
uv run python tests/test_scoring.py
uv run python tests/test_search_guards.py
uv run python tests/test_usage_attribution.py
uv run python tests/test_address_ground_truth_and_results_csv.py
```

All four are offline — no API key, no network.

## Known gaps

- **Ground truth format mismatch.** `--ground-truth` expects a *tab-delimited, headerless* file
  with 9 positional columns (`supplier_name, street_address, address_line_2, city, state,
  postal_code, country, confidence, category`). The `ground_truth.csv` in this repo is
  comma-delimited with a 17-column header and will not load as-is; convert it first.
- **No `--skip-verification` flag.** The adversarial verification pass was removed on purpose — it
  refuted only ~5% of candidates, which did not justify its share of the cost. There is nothing
  left to skip.
- **`address_line_2` and `category` are always blank.** They exist so rows align column-for-column
  with the ground-truth layout.

## Layout

```
src/site_extraction_one_agent/
  agent.py                 the agent, its prompt, and the reasoning behind each rule
  cli.py                   entrypoint, batch driver, reporting
  config.py                model, budgets, tier weights, cost rates
  search_tool.py           web_search and fetch_page, with their budget enforcement
  scoring.py               source tiers, scoring, address dedupe
  schemas.py               the structured output the agent must return
  results_csv.py           the 21-column results writer
  address_ground_truth.py  optional ground-truth comparison
  usage.py                 token/tool/cost tracking and the JSONL logs
```
