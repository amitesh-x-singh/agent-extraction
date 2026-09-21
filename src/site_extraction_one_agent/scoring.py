"""Deterministic source-tier scoring and ranking. Pure Python -- no LLM, no network calls."""

import re
import unicodedata
from urllib.parse import urlparse

from .config import ADDRESS_DEDUPE_JACCARD_THRESHOLD, TIER_WEIGHTS
from .schemas import CandidateAddress

Tier = str  # "A" | "B" | "C" | "D"

# Static domain -> tier rules, checked in order. First match wins.
_TIER_DOMAIN_PATTERNS: list[tuple[re.Pattern[str], Tier]] = [
    # Tier A: government registries, filings, official databases.
    (re.compile(r"(^|\.)sec\.gov$"), "A"),
    (re.compile(r"(^|\.)opencorporates\.com$"), "A"),
    (re.compile(r"(^|\.)find-and-update\.company-information\.service\.gov\.uk$"), "A"),
    (re.compile(r"(^|\.)companieshouse\.gov\.uk$"), "A"),
    (re.compile(r"\.gov$"), "A"),
    (re.compile(r"\.gov\.[a-z]{2}$"), "A"),
    # Tier B: business-profile / social listings, third-party press coverage.
    (re.compile(r"(^|\.)google\.[a-z.]+$"), "B"),
    (re.compile(r"(^|\.)linkedin\.com$"), "B"),
    # Tier C: directory aggregators, job boards, scraped listings.
    (re.compile(r"(^|\.)zoominfo\.com$"), "C"),
    (re.compile(r"(^|\.)manta\.com$"), "C"),
    (re.compile(r"(^|\.)dnb\.com$"), "C"),
    (re.compile(r"(^|\.)yelp\.com$"), "C"),
    (re.compile(r"(^|\.)indeed\.com$"), "C"),
    (re.compile(r"(^|\.)glassdoor\.com$"), "C"),
    (re.compile(r"(^|\.)crunchbase\.com$"), "C"),
    (re.compile(r"(^|\.)yellowpages\.com$"), "C"),
]

_CONTACT_PATH_HINTS = re.compile(
    r"(contact|imprint|impressum|legal|locations?|about[-_]?us|our[-_]?offices|find[-_]?us)",
    re.IGNORECASE,
)


def _host(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
    except ValueError:
        return ""
    return netloc.split("@")[-1].split(":")[0]


def _is_first_party(host: str, domains: list[str] | None) -> bool:
    """True if `host` is one of the company's own domains, or a subdomain of one.

    `domains` is the agent-reported `company_domains` list (plus the primary official domain).
    Matching a LIST rather than a single domain matters for any multinational: a company's own
    country sites and acquired-brand sites live on separate registrable domains
    (acme.com.my, acme.co.uk, acmebrand.com), and treating those as third-party scored genuine
    first-party contact pages at Tier D -- 0.05 -- alongside scraped directory listings. On one
    measured ELLSWORTH run that was 27 of 36 candidates.

    The list is used verbatim rather than inferred from a brand substring, which would be
    unsafe in both directions: 'intel' is a prefix of 'intelligencer.com', and no substring rule
    would connect 'ellsworth.com' to its own UK site at all.
    """
    for domain in domains or []:
        d = (domain or "").strip().lower().lstrip(".")
        if d and (host == d or host.endswith("." + d)):
            return True
    return False


def classify_domain(url: str, official_domain: str | None, company_domains: list[str] | None = None) -> Tier:
    """Classify a source URL into a source tier.

    A match against the company's own domains -- `official_domain` (e.g. "3m.com") or any entry
    in `company_domains` -- is promoted to Tier A if the URL looks like a contact/imprint/
    locations page, otherwise Tier B (e.g. a press release on the company's own domain).
    """
    host = _host(url)
    if not host:
        return "D"

    own = ([official_domain] if official_domain else []) + list(company_domains or [])
    if _is_first_party(host, own):
        return "A" if _CONTACT_PATH_HINTS.search(url) else "B"

    for pattern, tier in _TIER_DOMAIN_PATTERNS:
        if pattern.search(host):
            return tier

    return "D"


def score_candidate(
    candidate: CandidateAddress, official_domain: str | None, company_domains: list[str] | None = None
) -> tuple[float, Tier]:
    tier = classify_domain(candidate.source_url, official_domain, company_domains)
    return TIER_WEIGHTS[tier], tier


def rank_candidates(
    candidates: list[CandidateAddress],
    official_domain: str | None,
    company_domains: list[str] | None = None,
) -> list[dict]:
    """Score (by source tier) and sort candidates descending. Returns a list of dicts with
    candidate/tier/score."""
    ranked = []
    for candidate in candidates:
        score, tier = score_candidate(candidate, official_domain, company_domains)
        ranked.append({"candidate": candidate, "tier": tier, "score": score})
    ranked.sort(key=lambda r: r["score"], reverse=True)
    return ranked


_ADDRESS_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _address_tokens(address: str) -> frozenset[str]:
    """Bag-of-words for near-duplicate address comparison: diacritics stripped, case-folded,
    punctuation dropped -- so word ORDER (e.g. a reordered city/postcode/country tail) and
    stray punctuation don't block a match that a plain string-equality check would miss."""
    text = unicodedata.normalize("NFKD", address)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return frozenset(_ADDRESS_TOKEN_RE.findall(text.lower()))


# Below this many shared tokens, treat two addresses as unrelated regardless of the containment
# ratio -- otherwise two short, low-information addresses (e.g. "London Office" vs any other
# London address) could score a spurious 100% containment off just one or two common words.
_MIN_SHARED_TOKENS = 4

# When a group being deduped has at least this many addresses, filter out "common" tokens (see
# _COMMON_TOKEN_DF_RATIO) before comparing. Below it, there usually isn't enough data to tell
# genuine boilerplate apart from a coincidence, and two truly-identical short addresses would
# have every token flagged common and end up with nothing left to compare -- so small groups
# compare on raw tokens instead.
_MIN_GROUP_SIZE_FOR_COMMON_TOKEN_FILTER = 5

# A token appearing in at least this fraction of a group's addresses is treated as generic
# locale boilerplate (a city/province/country name repeated across every site in that area, or
# a filler word like "street"/"road"), not something that distinguishes one site from another.
# Needed for datasets with many addresses per city/country (e.g. a geocoded site list) -- there,
# raw token overlap is dominated by shared administrative-area words, which can make two
# genuinely different streets in the same city look like a near-total containment match.
_COMMON_TOKEN_DF_RATIO = 0.3

# Once common tokens are filtered out, require at least this many shared SIGNIFICANT tokens.
# Kept close to _MIN_SHARED_TOKENS (rather than dropped to 1-2) for two reasons found against
# real data in site_extraction.csv: (1) pinyin-transliterated Chinese addresses split into
# short, semantically weak syllables (e.g. "hua", "lu", "dong") that recur across genuinely
# unrelated street names -- "Long Hua Qu" (a district) and "Dong Hua Lu" (an unrelated street)
# shared "hua"/"lu" by coincidence. (2) A city/province name pair (e.g. "Guangzhou" +
# "Guangdong") can dodge the _COMMON_TOKEN_DF_RATIO filter in a country-wide group that spans
# many cities/provinces, even though it carries no more specific information than "somewhere in
# this city" -- letting a bare "Guangzhou, Guangdong Province, China" placeholder (3 tokens,
# all shared) swallow a fully-detailed address in that same city. Requiring 4 forces at least
# one token of real specificity (a street/building name, district, or number) beyond a bare
# city+province+country match.
_MIN_SIGNIFICANT_SHARED_TOKENS = 4


def _containment(a: frozenset[str], b: frozenset[str], min_shared: int = _MIN_SHARED_TOKENS) -> float:
    """Fraction of the SMALLER token set that's also in the larger one.

    Unlike Jaccard (intersection / union), this isn't diluted when one address is a superset of
    the other -- e.g. one copy of a candidate gets a company-name prefix ("Capgemini Singapore
    Pte Ltd, Marina Bay ...") that the other lacks. Jaccard would score that pair ~0.8 (union
    grows by the 3 extra words); containment correctly scores it 1.0 since every word of the
    shorter address appears in the longer one.
    """
    shared = len(a & b)
    if shared < min_shared:
        return 0.0
    smaller = min(len(a), len(b))
    return shared / smaller if smaller else 0.0


_NUMERIC_TOKEN_MIN_DIGITS = 3


def _numeric_tokens(tokens: frozenset[str]) -> frozenset[str]:
    return frozenset(t for t in tokens if t.isdigit() and len(t) >= _NUMERIC_TOKEN_MIN_DIGITS)


def _drops_unmatched_number(candidate_raw: frozenset[str], anchor_raw: frozenset[str]) -> bool:
    """True if `candidate_raw` carries a street-number/suite/postal-code-like token (3+ digits)
    while `anchor_raw` -- the entry that would survive the merge -- has none at all.

    Found against real data (EATON CORPORATION/China): several genuinely different Shenzhen
    streets, each with a distinct postal code, all matched a bare "Bao Shi Lu, Shenzhen,
    Guangdong, China" anchor on word overlap alone (city/province words plus a coincidental
    shared pinyin syllable), because that anchor -- lacking any number of its own -- had nothing
    to disagree with. A number the anchor can't represent at all is a stronger "these might
    differ" signal than raw word overlap is a "these might match" one, so it wins as a veto.
    """
    return bool(_numeric_tokens(candidate_raw)) and not _numeric_tokens(anchor_raw)


def dedupe_by_address_similarity(items: list, address_of, threshold: float = ADDRESS_DEDUPE_JACCARD_THRESHOLD) -> list:
    """Collapse near-duplicate addresses from `items`, keeping the first occurrence of each
    duplicate cluster (see `_containment` for what counts as "near-duplicate").

    `address_of(item)` extracts the address string to compare -- callers aren't assumed to be
    working with `CandidateAddress`/ranked-entry dicts specifically. Order matters: pass `items`
    pre-sorted by whatever "most credible/preferred first" means for the data (e.g.
    `rank_candidates`'s best-score-first output), since that's the entry each cluster keeps.

    `items` should already be scoped to addresses that could plausibly be the same site (e.g.
    one company's candidates, or one supplier+country's site list) -- the common-token filter
    below treats whatever's shared across most of `items` as boilerplate, so mixing in unrelated
    locales would wash out real city/country signal.
    """
    raw_tokens = [_address_tokens(address_of(item)) for item in items]

    n = len(items)
    if n >= _MIN_GROUP_SIZE_FOR_COMMON_TOKEN_FILTER:
        doc_frequency: dict[str, int] = {}
        for tokens in raw_tokens:
            for token in tokens:
                doc_frequency[token] = doc_frequency.get(token, 0) + 1
        common = {token for token, count in doc_frequency.items() if count / n >= _COMMON_TOKEN_DF_RATIO}
        compare_tokens = [tokens - common for tokens in raw_tokens]
        min_shared = _MIN_SIGNIFICANT_SHARED_TOKENS
    else:
        compare_tokens = raw_tokens
        min_shared = _MIN_SHARED_TOKENS

    kept: list = []
    kept_tokens: list[frozenset[str]] = []
    kept_raw: list[frozenset[str]] = []
    for item, tokens, raw in zip(items, compare_tokens, raw_tokens):
        match = any(
            _containment(tokens, seen, min_shared) >= threshold and not _drops_unmatched_number(raw, seen_raw)
            for seen, seen_raw in zip(kept_tokens, kept_raw)
        )
        if match:
            continue
        kept.append(item)
        kept_tokens.append(tokens)
        kept_raw.append(raw)
    return kept


def dedupe_ranked_by_address(
    ranked: list[dict], threshold: float = ADDRESS_DEDUPE_JACCARD_THRESHOLD
) -> list[dict]:
    """Collapse near-duplicate addresses from an already-ranked (best-score-first) list.

    The agent can report the same physical site more than once -- reworded, with its address
    components in a different order, or (e.g.) with a company-name prefix added by only one of
    two sources -- so plain string equality on (address, source_url) misses these. Comparing
    each address as an unordered bag of words catches that while still telling apart genuinely
    different sites, since a different street number/city/postal code is usually enough tokens
    out of the total to drop below `threshold`.

    Assumes `ranked` is already sorted best-score-first (as `rank_candidates` returns it), so
    keeping the first entry in each duplicate cluster keeps the most credible source.
    """
    return dedupe_by_address_similarity(ranked, lambda entry: entry["candidate"].full_address(), threshold)
