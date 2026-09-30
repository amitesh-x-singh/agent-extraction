"""Ad-hoc verification script (not pytest) for scoring.py -- no LLM call needed."""

from site_extraction_one_agent.schemas import CandidateAddress
from site_extraction_one_agent.scoring import (
    classify_domain,
    dedupe_ranked_by_address,
    rank_candidates,
    score_candidate,
)

# --- scoring.classify_domain ---
assert classify_domain("https://www.sec.gov/cgi-bin/browse-edgar", None) == "A"
assert classify_domain("https://find-and-update.company-information.service.gov.uk/company/123", None) == "A"
assert classify_domain("https://www.google.com/maps/place/x", None) == "B"
assert classify_domain("https://www.linkedin.com/company/acme", None) == "B"
assert classify_domain("https://www.zoominfo.com/c/acme-inc", None) == "C"
assert classify_domain("https://randomblog.example.com/post", None) == "D"
assert classify_domain("https://www.3m.com/3M/en_US/contact-us/", "3m.com") == "A"
assert classify_domain("https://news.3m.com/press-release", "3m.com") == "B"
assert classify_domain("https://3m.com.evil-lookalike.com/contact", "3m.com") == "D"
print("classify_domain: OK")

# --- scoring.score_candidate ---
c = CandidateAddress(street_address="1 Foo St", source_url="https://sec.gov/x", evidence_quote="q")
score, tier = score_candidate(c, None)
assert score == 1.0 and tier == "A", (score, tier)
print("score_candidate: OK")

# --- scoring.rank_candidates ---
candidates = [
    CandidateAddress(street_address="A", source_url="https://sec.gov/a", evidence_quote="a"),
    CandidateAddress(street_address="B", source_url="https://zoominfo.com/b", evidence_quote="b"),
    CandidateAddress(street_address="C", source_url="https://linkedin.com/c", evidence_quote="c"),
]
ranked = rank_candidates(candidates, None)
assert ranked[0]["candidate"].street_address == "A"  # sec.gov -- Tier A, 1.0
assert ranked[0]["score"] == 1.0
assert ranked[-1]["candidate"].street_address == "B"  # zoominfo.com -- Tier C, lowest
assert ranked[-1]["score"] == 0.25
print("rank_candidates: OK")

# --- scoring.dedupe_ranked_by_address ---


def _ranked_entry(address: str, score: float, source_url: str = "https://example.com/x") -> dict:
    return {
        "candidate": CandidateAddress(street_address=address, source_url=source_url, evidence_quote="q"),
        "tier": "A",
        "score": score,
    }


# Real duplicate from a Capgemini run: same office, reordered postcode/country tail, reported
# by two different sources (own site + Companies House) -- exact (address, source_url) match
# would miss this.
reordered = dedupe_ranked_by_address(
    [
        _ranked_entry("95 Queen Victoria Street, London, EC4V 4HN, United Kingdom", 1.0, "https://capgemini.com/locations"),
        _ranked_entry(
            "95 Queen Victoria Street, London, United Kingdom, EC4V 4HN",
            1.0,
            "https://find-and-update.company-information.service.gov.uk/company/00943935",
        ),
    ]
)
assert len(reordered) == 1, reordered
# The dropped duplicate's source is kept as an extra source of the surviving row, best first.
assert reordered[0]["candidate"].all_sources() == [
    "https://capgemini.com/locations",
    "https://find-and-update.company-information.service.gov.uk/company/00943935",
], reordered[0]["candidate"].all_sources()

# Agent-reported extra sources survive, URLs appear once, and the CSV cell is quoted + comma-separated.
from site_extraction_one_agent.results_csv import _row  # noqa: E402

multi = CandidateAddress(
    street_address="1 Foo St",
    source_url="https://acme.com/cert.pdf",
    other_source_urls=["https://acme.com/locations", "https://acme.com/cert.pdf", " "],
    evidence_quote="q",
)
assert multi.all_sources() == ["https://acme.com/cert.pdf", "https://acme.com/locations"]
cell = _row("ACME", {"candidate": multi, "tier": "B", "score": 0.6})["all_sources"]
assert cell == '"https://acme.com/cert.pdf", "https://acme.com/locations"', cell
print("all_sources: merged on dedupe, deduplicated, quoted in CSV: OK")

# Real duplicate from the same run: same source, one copy has a company-name prefix the other
# lacks.
prefixed = dedupe_ranked_by_address(
    [
        _ranked_entry("Marina Bay Financial Centre, Tower 3, 12 Marina Boulevard, #30-01, Singapore 018982", 1.0),
        _ranked_entry(
            "Capgemini Singapore Pte Ltd, Marina Bay Financial Centre, Tower 3, 12 Marina Boulevard, #30-01, Singapore 018982",
            0.6,
        ),
    ]
)
assert len(prefixed) == 1 and prefixed[0]["score"] == 1.0, prefixed

# Genuinely different sites (different street number/city) must NOT be collapsed.
distinct = dedupe_ranked_by_address(
    [
        _ranked_entry("128 South Tryon Street, Charlotte, NC 28202, United States", 1.0),
        _ranked_entry("333 West Wacker Drive, Chicago, IL 60606, United States", 0.6),
    ]
)
assert len(distinct) == 2, distinct

# Large-group regression found in site_extraction.csv: many addresses in the same city/province
# share most of their tokens (city/province/country boilerplate repeated on every row), which
# let two genuinely DIFFERENT streets ("Zhong Quan Jie" / "Zhong Shan Lu" in Urumqi) look like a
# near-total containment match on raw tokens. With >= 5 addresses in the group, common tokens
# should be filtered out first so only the real street-name tokens decide the match.
boilerplate_heavy = dedupe_ranked_by_address(
    [
        _ranked_entry("Zhong Quan Jie, Tian Shan Qu, Wu Lu Mu Qi Shi, Xin Jiang Wei Wu Er Zi Zhi Qu, China, 830004", 1.0),
        _ranked_entry("Zhong Shan Lu, Wu Lu Mu Qi Shi, Xin Jiang Wei Wu Er Zi Zhi Qu, China", 0.9),
        _ranked_entry("Ba Yi Lu, Wu Lu Mu Qi Shi, Xin Jiang Wei Wu Er Zi Zhi Qu, China", 0.8),
        _ranked_entry("Jian She Lu, Wu Lu Mu Qi Shi, Xin Jiang Wei Wu Er Zi Zhi Qu, China", 0.7),
        _ranked_entry("You Hao Lu, Wu Lu Mu Qi Shi, Xin Jiang Wei Wu Er Zi Zhi Qu, China", 0.6),
    ]
)
assert len(boilerplate_heavy) == 5, boilerplate_heavy

# A placeholder address with only city/province/country (no street) must not swallow a more
# specific address just because all of its few tokens happen to appear in the longer one.
placeholder = dedupe_ranked_by_address(
    [
        _ranked_entry("Guangzhou, Guangdong Province, China", 1.0),
        _ranked_entry("Guangzhou Science City, Huangpu, Guangzhou, Guangdong Province, China, 510653", 0.9),
        _ranked_entry("Tianhe District, Guangzhou, Guangdong Province, China", 0.8),
        _ranked_entry("Nansha District, Guangzhou, Guangdong Province, China", 0.7),
        _ranked_entry("Panyu District, Guangzhou, Guangdong Province, China", 0.6),
    ]
)
assert len(placeholder) == 5, placeholder

# Real false-positive found in site_extraction.csv (FOXCONN/China): a district name and an
# unrelated street name shared only 2 coincidental pinyin syllables ("hua", "lu") once
# boilerplate was stripped -- both are short, low-information tokens common across many
# unrelated Chinese place names, not evidence they're the same site.
coincidental_pinyin = dedupe_ranked_by_address(
    [
        _ranked_entry("302 Gui Yue Lu, Long Hua Qu, Shen Zhen Shi, Guang Dong Sheng, China, 518110", 1.0),
        _ranked_entry("Dong Hua Lu, Guang Dong Sheng, China", 0.9),
        _ranked_entry("Ba Yi Lu, Guang Dong Sheng, China", 0.8),
        _ranked_entry("Jian She Lu, Guang Dong Sheng, China", 0.7),
        _ranked_entry("You Hao Lu, Guang Dong Sheng, China", 0.6),
    ]
)
assert len(coincidental_pinyin) == 5, coincidental_pinyin

# Real false-positive found in site_extraction.csv (AMPHENOL/China): in a country-wide group
# spanning many cities/provinces, "Guangzhou"/"Guangdong" don't recur often enough to hit
# _COMMON_TOKEN_DF_RATIO, so a bare "Guangzhou, Guangdong Province, China" placeholder (with no
# street/district/number of its own) swallowed a fully-detailed address in that same city --
# losing its district, sub-locality, and postal code.
city_province_placeholder = dedupe_ranked_by_address(
    [
        _ranked_entry("Guangzhou, Guangdong Province, China", 1.0),
        _ranked_entry("Wan'ancun, Nansha District, Guangzhou, Guangdong Province, China, 511485", 0.9),
        _ranked_entry("Shenzhen, Guangdong Province, China", 0.8),
        _ranked_entry("Dongguan, Guangdong Province, China", 0.7),
        _ranked_entry("Foshan, Guangdong Province, China", 0.6),
    ]
)
assert len(city_province_placeholder) == 5, city_province_placeholder

# Real false-positive found in site_extraction.csv (EATON CORPORATION/China): several
# genuinely different Shenzhen streets, each with a distinct postal code, all matched a bare
# "Bao Shi Lu, Shenzhen, Guangdong, China" anchor -- which has no postal code of its own -- on
# shared city/province words plus one coincidental pinyin-syllable overlap ("bao"). A numbered
# address should never be swallowed by a number-less anchor that has nothing to disagree with.
numberless_anchor = dedupe_ranked_by_address(
    [
        _ranked_entry("Bao Shi Lu, Shen Zhen Shi, Guang Dong Sheng, China", 1.0),
        _ranked_entry("Xin An Lu, Bao An Qu, Shen Zhen Shi, Guang Dong Sheng, China, 518104", 0.9),
        _ranked_entry("Le Zhu Jiao Lu, Bao An Qu, Shen Zhen Shi, Guang Dong Sheng, China, 518128", 0.8),
        _ranked_entry("Bao Shi Xi Lu, Bao An Qu, Shen Zhen Shi, Guang Dong Sheng, China, 518108", 0.7),
        _ranked_entry("Jin Shui Lu, Zheng Zhou Shi, He Nan Sheng, China", 0.6),
    ]
)
assert len(numberless_anchor) == 5, numberless_anchor
print("dedupe_ranked_by_address: OK")

# --- first-party company_domains (country sites and acquired-brand sites) ----------------------
# Without this list a multinational's own country site scores Tier D, level with a scraped
# directory listing: on one measured run that was 27 of 36 candidates.
OWN = ["acme.com", "acme.co.uk", "acme.com.my", "acmebrand.com"]
assert classify_domain("https://www.acme.co.uk/contact-us/", "acme.com", OWN) == "A"
assert classify_domain("https://www.acme.com.my/company/contact.html", "acme.com", OWN) == "A"
assert classify_domain("https://acmebrand.com/our-locations", "acme.com", OWN) == "A"
# Own domain, but not a contact/locations page -> B, same rule as the primary domain.
assert classify_domain("https://careers.acme.co.uk/openings", "acme.com", OWN) == "B"
assert classify_domain("https://www.acme.co.uk/contact-us/", "acme.com", None) == "D"  # old behaviour
# A lookalike domain must not sneak in, and third-party tiers still apply.
assert classify_domain("https://acme.co.uk.evil.com/contact", "acme.com", OWN) == "D"
assert classify_domain("https://www.zoominfo.com/c/acme", "acme.com", OWN) == "C"
# The list can be empty (an older run, or a company whose domain was never identified).
assert classify_domain("https://www.acme.com/contact", "acme.com", []) == "A"
print("classify_domain with company_domains: OK")

print("\nALL SCORING CHECKS PASSED")
