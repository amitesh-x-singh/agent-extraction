"""Writes ranked research results to a CSV, using groundtruth.csv's column layout for the
first columns (supplier_name, street_address, address_line_2, city, state, postal_code,
country, confidence, category) so rows can be visually/structurally compared against it,
with our own evidence/entity columns appended after. No coordinate/geocoding columns -- this
project doesn't geocode candidates against a lat/long ground truth."""

import csv
from pathlib import Path

# Mirrors groundtruth.csv's 9-column layout exactly (see address_ground_truth.py), plus our
# own evidence/entity/scoring columns appended. groundtruth.csv itself has no header row; we add
# one here since these extra columns need labels.
FIELDNAMES = [
    "supplier_name",
    "street_address",
    "address_line_2",
    "city",
    "state",
    "postal_code",
    "country",
    "confidence",
    "category",
    # --- appended: our own evidence/entity/scoring columns, not in groundtruth.csv ---
    "legal_entity",
    "site_name",
    "site_type",
    "ownership_type",
    "operational_status",
    "status_as_of",
    "tier",
    "score",
    "source_url",
    "evidence_quote",
    "gt_address_city_match",
    "gt_address_country_match",
    "all_sources",
]

_TIER_TO_CONFIDENCE = {"A": "High", "B": "Medium", "C": "Low", "D": "Low"}
# A site found on this many distinct pages/documents is corroborated whatever its best source's tier.
HIGH_CONFIDENCE_MIN_SOURCES = 3


def confidence_for(tier: str, n_sources: int) -> str:
    """High/Medium/Low for a row: from the best source's tier, promoted to High when at least
    HIGH_CONFIDENCE_MIN_SOURCES distinct sources list the same address. The one place this is
    decided, so the CSV and the API/UI never disagree."""
    if n_sources >= HIGH_CONFIDENCE_MIN_SOURCES:
        return "High"
    return _TIER_TO_CONFIDENCE.get(tier, "")


def _row(company: str, entry: dict) -> dict:
    """One CSV row from one ranked entry. Factored out so the per-company and combined writers
    below cannot drift apart in column contents."""
    c = entry["candidate"]
    return {
        "supplier_name": company,
        "street_address": c.street_address,
        "address_line_2": "",
        "city": c.city or "",
        "state": c.state_province or "",
        "postal_code": c.postal_code or "",
        "country": c.country or "",
        "confidence": confidence_for(entry["tier"], len(c.all_sources())),
        "category": "",
        "legal_entity": c.legal_entity or "",
        "site_name": c.site_name or "",
        "site_type": c.site_type or "",
        "ownership_type": c.ownership_type or "",
        "operational_status": c.operational_status or "",
        "status_as_of": c.status_as_of or "",
        "tier": entry["tier"],
        "score": f"{entry['score']:.2f}",
        "source_url": c.source_url,
        "evidence_quote": c.evidence_quote,
        "gt_address_city_match": entry.get("gt_address_city_match", ""),
        "gt_address_country_match": entry.get("gt_address_country_match", ""),
        # Every source for this site, best first, each URL quoted and comma-separated.
        "all_sources": ", ".join(f'"{u}"' for u in c.all_sources()),
    }


def write_results_csv(path: Path, company: str, ranked: list[dict], *, delimiter: str = "\t") -> Path:
    return write_combined_results_csv(path, [(company, ranked)], delimiter=delimiter)


def write_combined_results_csv(
    path: Path, per_company: list[tuple[str, list[dict]]], *, delimiter: str = "\t"
) -> Path:
    """One file, one header, every company's ranked rows concatenated -- the same shape as a
    single-company file, so a combined run and a per-company run stay directly comparable.
    Rewritten from scratch on every call rather than appended to, so a batch run that rewrites
    it after each company always leaves a complete, valid file behind if it dies partway.

    `delimiter` defaults to a TAB, not a comma, despite the .csv extension: every existing file
    in results/ is tab-delimited and is read back that way by the comparison tooling, so the
    default cannot change without invalidating them. Pass "," for a conventional CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter=delimiter)
        writer.writeheader()
        for company, ranked in per_company:
            for entry in ranked:
                writer.writerow(_row(company, entry))
    return path
