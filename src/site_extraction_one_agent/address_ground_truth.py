"""Load and compare against groundtruth.csv -- a tab-delimited, structured-address ground
truth file (supplier_name, street_address, address_line_2, city, state, postal_code,
country, confidence, category), distinct from the lat/long-only supplier_site_locations.csv
handled by ground_truth.py.

This file has no coordinates, so comparison is by normalized city+country text match rather
than geocoding -- geocoding every ground-truth row (hundreds per company) would blow through
Nominatim's ~1 req/sec rate limit for no real benefit, since we only need to know whether a
candidate's city plausibly corresponds to a known site, not compute a precise distance.
"""

import csv
from pathlib import Path

_COLUMNS = (
    "supplier_name",
    "street_address",
    "address_line_2",
    "city",
    "state",
    "postal_code",
    "country",
    "confidence",
    "category",
)


def load_address_ground_truth(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Address ground truth CSV not found at {csv_path}.")
    rows: list[dict] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for raw_row in csv.reader(f, delimiter="\t"):
            if len(raw_row) < len(_COLUMNS):
                continue
            rows.append({col: raw_row[i].strip() for i, col in enumerate(_COLUMNS)})
    return rows


def rows_for_company(rows: list[dict], company_name: str) -> list[dict]:
    """Case-insensitive exact match on supplier_name, falling back to substring match."""
    name = company_name.strip().upper()
    exact = [r for r in rows if r["supplier_name"].strip().upper() == name]
    if exact:
        return exact
    return [r for r in rows if name in r["supplier_name"].strip().upper()]


def _norm(text: str) -> str:
    return " ".join(text.strip().upper().split())


def compare_by_city_country(ranked: list[dict], gt_rows: list[dict]) -> list[dict]:
    """Augment each ranked entry with whether its city/country match any known ground-truth
    row for this company. Cheap text comparison -- no network calls."""
    gt_by_city_country: dict[tuple[str, str], list[dict]] = {}
    for row in gt_rows:
        key = (_norm(row["city"]), _norm(row["country"]))
        gt_by_city_country.setdefault(key, []).append(row)

    for entry in ranked:
        candidate = entry["candidate"]
        city = _norm(candidate.city or "")
        country = _norm(candidate.country or "")
        matches = gt_by_city_country.get((city, country), []) if city else []
        # Fall back to country-only match if no exact city match, still informative.
        if not matches and country:
            matches = [r for r in gt_rows if _norm(r["country"]) == country]
        entry["gt_address_city_match"] = bool(matches) and bool(city)
        entry["gt_address_country_match"] = bool(matches)
        entry["gt_address_sample_match"] = matches[0] if matches else None

    return ranked
