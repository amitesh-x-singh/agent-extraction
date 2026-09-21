"""Offline checks (no API cost) for address_ground_truth.py and results_csv.py."""

import csv
import tempfile
from pathlib import Path

from site_extraction_one_agent.address_ground_truth import compare_by_city_country, load_address_ground_truth, rows_for_company
from site_extraction_one_agent.results_csv import write_results_csv
from site_extraction_one_agent.schemas import CandidateAddress

# --- address_ground_truth.load_address_ground_truth / rows_for_company ---
with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp) / "gt.csv"
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        f.write("CAPGEMINI\t123 Main St\t\tPARIS\tILE DE FRANCE\t75001\tFRANCE\tHigh\tAutomation\n")
        f.write("CAPGEMINI\t456 Side St\t\tLONDON\tENGLAND\tSW1A\tUNITED KINGDOM\tHigh\tAutomation\n")
        f.write("OTHERCO\t789 Other St\t\tBERLIN\tBERLIN\t10115\tGERMANY\tMedium\tConsulting\n")

    rows = load_address_ground_truth(tmp_path)
    assert len(rows) == 3, rows
    assert rows[0]["supplier_name"] == "CAPGEMINI"
    assert rows[0]["city"] == "PARIS"

    capgemini_rows = rows_for_company(rows, "CAPGEMINI")
    assert len(capgemini_rows) == 2, capgemini_rows
    print("load_address_ground_truth / rows_for_company: OK")

    # --- compare_by_city_country ---
    candidates = [
        CandidateAddress(street_address="123 Main St, Paris", city="Paris", country="France",
                          source_url="https://x.com", evidence_quote="q"),
        CandidateAddress(street_address="999 Nowhere Ave, Tokyo", city="Tokyo", country="Japan",
                          source_url="https://y.com", evidence_quote="q"),
    ]
    ranked = [
        {"candidate": candidates[0], "tier": "A", "score": 1.0},
        {"candidate": candidates[1], "tier": "D", "score": 0.05},
    ]
    ranked = compare_by_city_country(ranked, capgemini_rows)
    assert ranked[0]["gt_address_city_match"] is True, ranked[0]
    assert ranked[1]["gt_address_city_match"] is False, ranked[1]
    assert ranked[1]["gt_address_country_match"] is False, ranked[1]
    print("compare_by_city_country: OK")

    # --- results_csv.write_results_csv ---
    out_path = Path(tmp) / "results.csv"
    write_results_csv(out_path, "CAPGEMINI", ranked)
    with out_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        written_rows = list(reader)
    assert len(written_rows) == 2, written_rows
    assert written_rows[0]["supplier_name"] == "CAPGEMINI"
    assert written_rows[0]["street_address"] == "123 Main St, Paris"
    assert written_rows[0]["tier"] == "A"
    assert written_rows[0]["gt_address_city_match"] == "True"
    assert "gt_distance_km" not in written_rows[0], "geocoding columns must not appear in this project"
    assert "gt_coord_matched" not in written_rows[0], "geocoding columns must not appear in this project"
    print("write_results_csv: OK")

print("\nALL ADDRESS-GT / RESULTS-CSV CHECKS PASSED")
