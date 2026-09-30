"""Score this validation run against 'Sites Sample Data.xlsx' ground truth with the exact matching
rules of results/sites_sample_20260923/compare.py, and put it next to that run's numbers.

    uv run python results/search_ladder_test_20260924/score.py
"""
import csv
import glob
import json
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parent
BASE = RUN.parent / "sites_sample_20260923"
sys.path.insert(0, str(BASE))
import compare as cmp  # noqa: E402  (its normalizers and street_match)

cmp.RUN = BASE  # load_gt() reads the normalized ground truth from the baseline run's folder


def load_agent() -> dict:
    per: dict = {}
    for path in sorted(glob.glob(str(RUN / "*__effort-medium.csv"))):
        with open(path, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                sup = (r.get("supplier_name") or "").strip().upper()
                tok, dig = cmp.street_tokens(r.get("street_address", ""), r.get("address_line_2", ""))
                per.setdefault(sup, []).append({
                    "tok": tok, "digits": dig,
                    "city": cmp.norm_city(r.get("city", "")),
                    "country": cmp.norm_country(r.get("country", "")),
                    "postal": cmp.norm_postal(r.get("postal_code", "")),
                    "confidence": r.get("confidence", ""),
                    "n_sources": (r.get("all_sources") or "").count('"') // 2,
                })
    return per


def load_usage(since: float) -> dict:
    per = {}
    with open(RUN.parent.parent / "usage_log.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["timestamp"] >= since:
                per[r["company"].strip().upper()] = r
    return per


def recall(g: list, a: list) -> tuple[int, int]:
    city = sum(
        any(
            not (row["country"] and c["country"] and row["country"] != c["country"])
            and ((row["postal"] and row["postal"] == c["postal"])
                 or (row["city"] and c["city"] and (row["city"] == c["city"] or row["city"] in c["city"] or c["city"] in row["city"])))
            for c in a
        )
        for row in g
    )
    pairs = sorted(
        ((cmp.containment(row["tok"], c["tok"]), gi, ai)
         for gi, row in enumerate(g) for ai, c in enumerate(a) if cmp.street_match(row, c)),
        key=lambda p: -p[0],
    )
    used_g, used_a = set(), set()
    for _, gi, ai in pairs:
        if gi not in used_g and ai not in used_a:
            used_g.add(gi)
            used_a.add(ai)
    return city, len(used_g)


if __name__ == "__main__":
    gt, ag = cmp.load_gt(), load_agent()
    since = min(Path(p).stat().st_ctime for p in glob.glob(str(RUN / "suppliers.csv")))
    usage = load_usage(since)
    base = {r["supplier_name"]: r for r in csv.DictReader(open(BASE / "comparison.csv", encoding="utf-8"))}
    cols = ["supplier", "gt", "sites_old", "sites_new", "rc_old", "rc_new", "rs_old", "rs_new",
            "searches_old", "searches_new", "paid_fb_old", "paid_fb_new", "cost_old", "cost_new",
            "high_conf_new", "rows_3plus_sources", "refused_dup", "refused_cert", "ocr_pdfs"]
    out = []
    for sup in sorted(ag):
        g, a, u, b = gt.get(sup, []), ag[sup], usage.get(sup, {}), base.get(sup, {})
        city, street = recall(g, a) if g else (0, 0)
        counts = u.get("tool_call_counts", {})
        out.append({
            "supplier": sup, "gt": len(g),
            "sites_old": b.get("agent_sites", ""), "sites_new": len(a),
            "rc_old": b.get("recall_city", ""), "rc_new": round(city / len(g), 2) if g else "",
            "rs_old": b.get("recall_street", ""), "rs_new": round(street / len(g), 2) if g else "",
            "searches_old": b.get("web_search_calls", ""), "searches_new": u.get("web_search_calls", ""),
            "paid_fb_old": b.get("paid_fallbacks", ""), "paid_fb_new": counts.get("fetch_page_search_fallback", 0),
            "cost_old": b.get("cost_usd", ""), "cost_new": u.get("cost_usd", ""),
            "high_conf_new": sum(1 for x in a if x["confidence"] == "High"),
            "rows_3plus_sources": sum(1 for x in a if x["n_sources"] >= 3),
            "refused_dup": u.get("web_searches_refused_duplicate", ""),
            "refused_cert": u.get("web_searches_refused_no_certificates", ""),
            "ocr_pdfs": counts.get("fetch_page_pdf_ocr", 0),
        })
    with open(RUN / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(r)
