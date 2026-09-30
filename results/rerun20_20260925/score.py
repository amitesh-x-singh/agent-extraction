"""Score the 20-supplier re-run against 'Sites Sample Data.xlsx' with the exact matching rules of
results/sites_sample_20260923/compare.py, next to that 122-supplier run's numbers for the same 20.
Free: reads files on disk only.

    uv run python results/rerun20_20260925/score.py
"""
import csv
import glob
import json
import re
import sys
from pathlib import Path

csv.field_size_limit(10**9)
sys.stdout.reconfigure(encoding="utf-8")
RUN = Path(__file__).resolve().parent
BASE = RUN.parent / "sites_sample_20260923"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(RUN.parent / "_analysis"))
import compare as cmp  # noqa: E402
import cert_pdf_yield as cy  # noqa: E402

cmp.RUN = BASE
OLD_LOGS = Path(r"C:\Users\AMITES~1\AppData\Local\Temp\claude"
                r"\c--Users-AmiteshSingh-Development-site-extraction-one-agent"
                r"\9ffc8afd-0d9b-4fda-b6cf-f2a3d00f6c8a\scratchpad")
INPUT, CACHED, CACHE_WRITE, OUTPUT, SEARCH_FEE, BEE = 0.20, 0.02, 0.25, 1.20, 0.01, 0.0002
CERT_QUERY = re.compile(r"ISO 9001|IATF|AS9100|certificat", re.I)


def load_rows(paths, keep):
    per = {}
    for p in paths:
        for r in cy.load(p):
            sup = (r.get("supplier_name") or "").strip().upper()
            if sup not in keep:
                continue
            tok, dig = cmp.street_tokens(r.get("street_address", ""), r.get("address_line_2", ""))
            per.setdefault(sup, []).append({
                "raw": ", ".join(x for x in [r.get("street_address", ""), r.get("city", ""),
                                             r.get("postal_code", ""), r.get("country", "")] if x),
                "tok": tok, "digits": dig, "city": cmp.norm_city(r.get("city", "")),
                "country": cmp.norm_country(r.get("country", "")),
                "postal": cmp.norm_postal(r.get("postal_code", "")),
                "cert": cy.has_any_cert_pdf(r), "source_url": r.get("source_url", ""),
            })
    return per


def city_hit(row, c):
    if row["country"] and c["country"] and row["country"] != c["country"]:
        return False
    if row["postal"] and row["postal"] == c["postal"]:
        return True
    rc, ac = row["city"], c["city"]
    return bool(rc and ac and (rc == ac or rc in ac or ac in rc))


def match(g, a):
    """compare.py's rules: (city-matched GT indices, {GT index: agent index} one-to-one street pairs)."""
    city = {gi for gi, row in enumerate(g) if any(city_hit(row, c) for c in a)}
    pairs = sorted(((cmp.containment(row["tok"], c["tok"]), gi, ai)
                    for gi, row in enumerate(g) for ai, c in enumerate(a) if cmp.street_match(row, c)),
                   key=lambda p: -p[0])
    st, used = {}, set()
    for _, gi, ai in pairs:
        if gi not in st and ai not in used:
            st[gi] = ai
            used.add(ai)
    return city, st


def usage(paths):
    per = {}
    for p in paths:
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                per[r["company"].strip().upper()] = r
    return per


def old_bracket(u):
    """Same bracket as sites_sample_20260923/cost_report.py, per supplier."""
    exact = ((u["chat_output_tokens"] + u["web_search_output_tokens"]) / 1e6 * OUTPUT
             + u["web_search_calls"] * SEARCH_FEE + u["scrapingbee_credits"] * BEE)
    return (exact + u["web_search_input_tokens"] / 1e6 * INPUT + u["chat_input_tokens"] / 1e6 * CACHED,
            exact + (u["web_search_input_tokens"] + u["chat_input_tokens"]) / 1e6 * CACHE_WRITE)


def uncached_equiv(u):
    """The new run priced the way the old run logged its estimate (all input uncached): like-for-like."""
    return ((u["chat_input_tokens"] + u["web_search_input_tokens"]) / 1e6 * INPUT
            + (u["chat_output_tokens"] + u["web_search_output_tokens"]) / 1e6 * OUTPUT
            + u["web_search_calls"] * SEARCH_FEE + u["scrapingbee_credits"] * BEE)


if __name__ == "__main__":
    lines = open(RUN / "suppliers.csv", encoding="utf-8").read().splitlines()[1:]
    picks = [x.strip() for x in lines if x.strip()]
    keep = set(picks)
    gt = cmp.load_gt()
    old = load_rows(sorted(glob.glob(str(BASE / "per_company" / "*.csv"))), keep)
    new = load_rows(sorted(glob.glob(str(RUN / "*__effort-medium.csv"))), keep)
    base = {r["supplier_name"]: r for r in csv.DictReader(open(BASE / "comparison.csv", encoding="utf-8"))}
    uo = usage(glob.glob(str(OLD_LOGS / "shard*" / "usage_log.jsonl")))
    un = usage(glob.glob(str(RUN / "shard*" / "usage_log.jsonl")))

    out, lost, cert_rows = [], [], []
    for sup in picks:
        g, a_old, a_new = gt.get(sup, []), old.get(sup, []), new.get(sup, [])
        c_old, s_old = match(g, a_old)
        c_new, s_new = match(g, a_new)
        # the old side must reproduce the baseline run's own scores exactly
        assert len(s_old) == int(base[sup]["matched_street"]), (sup, len(s_old), base[sup]["matched_street"])
        assert len(c_old) == int(base[sup]["matched_city"]), (sup, len(c_old), base[sup]["matched_city"])
        u_o, u_n = uo[sup], un.get(sup)
        lo, hi = old_bracket(u_o)
        q = u_n.get("queries", []) if u_n else []
        cnt = u_n.get("tool_call_counts", {}) if u_n else {}
        pdf_paid = sum(1 for d in (u_n or {}).get("scrapingbee_details", []) if ".pdf" in d.get("url", "").lower())
        st_cert = {gi for gi, ai in s_new.items() if a_new[ai]["cert"]}
        ci_cert = {gi for gi in c_new if any(c["cert"] and city_hit(g[gi], c) for c in a_new)}
        out.append({
            "supplier": sup, "ran_new": u_n is not None, "gt": len(g),
            "sites_old": len(a_old), "sites_new": len(a_new),
            "city_old": len(c_old), "city_new": len(c_new),
            "street_old": len(s_old), "street_new": len(s_new),
            "rc_old": round(len(c_old) / len(g), 2), "rc_new": round(len(c_new) / len(g), 2),
            "rs_old": round(len(s_old) / len(g), 2), "rs_new": round(len(s_new) / len(g), 2),
            "searches_old": u_o["web_search_calls"], "searches_new": u_n and u_n["web_search_calls"],
            "sb_credits_old": u_o["scrapingbee_credits"], "sb_credits_new": u_n and u_n["scrapingbee_credits"],
            "cost_old_logged_uncached": u_o["estimated_cost_usd"],
            "cost_old_low": round(lo, 4), "cost_old_high": round(hi, 4),
            "cost_new_exact": u_n and u_n["cost_usd"],
            "cost_new_uncached_equiv": u_n and round(uncached_equiv(u_n), 4),
            "cert_sites_old": sum(r["cert"] for r in a_old), "cert_sites_new": sum(r["cert"] for r in a_new),
            "gt_street_via_cert_new": len(st_cert), "gt_street_via_cert_not_in_old": len(st_cert - set(s_old)),
            "gt_city_via_cert_new": len(ci_cert), "gt_city_via_cert_not_in_old": len(ci_cert - c_old),
            "cert_searches": sum(1 for x in q if x.get("tool") == "web_search" and CERT_QUERY.search(x.get("query", ""))),
            "cert_refused": u_n and u_n.get("web_searches_refused_no_certificates"),
            "pdfs_local": u_n and u_n.get("pdfs_extracted_locally"),
            "pdfs_ocr": cnt.get("fetch_page_pdf_ocr", 0), "pdf_paid_fallbacks": pdf_paid,
        })
        if u_n is None:
            continue
        for gi in sorted(set(s_old) - set(s_new)):
            near = sorted(a_new, key=lambda c: -cmp.containment(g[gi]["tok"], c["tok"]))
            lost.append({"supplier": sup, "gt_address": g[gi]["raw"], "old_match": a_old[s_old[gi]]["raw"],
                         "city_still_matched": gi in c_new,
                         "nearest_new_row": near[0]["raw"] if near else "",
                         "nearest_containment": round(cmp.containment(g[gi]["tok"], near[0]["tok"]), 2) if near else ""})
        for gi in sorted(st_cert | ci_cert):
            ai = s_new.get(gi)
            cert_rows.append({"supplier": sup, "gt_address": g[gi]["raw"],
                              "street_matched_by_cert_row": gi in st_cert,
                              "matched_in_old_street": gi in s_old, "matched_in_old_city": gi in c_old,
                              "new_row": a_new[ai]["raw"] if ai is not None else "",
                              "source_url": a_new[ai]["source_url"] if ai is not None else ""})

    for name, rows in (("comparison.csv", out), ("lost_sites.csv", lost), ("cert_pdf_attribution.csv", cert_rows)):
        with open(RUN / name, "w", newline="", encoding="utf-8") as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)

    done = [r for r in out if r["ran_new"]]
    if not done:
        print("old side reproduces baseline for all", len(out), "suppliers; no new results yet")
        sys.exit()

    def tot(k):
        return sum(r[k] or 0 for r in done)

    G = tot("gt")
    print(f"suppliers done {len(done)}/{len(out)}  gt sites {G}")
    for lvl, k in (("city", "rc"), ("street", "rs")):
        mo, mn = tot(f"{lvl}_old"), tot(f"{lvl}_new")
        print(f"{lvl:6s} micro {mo}/{G}={mo / G:.1%} -> {mn}/{G}={mn / G:.1%}   "
              f"macro {tot(k + '_old') / len(done):.1%} -> {tot(k + '_new') / len(done):.1%}")
    print(f"cost old: logged (all input uncached) ${tot('cost_old_logged_uncached'):.2f}; "
          f"true range ${tot('cost_old_low'):.2f}-${tot('cost_old_high'):.2f}")
    print(f"cost new: exact ${tot('cost_new_exact'):.2f}; priced like the old estimate ${tot('cost_new_uncached_equiv'):.2f}")
    print(f"searches {tot('searches_old')} -> {tot('searches_new')}   "
          f"scrapingbee credits {tot('sb_credits_old')} -> {tot('sb_credits_new')}")
    print(f"cert-pdf sites {tot('cert_sites_old')} -> {tot('cert_sites_new')}; "
          f"GT street via cert {tot('gt_street_via_cert_new')} (not matched in old {tot('gt_street_via_cert_not_in_old')}); "
          f"GT city via cert {tot('gt_city_via_cert_new')} (not matched in old {tot('gt_city_via_cert_not_in_old')})")
    print(f"pdfs local {tot('pdfs_local')}  ocr {tot('pdfs_ocr')}  paid pdf fallbacks {tot('pdf_paid_fallbacks')}  "
          f"cert searches {tot('cert_searches')}  cert refused {tot('cert_refused')}")
    print(f"street matches lost vs old: {len(lost)}")
