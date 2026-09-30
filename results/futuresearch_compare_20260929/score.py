"""Score FutureSearch's site research for the 20 re-run suppliers against 'Sites Sample Data.xlsx',
next to our agent's latest run (results/rerun20_20260925) and the union of the two. Same matching
rules as results/sites_sample_20260923/compare.py. Free: reads files on disk only.

    uv run --with pandas --with openpyxl python results/futuresearch_compare_20260929/score.py
"""
import csv
import glob
import importlib.util
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RERUN = ROOT / "results" / "rerun20_20260925"
FS_XLSX = ROOT / "agent_test_sample_SiteResearch_Result_13-05-46__29-09-2026.xlsx"

# rerun20's scorer is also called score.py, so load it by path under another name
_spec = importlib.util.spec_from_file_location("rerun20_score", RERUN / "score.py")
rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rs)
cmp = rs.cmp


def load_futuresearch():
    df = pd.read_excel(FS_XLSX, dtype=str).fillna("")
    per = {}
    for r in df.to_dict("records"):
        sup = r["raw_name"].strip().upper()
        tok, dig = cmp.street_tokens(r["street_address"])
        per.setdefault(sup, []).append({
            "raw": ", ".join(x for x in [r["street_address"], r["city"], r["state_province"],
                                         r["postal_code"], r["country"]] if x),
            "tok": tok, "digits": dig, "city": cmp.norm_city(r["city"]),
            "country": cmp.norm_country(r["country"]), "postal": cmp.norm_postal(r["postal_code"]),
        })
    return per


def dedup_pairs(a, f):
    """One-to-one agent/FutureSearch rows that are the same site under the street rule."""
    pairs = sorted(((cmp.containment(x["tok"], y["tok"]), i, j)
                    for i, x in enumerate(a) for j, y in enumerate(f) if cmp.street_match(x, y)),
                   key=lambda p: -p[0])
    ui, uj, n = set(), set(), 0
    for _, i, j in pairs:
        if i not in ui and j not in uj:
            ui.add(i)
            uj.add(j)
            n += 1
    return n


def pct(n, d):
    return round(n / d, 3) if d else 0.0


if __name__ == "__main__":
    picks = [x.strip() for x in open(RERUN / "suppliers.csv", encoding="utf-8").read().splitlines()[1:] if x.strip()]
    gt = cmp.load_gt()
    agent = rs.load_rows(sorted(glob.glob(str(RERUN / "*__effort-medium.csv"))), set(picks))
    fs = load_futuresearch()
    assert set(fs) == set(picks), set(fs) ^ set(picks)
    base = {r["supplier"]: r for r in csv.DictReader(open(RERUN / "comparison.csv", encoding="utf-8"))}

    out, detail = [], []
    for sup in picks:
        g, a, f = gt[sup], agent.get(sup, []), fs[sup]
        ca, sa = rs.match(g, a)
        cf, sf = rs.match(g, f)
        # our side must reproduce the re-run's own scores exactly
        assert len(ca) == int(base[sup]["city_new"]) and len(sa) == int(base[sup]["street_new"]), sup
        n = len(g)
        row = {"supplier": sup, "gt_sites": n, "agent_sites": len(a), "fs_sites": len(f)}
        for lvl, A, F in (("city", ca, cf), ("street", set(sa), set(sf))):
            row.update({
                f"agent_{lvl}_matched": len(A), f"agent_{lvl}_recall": pct(len(A), n),
                f"fs_{lvl}_matched": len(F), f"fs_{lvl}_recall": pct(len(F), n),
                f"both_{lvl}": len(A & F), f"agent_only_{lvl}": len(A - F),
                f"fs_only_{lvl}": len(F - A), f"neither_{lvl}": n - len(A | F),
                f"combined_{lvl}_matched": len(A | F), f"combined_{lvl}_recall": pct(len(A | F), n),
            })
            assert len(A & F) + len(A - F) + len(F - A) + row[f"neither_{lvl}"] == n
        dup = dedup_pairs(a, f)
        row.update({
            "agent_fs_same_site": dup, "agent_unique_sites": len(a) - dup, "fs_unique_sites": len(f) - dup,
            "combined_unique_sites": len(a) + len(f) - dup,
        })
        out.append(row)
        for gi, s in enumerate(g):
            detail.append({
                "supplier": sup, "gt_address": s["raw"],
                "agent_city": gi in ca, "agent_street": gi in sa,
                "fs_city": gi in cf, "fs_street": gi in sf,
                "agent_match": a[sa[gi]]["raw"] if gi in sa else "",
                "fs_match": f[sf[gi]]["raw"] if gi in sf else "",
            })

    for name, rows in (("comparison.csv", out), ("gt_site_detail.csv", detail)):
        with open(HERE / name, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    def tot(k):
        return sum(r[k] for r in out)

    G, S = tot("gt_sites"), len(out)
    print(f"suppliers {S}  gt sites {G}  agent rows {tot('agent_sites')}  fs rows {tot('fs_sites')}  "
          f"combined unique rows {tot('combined_unique_sites')} (same site in both: {tot('agent_fs_same_site')}, "
          f"agent only: {tot('agent_unique_sites')}, fs only: {tot('fs_unique_sites')})")
    for lvl in ("city", "street"):
        print(f"\n[{lvl}]            matched   micro   macro")
        for who in ("agent", "fs", "combined"):
            m = tot(f"{who}_{lvl}_matched")
            print(f"  {who:9s}       {m:4d}/{G}  {m / G:6.1%}  {tot(f'{who}_{lvl}_recall') / S:6.1%}")
        print(f"  both {tot(f'both_{lvl}')}  agent-only {tot(f'agent_only_{lvl}')}  "
              f"fs-only {tot(f'fs_only_{lvl}')}  neither {tot(f'neither_{lvl}')}")
