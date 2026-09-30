"""Pick 20 suppliers from the 122-supplier run (results/sites_sample_20260923) for a re-run with
the current pipeline. Free: reads files on disk only. Fixed seed, so the pick is reproducible.

    uv run python results/rerun20_20260925/select.py
"""
import csv
import glob
import random
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parent
BASE = RUN.parent / "sites_sample_20260923"
sys.path.insert(0, str(RUN.parent / "_analysis"))
import cert_pdf_yield as cy  # noqa: E402

ALREADY_RERUN = {"SCM GROUP", "KANTAR", "HUSKY INJECTION MOLDING SYSTEM",
                 "TOPPAN LEEFUNG PRINTING", "COSTACURTA SPA VICO"}
rng = random.Random(20260925)

rows = [r for r in csv.DictReader(open(BASE / "comparison.csv", encoding="utf-8"))
        if r["ran"] == "True" and int(r["gt_sites"]) > 0 and r["supplier_name"] not in ALREADY_RERUN]
cert_sup = {r["supplier_name"] for p in glob.glob(str(BASE / "shard_0*_combined.csv"))
            for r in cy.load(p) if cy.is_cert_pdf(r)}

cert = sorted(r["supplier_name"] for r in rows if r["supplier_name"] in cert_sup)
pick = rng.sample(cert, min(6, len(cert)))
rest = [r for r in rows if r["supplier_name"] not in pick]
strata = {"low": [r for r in rest if float(r["recall_street"]) < 0.4],
          "mid": [r for r in rest if 0.4 <= float(r["recall_street"]) < 0.8],
          "high": [r for r in rest if float(r["recall_street"]) >= 0.8]}
for name, k in (("low", 6), ("mid", 4), ("high", 4)):
    pick += rng.sample(sorted(r["supplier_name"] for r in strata[name]), k)

with open(RUN / "suppliers.csv", "w", newline="", encoding="utf-8") as f:
    f.write("supplier_name\n" + "\n".join(pick) + "\n")
for i in range(4):
    d = RUN / f"shard{i:02d}"
    d.mkdir(exist_ok=True)
    with open(d / "suppliers.csv", "w", newline="", encoding="utf-8") as f:
        f.write("supplier_name\n" + "\n".join(pick[i::4]) + "\n")
base = {r["supplier_name"]: r for r in rows}
for s in pick:
    b = base[s]
    print(f"{s:50s} gt={b['gt_sites']:>3} rs_old={b['recall_street']:>5} cost_old={b['cost_usd']} cert={s in cert_sup}")
print({k: len(v) for k, v in strata.items()}, "cert pool", len(cert))
