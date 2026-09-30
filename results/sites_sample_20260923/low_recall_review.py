"""Build a manual-verification sheet for the suppliers the agent scored worst on.

One row per address, ground truth and agent side by side, so each line can be judged by eye:

  MATCHED    a ground-truth site the agent found, with the address it returned for it. Check
             whether the pairing is genuine or the matcher was too generous.
  NEAR-MISS? a ground-truth site the scorer counted as missed, next to an agent address that
             looks like it anyway -- same locality plus a shared house number or postal code,
             but not enough shared wording to clear the matching threshold. These are where
             reported recall most likely understates the agent, so they are worth reading
             first. Real example: ground truth "22ND FLOOR, RAFFLES CITY EAST TOWER, NO 1089,
             DONGDAMING ROAD, SHANGHAI" against the agent's "1089 Dongdaming Road, Floors
             21-22, Shanghai" -- the same building, scored as a miss because the ground truth
             names the tower and the agent does not.
  MISSED     a ground-truth site with no plausible agent address at all.
  EXTRA      an agent address matching no ground-truth site. Check whether it is a real site
             absent from the ground truth, or something the agent should not have returned.

A NEAR-MISS? pairing is advisory and did NOT count towards the recall figures in the report.
Nothing here changes those numbers; it marks the rows a human should rule on.

Suppliers are ordered worst street-level recall first, so the sheet can be worked top-down and
abandoned at any point.
"""
import csv
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN))
import compare as C  # noqa: E402

THRESHOLD = 0.75  # street-level recall below this counts as "low"
OUT = RUN / "low_recall_review.csv"

FIELDS = [
    "supplier_name",
    "street_recall",
    "status",
    "gt_address",
    "agent_address",
    "match_score",
    "agent_source_url",
    "agent_tier",
]


def main():
    gt, ag = C.load_gt(), C.load_agent()
    costs = C.load_costs()
    scored = {r["supplier_name"]: r for r in C.score() if r["ran"]}
    low = sorted((s for s, r in scored.items() if r["recall_street"] < THRESHOLD),
                 key=lambda s: (scored[s]["recall_street"], -scored[s]["gt_sites"]))

    out_rows = []
    for sup in low:
        g, a = gt.get(sup, []), ag.get(sup, [])
        recall = "%.0f%%" % (100 * scored[sup]["recall_street"])

        # Same greedy one-to-one matching score() uses, kept here so the sheet shows exactly
        # the pairings the reported recall was computed from.
        pairs = []
        for gi, row in enumerate(g):
            for ai, cand in enumerate(a):
                if C.street_match(row, cand):
                    pairs.append((C.containment(row["tok"], cand["tok"]), gi, ai))
        pairs.sort(key=lambda p: -p[0])
        gt_to_agent, used_a = {}, set()
        for score, gi, ai in pairs:
            if gi in gt_to_agent or ai in used_a:
                continue
            gt_to_agent[gi] = (ai, score)
            used_a.add(ai)

        # Second, looser pass over what the strict pass left behind, to surface probable
        # matcher failures rather than leave a reviewer to find them by reading everything.
        # Deliberately permissive: it only has to be worth a human glance, and every hit is
        # labelled as a question rather than an answer.
        near = []
        for gi, row in enumerate(g):
            if gi in gt_to_agent:
                continue
            for ai, cand in enumerate(a):
                if ai in used_a:
                    continue
                if row["country"] and cand["country"] and row["country"] != cand["country"]:
                    continue
                same_postal = bool(row["postal"]) and row["postal"] == cand["postal"]
                shared_number = bool(row["digits"] & cand["digits"])
                if not (same_postal or shared_number):
                    continue
                near.append((C.containment(row["tok"], cand["tok"]), gi, ai))
        near.sort(key=lambda p: -p[0])
        gt_to_near = {}
        for score, gi, ai in near:
            if gi in gt_to_near or ai in used_a:
                continue
            gt_to_near[gi] = (ai, score)
            used_a.add(ai)

        for gi, row in enumerate(g):
            if gi in gt_to_agent:
                ai, score = gt_to_agent[gi]
                cand = a[ai]
                status = "MATCHED"
            elif gi in gt_to_near:
                ai, score = gt_to_near[gi]
                cand = a[ai]
                status = "NEAR-MISS?"
            else:
                out_rows.append({
                    "supplier_name": sup, "street_recall": recall, "status": "MISSED",
                    "gt_address": row["raw"], "agent_address": "", "match_score": "",
                    "agent_source_url": "", "agent_tier": "",
                })
                continue
            out_rows.append({
                "supplier_name": sup, "street_recall": recall, "status": status,
                "gt_address": row["raw"], "agent_address": cand["raw"],
                "match_score": "%.2f" % score,
                "agent_source_url": cand["source_url"], "agent_tier": cand["tier"],
            })
        for ai, cand in enumerate(a):
            if ai not in used_a:
                out_rows.append({
                    "supplier_name": sup, "street_recall": recall, "status": "EXTRA",
                    "gt_address": "", "agent_address": cand["raw"], "match_score": "",
                    "agent_source_url": cand["source_url"], "agent_tier": cand["tier"],
                })

    # utf-8-sig so Excel renders the accented characters in these addresses correctly.
    with OUT.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out_rows)

    counts = {}
    for r in out_rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("wrote %s" % OUT)
    print("  %d suppliers below %.0f%% street recall" % (len(low), 100 * THRESHOLD))
    print("  %d rows: %d MATCHED, %d NEAR-MISS?, %d MISSED, %d EXTRA"
          % (len(out_rows), counts.get("MATCHED", 0), counts.get("NEAR-MISS?", 0),
             counts.get("MISSED", 0), counts.get("EXTRA", 0)))
    print("\nworst first:")
    for sup in low[:12]:
        r = scored[sup]
        print("  %-42s street %3.0f%%  city %3.0f%%  gt %3d  agent %3d"
              % (sup[:41], 100 * r["recall_street"], 100 * r["recall_city"],
                 r["gt_sites"], r["agent_sites"]))


if __name__ == "__main__":
    main()
