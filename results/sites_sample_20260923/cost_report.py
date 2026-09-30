"""What the completed run actually cost, separating what is exact from what is not.

The run finished before usage.py recorded the cached / cache-write split of input tokens, so
its per-call input prices cannot be reconstructed. Everything else in the bill can: output
tokens, the web_search per-call fee and ScrapingBee credits are all exact counts. This reports
the exact lines, then brackets the one unknown line between its two extremes.
"""
import glob
import json
import os
import time

SC = (r"C:\Users\AMITES~1\AppData\Local\Temp\claude"
      r"\c--Users-AmiteshSingh-Development-site-extraction-one-agent"
      r"\9ffc8afd-0d9b-4fda-b6cf-f2a3d00f6c8a\scratchpad")

INPUT, CACHED, CACHE_WRITE, OUTPUT = 0.20, 0.02, 0.25, 1.20
WEB_SEARCH_CALL, BEE_CREDIT = 0.01, 0.0002


def _load():
    rows = []
    for p in (glob.glob(os.path.join(SC, "shard*", "usage_log.jsonl"))
              + glob.glob(os.path.join(SC, "pilot", "usage_log.jsonl"))):
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def summarize():
    rows = _load()
    t = {}
    for r in rows:
        for k, v in (("chat_in", r["chat_input_tokens"]),
                     ("chat_out", r["chat_output_tokens"]),
                     ("srch_in", r["web_search_input_tokens"]),
                     ("srch_out", r["web_search_output_tokens"]),
                     ("srch_calls", r["web_search_calls"]),
                     ("credits", r["scrapingbee_credits"]),
                     ("old", r["estimated_cost_usd"])):
            t[k] = t.get(k, 0) + v

    total_in = t["chat_in"] + t["srch_in"]
    out_cost = (t["chat_out"] + t["srch_out"]) / 1e6 * OUTPUT
    fee_cost = t["srch_calls"] * WEB_SEARCH_CALL
    bee_cost = t["credits"] * BEE_CREDIT
    exact = out_cost + fee_cost + bee_cost
    ts = [r["timestamp"] for r in rows]
    return {
        "runs": len(rows),
        "suppliers": len({r["company"] for r in rows}),
        "start": time.strftime("%H:%M", time.localtime(min(ts))),
        "end": time.strftime("%H:%M", time.localtime(max(ts))),
        "date": time.strftime("%d %B %Y", time.localtime(min(ts))),
        "output_tokens": t["chat_out"] + t["srch_out"],
        "output_cost": round(out_cost, 3),
        "search_calls": t["srch_calls"],
        "search_fee": round(fee_cost, 3),
        "credits": t["credits"],
        "bee_cost": round(bee_cost, 3),
        "exact_subtotal": round(exact, 2),
        "input_tokens": total_in,
        "chat_input_tokens": t["chat_in"],
        "search_input_tokens": t["srch_in"],
        # Search content cannot be a cache READ: a cache hit needs an identical prefix that
        # was already sent, and across 4,043 queries in this run exactly one repeated, so
        # every search call carried a unique query and unique retrieved page text. Its
        # tokens are therefore uncached or cache-write, never the $0.02 rate. Chat input is
        # the opposite case -- one conversation re-sent ~7 times -- so it spans the full range.
        "search_low": round(t["srch_in"] / 1e6 * INPUT, 2),
        "search_high": round(t["srch_in"] / 1e6 * CACHE_WRITE, 2),
        "chat_low": round(t["chat_in"] / 1e6 * CACHED, 2),
        "chat_high": round(t["chat_in"] / 1e6 * CACHE_WRITE, 2),
        "low": round(exact + t["srch_in"] / 1e6 * INPUT + t["chat_in"] / 1e6 * CACHED, 2),
        "high": round(exact + t["srch_in"] / 1e6 * CACHE_WRITE + t["chat_in"] / 1e6 * CACHE_WRITE, 2),
        "naive_low": round(exact + total_in / 1e6 * CACHED, 2),
        "logged": round(t["old"], 2),
    }


if __name__ == "__main__":
    s = summarize()
    print("runs logged: %d (%d unique suppliers; ADAFRUIT ran twice, pilot + shard)"
          % (s["runs"], s["suppliers"]))
    print("window: %s to %s on %s\n" % (s["start"], s["end"], s["date"]))
    print("EXACT lines (counts the providers reported, times published rates)")
    print("  output tokens        %14s  $%8.3f" % ("{:,}".format(s["output_tokens"]), s["output_cost"]))
    print("  web_search call fee  %14s  $%8.3f" % ("{:,} calls".format(s["search_calls"]), s["search_fee"]))
    print("  scrapingbee          %14s  $%8.3f" % ("{:,} credits".format(s["credits"]), s["bee_cost"]))
    print("  %-35s $%8.3f" % ("subtotal, exact", s["exact_subtotal"]))
    print("\nINPUT TOKENS %s  (split not recorded at run time)" % "{:,}".format(s["input_tokens"]))
    print("  search content %12s  $%6.2f - $%6.2f   cannot be a cache read: 1 repeated"
          % ("{:,}".format(s["search_input_tokens"]), s["search_low"], s["search_high"]))
    print("  %31s query in 4,043, so no prefix to hit" % "")
    print("  chat           %12s  $%6.2f - $%6.2f   one conversation re-sent ~7x, so"
          % ("{:,}".format(s["chat_input_tokens"]), s["chat_low"], s["chat_high"]))
    print("  %31s most of it is genuinely cacheable" % "")
    print("\nRUN TOTAL   $%.2f - $%.2f" % (s["low"], s["high"]))
    print("  (a naive bracket that let search cache too would start at $%.2f; ruled out above)"
          % s["naive_low"])
    print("  figure this run logged, pricing all input as uncached: $%.2f" % s["logged"])
