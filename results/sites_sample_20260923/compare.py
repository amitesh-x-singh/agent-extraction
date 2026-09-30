"""Score the agent's run against 'Sites Sample Data.xlsx' ground truth.

Two recall levels per supplier:
  * city level   -- a GT site counts as found if the agent reported any site in the same
                    (city, country).
  * street level -- a GT site counts as found only if some agent row matches its actual
                    street address, matched one-to-one (greedy, best containment first), so
                    five GT sites in one city cannot all be satisfied by a single agent row.

'New sites' are agent rows that matched no GT row at street level.
"""
import csv
import glob
import json
import re
import unicodedata
from pathlib import Path

RUN = Path(__file__).resolve().parent
SCRATCH = Path(
    r"C:\Users\AMITES~1\AppData\Local\Temp\claude"
    r"\c--Users-AmiteshSingh-Development-site-extraction-one-agent"
    r"\9ffc8afd-0d9b-4fda-b6cf-f2a3d00f6c8a\scratchpad"
)

# ---------------------------------------------------------------- normalizers
# The answer key is written in unaccented capitals ('DUSSELDORF', 'BRUHLER STRASSE'); the agent
# reports the real spelling ('Dusseldorf' with an umlaut, 'Bruhler Strasse' with an eszett).
# Without folding, every accented character became a space and split the token, so whole
# German/French/Spanish cities silently failed to match. NFKD strips combining marks; the
# letters below do not decompose and need naming.
_LIGATURES = {
    "ß": "ss", "ẞ": "SS",      # eszett
    "ø": "o", "Ø": "O",        # o-slash
    "æ": "ae", "Æ": "AE",      # ae
    "œ": "oe", "Œ": "OE",      # oe
    "ł": "l", "Ł": "L",        # l-stroke
    "đ": "d", "Đ": "D",        # d-stroke
    "ð": "d", "Ð": "D", "þ": "th", "Þ": "TH",
}


def fold(s):
    """Accent- and ligature-folded ASCII, so the two spellings of a name compare equal."""
    s = s or ""
    for ch, rep in _LIGATURES.items():
        s = s.replace(ch, rep)
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


_COUNTRY_CANON = {
    "UNITED STATES OF AMERICA": "US", "UNITED STATES": "US", "USA": "US", "U.S.A.": "US",
    "US": "US", "U.S.": "US", "AMERICA": "US",
    "UNITED KINGDOM OF GREAT BRITAIN AND NORTHERN IRELAND": "GB", "UNITED KINGDOM": "GB",
    "UK": "GB", "U.K.": "GB", "GREAT BRITAIN": "GB", "ENGLAND": "GB", "SCOTLAND": "GB",
    "WALES": "GB", "NORTHERN IRELAND": "GB",
    "KOREA (REPUBLIC OF)": "KR", "SOUTH KOREA": "KR", "KOREA": "KR", "REPUBLIC OF KOREA": "KR",
    "VIET NAM": "VN", "VIETNAM": "VN",
    "RUSSIAN FEDERATION": "RU", "RUSSIA": "RU",
    "CZECHIA": "CZ", "CZECH REPUBLIC": "CZ",
    "IRAN (ISLAMIC REPUBLIC OF)": "IR", "IRAN": "IR",
    "BOLIVIA (PLURINATIONAL STATE OF)": "BO", "BOLIVIA": "BO",
    "VENEZUELA (BOLIVARIAN REPUBLIC OF)": "VE", "VENEZUELA": "VE",
    "TAIWAN": "TW", "TAIWAN, PROVINCE OF CHINA": "TW", "CHINESE TAIPEI": "TW",
    "HONG KONG": "HK", "HONG KONG SAR": "HK",
    "MACAO": "MO", "MACAU": "MO",
    "UNITED ARAB EMIRATES": "AE", "UAE": "AE",
    "NETHERLANDS": "NL", "THE NETHERLANDS": "NL", "HOLLAND": "NL",
    "TURKEY": "TR", "TURKIYE": "TR",
    "SLOVAKIA": "SK", "SLOVAK REPUBLIC": "SK",
    "IVORY COAST": "CI",
    "PUERTO RICO": "PR",
}
_COUNTRY_WORD = {
    "GERMANY": "DE", "DEUTSCHLAND": "DE", "FRANCE": "FR", "ITALY": "IT", "ITALIA": "IT",
    "SPAIN": "ES", "INDIA": "IN", "CHINA": "CN", "JAPAN": "JP",
    "CANADA": "CA", "MEXICO": "MX", "BRAZIL": "BR", "BRASIL": "BR", "AUSTRALIA": "AU",
    "SWEDEN": "SE", "SWITZERLAND": "CH", "SINGAPORE": "SG", "POLAND": "PL", "BELGIUM": "BE",
    "AUSTRIA": "AT", "DENMARK": "DK", "NORWAY": "NO", "FINLAND": "FI", "IRELAND": "IE",
    "PORTUGAL": "PT", "MALAYSIA": "MY", "THAILAND": "TH", "INDONESIA": "ID",
    "PHILIPPINES": "PH", "SOUTH AFRICA": "ZA", "NEW ZEALAND": "NZ", "ARGENTINA": "AR",
    "CHILE": "CL", "COLOMBIA": "CO", "PERU": "PE", "HUNGARY": "HU", "ROMANIA": "RO",
    "GREECE": "GR", "ISRAEL": "IL", "EGYPT": "EG", "SAUDI ARABIA": "SA", "QATAR": "QA",
    "KUWAIT": "KW", "BAHRAIN": "BH", "OMAN": "OM", "JORDAN": "JO", "MOROCCO": "MA",
    "NIGERIA": "NG", "KENYA": "KE", "GHANA": "GH", "UKRAINE": "UA", "LITHUANIA": "LT",
    "LATVIA": "LV", "ESTONIA": "EE", "SLOVENIA": "SI", "CROATIA": "HR", "SERBIA": "RS",
    "BULGARIA": "BG", "LUXEMBOURG": "LU", "MALTA": "MT", "CYPRUS": "CY", "ICELAND": "IS",
    "URUGUAY": "UY", "PARAGUAY": "PY", "ECUADOR": "EC", "COSTA RICA": "CR", "PANAMA": "PA",
    "GUATEMALA": "GT", "HONDURAS": "HN", "NICARAGUA": "NI", "EL SALVADOR": "SV",
    "PAKISTAN": "PK", "BANGLADESH": "BD", "SRI LANKA": "LK", "MONGOLIA": "MN",
    "ARMENIA": "AM", "AZERBAIJAN": "AZ", "TUNISIA": "TN", "ALGERIA": "DZ", "ANGOLA": "AO",
    "ZAMBIA": "ZM", "ZIMBABWE": "ZW", "UGANDA": "UG", "NAMIBIA": "NA", "BOTSWANA": "BW",
    "SENEGAL": "SN", "MAURITIUS": "MU", "FIJI": "FJ", "HAITI": "HT", "JAMAICA": "JM",
    "GUYANA": "GY", "LIECHTENSTEIN": "LI", "SAN MARINO": "SM", "GUERNSEY": "GG",
    "JERSEY": "JE", "PAPUA NEW GUINEA": "PG", "NEW CALEDONIA": "NC",
    "FRENCH POLYNESIA": "PF",
}
_COUNTRY_CANON.update(_COUNTRY_WORD)


def norm_country(s):
    t = " ".join(fold(s).upper().split()).strip(" ,.")
    if not t:
        return ""
    if t in _COUNTRY_CANON:
        return _COUNTRY_CANON[t]
    # last resort token scan, so 'Shanghai, China' or 'USA (Texas)' still resolve
    for key, code in _COUNTRY_CANON.items():
        if re.search(r"\b" + re.escape(key) + r"\b", t):
            return code
    return t


_CITY_STRIP = re.compile(r"[^A-Z0-9 ]+")
_CITY_DROP = {"CITY", "TOWN", "SHI", "KU", "DISTRICT", "PROVINCE", "COUNTY", "THE"}


def norm_city(s):
    t = _CITY_STRIP.sub(" ", fold(s).upper())
    return " ".join(w for w in t.split() if w not in _CITY_DROP)


_ABBR = {
    "STREET": "ST", "STRASSE": "STR", "ROAD": "RD", "AVENUE": "AVE", "AV": "AVE",
    "BOULEVARD": "BLVD", "BOUL": "BLVD", "DRIVE": "DR", "LANE": "LN", "COURT": "CT",
    "SUITE": "STE", "BUILDING": "BLDG", "FLOOR": "FL", "HIGHWAY": "HWY", "PARKWAY": "PKWY",
    "SQUARE": "SQ", "PLACE": "PL", "NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W",
    "SAINT": "ST", "INDUSTRIAL": "IND", "AVENIDA": "AVE", "VIALE": "VIA", "STRADA": "VIA",
}
_STREET_NOISE = {
    "STE", "BLDG", "FL", "UNIT", "NO", "PO", "BOX", "IND", "ESTATE", "ZONE", "AREA",
    "PHASE", "BLOCK", "PLOT", "LTD", "GMBH", "INC", "LLC", "SA", "BV", "AG", "CO",
    "CORP", "AND", "OF", "THE", "DE", "DEL", "LA", "EL", "DI", "N", "S", "E", "W",
}
_PUNCT = re.compile(r"[^A-Z0-9]+")


def street_tokens(*parts):
    """(meaningful tokens, digit-bearing tokens) for an address's street portion."""
    raw = _PUNCT.sub(" ", fold(" ".join(p or "" for p in parts)).upper())
    toks = []
    for w in raw.split():
        w = _ABBR.get(w, w)
        if len(w) == 1 and not w.isdigit():
            continue
        toks.append(w)
    allt = set(toks)
    digits = {w for w in allt if any(ch.isdigit() for ch in w)}
    meaningful = allt - _STREET_NOISE
    return (meaningful or allt), digits


def norm_postal(s):
    return re.sub(r"[^A-Z0-9]", "", fold(s).upper())


def containment(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def street_match(gt, ag):
    if gt["country"] and ag["country"] and gt["country"] != ag["country"]:
        return False
    c = containment(gt["tok"], ag["tok"])
    same_postal = bool(gt["postal"]) and gt["postal"] == ag["postal"]
    same_city = bool(gt["city"]) and gt["city"] == ag["city"]
    if not (same_city or same_postal):
        # allow one city string to contain the other ('BROOKLYN' vs 'BROOKLYN NY')
        if not (gt["city"] and ag["city"]
                and (gt["city"] in ag["city"] or ag["city"] in gt["city"])):
            return False
    if same_postal and c >= 0.40:
        return True
    if c >= 0.60:
        d_gt, d_ag = gt["digits"], ag["digits"]
        if d_gt and d_ag and not (d_gt & d_ag):
            return c >= 0.85  # both name a house number and they disagree
        return True
    return False


# ---------------------------------------------------------------- load inputs
def load_gt():
    per = {}
    with open(RUN / "gt_normalized.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sup = r["supplier_name"].strip().upper()
            tok, dig = street_tokens(r["street_address1"], r["street_address2"])
            per.setdefault(sup, []).append({
                "raw": ", ".join(x for x in [r["street_address1"], r["street_address2"],
                                             r["city"], r["state"], r["postal_code"],
                                             r["country"]] if x),
                "tok": tok, "digits": dig,
                "city": norm_city(r["city"]), "country": norm_country(r["country"]),
                "postal": norm_postal(r["postal_code"]),
            })
    return per


def load_agent():
    """Read the per-company files rather than the per-shard combined ones. The batch driver
    writes a company's own file first and only then rewrites its shard's combined file, so
    when the run was stopped partway the per-company directory is the complete record."""
    per = {}
    for path in sorted(glob.glob(str(RUN / "per_company" / "*.csv"))):
        with open(path, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                sup = (r.get("supplier_name") or "").strip().upper()
                if not sup:
                    continue
                tok, dig = street_tokens(r.get("street_address", ""), r.get("address_line_2", ""))
                per.setdefault(sup, []).append({
                    "raw": ", ".join(x for x in [r.get("street_address", ""), r.get("city", ""),
                                                 r.get("state", ""), r.get("postal_code", ""),
                                                 r.get("country", "")] if x),
                    "tok": tok, "digits": dig,
                    "city": norm_city(r.get("city", "")),
                    "country": norm_country(r.get("country", "")),
                    "postal": norm_postal(r.get("postal_code", "")),
                    "tier": r.get("tier", ""), "source_url": r.get("source_url", ""),
                })
    return per


def load_costs():
    """A usage-log line is appended only after a company's results are on disk, so its
    presence is what marks that supplier as having run to completion."""
    per = {}
    for p in (glob.glob(str(SCRATCH / "shard*" / "usage_log.jsonl"))
              + glob.glob(str(SCRATCH / "pilot" / "usage_log.jsonl"))):
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                counts = r.get("tool_call_counts", {})
                per[r["company"].strip().upper()] = {
                    # Runs made before usage.py priced cached input separately wrote
                    # 'estimated_cost_usd'; runs made after write the exact 'cost_usd'.
                    "cost": r.get("cost_usd", r.get("estimated_cost_usd")),
                    "cost_is_exact": "cost_usd" in r,
                    "breakdown": r.get("cost_breakdown_usd"),
                    "web_search_calls": r["web_search_calls"],
                    "fetch_calls": counts.get("fetch_page", 0),
                    "paid_fallbacks": counts.get("fetch_page_search_fallback", 0),
                }
    return per


def score():
    gt, ag, costs = load_gt(), load_agent(), load_costs()
    out = []
    for sup in sorted(gt):
        g, a = gt[sup], ag.get(sup, [])
        # --- city level: same country, and either the same city (exact, or one name
        # contained in the other) or the same postal code. Deliberately the same locality
        # test street_match() applies before it looks at the street itself, so city-level
        # recall is always >= street-level recall rather than the two disagreeing.
        city_hit = [False] * len(g)
        for gi, row in enumerate(g):
            for cand in a:
                if row["country"] and cand["country"] and row["country"] != cand["country"]:
                    continue
                if row["postal"] and row["postal"] == cand["postal"]:
                    city_hit[gi] = True
                    break
                rc, ac = row["city"], cand["city"]
                if rc and ac and (rc == ac or rc in ac or ac in rc):
                    city_hit[gi] = True
                    break
        city_hits = sum(city_hit)
        # --- street level, one-to-one greedy
        pairs = []
        for gi, row in enumerate(g):
            for ai, cand in enumerate(a):
                if street_match(row, cand):
                    pairs.append((containment(row["tok"], cand["tok"]), gi, ai))
        pairs.sort(key=lambda p: -p[0])
        used_g, used_a = set(), set()
        for _, gi, ai in pairs:
            if gi in used_g or ai in used_a:
                continue
            used_g.add(gi)
            used_a.add(ai)
        c = costs.get(sup, {})
        out.append({
            "supplier_name": sup,
            "gt_sites": len(g),
            "agent_sites": len(a),
            "new_sites": len(a) - len(used_a),
            "recall_city": round(city_hits / len(g), 4) if g else 0.0,
            "recall_street": round(len(used_g) / len(g), 4) if g else 0.0,
            "matched_city": city_hits,
            "matched_street": len(used_g),
            "cost_usd": c.get("cost"),
            "cost_is_exact": c.get("cost_is_exact", False),
            "web_search_calls": c.get("web_search_calls"),
            "fetch_calls": c.get("fetch_calls"),
            "paid_fallbacks": c.get("paid_fallbacks"),
            "ran": bool(c),
        })
    return out


if __name__ == "__main__":
    rows = score()
    ran = [r for r in rows if r["ran"]]
    with open(RUN / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(RUN / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    print("suppliers scored: %d/%d" % (len(ran), len(rows)))
    if ran:
        gts = sum(r["gt_sites"] for r in ran)
        print("GT sites %d  agent sites %d  new %d"
              % (gts, sum(r["agent_sites"] for r in ran), sum(r["new_sites"] for r in ran)))
        print("micro recall  city %.1f%%  street %.1f%%"
              % (100 * sum(r["matched_city"] for r in ran) / gts,
                 100 * sum(r["matched_street"] for r in ran) / gts))
        print("macro recall  city %.1f%%  street %.1f%%"
              % (100 * sum(r["recall_city"] for r in ran) / len(ran),
                 100 * sum(r["recall_street"] for r in ran) / len(ran)))
        tc = [r["cost_usd"] for r in ran if r["cost_usd"] is not None]
        print("cost total $%.2f  mean $%.3f" % (sum(tc), sum(tc) / max(len(tc), 1)))
