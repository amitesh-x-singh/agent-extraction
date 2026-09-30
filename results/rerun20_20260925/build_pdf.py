"""Build site_extraction_benchmark_20.pdf: our agent (this re-run) vs FutureSearch on the same 20 suppliers.
Reads comparison.csv here (exact agent cost) and ../futuresearch_compare_20260929/comparison.csv (recall,
overlap). Free: reads files on disk only.

    uv run --with reportlab python results/rerun20_20260925/build_pdf.py
"""
import csv
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

RUN = Path(__file__).resolve().parent
FSC = RUN.parent / "futuresearch_compare_20260929"
OUT = RUN / "site_extraction_benchmark_20.pdf"
cost = {r["supplier"]: float(r["cost_new_exact"])
        for r in csv.DictReader(open(RUN / "comparison.csv", encoding="utf-8"))}
rows = list(csv.DictReader(open(FSC / "comparison.csv", encoding="utf-8-sig")))
for r in rows:
    for k, v in r.items():
        if k != "supplier":
            r[k] = float(v)
    r["cost"] = cost[r["supplier"]]

# supplied by the user: FutureSearch's bill and run time for these 20, our agent's average run time
FS_COST, FS_MINUTES, AGENT_MIN_PER_SUPPLIER = 26.00, 8.4, 3.6

NAVY = colors.HexColor("#1F3A5F")
GRID = colors.HexColor("#C9D1DC")
STRIPE = colors.HexColor("#F3F6FA")
UP = colors.HexColor("#1E7B45")

ss = getSampleStyleSheet()
title = ParagraphStyle("t", parent=ss["Title"], textColor=NAVY, fontSize=18, spaceAfter=2, alignment=0)
sub = ParagraphStyle("s", parent=ss["Normal"], textColor=colors.HexColor("#5A6573"), fontSize=9.5)
h2 = ParagraphStyle("h", parent=ss["Heading2"], textColor=NAVY, fontSize=12, spaceBefore=10, spaceAfter=4)
body = ParagraphStyle("b", parent=ss["Normal"], fontSize=9.5, leading=13, leftIndent=10, spaceAfter=3)
cell = ParagraphStyle("c", parent=body, fontSize=7.5, leading=9, leftIndent=0, spaceAfter=0)
note = ParagraphStyle("n", parent=body, leftIndent=0, fontSize=7.5, leading=10, textColor=colors.HexColor("#5A6573"))


def tot(k):
    return sum(r[k] for r in rows)


N, G = len(rows), tot("gt_sites")
WHO = ("agent", "fs", "combined")
pooled = {(w, l): tot(f"{w}_{l}_matched") / G for w in WHO for l in ("street", "city")}
avg = {(w, l): sum(r[f"{w}_{l}_matched"] / r["gt_sites"] for r in rows) / N for w in WHO for l in ("street", "city")}
agent_cost = tot("cost")


story = [
    Paragraph("Site Extraction Benchmark: 20 Suppliers", title),
    Paragraph("Our agent (25 Sep 2026 run) vs FutureSearch, scored against "
              "<i>Sites Sample Data.xlsx</i> (225 ground-truth sites) &nbsp;|&nbsp; 29 Sep 2026", sub),
    Spacer(1, 8),
]
# summary: one row per measure, agent vs FutureSearch vs using both; the better of the two is green
a_min, f_min = AGENT_MIN_PER_SUPPLIER, FS_MINUTES / N
summary = [  # (label, agent, fs, both, agent value, fs value, higher is better; None = no winner)
    ("Street recall", *(f"{pooled[(w, 'street')]:.1%}" for w in WHO),
     pooled[("agent", "street")], pooled[("fs", "street")], True),
    ("City recall", *(f"{pooled[(w, 'city')]:.1%}" for w in WHO),
     pooled[("agent", "city")], pooled[("fs", "city")], True),
    ("Sites found", f"{tot('agent_sites'):.0f}", f"{tot('fs_sites'):.0f}",
     f"{tot('combined_unique_sites'):.0f} unique", 0, 0, None),
    ("Cost per supplier", f"${agent_cost / N:.2f}", f"${FS_COST / N:.2f}",
     f"${(agent_cost + FS_COST) / N:.2f}", agent_cost, FS_COST, False),
    ("Cost, 20 suppliers", f"${agent_cost:.2f}", f"${FS_COST:.2f}",
     f"${agent_cost + FS_COST:.2f}", agent_cost, FS_COST, False),
    ("Run time per supplier", f"{a_min:.1f} min *", f"{f_min:.2f} min", f"{a_min + f_min:.1f} min",
     a_min, f_min, False),
    ("Run time, 20 suppliers", f"{a_min * N:.0f} min *", f"{FS_MINUTES:.1f} min",
     f"{a_min * N + FS_MINUTES:.1f} min", a_min, f_min, False),
]
WIN_BG = colors.HexColor("#E3F2E8")
BOTH_BG = colors.HexColor("#E8ECF7")
sdata = [["", "Our agent", "FutureSearch", "Using both"]]
sstyle = [
    ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 9.5),
    ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"), ("FONTNAME", (3, 1), (3, -1), "Helvetica-Bold"),
    ("BACKGROUND", (3, 1), (3, -1), BOTH_BG),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("BOX", (0, 0), (-1, -1), 0.6, GRID), ("LINEBELOW", (0, 1), (-1, -1), 0.3, GRID),
    ("LINEBEFORE", (1, 1), (-1, -1), 0.3, GRID),
    ("LINEBELOW", (0, 3), (-1, 3), 1, NAVY), ("LINEBELOW", (0, 5), (-1, 5), 1, NAVY),  # group breaks
    ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]
for i, (label, a, f, both, av, fv, higher) in enumerate(summary, start=1):
    sdata.append([label, a, f, both])
    if higher is not None and av != fv:
        win = 1 if (av > fv) == higher else 2
        sstyle += [("BACKGROUND", (win, i), (win, i), WIN_BG), ("TEXTCOLOR", (win, i), (win, i), UP),
                   ("FONTNAME", (win, i), (win, i), "Helvetica-Bold")]
st = Table(sdata, colWidths=[55 * mm, 40 * mm, 40 * mm, 40 * mm], hAlign="LEFT")
st.setStyle(TableStyle(sstyle))
story += [st, Spacer(1, 2), Paragraph(
    f"* Sequential time. Our agent ran the 20 in 4 parallel workers: 24.6 min wall clock. "
    f"Green = better of our agent and FutureSearch.", note), Spacer(1, 4)]


def gap(sup, level="street"):
    r = next(x for x in rows if x["supplier"] == sup)
    return r[f"fs_{level}_matched"] - r[f"agent_{level}_matched"]


story.append(Paragraph("Key findings", h2))
findings = [
    f"<b>Combined recall:</b> street {pooled[('combined', 'street')]:.1%}; FutureSearch adds "
    f"{tot('fs_only_street'):.0f} GT sites our agent missed, our agent adds {tot('agent_only_street'):.0f}.",
    f"<b>FutureSearch's lead:</b> mostly BYK Additives (+{gap('BYK ADDITIVES'):.0f}) "
    f"and Hoshizaki (+{gap('HOSHIZAKI'):.0f}).",
    "<b>WeylChem:</b> our agent found 6 of 7; FutureSearch found 0, reporting sites of the renamed Catexel instead.",
    f"<b>Per supplier:</b> our agent averages {avg[('agent', 'street')]:.1%} street recall vs "
    f"FutureSearch {avg[('fs', 'street')]:.1%}; FutureSearch leads on the total.",
    f"<b>Cost:</b> ${agent_cost / N:.2f} vs ${FS_COST / N:.2f} per supplier "
    f"({FS_COST / agent_cost:.1f}&times; cheaper).",
    "<b>Missing streets:</b> FutureSearch gave no street for 20 of 337 sites (13 at CPH Chemicals); our agent for 1 of 330.",
    "<b>Neither found:</b> Fine Will Industrial and Cafe de Mi Tierra.",
]
story += [Paragraph(b, body, bulletText="•") for b in findings]

story += [PageBreak(), Paragraph("Results by supplier", h2)]
head1 = ["Supplier", "GT\nsites", "Sites found", "", "Unique sites", "", "Overlap", "Street recall", "", "",
         "City recall", "", "", "Agent\ncost"]
head2 = ["", "", "Agent", "FS", "Agent", "FS", "", "Agent", "FS", "Combined", "Agent", "FS", "Combined", ""]
data = [head1, head2]
style = [
    ("BACKGROUND", (0, 0), (-1, 1), NAVY), ("TEXTCOLOR", (0, 0), (-1, 1), colors.white),
    ("FONTNAME", (0, 0), (-1, 1), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 7.5),
    ("ALIGN", (1, 0), (-1, -1), "RIGHT"), ("ALIGN", (2, 0), (-2, 0), "CENTER"),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("SPAN", (0, 0), (0, 1)), ("SPAN", (1, 0), (1, 1)), ("SPAN", (6, 0), (6, 1)), ("SPAN", (13, 0), (13, 1)),
    ("SPAN", (2, 0), (3, 0)), ("SPAN", (4, 0), (5, 0)), ("SPAN", (7, 0), (9, 0)), ("SPAN", (10, 0), (12, 0)),
    ("LINEBELOW", (2, 0), (5, 0), 0.4, colors.white), ("LINEBELOW", (7, 0), (12, 0), 0.4, colors.white),
    ("LINEBELOW", (0, 1), (-1, -1), 0.3, GRID), ("TOPPADDING", (0, 0), (-1, -1), 1.0),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 1.0),
]
# light vertical rules between the column groups
for c in (2, 4, 6, 7, 10, 13):
    style.append(("LINEBEFORE", (c, 2), (c, -1), 0.4, GRID))


def rc(r, who, level):
    return r[f"{who}_{level}_matched"] / r["gt_sites"]


for i, r in enumerate(rows, start=2):
    data.append([Paragraph(r["supplier"], cell), f"{r['gt_sites']:.0f}",
                 f"{r['agent_sites']:.0f}", f"{r['fs_sites']:.0f}",
                 f"{r['agent_unique_sites']:.0f}", f"{r['fs_unique_sites']:.0f}", f"{r['agent_fs_same_site']:.0f}",
                 *(f"{rc(r, w, 'street'):.0%}" for w in WHO), *(f"{rc(r, w, 'city'):.0%}" for w in WHO),
                 f"${r['cost']:.2f}"])
    if i % 2 == 1:
        style.append(("BACKGROUND", (0, i), (-1, i), STRIPE))
    for col, level in ((7, "street"), (10, "city")):
        a, f = rc(r, "agent", level), rc(r, "fs", level)
        if a != f:
            win = col if a > f else col + 1
            style += [("TEXTCOLOR", (win, i), (win, i), UP), ("FONTNAME", (win, i), (win, i), "Helvetica-Bold")]
n = len(data)
data.append(["Total (all 225 GT sites)", f"{G:.0f}", f"{tot('agent_sites'):.0f}", f"{tot('fs_sites'):.0f}",
             f"{tot('agent_unique_sites'):.0f}", f"{tot('fs_unique_sites'):.0f}", f"{tot('agent_fs_same_site'):.0f}",
             *(f"{pooled[(w, 'street')]:.1%}" for w in WHO), *(f"{pooled[(w, 'city')]:.1%}" for w in WHO),
             f"${agent_cost:.2f}"])
data.append(["Average recall per supplier", "", "", "", "", "", "",
             *(f"{avg[(w, 'street')]:.1%}" for w in WHO), *(f"{avg[(w, 'city')]:.1%}" for w in WHO), ""])
style += [("FONTNAME", (0, n), (-1, n + 1), "Helvetica-Bold"), ("BACKGROUND", (0, n), (-1, n + 1), STRIPE),
          ("LINEABOVE", (0, n), (-1, n), 1, NAVY)]
t = Table(data, colWidths=[72 * mm] + [15 * mm] * 13, repeatRows=2)
t.setStyle(TableStyle(style))
story += [t, PageBreak(), Paragraph("Notes", h2)]

notes = [
    "<b>Recall:</b> street = a ground-truth address matched one-to-one; city = any site in the same city. "
    "Total = GT sites found &divide; 225; average = mean of each supplier's recall.",
    "<b>Combined:</b> a GT site counts as found if either tool found it.",
    "<b>Unique sites:</b> sites the other tool did not report, matched by street address with the scoring rule.",
    "<b>Overlap:</b> sites both tools reported, matched the same way.",
    "<b>Green:</b> the higher of agent and FutureSearch recall for that supplier.",
    "<b>Cost and time:</b> agent cost is exact per supplier; FutureSearch's $26.00 and 8.4 min are totals for all 20. "
    "Our agent ran the 20 in 4 parallel workers (24.6 min wall clock).",
]
story += [Paragraph(b, body, bulletText="•") for b in notes]

story.append(Paragraph("How ISO certificates improved the results", h2))
cert = [
    "<b>Multiple sources:</b> each site lists every page or document showing its address; "
    "97 of 330 sites (29%) have 2 or more sources.",
    "<b>Confirmation:</b> 17 of the 167 matched sites are backed by a certificate PDF, "
    "a second independent source for the address.",
    "<b>Cost:</b> near zero; 53 PDFs read locally for free and only 3 needed a paid fallback.",
]
story += [Paragraph(b, body, bulletText="•") for b in cert]

story.append(Paragraph("Cost-saving changes: free sources before paid search", h2))
saving = [
    "<b>Free registry lookups:</b> the agent fetches GLEIF (LEI register), SEC EDGAR (10-K properties), "
    "Wikipedia and UK Companies House directly by URL, with no web search; "
    "GLEIF and Wikipedia were checked for all 20 suppliers.",
    "<b>Sitemap first:</b> it reads the supplier's sitemap.xml to find locations, contact and certificate "
    "pages without searching for them (19 of 20 suppliers).",
    "<b>Free page and PDF reading:</b> pages are read with a free plain fetch, then a ~$0.001 browser fetch; "
    "PDFs are read locally, with OCR for scans.",
    "<b>Links reused, not searched:</b> PDF and certificate links on fetched pages are handed to the agent "
    "to open directly.",
    "<b>Search guards:</b> one certificate probe per supplier, near-duplicate searches refused, "
    "and no guessed domains.",
    "<b>Paid fallback capped:</b> at most 10 paid page retrievals per supplier and 2 per website.",
    "<b>Effect:</b> 515 page fetches, 420 web searches and 33 paid fallbacks for all 20 suppliers.",
]
story += [Paragraph(b, body, bulletText="•") for b in saving]


def footer(canv, doc):
    canv.saveState()
    canv.setFont("Helvetica", 7)
    canv.setFillColor(colors.HexColor("#8A94A3"))
    canv.drawString(15 * mm, 8 * mm,
                    "Site extraction benchmark | results/rerun20_20260925 + results/futuresearch_compare_20260929")
    canv.drawRightString(landscape(A4)[0] - 15 * mm, 8 * mm, f"Page {doc.page}")
    canv.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=landscape(A4),
                        leftMargin=15 * mm, rightMargin=15 * mm, topMargin=12 * mm, bottomMargin=14 * mm,
                        title="Site Extraction Benchmark: 20 Suppliers", author="Amitesh Singh")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT)
