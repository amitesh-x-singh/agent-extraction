"""Render the benchmark report page from comparison.json into report.html."""
import json
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN))
import cost_report  # noqa: E402


def main():
    rows = json.loads((RUN / "comparison.json").read_text(encoding="utf-8"))
    ran = [r for r in rows if r["ran"]]
    gts = sum(r["gt_sites"] for r in ran)
    summary = {
        "suppliers": len(ran),
        "suppliers_total": len(rows),
        "gt_sites": gts,
        "agent_sites": sum(r["agent_sites"] for r in ran),
        "new_sites": sum(r["new_sites"] for r in ran),
        "matched_city": sum(r["matched_city"] for r in ran),
        "matched_street": sum(r["matched_street"] for r in ran),
        "recall_city_micro": sum(r["matched_city"] for r in ran) / gts if gts else 0,
        "recall_street_micro": sum(r["matched_street"] for r in ran) / gts if gts else 0,
        "recall_city_macro": sum(r["recall_city"] for r in ran) / len(ran) if ran else 0,
        "recall_street_macro": sum(r["recall_street"] for r in ran) / len(ran) if ran else 0,
    }
    table = [{
        "s": r["supplier_name"], "g": r["gt_sites"], "a": r["agent_sites"],
        "n": r["new_sites"], "rc": r["recall_city"], "rs": r["recall_street"],
        "c": r["cost_usd"],
    } for r in ran]
    payload = {"summary": summary, "rows": table, "cost": cost_report.summarize()}
    html = TEMPLATE.replace("__DATA__", json.dumps(payload, ensure_ascii=False))
    (RUN / "report.html").write_text(html, encoding="utf-8")
    print("wrote", RUN / "report.html", len(html), "bytes,", len(table), "suppliers")


TEMPLATE = r"""<title>Supplier Site Discovery Benchmark</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=Source+Sans+3:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{
  --ground:#f3f5f6; --surface:#ffffff; --surface-2:#fafbfb;
  --ink:#14181b; --ink-2:#4d585f; --ink-3:#77838a;
  --line:#dde3e6; --line-2:#eaeff1;
  --accent:#0b6e6a;
  --good:#2c7a43; --warn:#9a6600; --bad:#a7302a;
  --shadow:0 1px 2px rgba(20,24,27,.06),0 8px 24px -16px rgba(20,24,27,.25);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#0e1113; --surface:#161a1d; --surface-2:#1b2023;
    --ink:#e8edef; --ink-2:#a3aeb4; --ink-3:#7d888e;
    --line:#252b2f; --line-2:#1f2428;
    --accent:#49bdb4;
    --good:#63b97c; --warn:#d09a3c; --bad:#e0796f;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px -16px rgba(0,0,0,.8);
  }
}
:root[data-theme="dark"]{
  --ground:#0e1113; --surface:#161a1d; --surface-2:#1b2023;
  --ink:#e8edef; --ink-2:#a3aeb4; --ink-3:#7d888e;
  --line:#252b2f; --line-2:#1f2428;
  --accent:#49bdb4;
  --good:#63b97c; --warn:#d09a3c; --bad:#e0796f;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px -16px rgba(0,0,0,.8);
}
*{box-sizing:border-box}
body{
  background:var(--ground); color:var(--ink);
  font-family:"Source Sans 3",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  font-size:15px; line-height:1.55; margin:0;
}
.wrap{max-width:1180px; margin:0 auto; padding-inline:16px; padding-block:0 64px}
h1,h2,h3{font-family:Archivo,"Source Sans 3",sans-serif; text-wrap:balance; margin:0}
h1{font-size:clamp(26px,4.4vw,38px); font-weight:700; letter-spacing:-.018em; line-height:1.12}
h2{font-size:19px; font-weight:600; letter-spacing:-.008em}
.eyebrow{
  font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:11px; font-weight:500;
  letter-spacing:.14em; text-transform:uppercase; color:var(--accent);
}

/* ---------- masthead ---------- */
header.top{border-bottom:1px solid var(--line); background:var(--surface)}
header.top .wrap{display:flex; flex-direction:column; gap:14px; padding-block:34px 28px}
.sub{color:var(--ink-2); max-width:68ch; margin:0}
.meta{display:flex; flex-wrap:wrap; gap:6px 20px; font-size:13px; color:var(--ink-3)}
.meta b{color:var(--ink-2); font-weight:600}

/* ---------- summary strip ---------- */
.tiles{display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:1px;
  background:var(--line); border:1px solid var(--line); border-radius:10px; overflow:hidden;
  margin-block:26px}
.tile{background:var(--surface); padding:16px 18px; display:flex; flex-direction:column; gap:3px}
.tile .k{font-size:12px; color:var(--ink-3); letter-spacing:.02em}
.tile .v{font-family:Archivo,sans-serif; font-size:26px; font-weight:600; letter-spacing:-.02em;
  font-variant-numeric:tabular-nums; line-height:1.15}
.tile .note{font-size:12px; color:var(--ink-3)}
.tile.accent .v{color:var(--accent)}

/* ---------- controls ---------- */
h2.tablehead{margin:30px 0 10px}
.controls{display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin:0 0 12px}
input[type=search]{
  font:inherit; font-size:14px; padding:7px 11px; border:1px solid var(--line);
  border-radius:7px; background:var(--surface); color:var(--ink); min-width:210px; flex:1 1 210px}
input[type=search]:focus-visible{outline:2px solid var(--accent); outline-offset:1px}
.count{font-size:13px; color:var(--ink-3); margin-left:auto}

/* ---------- table ---------- */
.tablewrap{overflow-x:auto; border:1px solid var(--line); border-radius:10px;
  background:var(--surface); box-shadow:var(--shadow)}
table{border-collapse:collapse; width:100%; min-width:760px; font-size:14px}
thead th{
  position:sticky; top:env(safe-area-inset-top,0px); z-index:2;
  background:var(--surface-2); border-bottom:1px solid var(--line);
  font-family:"IBM Plex Mono",monospace; font-size:10.5px; font-weight:500;
  letter-spacing:.09em; text-transform:uppercase; color:var(--ink-3);
  padding:10px 12px; text-align:right; white-space:nowrap; cursor:pointer; user-select:none}
thead th:first-child{text-align:left}
thead th:hover{color:var(--ink)}
thead th[aria-sort]{color:var(--accent)}
thead th .arr{opacity:.55; font-size:9px; margin-left:3px}
tbody td{padding:9px 12px; border-bottom:1px solid var(--line-2); text-align:right;
  font-variant-numeric:tabular-nums; white-space:nowrap}
tbody td:first-child{text-align:left; white-space:normal; min-width:200px;
  font-weight:600; letter-spacing:-.005em}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover td{background:var(--surface-2)}
td.mono{font-family:"IBM Plex Mono",monospace; font-size:13px}
.rec{display:inline-flex; align-items:center; gap:8px; justify-content:flex-end; width:100%}
.bar{width:44px; height:5px; border-radius:3px; background:var(--line); overflow:hidden; flex:none}
.bar i{display:block; height:100%; border-radius:3px}
.v-good i{background:var(--good)} .v-warn i{background:var(--warn)} .v-bad i{background:var(--bad)}
.v-good .pc{color:var(--good)} .v-warn .pc{color:var(--warn)} .v-bad .pc{color:var(--bad)}
.pc{font-family:"IBM Plex Mono",monospace; font-size:13px; font-weight:500; min-width:38px}
tfoot td{padding:11px 12px; text-align:right; font-weight:600; border-top:1px solid var(--line);
  background:var(--surface-2); font-variant-numeric:tabular-nums;
  font-family:"IBM Plex Mono",monospace; font-size:13px}
tfoot td:first-child{text-align:left; font-family:Archivo,sans-serif; font-size:13px}

/* ---------- notes ---------- */
section.method{margin-top:40px; display:grid; gap:18px}
.card{background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:18px 20px}
.card p{margin:8px 0 0; font-size:14px; color:var(--ink-2); max-width:78ch}
.card ul{margin:10px 0 0; padding-left:18px; font-size:14px; color:var(--ink-2); max-width:78ch}
.card li{margin:6px 0}
code{font-family:"IBM Plex Mono",monospace; font-size:12.5px;
  background:var(--surface-2); border:1px solid var(--line-2); border-radius:4px; padding:1px 4px}
.ledger{margin-top:14px; display:grid; gap:1px; background:var(--line);
  border:1px solid var(--line); border-radius:8px; overflow:hidden}
.led{background:var(--surface); display:grid;
  grid-template-columns:1fr auto auto; gap:6px 14px; align-items:baseline; padding:9px 14px}
.led .n{font-family:"IBM Plex Mono",monospace; font-size:12.5px; color:var(--ink-3);
  font-variant-numeric:tabular-nums; text-align:right; white-space:nowrap}
.led .amt{font-family:"IBM Plex Mono",monospace; font-size:13.5px; font-weight:500;
  font-variant-numeric:tabular-nums; text-align:right; min-width:104px; white-space:nowrap}
.led.head{background:var(--surface-2); font-family:"IBM Plex Mono",monospace; font-size:10.5px;
  letter-spacing:.09em; text-transform:uppercase; color:var(--ink-3)}
.led.sum{background:var(--surface-2); font-weight:600}
.led.unknown .amt{color:var(--warn)}
.led.unknown{border-left:3px solid var(--warn)}
@media (max-width:560px){
  .led{grid-template-columns:1fr auto}
  .led .n{grid-column:1/-1; text-align:left}
}
</style>

<header class="top">
  <div class="wrap">
    <div class="eyebrow">Benchmark result &middot; 23 September 2026</div>
    <h1>Supplier Site Discovery Benchmark</h1>
    <p class="sub" id="sub"></p>
    <div class="meta" id="meta"></div>
  </div>
</header>

<div class="wrap">
  <div class="tiles" id="tiles"></div>

  <h2 class="tablehead">Per-supplier detail</h2>
  <div class="controls">
    <input type="search" id="q" placeholder="Filter suppliers" aria-label="Filter suppliers">
    <span class="count" id="count"></span>
  </div>

  <div class="tablewrap">
    <table>
      <thead><tr id="hrow"></tr></thead>
      <tbody id="tbody"></tbody>
      <tfoot id="tfoot"></tfoot>
    </table>
  </div>

  <section class="method">
    <div class="card">
      <h2>How a site counts as found</h2>
      <p>Both levels require the same country, then:</p>
      <ul>
        <li><strong>City</strong> &mdash; the agent reported a site in the same city or postal
          code. Several ground-truth sites in one city can be satisfied by a single agent row.</li>
        <li><strong>Full street</strong> &mdash; the street address matches too, compared as
          normalised token sets so spelling and accents do not block a match. Matching is
          one-to-one, so each agent row can satisfy only one ground-truth site.</li>
        <li><strong>New sites</strong> &mdash; agent rows matching no ground-truth site. Counted,
          not judged: confirming them is a separate exercise.</li>
      </ul>
    </div>

    <div class="card">
      <h2>What the cost column is</h2>
      <p id="costnote"></p>
      <div class="ledger" id="ledger"></div>
    </div>
  </section>
</div>

<script>
const D = __DATA__;
const S = D.summary, C = D.cost;
const pct = v => (v*100).toFixed(0) + '%';
const usd = v => v == null ? '—' : '$' + v.toFixed(2);
const n = v => v.toLocaleString();
const band = v => v >= 0.75 ? 'v-good' : v >= 0.45 ? 'v-warn' : 'v-bad';

document.getElementById('sub').innerHTML =
  'How completely an automated research agent rediscovers a supplier’s physical sites from ' +
  'public sources, and what that costs per supplier. Measured against ground truth over ' +
  S.suppliers + ' suppliers and their ' + n(S.gt_sites) + ' known sites.';

document.getElementById('meta').innerHTML = [
  ['Scope', S.suppliers + ' of ' + S.suppliers_total + ' suppliers'],
  ['Ground-truth sites', n(S.gt_sites)],
  ['Run time', '57 minutes, 8 workers in parallel'],
  ['Model', 'gpt-5.6-luna'],
].map(([k,v]) => '<span><b>' + k + '</b> ' + v + '</span>').join('');

document.getElementById('tiles').innerHTML = [
  {k:'Found to the street address', v:pct(S.recall_street_micro),
   note:n(S.matched_street) + ' of ' + n(S.gt_sites) + ' known sites', accent:true},
  {k:'Found to the right city', v:pct(S.recall_city_micro),
   note:n(S.matched_city) + ' of ' + n(S.gt_sites) + ' known sites', accent:true},
  {k:'Sites returned', v:n(S.agent_sites),
   note:n(S.new_sites) + ' of them not in the ground truth'},
  {k:'Run cost', v:'$' + C.low.toFixed(0) + '–' + C.high.toFixed(0),
   note:S.suppliers + ' suppliers, about $0.55 each'},
].map(t => '<div class="tile' + (t.accent ? ' accent' : '') + '"><span class="k">' + t.k +
  '</span><span class="v">' + t.v + '</span><span class="note">' + t.note + '</span></div>').join('');

const COLS = [
  {id:'s',  label:'Supplier name'},
  {id:'g',  label:'GT sites'},
  {id:'a',  label:'Total sites found by agent'},
  {id:'n',  label:'New sites found by agent'},
  {id:'rc', label:'Recall — city'},
  {id:'rs', label:'Recall — street'},
  {id:'c',  label:'Agent cost USD'},
];
let sortKey = 'rs', sortDir = 1;

document.getElementById('hrow').innerHTML = COLS.map(c =>
  '<th data-k="' + c.id + '" scope="col">' + c.label + '<span class="arr"></span></th>').join('');

function render() {
  const q = document.getElementById('q').value.trim().toLowerCase();
  const rows = D.rows.filter(r => !q || r.s.toLowerCase().includes(q))
    .sort((a,b) => {
      const x = a[sortKey], y = b[sortKey];
      if (typeof x === 'string') return sortDir * x.localeCompare(y);
      return sortDir * ((x ?? -1) - (y ?? -1));
    });
  document.getElementById('tbody').innerHTML = rows.map(r => {
    const rec = (v) => '<td><span class="rec ' + band(v) + '"><span class="bar"><i style="width:' +
      (v*100).toFixed(0) + '%"></i></span><span class="pc">' + pct(v) + '</span></span></td>';
    return '<tr><td>' + r.s + '</td>' +
      '<td class="mono">' + r.g + '</td>' +
      '<td class="mono">' + r.a + '</td>' +
      '<td class="mono">' + r.n + '</td>' +
      rec(r.rc) + rec(r.rs) +
      '<td class="mono">' + usd(r.c) + '</td></tr>';
  }).join('');
  const sum = (f) => rows.reduce((t,r) => t + (f(r) ?? 0), 0);
  const g = sum(r => r.g);
  document.getElementById('tfoot').innerHTML = rows.length ? '<tr><td>' + rows.length +
    ' suppliers</td><td>' + g + '</td><td>' + sum(r => r.a) + '</td><td>' + sum(r => r.n) +
    '</td><td>' + pct(g ? rows.reduce((t,r) => t + r.rc*r.g, 0)/g : 0) +
    '</td><td>' + pct(g ? rows.reduce((t,r) => t + r.rs*r.g, 0)/g : 0) +
    '</td><td>' + usd(sum(r => r.c)) + '</td></tr>' : '';
  document.getElementById('count').textContent = rows.length + ' of ' + D.rows.length + ' suppliers';
  document.querySelectorAll('thead th').forEach(th => {
    const on = th.dataset.k === sortKey;
    if (on) th.setAttribute('aria-sort', sortDir > 0 ? 'ascending' : 'descending');
    else th.removeAttribute('aria-sort');
    th.querySelector('.arr').textContent = on ? (sortDir > 0 ? '▲' : '▼') : '';
  });
}

document.querySelectorAll('thead th').forEach(th => th.addEventListener('click', () => {
  const k = th.dataset.k;
  if (k === sortKey) sortDir = -sortDir;
  else { sortKey = k; sortDir = k === 's' ? 1 : -1; }
  render();
}));
document.getElementById('q').addEventListener('input', render);

document.getElementById('costnote').innerHTML =
  'The first three lines are exact &mdash; a count the provider reported, times a published ' +
  'rate &mdash; and they are ' +
  (100 * C.exact_subtotal / ((C.low + C.high) / 2)).toFixed(0) + '% of the bill. Input tokens ' +
  'are bracketed: they are billed at three prices depending on whether each token was cached, ' +
  'a cache write or uncached, and this run predates that split being recorded. It is recorded ' +
  'now, so later runs report one exact figure. The per-supplier column above prices all input ' +
  'at the uncached rate, a position inside the bracket.';

document.getElementById('ledger').innerHTML = [
  '<div class="led head"><span>Line</span><span class="n">Count</span><span class="amt">USD</span></div>',
  '<div class="led"><span>Output tokens</span><span class="n">' + n(C.output_tokens) +
    '</span><span class="amt">$' + C.output_cost.toFixed(2) + '</span></div>',
  '<div class="led"><span>Search call fee</span><span class="n">' + n(C.search_calls) +
    ' calls</span><span class="amt">$' + C.search_fee.toFixed(2) + '</span></div>',
  '<div class="led"><span>Page fetching</span><span class="n">' + n(C.credits) +
    ' credits</span><span class="amt">$' + C.bee_cost.toFixed(2) + '</span></div>',
  '<div class="led sum"><span>Exact subtotal</span><span class="n"></span><span class="amt">$' +
    C.exact_subtotal.toFixed(2) + '</span></div>',
  '<div class="led unknown"><span>Input tokens</span><span class="n">' + n(C.input_tokens) +
    '</span><span class="amt">$' + (C.low - C.exact_subtotal).toFixed(2) + '–$' +
    (C.high - C.exact_subtotal).toFixed(2) + '</span></div>',
  '<div class="led sum"><span>Run total</span><span class="n">' + C.runs + ' runs, ' +
    C.start + '–' + C.end + '</span><span class="amt">$' + C.low.toFixed(2) + '–$' +
    C.high.toFixed(2) + '</span></div>',
].join('');

render();
</script>
"""

if __name__ == "__main__":
    main()
