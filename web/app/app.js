// Nazar product UI.
// Rule for this file: no figure shown to the user is written here. Everything the page
// prints comes out of data/ at runtime. Numbers below are layout, formatting or unit
// multipliers only.

const DATA = 'data/';

// categories.json and compression.json live in artifacts/ in the repo and next to the page
// once it ships. Same two roots live.js uses, looked up the same way.
const ART_ROOTS = ['data/', '../../artifacts/'];

const PAD = { l: 46, r: 12, t: 14, b: 26 };
const BINS = 36;
const PER_PARTS = 10000;
const NEAR_ROWS = 10;
const WORST_ROWS = 8;
const SERIES_COLORS = ['#4a8fc7', '#d1803b', '#4f9d69', '#9b7fc4'];
// one per category, and there are eight of them
const CAT_COLORS = ['#4a8fc7', '#d1803b', '#4f9d69', '#9b7fc4', '#c9a227', '#5fb3b3',
  '#c1493f', '#8a94a6'];
const PAD_C = { l: 46, r: 92, t: 14, b: 30 };

const NUMERIC = new Set(['label', 'd_star', 'w', 'score_eq7']);
const LABEL_GOOD = 0;

// score column in scores.csv -> the auroc field the pipeline wrote for that same score
const AUROC_FIELD = { score_eq7: 'image_auroc_eq7', d_star: 'image_auroc_plain' };

const S = {
  manifest: null,
  runs: [],
  primary: null,
  maps: null,
  artRoot: null,
  cats: null,
  comp: null,
  scoreKey: 'score_eq7',
  threshold: 0,
  domain: [0, 1],
};

const $ = (id) => document.getElementById(id);
const f2 = (v) => v.toFixed(2);
const f3 = (v) => v.toFixed(3);
const f4 = (v) => v.toFixed(4);
const pct = (v) => (v * 100).toFixed(1) + '%';
const esc = (s) => String(s).replace(/[<>&]/g, (c) => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;' }[c]));

async function getJSON(path) {
  const r = await fetch(DATA + path);
  if (!r.ok) throw new Error(path + ' -> HTTP ' + r.status);
  return r.json();
}

async function getText(path) {
  const r = await fetch(DATA + path);
  if (!r.ok) throw new Error(path + ' -> HTTP ' + r.status);
  return r.text();
}

async function maybeJSON(path) {
  try { return await getJSON(path); } catch (e) { return null; }
}

// The first root that actually serves the file wins, and the root that answered is kept so
// the source line under each panel names the path the numbers really came from.
async function findArt(name) {
  for (const root of ART_ROOTS) {
    try {
      const r = await fetch(root + name);
      if (r.ok) return { root, path: root + name, data: await r.json() };
    } catch (e) { /* try the next one */ }
  }
  return null;
}

// A published figure that is not in the file is a dash. Nothing here guesses one.
const cell = (v, dp) => (typeof v === 'number' ? v.toFixed(dp) : '-');
const auroc100 = (v) => (typeof v === 'number' ? (v * 100).toFixed(2) : '-');
const signed = (v, dp) => (typeof v === 'number' ? (v >= 0 ? '+' : '') + v.toFixed(dp) : '-');
const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);
const pctLabel = (p) => (p * 100).toFixed(0) + '%';

function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  const head = lines.shift().split(',').map((h) => h.trim());
  return lines.map((line) => {
    const cells = line.split(',');
    const row = {};
    head.forEach((h, i) => {
      const raw = (cells[i] || '').trim();
      row[h] = NUMERIC.has(h) ? Number(raw) : raw;
    });
    return row;
  });
}

const score = (row) => row[S.scoreKey];
const isGood = (row) => row.label === LABEL_GOOD;

// ---- stats ------------------------------------------------------------------

function extent(values) {
  let lo = Infinity, hi = -Infinity;
  for (const v of values) { if (v < lo) lo = v; if (v > hi) hi = v; }
  return [lo, hi];
}

function rates(rows, thr) {
  let escapes = 0, falseRejects = 0, nGood = 0, nDefect = 0;
  for (const row of rows) {
    const s = score(row);
    if (isGood(row)) { nGood++; if (s >= thr) falseRejects++; }
    else { nDefect++; if (s < thr) escapes++; }
  }
  return {
    escapes, falseRejects, nGood, nDefect,
    escapeRate: nDefect ? escapes / nDefect : 0,
    falseRejectRate: nGood ? falseRejects / nGood : 0,
  };
}

function costInputs() {
  return {
    defectRate: Number($('defectRate').value) / 100,
    costEscape: Number($('costEscape').value),
    costFalse: Number($('costFalse').value),
  };
}

// expected cost per part inspected. the test set mix is not a line mix, so the rates
// come from the data and the prior comes from the operator.
function expectedCost(rows, thr, c) {
  const r = rates(rows, thr);
  return c.defectRate * r.escapeRate * c.costEscape
       + (1 - c.defectRate) * r.falseRejectRate * c.costFalse;
}

// cost is a step function of the threshold, so it can only turn at a score value.
// ties resolve to the lowest threshold, which is the fewer-escapes side.
function costMinimum(rows, c) {
  // the padded domain top sits above every score, so it is the reject-nothing end
  const cuts = [...new Set(rows.map(score))].sort((a, b) => a - b);
  cuts.push(S.domain[1]);
  let best = null;
  for (const thr of cuts) {
    const cost = expectedCost(rows, thr, c);
    if (best === null || cost < best.cost) best = { thr, cost };
  }
  return best;
}

// ---- canvas -----------------------------------------------------------------

function fit(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function geom(w, h, xd, yd) {
  const x0 = PAD.l, x1 = w - PAD.r, y0 = h - PAD.b, y1 = PAD.t;
  return {
    x0, x1, y0, y1,
    sx: (v) => x0 + ((v - xd[0]) / (xd[1] - xd[0] || 1)) * (x1 - x0),
    sy: (v) => y0 - ((v - yd[0]) / (yd[1] - yd[0] || 1)) * (y0 - y1),
    ux: (px) => xd[0] + ((px - x0) / (x1 - x0)) * (xd[1] - xd[0]),
  };
}

function drawFrame(ctx, fr) {
  ctx.strokeStyle = '#2f3941';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(fr.x0 + 0.5, fr.y1);
  ctx.lineTo(fr.x0 + 0.5, fr.y0 + 0.5);
  ctx.lineTo(fr.x1, fr.y0 + 0.5);
  ctx.stroke();
  return fr;
}

function ticks(lo, hi, count) {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const raw = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm >= 5 ? 5 : norm >= 2 ? 2 : 1) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(v);
  return out;
}

function axisLabels(ctx, fr, xt, yt, fmtX, fmtY) {
  ctx.fillStyle = '#7d8992';
  ctx.font = '10px ui-monospace, Consolas, monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (const t of xt) {
    const x = fr.sx(t);
    if (x < fr.x0 - 1 || x > fr.x1 + 1) continue;
    ctx.fillText(fmtX(t), x, fr.y0 + 6);
    ctx.strokeStyle = '#222a31';
    ctx.beginPath();
    ctx.moveTo(x + 0.5, fr.y0);
    ctx.lineTo(x + 0.5, fr.y0 + 3);
    ctx.stroke();
  }
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (const t of yt) {
    const y = fr.sy(t);
    if (y > fr.y0 + 1 || y < fr.y1 - 1) continue;
    ctx.fillText(fmtY(t), fr.x0 - 6, y);
    ctx.strokeStyle = '#1b2127';
    ctx.beginPath();
    ctx.moveTo(fr.x0, Math.round(y) + 0.5);
    ctx.lineTo(fr.x1, Math.round(y) + 0.5);
    ctx.stroke();
  }
}

// ---- histogram --------------------------------------------------------------

function histogram(rows, domain) {
  const good = new Array(BINS).fill(0);
  const defect = new Array(BINS).fill(0);
  const span = domain[1] - domain[0] || 1;
  let nGood = 0, nDefect = 0;
  for (const row of rows) {
    let b = Math.floor(((score(row) - domain[0]) / span) * BINS);
    if (b >= BINS) b = BINS - 1;
    if (b < 0) b = 0;
    if (isGood(row)) { good[b]++; nGood++; } else { defect[b]++; nDefect++; }
  }
  return { good, defect, nGood, nDefect };
}

function drawHist() {
  const { ctx, w, h } = fit($('histCanvas'));
  const rows = S.primary.rows;
  const xd = S.domain;
  const hist = histogram(rows, xd);
  // per class share, because the two classes have very different n here
  const gShare = hist.good.map((c) => (hist.nGood ? c / hist.nGood : 0));
  const dShare = hist.defect.map((c) => (hist.nDefect ? c / hist.nDefect : 0));
  const yMax = Math.max(...gShare, ...dShare) || 1;
  const fr = drawFrame(ctx, geom(w, h, xd, [0, yMax]));
  axisLabels(ctx, fr, ticks(xd[0], xd[1], 6), ticks(0, yMax, 4), f2, (v) => pct(v));

  const bw = (fr.x1 - fr.x0) / BINS;
  const bar = (shares, color) => {
    ctx.fillStyle = color;
    shares.forEach((v, i) => {
      if (v <= 0) return;
      const x = fr.x0 + i * bw;
      const y = fr.sy(v);
      ctx.fillRect(x + 0.5, y, Math.max(bw - 1, 1), fr.y0 - y);
    });
  };
  ctx.globalAlpha = 0.62;
  bar(dShare, '#d1803b');
  bar(gShare, '#4a8fc7');
  ctx.globalAlpha = 1;

  // the line
  const tx = fr.sx(S.threshold);
  ctx.strokeStyle = '#eaf0f4';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(tx, fr.y1 - 4);
  ctx.lineTo(tx, fr.y0);
  ctx.stroke();
  ctx.fillStyle = '#eaf0f4';
  ctx.fillRect(tx - 3, fr.y1 - 8, 6, 6);

  ctx.font = '10px ui-monospace, Consolas, monospace';
  ctx.textBaseline = 'top';
  ctx.fillStyle = '#7d8992';
  ctx.textAlign = 'right';
  ctx.fillText('SHIP', tx - 6, fr.y1 - 2);
  ctx.textAlign = 'left';
  ctx.fillText('PULL', tx + 6, fr.y1 - 2);

  // legend, counts straight off the parsed rows
  const items = [['#4a8fc7', 'good n=' + hist.nGood], ['#d1803b', 'defect n=' + hist.nDefect]];
  let ly = fr.y1 + 4;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  for (const [color, text] of items) {
    const lx = fr.x1 - 92;
    ctx.globalAlpha = 0.62;
    ctx.fillStyle = color;
    ctx.fillRect(lx, ly + 2, 8, 8);
    ctx.globalAlpha = 1;
    ctx.fillStyle = '#7d8992';
    ctx.fillText(text, lx + 12, ly);
    ly += 13;
  }
}

// ---- cost curve -------------------------------------------------------------

function drawCost() {
  const { ctx, w, h } = fit($('costCanvas'));
  const rows = S.primary.rows;
  const c = costInputs();
  const xd = S.domain;
  // sample the staircase at pixel resolution, then scale y to what it actually reaches
  const probe = geom(w, h, xd, [0, 1]);
  const pts = [];
  for (let px = probe.x0; px <= probe.x1; px++) pts.push([px, expectedCost(rows, probe.ux(px), c)]);
  const yHi = Math.max(...pts.map((p) => p[1])) || 1;
  const fr2 = drawFrame(ctx, geom(w, h, xd, [0, yHi]));
  axisLabels(ctx, fr2, ticks(xd[0], xd[1], 6), ticks(0, yHi, 4), f2, (v) => v.toFixed(1));

  ctx.strokeStyle = '#c9a227';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  pts.forEach(([px, v], i) => (i ? ctx.lineTo(px, fr2.sy(v)) : ctx.moveTo(px, fr2.sy(v))));
  ctx.stroke();

  const min = costMinimum(rows, c);
  const mx = fr2.sx(min.thr);
  ctx.strokeStyle = '#4f9d69';
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(mx, fr2.y0);
  ctx.lineTo(mx, fr2.y1);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#4f9d69';
  ctx.beginPath();
  ctx.arc(mx, fr2.sy(min.cost), 3, 0, Math.PI * 2);
  ctx.fill();
  ctx.font = '10px ui-monospace, Consolas, monospace';
  ctx.textAlign = mx > fr2.x1 - 60 ? 'right' : 'left';
  ctx.textBaseline = 'top';
  ctx.fillText(' min ' + f3(min.thr), mx + (mx > fr2.x1 - 60 ? -4 : 4), fr2.y1);

  const tx = fr2.sx(S.threshold);
  ctx.strokeStyle = '#eaf0f4';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(tx, fr2.y0);
  ctx.lineTo(tx, fr2.y1);
  ctx.stroke();
}

// ---- readouts ---------------------------------------------------------------

function ro(label, value, sub, cls) {
  return '<div class="ro ' + (cls || '') + '"><span>' + esc(label) + '</span><b>' + value + '</b>'
       + (sub ? '<small>' + esc(sub) + '</small>' : '') + '</div>';
}

function drawReadouts() {
  const rows = S.primary.rows;
  const r = rates(rows, S.threshold);
  const c = costInputs();
  const cost = expectedCost(rows, S.threshold, c);
  const escPerBatch = c.defectRate * r.escapeRate * PER_PARTS;
  const stopPerBatch = (1 - c.defectRate) * r.falseRejectRate * PER_PARTS;

  $('rateReadouts').innerHTML = [
    ro('Threshold', f3(S.threshold), S.scoreKey),
    ro('Escape rate', pct(r.escapeRate), r.escapes + ' of ' + r.nDefect + ' defects', r.escapes ? 'alarm' : ''),
    ro('False reject rate', pct(r.falseRejectRate), r.falseRejects + ' of ' + r.nGood + ' goods', r.falseRejects ? 'hot' : ''),
    ro('Expected cost', cost.toFixed(2), 'per part inspected'),
    ro('Escapes', escPerBatch.toFixed(1), 'per ' + PER_PARTS.toLocaleString() + ' parts'),
    ro('Line stops', stopPerBatch.toFixed(1), 'per ' + PER_PARTS.toLocaleString() + ' parts'),
  ].join('');

  $('costEscapeOut').value = c.costEscape;
  $('costFalseOut').value = c.costFalse;

  const m = S.primary.metrics;
  const field = AUROC_FIELD[S.scoreKey];
  const auroc = m[field];
  const cs = typeof m.coreset_pct === 'number' ? ', coreset ' + pctLabel(m.coreset_pct) : '';
  $('aurocReadout').textContent = auroc === undefined
    ? 'image AUROC not in metrics for this score'
    : 'image AUROC ' + f4(auroc) + ' (' + field + ', ' + m.config.backbone + cs + ')';
}

// ---- where it breaks --------------------------------------------------------

function tableHTML(cols, rowsHTML) {
  return '<thead><tr>' + cols.map((c) => '<th>' + esc(c) + '</th>').join('') + '</tr></thead>'
       + '<tbody>' + (rowsHTML || '<tr class="empty-row"><td colspan="' + cols.length + '">no rows</td></tr>') + '</tbody>';
}

function drawBreaks() {
  const rows = S.primary.rows;
  const goods = rows.filter(isGood);
  const defects = rows.filter((r) => !isGood(r));
  const worstGood = Math.max(...goods.map(score));
  const under = defects.filter((r) => score(r) < worstGood);

  $('breakSummary').innerHTML = [
    ro('Worst good part', f3(worstGood), 'no line can sit above this'),
    ro('Defects under it', String(under.length), 'of ' + defects.length + ' defects', 'alarm'),
    ro('Share of defects', pct(defects.length ? under.length / defects.length : 0), 'unreachable at any threshold', 'alarm'),
    ro('Lowest defect', f3(Math.min(...defects.map(score))), 'scores like a good part'),
  ].join('');

  // by defect type
  const types = new Map();
  for (const row of defects) {
    const t = types.get(row.defect_type) || { total: 0, under: 0 };
    t.total++;
    if (score(row) < worstGood) t.under++;
    types.set(row.defect_type, t);
  }
  const typeRows = [...types.entries()]
    .sort((a, b) => b[1].under / b[1].total - a[1].under / a[1].total)
    .map(([name, t]) => {
      const share = t.under / t.total;
      const cls = share >= 0.5 ? 'alarm' : share > 0 ? 'warn' : '';
      return '<tr><td>' + esc(name) + '</td><td>' + t.total + '</td>'
           + '<td class="' + cls + '">' + t.under + '</td>'
           + '<td class="' + cls + '">' + pct(share) + '</td></tr>';
    }).join('');
  $('tblByType').innerHTML = tableHTML(['Defect type', 'Tested', 'Under worst good', 'Share'], typeRows);

  // worst false negatives
  const fnRows = defects.slice().sort((a, b) => score(a) - score(b)).slice(0, WORST_ROWS)
    .map((r) => '<tr><td>' + esc(r.defect_type) + '</td><td>' + esc(r.file) + '</td>'
      + '<td class="alarm">' + f3(score(r)) + '</td>'
      + '<td>' + (score(r) < worstGood ? 'yes' : 'no') + '</td></tr>').join('');
  $('tblWorstFN').innerHTML = tableHTML(['Defect type', 'File', 'Score', 'Under worst good'], fnRows);

  // near the line
  const near = rows.slice()
    .sort((a, b) => Math.abs(score(a) - S.threshold) - Math.abs(score(b) - S.threshold))
    .slice(0, NEAR_ROWS)
    .sort((a, b) => score(a) - score(b))
    .map((r) => {
      const s = score(r);
      const pull = s >= S.threshold;
      const wrong = (isGood(r) && pull) || (!isGood(r) && !pull);
      const verdict = pull ? 'PULL' : 'SHIP';
      const call = !wrong ? 'ok' : isGood(r) ? 'false reject' : 'escape';
      const cls = !wrong ? '' : isGood(r) ? 'warn' : 'alarm';
      return '<tr><td>' + esc(isGood(r) ? 'good' : r.defect_type) + '</td><td>' + esc(r.file) + '</td>'
        + '<td>' + f3(s) + '</td><td>' + (s - S.threshold >= 0 ? '+' : '') + f3(s - S.threshold) + '</td>'
        + '<td>' + verdict + '</td><td class="' + cls + '">' + call + '</td></tr>';
    }).join('');
  $('tblNear').innerHTML = tableHTML(['Part', 'File', 'Score', 'To line', 'Verdict', 'Call'], near);
}

// ---- few shot curve ---------------------------------------------------------

function drawCurve(records) {
  const canvas = $('curveCanvas');
  if (!records || !records.length) {
    canvas.classList.add('hidden');
    $('curveEmpty').classList.remove('hidden');
    $('curveEmpty').textContent =
      'curve.json not built yet.\n'
      + 'expected at ' + DATA + (S.manifest.curve || 'curve.json')
      + ', records of {category, backbone, n_shot, seed, bank_rows, image_auroc_plain, image_auroc_eq7, pixel_auroc}.\n'
      + 'the sweep is running offline. this panel fills itself when the file lands.';
    $('curveStatus').textContent = 'no data';
    $('curveLegend').innerHTML = '';
    return;
  }
  canvas.classList.remove('hidden');
  $('curveEmpty').classList.add('hidden');

  const field = AUROC_FIELD[S.scoreKey];
  const usable = records.filter((r) => typeof r[field] === 'number' && typeof r.n_shot === 'number');
  if (!usable.length) {
    $('curveStatus').textContent = 'curve.json has no ' + field;
    return;
  }
  $('curveStatus').textContent = usable.length + ' runs, ' + field;

  // group by series, then by n_shot across seeds
  const series = new Map();
  for (const r of usable) {
    const key = r.category + ' / ' + r.backbone;
    if (!series.has(key)) series.set(key, new Map());
    const byShot = series.get(key);
    if (!byShot.has(r.n_shot)) byShot.set(r.n_shot, []);
    byShot.get(r.n_shot).push(r[field]);
  }

  const shots = [...new Set(usable.map((r) => r.n_shot))].sort((a, b) => a - b);
  const allValues = usable.map((r) => r[field]);
  const [vLo, vHi] = extent(allValues);
  const padY = (vHi - vLo) * 0.12 || 0.02;
  const yd = [Math.max(0, vLo - padY), Math.min(1, vHi + padY)];

  const { ctx, w, h } = fit(canvas);
  // ordinal x, so 1 and full sit the same distance apart as everything else
  const xd = [0, Math.max(shots.length - 1, 1)];
  const fr = drawFrame(ctx, geom(w, h, xd, yd));
  axisLabels(ctx, fr, [], ticks(yd[0], yd[1], 4), f2, f3);
  ctx.fillStyle = '#7d8992';
  ctx.font = '10px ui-monospace, Consolas, monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  shots.forEach((s, i) => ctx.fillText(String(s), fr.sx(i), fr.y0 + 6));

  const legend = [];
  [...series.entries()].forEach(([name, byShot], si) => {
    const color = SERIES_COLORS[si % SERIES_COLORS.length];
    const pts = shots.map((s, i) => {
      const vals = byShot.get(s);
      if (!vals) return null;
      const [lo, hi] = extent(vals);
      const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
      return { x: fr.sx(i), lo, hi, mean, n: vals.length };
    }).filter(Boolean);
    if (!pts.length) return;

    // min to max band across seeds, no error bars on this few samples
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.16;
    ctx.beginPath();
    pts.forEach((p, i) => (i ? ctx.lineTo(p.x, fr.sy(p.hi)) : ctx.moveTo(p.x, fr.sy(p.hi))));
    for (let i = pts.length - 1; i >= 0; i--) ctx.lineTo(pts[i].x, fr.sy(pts[i].lo));
    ctx.closePath();
    ctx.fill();
    ctx.globalAlpha = 1;

    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    pts.forEach((p, i) => (i ? ctx.lineTo(p.x, fr.sy(p.mean)) : ctx.moveTo(p.x, fr.sy(p.mean))));
    ctx.stroke();
    ctx.fillStyle = color;
    pts.forEach((p) => { ctx.beginPath(); ctx.arc(p.x, fr.sy(p.mean), 2.5, 0, Math.PI * 2); ctx.fill(); });

    const seedMax = Math.max(...pts.map((p) => p.n));
    legend.push('<span><i style="background:' + color + '"></i>' + esc(name)
      + ' (mean of up to ' + seedMax + ' seeds, band is min to max)</span>');
  });
  $('curveLegend').innerHTML = legend.join('');
}

// ---- heatmap strip ----------------------------------------------------------

// The PNGs were rendered by whichever run exported them, which is not necessarily the run
// driving the panels above. The scores under each image are joined from that same run, and
// the panel says which run it is, because a heatmap carries no label of its own.
function drawStrip() {
  const box = $('mapStrip');
  const run = S.maps;
  if (!run || !run.heatmaps || !run.heatmaps.length) {
    box.innerHTML = '<div class="empty">no run in ' + DATA + 'index.json exported a heatmap index</div>';
    $('mapsScope').textContent = '';
    return;
  }
  const byKey = new Map((run.rows || []).map((r) => [r.defect_type + '/' + r.file, r]));
  box.innerHTML = run.heatmaps.map((entry) => {
    const row = byKey.get(entry.defect_type + '/' + entry.file);
    const s = row ? f3(score(row)) : 'no score row';
    return '<figure><img loading="lazy" src="' + DATA + esc(entry.png) + '" alt="'
      + esc(entry.defect_type + ' ' + entry.file) + '">'
      + '<figcaption><b>' + esc(entry.defect_type) + '</b> ' + esc(entry.file)
      + '<br>' + S.scoreKey + ' ' + s + '</figcaption></figure>';
  }).join('');

  const m = run.metrics;
  const pm = S.primary.metrics;
  const desc = (x) => x.category + ' / ' + (x.config ? x.config.backbone : 'backbone not in metrics')
    + (typeof x.coreset_pct === 'number' ? ' / coreset ' + pctLabel(x.coreset_pct) : '');
  const el = $('mapsScope');
  // against S.primary.id, not the manifest. the category switcher moves the
  // panels off the shipped primary and this line has to follow them.
  if (run.id === S.primary.id) {
    el.textContent = 'These are ' + desc(m) + ', the same run as the panels above ('
      + run.files.metrics + ').';
  } else {
    // the scores printed under these images are on the strip's own scale, not the one the
    // threshold above uses, and nothing in the picture would say so
    el.textContent = 'These images were rendered by ' + run.id + ', which is ' + desc(m) + ' ('
      + run.files.metrics + '). The panels above run ' + S.primary.id + ', which is '
      + desc(pm) + '. The scores under these images come from ' + run.files.scores
      + ' and are not on the same scale as the threshold above.';
  }
}

// ---- runs table -------------------------------------------------------------

function drawRuns() {
  const cols = ['Run', 'Category', 'Backbone', 'Coreset', 'Bank rows', 'Train', 'Test good', 'Test defect',
    'AUROC plain', 'AUROC eq7', 'Pixel AUROC', 'Notes'];
  const body = S.runs.map((run) => {
    const m = run.metrics;
    const notes = [];
    let sabotaged = false;
    if (m.ablate_bank && m.ablate_bank !== 'none') { notes.push('bank ablated: ' + m.ablate_bank); sabotaged = true; }
    if (m.permute_labels) { notes.push('labels permuted'); sabotaged = true; }
    if (m.config && m.config.balance_blocks) notes.push('blocks balanced');
    if (run.id === S.manifest.primary_run) notes.push('drives the panels above');
    if (S.maps && run.id === S.maps.id) notes.push('heatmap strip');
    return '<tr><td>' + esc(run.id) + '</td><td>' + esc(m.category) + '</td>'
      + '<td class="hi">' + esc(m.config ? m.config.backbone : '-') + '</td>'
      + '<td>' + (typeof m.coreset_pct === 'number' ? pctLabel(m.coreset_pct) : '-') + '</td>'
      + '<td>' + (m.bank_size !== undefined ? m.bank_size.toLocaleString() : '-') + '</td>'
      + '<td>' + (m.n_train !== undefined ? m.n_train : '-') + '</td>'
      + '<td>' + (m.n_test_good !== undefined ? m.n_test_good : '-') + '</td>'
      + '<td>' + (m.n_test_defect !== undefined ? m.n_test_defect : '-') + '</td>'
      + '<td>' + cell(m.image_auroc_plain !== undefined ? m.image_auroc_plain : m.image_auroc, 4) + '</td>'
      + '<td>' + cell(m.image_auroc_eq7, 4) + '</td>'
      + '<td>' + cell(m.pixel_auroc, 4) + '</td>'
      + '<td class="' + (sabotaged ? 'warn' : '') + '">' + esc(notes.join(', ')) + '</td></tr>';
  }).join('');
  $('tblRuns').innerHTML = tableHTML(cols, body);
}

// ---- ours against published -------------------------------------------------

const uniq = (xs) => [...new Set(xs)];
// one value if every record agrees, null if they do not, so a label never claims a
// coreset the rows do not all share
const oneOf = (xs) => (uniq(xs).length === 1 ? xs[0] : null);

// categories.json holds two things per category. The top level is the bank this page
// ships. paper_comparison inside it is the same category rebuilt at the coreset the paper
// reports, which is the only coreset the paper gives a figure for. The published column
// therefore lines up with the inner run and not with the shipped one.
function drawPublished() {
  const empty = $('pubEmpty');
  if (!S.cats || !S.cats.length) {
    empty.classList.remove('hidden');
    empty.textContent = 'categories.json is not being served, so there is nothing to compare.\n'
      + 'looked in: ' + ART_ROOTS.map((r) => r + 'categories.json').join(', ');
    $('tblPublished').innerHTML = '';
    $('pubSummary').innerHTML = '';
    $('pubNote').textContent = '';
    return;
  }
  empty.classList.add('hidden');

  const rows = S.cats.map((c) => {
    const pc = c.paper_comparison || {};
    const pub = typeof pc.published_image_auroc === 'number' ? pc.published_image_auroc : null;
    const ref = typeof pc.image_auroc_plain === 'number' ? pc.image_auroc_plain : null;
    // The file carries the gap already worked out in points, so that is what gets printed.
    // Our column is a fraction and the published column is already per cent, and doing that
    // conversion twice is how the two would drift apart. Falls back to the subtraction only
    // if a record has both figures and no gap.
    const gap = typeof pc.gap_points === 'number' ? pc.gap_points
      : pub !== null && ref !== null ? ref * 100 - pub : null;
    return { c, pc, pub, ref, gap };
  });

  const shipPct = oneOf(S.cats.map((c) => c.coreset_pct));
  const refPct = oneOf(rows.map((r) => r.pc.coreset_pct).filter((v) => typeof v === 'number'));
  const shipLab = shipPct === null ? 'shipped' : pctLabel(shipPct);
  const refLab = refPct === null ? 'paper coreset' : pctLabel(refPct);

  $('tblPublished').innerHTML = tableHTML(
    ['Category', 'Ours, ' + shipLab + ' (shipped)', 'Ours, ' + refLab, 'Published, ' + refLab, 'Gap'],
    rows.map(({ c, pub, ref, gap }) => {
      const cls = gap === null ? '' : gap < 0 ? 'warn' : 'ok-cell';
      return '<tr><td>' + esc(c.category) + '</td>'
        + '<td class="hi">' + auroc100(c.image_auroc_plain) + '</td>'
        + '<td>' + auroc100(ref) + '</td>'
        + '<td>' + cell(pub, 2) + '</td>'
        + '<td class="' + cls + '">' + signed(gap, 2) + '</td></tr>';
    }).join(''));

  const gaps = rows.map((r) => r.gap).filter((v) => v !== null);
  const worst = rows.filter((r) => r.gap !== null)
    .slice().sort((a, b) => Math.abs(b.gap) - Math.abs(a.gap))[0];
  const m = mean(gaps);

  $('pubSummary').innerHTML = [
    ro('Mean gap', signed(m, 2), 'points of image AUROC, ' + gaps.length + ' categories'),
    ro('Largest gap', worst ? signed(worst.gap, 2) : '-', worst ? worst.c.category : 'none'),
    ro('Below published', String(gaps.filter((g) => g < 0).length), 'of ' + gaps.length + ' categories'),
    ro('Compared at', refLab, 'coreset, both sides'),
  ].join('');

  const srcs = uniq(rows.map((r) => r.pc.source).filter(Boolean));
  $('pubNote').textContent =
    'The published column is the paper at ' + refLab + ' coreset. The bank this page ships is '
    + shipLab + ', and the paper gives no figure at ' + shipLab + ', so the shipped column has'
    + ' nothing to compare against and the gap is not measured from it. The gap column is our '
    + refLab + ' run minus the published ' + refLab + ' figure, which is the same coreset on both'
    + ' sides. Our ' + shipLab + ' column is here because it is what the live panel at the top of'
    + ' this page actually runs.';
  $('srcPublished').textContent = 'source: ' + S.catsPath
    + (srcs.length ? '. Published figures: ' + srcs.join(' ') : '');
}

// ---- compression ------------------------------------------------------------

// x is bank megabytes on a log axis, because the coresets here span better than an order of
// magnitude and a linear axis puts every interesting point in the left margin.
function logTicks(lo, hi) {
  const out = [];
  for (let d = Math.floor(Math.log10(lo)); d <= Math.ceil(Math.log10(hi)); d++) {
    for (const mm of [1, 2, 5]) {
      const v = mm * Math.pow(10, d);
      if (v >= lo && v <= hi) out.push(v);
    }
  }
  return out;
}

const MBOF = (r) => r.bank_bytes_f16 / 1e6;
// the compression panel is pinned to the plain score, the same field the published
// comparison uses, so the two panels are read on one scale
const COMP_FIELD = 'image_auroc_plain';

function compSeries() {
  const byCat = new Map();
  for (const r of S.comp) {
    if (typeof r[COMP_FIELD] !== 'number' || typeof r.bank_bytes_f16 !== 'number') continue;
    if (!byCat.has(r.category)) byCat.set(r.category, []);
    byCat.get(r.category).push(r);
  }
  for (const list of byCat.values()) list.sort((a, b) => a.coreset_pct - b.coreset_pct);
  return byCat;
}

function drawCompression() {
  const canvas = $('compCanvas');
  const empty = $('compEmpty');
  if (!S.comp || !S.comp.length) {
    canvas.classList.add('hidden');
    empty.classList.remove('hidden');
    empty.textContent = 'compression.json is not being served, so this panel has nothing to plot.\n'
      + 'looked in: ' + ART_ROOTS.map((r) => r + 'compression.json').join(', ');
    $('compLegend').innerHTML = '';
    return;
  }
  canvas.classList.remove('hidden');
  empty.classList.add('hidden');

  const byCat = compSeries();
  const shipPct = S.cats && S.cats.length ? oneOf(S.cats.map((c) => c.coreset_pct)) : null;

  const mbs = S.comp.map(MBOF);
  const vals = S.comp.map((r) => r[COMP_FIELD]).filter((v) => typeof v === 'number');
  const [mLo, mHi] = extent(mbs);
  const xd = [Math.log10(mLo / 1.18), Math.log10(mHi * 1.18)];
  const [vLo, vHi] = extent(vals);
  const padY = (vHi - vLo) * 0.12 || 0.02;
  const yd = [Math.max(0, vLo - padY), Math.min(1, vHi + padY)];

  const { ctx, w, h } = fit(canvas);
  const x0 = PAD_C.l, x1 = w - PAD_C.r, y0 = h - PAD_C.b, y1 = PAD_C.t;
  const fr = {
    x0, x1, y0, y1,
    sx: (v) => x0 + ((v - xd[0]) / (xd[1] - xd[0] || 1)) * (x1 - x0),
    sy: (v) => y0 - ((v - yd[0]) / (yd[1] - yd[0] || 1)) * (y0 - y1),
  };
  drawFrame(ctx, fr);
  const xt = logTicks(Math.pow(10, xd[0]), Math.pow(10, xd[1]));
  axisLabels(ctx, { ...fr, sx: (v) => fr.sx(Math.log10(v)) }, xt, ticks(yd[0], yd[1], 4),
    (v) => String(v), f3);

  ctx.fillStyle = '#7d8992';
  ctx.font = '10px ui-monospace, Consolas, monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillText('bank megabytes, float16', (x0 + x1) / 2, fr.y0 + 17);

  const legend = [];
  const ends = [];
  [...byCat.entries()].forEach(([name, list], si) => {
    const color = CAT_COLORS[si % CAT_COLORS.length];
    const pts = list.map((r) => ({ x: fr.sx(Math.log10(MBOF(r))), y: fr.sy(r[COMP_FIELD]), r }));
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    pts.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
    ctx.stroke();
    pts.forEach((p) => {
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(p.x, p.y, 2.4, 0, Math.PI * 2);
      ctx.fill();
      // the shipped point gets a ring, so the bank the live panel downloads is findable
      if (shipPct !== null && p.r.coreset_pct === shipPct) {
        ctx.strokeStyle = '#eaf0f4';
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
        ctx.stroke();
      }
    });
    const last = pts[pts.length - 1];
    ends.push({ name, color, x: last.x, y: last.y, at: last.y });
    legend.push('<span><i style="background:' + color + '"></i>' + esc(name) + '</span>');
  });

  // The line ends are the only place a name fits without a key box over the plot, and two
  // categories can end at the same height. Push the labels apart just enough to read them
  // and leave a leader to the point each one belongs to.
  const LH = 11;
  ends.sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) {
    if (ends[i].y - ends[i - 1].y < LH) ends[i].y = ends[i - 1].y + LH;
  }
  const over = ends.length ? ends[ends.length - 1].y - fr.y0 : 0;
  if (over > 0) ends.forEach((e) => { e.y -= over; });
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  for (const e of ends) {
    const lx = Math.min(e.x + 7, x1 + 5);
    if (Math.abs(e.y - e.at) > 1.5) {
      ctx.strokeStyle = e.color;
      ctx.globalAlpha = 0.45;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(e.x + 2, e.at);
      ctx.lineTo(lx - 2, e.y);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
    ctx.fillStyle = e.color;
    ctx.fillText(e.name, lx, e.y);
  }
  if (shipPct !== null) {
    legend.push('<span><i class="ring"></i>shipped bank, ' + pctLabel(shipPct) + ' coreset</span>');
  }
  $('compLegend').innerHTML = legend.join('');
  $('compStatus').textContent = S.comp.length + ' runs, ' + byCat.size + ' categories, ' + COMP_FIELD;
}

// Everything printed here is computed off compression.json at runtime. The comparison
// between two coresets is only made across the categories that both were built for.
function drawCompressionText() {
  if (!S.comp || !S.comp.length) {
    $('compSummary').innerHTML = '';
    $('compFind').textContent = '';
    $('compShots').textContent = '';
    $('tblComp').innerHTML = '';
    return;
  }
  const byCat = compSeries();
  const nCat = byCat.size;
  const pcts = uniq(S.comp.map((r) => r.coreset_pct)).sort((a, b) => a - b);
  const at = (p) => S.comp.filter((r) => r.coreset_pct === p && typeof r[COMP_FIELD] === 'number');
  const full = pcts.filter((p) => at(p).length === nCat);

  const shipPct = S.cats && S.cats.length ? oneOf(S.cats.map((c) => c.coreset_pct)) : null;
  const refPct = S.cats && S.cats.length
    ? oneOf(S.cats.map((c) => c.paper_comparison && c.paper_comparison.coreset_pct)
      .filter((v) => typeof v === 'number'))
    : null;

  // the table lists every coreset in the file, and says outright which ones were not built
  // for all of the categories
  $('tblComp').innerHTML = tableHTML(
    ['Coreset', 'Mean MB', 'Mean AUROC', 'Categories'],
    pcts.map((p) => {
      const rs = at(p);
      const names = rs.map((r) => r.category).sort();
      const cov = rs.length === nCat ? 'all ' + nCat : names.join(', ');
      const isShip = shipPct !== null && p === shipPct;
      return '<tr><td>' + pctLabel(p) + (isShip ? ' (shipped)' : '') + '</td>'
        + '<td>' + cell(mean(rs.map(MBOF)), 1) + '</td>'
        + '<td class="' + (isShip ? 'hi' : '') + '">' + auroc100(mean(rs.map((r) => r[COMP_FIELD]))) + '</td>'
        + '<td class="' + (rs.length === nCat ? '' : 'warn') + '">' + esc(cov) + '</td></tr>';
    }).join(''));

  const meanA = (p) => mean(at(p).map((r) => r[COMP_FIELD]));
  const meanMB = (p) => mean(at(p).map(MBOF));

  const shipOK = shipPct !== null && full.includes(shipPct);
  const refOK = refPct !== null && full.includes(refPct);

  $('compSummary').innerHTML = [
    ro('Mean AUROC, ' + (shipOK ? pctLabel(shipPct) : '-'), shipOK ? auroc100(meanA(shipPct)) : '-',
      'shipped, ' + nCat + ' categories'),
    ro('Mean bank, ' + (shipOK ? pctLabel(shipPct) : '-'), shipOK ? cell(meanMB(shipPct), 1) : '-', 'MB, float16'),
    ro('Mean AUROC, ' + (refOK ? pctLabel(refPct) : '-'), refOK ? auroc100(meanA(refPct)) : '-',
      'the paper coreset'),
    ro('Mean bank, ' + (refOK ? pctLabel(refPct) : '-'), refOK ? cell(meanMB(refPct), 1) : '-', 'MB, float16'),
  ].join('');

  // the finding, and the reason not to lean on it
  if (shipOK && refOK) {
    const dA = (meanA(shipPct) - meanA(refPct)) * 100;
    const ratio = meanMB(refPct) / meanMB(shipPct);
    const spread = (() => {
      const [lo, hi] = extent(at(shipPct).map((r) => r[COMP_FIELD]));
      return (hi - lo) * 100;
    })();
    $('compFind').textContent =
      'Across all ' + nCat + ' categories the ' + pctLabel(shipPct) + ' bank means '
      + auroc100(meanA(shipPct)) + ' image AUROC and the ' + pctLabel(refPct) + ' bank means '
      + auroc100(meanA(refPct)) + '. The ' + pctLabel(shipPct) + ' banks average '
      + cell(meanMB(shipPct), 1) + ' MB against ' + cell(meanMB(refPct), 1) + ' MB, so the memory'
      + ' budget falls by a factor of ' + ratio.toFixed(1) + ' and the mean does not follow it down.'
      + ' The two means are ' + Math.abs(dA).toFixed(2) + ' points apart. At '
      + pctLabel(shipPct) + ' the categories are spread over ' + spread.toFixed(2)
      + ' points, so a gap this small between the means does not establish that '
      + pctLabel(shipPct) + ' is the better setting in general. On these ' + nCat
      + ' categories the bigger bank bought no accuracy.';
  } else {
    $('compFind').textContent = '';
  }

  // which categories a smaller bank actually costs, and whether those are the ones the paper
  // already found hard. Both halves are read out of the file and the second sentence only
  // prints if the two lists agree.
  const small = full.length ? full[0] : null;
  if (small !== null && shipOK && small !== shipPct) {
    const drops = [...byCat.entries()].map(([name, list]) => {
      const a = list.find((r) => r.coreset_pct === small);
      const b = list.find((r) => r.coreset_pct === shipPct);
      return a && b ? { name, d: (b[COMP_FIELD] - a[COMP_FIELD]) * 100 } : null;
    }).filter(Boolean).sort((x, y) => y.d - x.d);
    const top = drops.slice(0, 2);
    const rest = drops.slice(2);
    let t = 'Dropping to ' + pctLabel(small) + ' costs ' + top.map((x) => x.name).join(' and ')
      + ' the most: ' + top.map((x) => x.d.toFixed(2)).join(' and ') + ' points of image AUROC.';
    if (rest.length) {
      t += ' The other ' + rest.length + ' categories move by at most '
        + Math.max(...rest.map((x) => Math.abs(x.d))).toFixed(2) + ' points.';
    }
    // rank the categories by the figure the paper published, and only make the claim if the
    // bottom of that ranking is the same pair
    const pub = S.cats ? S.cats
      .map((c) => ({ name: c.category, v: c.paper_comparison && c.paper_comparison.published_image_auroc }))
      .filter((x) => typeof x.v === 'number')
      .sort((x, y) => x.v - y.v) : [];
    if (pub.length === nCat) {
      const bottom = pub.slice(0, 2).map((x) => x.name).sort();
      if (String(bottom) === String(top.map((x) => x.name).sort())) {
        // each figure is printed next to its own category, because this list is ranked by
        // the published score and the one above it is ranked by the drop
        t += ' Those are also the two lowest published figures in the paper: '
          + pub.slice(0, 2).map((x) => x.name + ' at ' + x.v.toFixed(1)).join(' and ') + '.';
      }
    }
    $('compShots').textContent = t;
  } else {
    $('compShots').textContent = '';
  }

  $('srcCompression').textContent = 'source: ' + S.compPath + '. Field plotted: ' + COMP_FIELD + '.';
}

// ---- plain language intro ---------------------------------------------------

// intro.json is the only copy on this page written for someone who does not read AUROC.
// It is printed as it was written, and the headings are the file's own key names with the
// underscores taken out, so no sentence and no label here was composed in this file. A key
// that is not in the file prints nothing rather than a heading over an empty space.
const INTRO = 'intro.json';

const headOf = (key) => {
  const s = key.replace(/_/g, ' ');
  return s.charAt(0).toUpperCase() + s.slice(1);
};

const paras = (list) => list.map((t) => '<p>' + esc(t) + '</p>').join('');
const sect = (key, inner) => '<h2>' + esc(headOf(key)) + '</h2>' + inner;

async function drawIntro() {
  const sec = $('panel-intro');
  const d = await maybeJSON(INTRO);
  if (!d) { sec.classList.add('hidden'); return; }

  const out = [];
  if (d.headline) out.push('<h1>' + esc(d.headline) + '</h1>');
  if (d.paragraphs) out.push(paras(d.paragraphs));
  if (d.what_you_are_looking_at) {
    out.push(sect('what_you_are_looking_at', paras(d.what_you_are_looking_at)));
  }
  const q = d.the_two_questions;
  if (q) {
    const items = (q.items || []).map((it) =>
      '<div class="qa"><p class="q">' + esc(it.q) + '</p><p class="a">' + esc(it.a) + '</p></div>').join('');
    out.push(sect('the_two_questions', (q.intro ? '<p>' + esc(q.intro) + '</p>' : '') + items));
  }
  if (d.honest_bits) {
    out.push(sect('honest_bits',
      '<ul class="bits">' + d.honest_bits.map((t) => '<li>' + esc(t) + '</li>').join('') + '</ul>'));
  }
  if (d.credits) out.push('<p class="credit">' + esc(d.credits) + '</p>');
  // the note in the file is addressed to whoever maintains the copy, not to a visitor, so
  // it goes in the source line with the filename rather than into the body text
  out.push('<p class="src">source: ' + DATA + INTRO + (d.note ? '. ' + esc(d.note) : '') + '</p>');

  sec.innerHTML = out.join('');
  sec.classList.remove('hidden');
}

// ---- status bar and provenance ---------------------------------------------

function drawStatus() {
  const m = S.primary.metrics;
  const cells = [
    ['Category', m.category],
    ['Backbone', m.config.backbone],
    ['Coreset', typeof m.coreset_pct === 'number' ? pctLabel(m.coreset_pct) : '-'],
    ['Bank rows', m.bank_size.toLocaleString()],
    ['Train (good only)', m.n_train],
    ['Test good', m.n_test_good],
    ['Test defect', m.n_test_defect],
    ['Pixel AUROC', f4(m.pixel_auroc)],
    ['Calibration split', 'none exported'],
  ];
  $('statusBar').innerHTML = cells
    .map(([k, v]) => '<div class="bar-cell"><b>' + esc(String(v)) + '</b><span>' + esc(k) + '</span></div>')
    .join('');
}

function drawSources() {
  const p = S.primary;
  const src = (label, files) => label + ': ' + files.join(', ');
  $('srcThreshold').textContent = src('source', [DATA + p.files.scores + ' (' + p.source + '/scores.csv)',
    DATA + p.files.metrics + ' (' + p.source + '/metrics.json)']);
  $('srcBreaks').textContent = src('source', [DATA + p.files.scores + ' (' + p.source + '/scores.csv)']);
  $('srcCurve').textContent = src('source', [DATA + (S.manifest.curve || 'curve.json')]);
  if (S.maps) {
    $('srcMaps').textContent = src('source', [DATA + S.maps.files.heatmaps + ' and ' + DATA + 'heatmaps/*.png ('
      + S.maps.source + '/heatmaps), scores joined from ' + DATA + S.maps.files.scores])
      + '. Derived from MVTec AD images, CC BY-NC-SA 4.0.';
  }
  $('srcRuns').textContent = src('source', S.runs.map((r) => DATA + r.files.metrics));

  // The panels below the live one run on an exported run in data/. The live panel at the
  // top downloads its own bank out of artifacts/. Those are different files at different
  // coresets, and the artifacts are shape identical, so the pair is spelled out here rather
  // than left to the reader.
  const pm = S.primary.metrics;
  const desc = pm.category + ' / ' + pm.config.backbone
    + (typeof pm.coreset_pct === 'number' ? ' / coreset ' + pctLabel(pm.coreset_pct) : '')
    + (pm.config.weights_sha256 ? ' / weights sha256 ' + pm.config.weights_sha256.slice(0, 12) : '');
  let t = 'Panels below the live one: ' + desc + ' (' + DATA + p.files.metrics + ').';
  if (S.cats && S.cats.length) {
    const bb = oneOf(S.cats.map((c) => c.backbone));
    const cp = oneOf(S.cats.map((c) => c.coreset_pct));
    t += ' Banks the live panel downloads: ' + S.cats.length + ' categories'
      + (bb ? ', all ' + bb : '')
      + (cp !== null ? ', all coreset ' + pctLabel(cp) : '')
      + ' (' + S.catsPath + ').';
  }
  $('footSidecar').textContent = t;
}

// ---- render -----------------------------------------------------------------

function renderCharts() {
  drawHist();
  drawCost();
}

function renderAll() {
  drawStatus();
  renderCharts();
  drawReadouts();
  drawBreaks();
  drawStrip();
}

// The category buttons sit in the live panel and used to move only the bank, so
// picking leather left the heatmaps, the threshold and the failure tables on
// screw with nothing saying so. live.js calls this on a switch. Every category
// ships its own metrics, scores and heatmap index, so the whole page can follow.
function useCategory(category) {
  const run = S.runs.find((r) => r.id === 'ship_' + category);
  if (!run || !run.rows) return false;
  S.primary = run;
  if (run.heatmaps && run.heatmaps.length) S.maps = run;
  recomputeDomain();
  S.threshold = costMinimum(S.primary.rows, costInputs()).thr;
  renderAll();
  drawSources();
  return true;
}

function setThreshold(v) {
  const [lo, hi] = S.domain;
  S.threshold = Math.min(Math.max(v, lo), hi);
  renderCharts();
  drawReadouts();
  drawBreaks();
}

function recomputeDomain() {
  const values = S.primary.rows.map(score);
  const [lo, hi] = extent(values);
  const pad = (hi - lo) * 0.04;
  S.domain = [lo - pad, hi + pad];
}

// ---- init -------------------------------------------------------------------

async function boot() {
  // first, and before the manifest, so the intro still stands if the run data fails to load
  await drawIntro();

  S.manifest = await getJSON('index.json');

  S.runs = await Promise.all(S.manifest.runs.map(async (r) => {
    const metrics = await getJSON(r.metrics);
    const rows = r.scores ? parseCSV(await getText(r.scores)) : null;
    const heatmaps = r.heatmaps ? await maybeJSON(r.heatmaps) : null;
    return { id: r.id, source: r.source, files: r, metrics, rows, heatmaps };
  }));

  S.primary = S.runs.find((r) => r.id === S.manifest.primary_run);
  if (!S.primary || !S.primary.rows) throw new Error('primary run ' + S.manifest.primary_run + ' has no scores.csv');
  // whichever run exported heatmaps drives that strip, independent of the primary run
  S.maps = S.runs.find((r) => r.heatmaps && r.heatmaps.length) || null;

  const cats = await findArt(S.manifest.categories || 'categories.json');
  const comp = await findArt(S.manifest.compression || 'compression.json');
  S.cats = cats ? cats.data : null;
  S.catsPath = cats ? cats.path : null;
  S.comp = comp ? comp.data : null;
  S.compPath = comp ? comp.path : null;

  recomputeDomain();
  S.threshold = costMinimum(S.primary.rows, costInputs()).thr;

  renderAll();
  drawRuns();
  drawPublished();
  drawCompression();
  drawCompressionText();
  drawSources();
  curveCache = await loadCurve();
  drawCurve(curveCache);
  wire();
}

// A1 writes this offline. accept a bare array or a records wrapper, and say so plainly when
// it is not there yet.
async function loadCurve() {
  const raw = await maybeJSON(S.manifest.curve || 'curve.json');
  if (!raw) return null;
  return Array.isArray(raw) ? raw : raw.records || null;
}

let curveCache = null;

function wire() {
  const canvas = $('histCanvas');
  let dragging = false;
  const pick = (ev) => {
    const rect = canvas.getBoundingClientRect();
    const x = ev.clientX - rect.left;
    const x0 = PAD.l, x1 = rect.width - PAD.r;
    const t = S.domain[0] + ((x - x0) / (x1 - x0)) * (S.domain[1] - S.domain[0]);
    setThreshold(t);
  };
  canvas.addEventListener('pointerdown', (ev) => {
    dragging = true;
    canvas.setPointerCapture(ev.pointerId);
    pick(ev);
  });
  canvas.addEventListener('pointermove', (ev) => { if (dragging) pick(ev); });
  canvas.addEventListener('pointerup', (ev) => {
    dragging = false;
    canvas.releasePointerCapture(ev.pointerId);
  });

  $('scoreKey').addEventListener('change', (ev) => {
    S.scoreKey = ev.target.value;
    recomputeDomain();
    S.threshold = costMinimum(S.primary.rows, costInputs()).thr;
    renderAll();
    drawCurve(curveCache);
  });

  ['defectRate', 'costEscape', 'costFalse'].forEach((id) =>
    $(id).addEventListener('input', () => { renderCharts(); drawReadouts(); }));

  $('snapMin').addEventListener('click', () => setThreshold(costMinimum(S.primary.rows, costInputs()).thr));

  let resizeTimer = null;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { renderCharts(); drawCurve(curveCache); drawCompression(); }, 80);
  });
}

boot().catch((err) => {
  const box = $('loadError');
  box.classList.remove('hidden');
  box.textContent = 'load failed: ' + err.message
    + '\nthis page reads data/ over fetch, so it needs a static server, not file://.'
    + '\nserve Nazar/web/app over http and reload.';
});
