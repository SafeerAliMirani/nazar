// Nazar live inspection panel.
//
// The kNN that scores a part runs here, in WGSL, on the visitor's GPU, against the
// same memory bank the pipeline used. The backbone does not run in the browser: patch
// features come out of queries/*.f16.bin, computed offline. Everything downstream of
// that is live.
//
// Same rule as app.js: no figure printed here is written in this file. Scores are
// measured, the reference values come from the per-set file named in queries/manifest.json,
// tolerances are derived from the float16 format and from vector norms read at runtime.
// Download sizes come from the HEAD response, or from the sidecar byte count if the server
// will not give a length.
//
// This file is deliberately independent of app.js. If the bank fails to download or the
// GPU refuses the buffer, this panel says so and the rest of the page carries on.

(() => {
'use strict';

// data/ first in case the artifacts get copied next to the page at ship time, then the
// repo layout the dev server uses.
const ROOTS = ['data/', '../../artifacts/'];

const KERNEL = '../kernels/knn.wgsl';
const ENTRY = 'knn';
const WG_SIZE = 256;              // matches @workgroup_size in knn.wgsl
const SENTINEL = -12345;          // if this survives, the kernel never wrote

// float16: 10 mantissa bits, so an ulp is 2^-10 relative and worst case rounding is
// half of that. Used to derive the parity tolerance, not tuned to make a table go green.
const F16_REL = Math.pow(2, -11);
const F32_EPS = Math.pow(2, -24);

const FADE_MS = 420;
const COUNT_MS = 480;

// same breakpoint the stylesheet stacks the page at. Below it the panel waits to be asked
// instead of pulling the bank down a phone connection.
const NARROW = '(max-width: 900px)';

// inferno-ish ramp. cold end is transparent so a clean part looks clean.
const RAMP = [
  [0.00, 4, 3, 26],
  [0.25, 66, 10, 104],
  [0.50, 147, 38, 103],
  [0.75, 221, 81, 58],
  [0.90, 252, 165, 10],
  [1.00, 252, 255, 164],
];
const TRUTH = [70, 220, 130];

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[<>&]/g, (c) => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;' }[c]));
const f3 = (v) => v.toFixed(3);
const f4 = (v) => v.toFixed(4);
const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);

const L = {
  root: null, meta: null, side: null, device: null, pipe: null, bind: null,
  bankU16: null, nb9: null, N: 0, D: 0, Q: 0,
  qGPU: null, minGPU: null, idxGPU: null, params: null, stageMin: null, stageIdx: null,
  bankGPU: null,
  scale: [0, 1], view: 'overlay', busy: false, current: null, cache: new Map(),
  fadeStart: 0, fadeRaf: 0,
  // the switcher
  sets: [], sel: -1, defaultCat: null, armed: false, switching: false, gen: 0,
};

// ---- float16 ----------------------------------------------------------------

// 65536 entries, so decoding the bank is a table lookup per value instead of bit
// twiddling 38 million times.
const F16 = (() => {
  const t = new Float32Array(65536);
  const SUB = Math.pow(2, -14);
  for (let h = 0; h < 65536; h++) {
    const s = (h & 0x8000) ? -1 : 1;
    const e = (h & 0x7c00) >> 10;
    const f = h & 0x03ff;
    t[h] = e === 0 ? s * SUB * (f / 1024)
         : e === 31 ? (f ? NaN : s * Infinity)
         : s * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  return t;
})();

const f16to32 = (u16) => {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) out[i] = F16[u16[i]];
  return out;
};

// ---- loading ----------------------------------------------------------------

function progress(label, frac, note) {
  $('loadLabel').textContent = label;
  $('loadPct').textContent = frac >= 0 ? (frac * 100).toFixed(0) + '%' : '';
  $('loadBar').style.width = (clamp01(frac >= 0 ? frac : 1) * 100) + '%';
  if (note !== undefined) $('loadNote').textContent = note;
}

// bytes on the wire for one url, straight off the response. null if the server will not
// say, which is the case for a chunked response.
async function sizeOf(url) {
  try {
    const r = await fetch(url, { method: 'HEAD' });
    if (!r.ok) return null;
    const n = Number(r.headers.get('content-length'));
    return Number.isFinite(n) && n > 0 ? n : null;
  } catch (e) {
    return null;
  }
}

// find whichever root actually serves the queries manifest
async function findRoot() {
  for (const root of ROOTS) {
    try {
      const r = await fetch(root + 'queries/manifest.json');
      if (r.ok) return { root, man: await r.json() };
    } catch (e) { /* try the next one */ }
  }
  return null;
}

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(url + ' -> HTTP ' + r.status);
  return r.json();
}

// streamed so the bank download shows a real bar instead of a dead page
async function getBytes(url, label) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(url + ' -> HTTP ' + r.status);
  const total = Number(r.headers.get('content-length')) || 0;
  if (!r.body) return r.arrayBuffer();
  const reader = r.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    progress(label, total ? got / total : -1,
      (got / 1e6).toFixed(1) + ' of ' + (total ? (total / 1e6).toFixed(1) + ' MB' : 'unknown size'));
    await null;
  }
  const out = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out.buffer;
}

function fallback(msg) {
  $('liveLoad').classList.add('hidden');
  $('liveBody').classList.add('hidden');
  $('liveDefer').classList.add('hidden');
  const box = $('liveFallback');
  box.classList.remove('hidden');
  box.textContent = msg + '\nThe heatmaps further down this page were rendered offline and still apply.';
  $('liveVerdict').textContent = 'live panel off';
  $('liveVerdict').className = 'pill off';
}

// ---- gpu --------------------------------------------------------------------

async function initGPU(N, D, Q) {
  if (!navigator.gpu) throw new Error('This browser has no WebGPU, so the kernel cannot run here.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('WebGPU is present but no GPU adapter was handed out.');

  const bankBytes = N * D * 4;   // the kernel binds array<f32>, so f16 is expanded on upload
  if (adapter.limits.maxStorageBufferBindingSize < bankBytes) {
    throw new Error('This GPU caps a storage buffer at '
      + (adapter.limits.maxStorageBufferBindingSize / 1e6).toFixed(0) + ' MB and the bank needs '
      + (bankBytes / 1e6).toFixed(0) + ' MB as f32.');
  }
  // the WebGPU default is 128 MB, under what the bank needs, so ask for it explicitly and
  // never ask for less than the default we would have had anyway
  const want = (need, adapterMax, dflt) => Math.min(adapterMax, Math.max(need, dflt));
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: want(bankBytes, adapter.limits.maxStorageBufferBindingSize, 134217728),
      maxBufferSize: want(bankBytes, adapter.limits.maxBufferSize, 268435456),
    },
  });
  device.addEventListener('uncapturederror', (e) => console.error('nazar gpu:', e.error.message));

  const info = adapter.info || {};
  const name = [info.description, info.vendor, info.architecture].filter(Boolean).join(' ');
  $('liveAdapter').textContent = name ? 'gpu: ' + name : 'gpu: adapter does not name itself';
  return device;
}

function uploadBank(device, u16, N, D) {
  // decoded straight into the mapped buffer, so the f32 copy, twice the size of the download,
  // never also exists as a JS array. the f16 array stays: eq.7 needs 9 bank rows on the CPU.
  const buf = device.createBuffer({
    size: N * D * 4, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true,
  });
  const dst = new Float32Array(buf.getMappedRange());
  for (let i = 0; i < dst.length; i++) dst[i] = F16[u16[i]];
  buf.unmap();
  return buf;
}

async function readBack(device, src, stage, Type, n) {
  const bytes = n * Type.BYTES_PER_ELEMENT;
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(src, 0, stage, 0, bytes);
  device.queue.submit([enc.finish()]);
  await stage.mapAsync(GPUMapMode.READ);
  const v = new Type(stage.getMappedRange().slice(0, bytes));
  stage.unmap();
  return v;
}

// one workgroup per query patch, as the kernel expects
async function runKNN(qF32) {
  const { device, Q } = L;
  device.queue.writeBuffer(L.qGPU, 0, qF32);
  device.queue.writeBuffer(L.minGPU, 0, new Float32Array(Q).fill(SENTINEL));
  device.queue.writeBuffer(L.idxGPU, 0, new Uint32Array(Q).fill(0xffffffff));

  const t0 = performance.now();
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(L.pipe);
  pass.setBindGroup(0, L.bind);
  pass.dispatchWorkgroups(Q);
  pass.end();
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
  const ms = performance.now() - t0;

  const mins = await readBack(device, L.minGPU, L.stageMin, Float32Array, Q);
  const idx = await readBack(device, L.idxGPU, L.stageIdx, Uint32Array, Q);
  // the readback liveness check: a buffer that still holds what we put in it is not a result
  for (let i = 0; i < Q; i++) {
    if (mins[i] === SENTINEL) throw new Error('the kernel did not write patch ' + i);
  }
  return { mins, idx, ms };
}

// ---- scoring ----------------------------------------------------------------

const bankVal = (r, k) => F16[L.bankU16[r * L.D + k]];

function distToRow(q, qoff, r) {
  let acc = 0;
  const base = r * L.D;
  for (let k = 0; k < L.D; k++) {
    const d = q[qoff + k] - F16[L.bankU16[base + k]];
    acc += d * d;
  }
  return Math.sqrt(acc);
}

function rowNorm(r) {
  let acc = 0;
  const base = r * L.D;
  for (let k = 0; k < L.D; k++) { const v = F16[L.bankU16[base + k]]; acc += v * v; }
  return Math.sqrt(acc);
}

function vecNorm(q, off, n) {
  let acc = 0;
  for (let k = 0; k < n; k++) { const v = q[off + k]; acc += v * v; }
  return Math.sqrt(acc);
}

// eq.7, same shift as the pipeline: exp is shifted by the largest distance so it cannot
// overflow, and the shift cancels in the ratio.
function eq7(dStar, dNb) {
  let c = dStar;
  for (const d of dNb) if (d > c) c = d;
  let sum = 0;
  for (const d of dNb) sum += Math.exp(d - c);
  return 1 - Math.exp(dStar - c) / sum;
}

// ---- map --------------------------------------------------------------------

// bilinear, align_corners=False, matching torch F.interpolate
function upsample(src, h, w, size) {
  const out = new Float32Array(size * size);
  const sy = h / size, sx = w / size;
  for (let y = 0; y < size; y++) {
    let fy = (y + 0.5) * sy - 0.5;
    if (fy < 0) fy = 0;
    const y0 = Math.min(Math.floor(fy), h - 1), y1 = Math.min(y0 + 1, h - 1), ly = fy - y0;
    for (let x = 0; x < size; x++) {
      let fx = (x + 0.5) * sx - 0.5;
      if (fx < 0) fx = 0;
      const x0 = Math.min(Math.floor(fx), w - 1), x1 = Math.min(x0 + 1, w - 1), lx = fx - x0;
      const a = src[y0 * w + x0], b = src[y0 * w + x1];
      const c = src[y1 * w + x0], d = src[y1 * w + x1];
      out[y * size + x] = (a + (b - a) * lx) * (1 - ly) + (c + (d - c) * lx) * ly;
    }
  }
  return out;
}

// scipy gaussian_filter defaults: truncate 4.0, reflect at the edges
function gaussKernel(sigma) {
  const r = Math.floor(4.0 * sigma + 0.5);
  const k = new Float32Array(2 * r + 1);
  let sum = 0;
  for (let i = -r; i <= r; i++) { const v = Math.exp(-0.5 * (i / sigma) * (i / sigma)); k[i + r] = v; sum += v; }
  for (let i = 0; i < k.length; i++) k[i] /= sum;
  return { k, r };
}

const reflect = (i, n) => (i < 0 ? -i - 1 : i >= n ? 2 * n - i - 1 : i);

function blur(img, size, sigma) {
  const { k, r } = gaussKernel(sigma);
  const tmp = new Float32Array(size * size);
  const out = new Float32Array(size * size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let acc = 0;
      for (let i = -r; i <= r; i++) acc += k[i + r] * img[y * size + reflect(x + i, size)];
      tmp[y * size + x] = acc;
    }
  }
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let acc = 0;
      for (let i = -r; i <= r; i++) acc += k[i + r] * tmp[reflect(y + i, size) * size + x];
      out[y * size + x] = acc;
    }
  }
  return out;
}

function ramp(t) {
  for (let i = 1; i < RAMP.length; i++) {
    if (t <= RAMP[i][0] || i === RAMP.length - 1) {
      const a = RAMP[i - 1], b = RAMP[i];
      const u = clamp01((t - a[0]) / (b[0] - a[0]));
      return [a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u, a[3] + (b[3] - a[3]) * u];
    }
  }
  return [0, 0, 0];
}

// ---- drawing ----------------------------------------------------------------

function loadImage(url) {
  return new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => res(im);
    im.onerror = () => rej(new Error(url + ' failed to load'));
    im.src = url;
  });
}

function imageData(im, size) {
  const c = document.createElement('canvas');
  c.width = size; c.height = size;
  const cx = c.getContext('2d', { willReadFrequently: true });
  cx.drawImage(im, 0, 0, size, size);
  return cx.getImageData(0, 0, size, size);
}

// mask pixels that touch an off pixel: the outline of the labelled defect
function maskEdges(mask, size) {
  const on = new Uint8Array(size * size);
  let count = 0;
  for (let i = 0; i < size * size; i++) { if (mask.data[i * 4] > 127) { on[i] = 1; count++; } }
  const edge = new Uint8Array(size * size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = y * size + x;
      if (!on[i]) continue;
      const l = x > 0 && on[i - 1], r = x < size - 1 && on[i + 1];
      const u = y > 0 && on[i - size], d = y < size - 1 && on[i + size];
      if (!(l && r && u && d)) edge[i] = 1;
    }
  }
  return { on, edge, count };
}

function paint(alpha) {
  const cur = L.current;
  if (!cur) return;
  const size = cur.size;
  const cv = $('liveCanvas');
  const cx = cv.getContext('2d');
  const out = cx.createImageData(size, size);
  const src = cur.img.data;
  const [lo, hi] = L.scale;
  const span = hi - lo || 1;
  const dim = L.view === 'truth' ? 0.45 : 1;

  for (let i = 0; i < size * size; i++) {
    let r = src[i * 4] * dim, g = src[i * 4 + 1] * dim, b = src[i * 4 + 2] * dim;
    if (L.view === 'overlay' && cur.map) {
      const t = clamp01((cur.map[i] - lo) / span);
      // colour carries the distance, opacity only decides how much of the part shows
      // through. cubed so the peak reads and the flat middle of the map does not fog
      // the whole screw.
      const a = t * t * t * 0.9 * alpha;
      if (a > 0) {
        const c = ramp(t);
        r = r * (1 - a) + c[0] * a;
        g = g * (1 - a) + c[1] * a;
        b = b * (1 - a) + c[2] * a;
      }
    }
    if (L.view === 'truth' && cur.mask && cur.mask.on[i]) {
      r = r * 0.4 + TRUTH[0] * 0.6; g = g * 0.4 + TRUTH[1] * 0.6; b = b * 0.4 + TRUTH[2] * 0.6;
    }
    if (L.view !== 'part' && cur.mask && cur.mask.edge[i]) {
      r = TRUTH[0]; g = TRUTH[1]; b = TRUTH[2];
    }
    out.data[i * 4] = r; out.data[i * 4 + 1] = g; out.data[i * 4 + 2] = b; out.data[i * 4 + 3] = 255;
  }
  cx.putImageData(out, 0, 0);
}

function fadeIn() {
  cancelAnimationFrame(L.fadeRaf);
  if (document.hidden) { paint(1); return; }   // a throttled rAF would leave the map at zero alpha
  L.fadeStart = performance.now();
  const step = () => {
    const t = clamp01((performance.now() - L.fadeStart) / FADE_MS);
    paint(t * t * (3 - 2 * t));
    if (t < 1) L.fadeRaf = requestAnimationFrame(step);
  };
  step();
}

// rAF is throttled in a hidden tab, which would freeze the animation part way and leave a
// number on screen that is not the score. The value goes up first, the animation only
// replays it.
function countUp(el, to, dp) {
  el.textContent = to.toFixed(dp);
  if (document.hidden) return;
  const t0 = performance.now();
  const step = () => {
    const t = clamp01((performance.now() - t0) / COUNT_MS);
    const e = 1 - Math.pow(1 - t, 3);
    el.textContent = (to * e).toFixed(dp);
    if (t < 1) requestAnimationFrame(step);
  };
  step();
}

// ---- panel ------------------------------------------------------------------

function ro(label, id, sub, cls) {
  return '<div class="ro ' + (cls || '') + '"><span>' + esc(label) + '</span>'
       + '<b id="' + id + '">-</b>' + (sub ? '<small>' + esc(sub) + '</small>' : '') + '</div>';
}

function strip() {
  $('liveStrip').innerHTML = L.meta.images.map((im, i) =>
    '<button type="button" class="thumb" data-i="' + i + '">'
    + '<img src="' + esc(L.root + 'queries/' + im.name + '.png') + '" alt="' + esc(im.name) + '" width="224" height="224">'
    + '<span class="' + (im.label ? 'bad' : 'ok') + '">' + esc(im.defect_type) + '</span>'
    + '</button>').join('');
}

function markThumb(i) {
  [...$('liveStrip').children].forEach((el, j) => el.classList.toggle('on', i === j));
}

async function pick(i) {
  if (L.busy) return;   // the strip stops taking clicks while this runs, see .busy below
  L.busy = true;
  $('liveStrip').classList.add('busy');
  markThumb(i);
  const meta = L.meta.images[i];
  const base = L.root + 'queries/' + meta.name;
  $('stageTag').textContent = meta.defect_type + '  ' + meta.file;
  $('liveVerdict').textContent = 'running on your gpu';
  $('liveVerdict').className = 'pill run';
  $('stageScan').classList.remove('hidden');

  try {
    if (!L.cache.has(meta.name)) {
      const [bin, im, mk] = await Promise.all([
        fetch(base + '.f16.bin').then((r) => r.arrayBuffer()),
        loadImage(base + '.png'),
        loadImage(base + '_mask.png').catch(() => null),
      ]);
      const size = L.meta.crop;
      L.cache.set(meta.name, {
        q: f16to32(new Uint16Array(bin)),
        img: imageData(im, size),
        mask: mk ? maskEdges(imageData(mk, size), size) : null,
        size,
      });
    }
    const cur = Object.assign({}, L.cache.get(meta.name), { meta });
    L.current = cur;
    paint(1);

    const { mins, idx, ms } = await runKNN(cur.q);

    // d_star is the worst patch, before any blur
    let p = 0;
    for (let j = 1; j < mins.length; j++) if (mins[j] > mins[p]) p = j;
    const dStar = mins[p];
    const nStar = idx[p];

    // eq.7 over the 9 bank neighbours of n*, gathered from the precomputed table
    const b = L.meta.b_nearest;
    const dNb = [];
    for (let j = 0; j < b; j++) dNb.push(distToRow(cur.q, p * L.D, L.nb9[nStar * b + j]));
    const w = eq7(dStar, dNb);
    const score = w * dStar;

    const [gh, gw] = meta.grid;
    const grid = new Float32Array(gh * gw);
    for (let j = 0; j < gh * gw; j++) grid[j] = mins[j];
    cur.map = blur(upsample(grid, gh, gw, cur.size), cur.size, L.meta.blur_sigma);
    let mLo = Infinity, mHi = -Infinity;
    for (const v of cur.map) { if (v < mLo) mLo = v; if (v > mHi) mHi = v; }

    // Tolerance, derived not tuned. The bank and the query ship float16, python scored
    // from float32. For a distance d = ||q - b||, perturbing b by db moves d by at most
    // ||db|| (Cauchy-Schwarz on the unit difference vector), and ||db|| <= F16_REL*||b||.
    // Same for the query. The f32 accumulation over D terms adds sqrt(D)*eps*d.
    const ref = meta.reference;
    const nB = rowNorm(nStar), nQ = vecNorm(cur.q, p * L.D, L.D);
    const tolD = F16_REL * (nB + nQ) + Math.sqrt(L.D) * F32_EPS * dStar;
    // w = 1 - p where p is a softmax term, so |dw| <= 2*p*tolD and p = 1 - w
    const tolW = 2 * (1 - w) * tolD;
    const tolS = w * tolD + dStar * tolW;

    const rows = [
      ['d_star', dStar, ref.d_star, tolD, 4],
      ['w (eq.7)', w, ref.w, tolW, 4],
      ['score, w * d_star', score, ref.score_eq7, tolS, 4],
    ];
    let bad = 0;
    const html = rows.map(([name, got, exp, tol, dp]) => {
      const diff = Math.abs(got - exp);
      const ok = diff <= tol;
      if (!ok) bad++;
      return '<tr><td>' + esc(name) + '</td><td>' + got.toFixed(dp) + '</td><td>' + exp.toFixed(dp)
        + '</td><td>' + diff.toExponential(1) + '</td><td>' + tol.toExponential(1) + '</td>'
        + '<td class="' + (ok ? 'ok' : 'alarm') + '">' + (ok ? 'match' : 'MISMATCH') + '</td></tr>';
    }).join('');
    $('tblParity').innerHTML =
      '<thead><tr><th>Quantity</th><th>This GPU</th><th>Python</th><th>Diff</th><th>Tolerance</th><th></th></tr></thead>'
      + '<tbody>' + html + '</tbody>';
    // The exported reference carries d_star, w and the score. It does not carry the map
    // range, so the peak is checked against d_star instead: the map is these same patch
    // distances upsampled and blurred, and neither step can lift a value above the grid
    // maximum, so the peak has to land at or under d_star.
    $('parityTol').textContent =
      'Tolerance is derived from the float16 bank, not tuned: |d error| <= 2^-11 * (||bank row|| + ||patch||) = '
      + tolD.toExponential(2) + ' at ||bank row|| ' + f3(nB) + ', ||patch|| ' + f3(nQ) + '.'
      + ' Heatmap peak here ' + f3(mHi) + ', floor ' + f3(mLo) + ', against d_star ' + f3(dStar)
      + ' before the blur.';

    $('liveVerdict').textContent = bad ? 'PARITY MISMATCH' : 'matches python';
    $('liveVerdict').className = 'pill ' + (bad ? 'bad' : 'ok');

    $('roMs').textContent = ms.toFixed(1);
    // every patch is compared against every bank row: 2 flops per dim, all of them
    $('roGf').textContent = ((2 * L.Q * L.N * L.D) / ms / 1e6).toFixed(0);
    $('roRows').textContent = (L.N * L.Q / 1e6).toFixed(1);
    countUp($('roD'), dStar, 3);
    countUp($('roW'), w, 4);
    countUp($('roScore'), score, 3);
    $('roPatch').textContent = p;
    $('roNstar').textContent = nStar;

    $('stageScan').classList.add('hidden');
    fadeIn();
  } catch (err) {
    $('stageScan').classList.add('hidden');
    $('liveVerdict').textContent = 'gpu run failed';
    $('liveVerdict').className = 'pill bad';
    $('parityTol').textContent = 'live run failed: ' + err.message;
    console.error(err);
  }
  L.busy = false;
  $('liveStrip').classList.remove('busy');
}

function wireViews() {
  $('viewBtns').addEventListener('click', (ev) => {
    const b = ev.target.closest('button');
    if (!b) return;
    L.view = b.dataset.view;
    [...$('viewBtns').children].forEach((el) => el.classList.toggle('on', el === b));
    paint(1);
  });
}

// ---- the page's own runs ----------------------------------------------------

// data/index.json names the runs behind the panels below this one. Loaded once and kept,
// because both notes under here need it and it is the same handful of small files app.js
// already pulled.
let runsCache = null;
async function pageRuns() {
  if (runsCache) return runsCache;
  const man = await getJSON('data/index.json');
  const runs = await Promise.all(man.runs.map(async (r) => (
    { id: r.id, file: r.metrics, m: await getJSON('data/' + r.metrics) }
  )));
  runsCache = { man, runs };
  return runsCache;
}

const backboneOf = (m) => (m.config && m.config.backbone ? m.config.backbone : null);

// The panel used to check the backbone and the coreset here against the run driving the rest
// of the page, because at one point they were different exports and a mixup between two
// shape-identical banks would have been invisible. Everything on the page is now one backbone
// at one coreset out of one publish step, so that check can no longer fire and the warning it
// printed is not a thing that can be true.
//
// The category still can differ, because the visitor picks it. The panels below are one run on
// one category, so picking anything else up here puts these scores on a different scale from
// the threshold down there. Read out of the data rather than assumed.
async function scopeNote() {
  const el = $('liveScope');
  el.classList.add('hidden');
  try {
    const { man, runs } = await pageRuns();
    const run = runs.find((r) => r.id === man.primary_run);
    if (!run || run.m.category === L.meta.category) return;
    el.textContent = 'This panel is scoring ' + L.meta.category + '. Every panel below it is '
      + run.m.category + ', from ' + run.file + ', so the scores here are not on the same scale as'
      + ' the threshold down there.';
    el.classList.remove('hidden');
  } catch (e) {
    console.warn('nazar: could not compare the live category against the page run', e);
  }
}

// If this category has both a clean run and a run with the bank deliberately destroyed,
// the gap between them is what says whether the bank is doing the work. On leather it
// barely moves, and someone who switches to leather should read that before they read the
// score. Pulled out of the metrics files rather than asserted, and it prints nothing for a
// category that has no ablation run.
async function categoryNote(set) {
  const el = $('setNote');
  el.classList.add('hidden');
  el.textContent = '';
  try {
    const { runs } = await pageRuns();
    const here = runs.filter((r) => r.m.category === set.category);
    const clean = here.find((r) => r.m.ablate_bank === 'none' && !r.m.permute_labels);
    const ablated = here.find((r) => r.m.ablate_bank && r.m.ablate_bank !== 'none');
    if (!clean || !ablated) return;
    const bb = backboneOf(clean.m);
    // only worth printing as a pair if the two runs differ by the ablation and nothing else
    if (!bb || bb !== backboneOf(ablated.m)) return;
    if (typeof clean.m.image_auroc !== 'number' || typeof ablated.m.image_auroc !== 'number') return;
    const cap = set.category.charAt(0).toUpperCase() + set.category.slice(1);
    let text = cap + ' scores image AUROC ' + f4(clean.m.image_auroc) + ' in ' + clean.file + '.'
      + ' With the bank shuffled along every feature dimension, which destroys it, ' + set.category
      + ' still scores ' + f4(ablated.m.image_auroc) + ' in ' + ablated.file + '.'
      + ' Both are ' + bb + ' runs on this category.'
      + ' A high score on ' + set.category + ' does not show that the bank is doing the work.';
    // the pointer to the default only makes sense from some other category
    if (L.defaultCat && L.defaultCat !== set.category) {
      text += ' The ' + L.defaultCat + ' set is the one worth reading here.';
    }
    el.textContent = text;
    el.classList.remove('hidden');
  } catch (e) {
    console.warn('nazar: could not read the ablation runs for ' + set.category, e);
  }
}

// ---- sets -------------------------------------------------------------------

const MB = (bytes) => (bytes / 1e6).toFixed(1) + ' MB';

// Which category the page opens on. Leather saturates at the top of the scale, so it is a
// poor first impression of what the bank is doing. This is a preference, not a measurement.
const DEFAULT_CATEGORY = 'screw';

// what a set actually costs to pull: the bank plus its neighbour table. Asked of the server
// with HEAD, and if the server will not give a length, taken from the byte counts the
// sidecar records. Never written here.
async function setBytes(set) {
  const stem = L.root + set.bank;
  const [bank, nb] = await Promise.all([sizeOf(stem + '.f16.bin'), sizeOf(stem + '.nb9.i32.bin')]);
  if (bank !== null && nb !== null) return { bytes: bank + nb, from: 'server' };
  try {
    const side = await getJSON(L.root + 'sidecar_' + set.bank.replace(/^bank_/, '') + '.json');
    const n = side.nb9.shape[0] * side.nb9.shape[1] * 4;
    return { bytes: side.bank.f16_bytes + n, from: 'sidecar' };
  } catch (e) {
    return { bytes: null, from: null };
  }
}

function renderSwitch() {
  $('setBtns').innerHTML = L.sets.map((s, i) =>
    '<button type="button" data-set="' + i + '">'
    + '<b>' + esc(s.category) + '</b>'
    + '<small>' + esc(s.backbone) + '</small>'
    + '<small>' + (s.bytes ? MB(s.bytes) : 'size unknown') + '</small>'
    + '</button>').join('');
}

function markSet() {
  [...$('setBtns').children].forEach((el, i) => el.classList.toggle('on', i === L.sel));
}

function lockSwitch(on) {
  [...$('setBtns').children].forEach((el) => { el.disabled = on; });
  const go = $('deferGo');
  if (go) go.disabled = on;
}

// Clicking a category before the panel has started should only move the selection. The
// download is the whole reason the panel waits on a narrow screen, so a click must not
// trigger it by accident.
async function selectSet(i) {
  if (L.switching || L.busy || i === L.sel) return;
  L.sel = i;
  markSet();
  // the rest of the page reads this category too. it does not wait for the bank,
  // since none of those panels need it. useCategory is app.js.
  if (typeof useCategory === 'function') useCategory(L.sets[i].category);
  await categoryNote(L.sets[i]);
  if (L.armed) {
    await loadSet(i);
  } else {
    $('deferNote').textContent = deferReason(L.sets[i]) || '';
  }
}

// Everything the old set put on the GPU goes here. The device is dropped with it: the
// storage buffer limit is fixed when the device is made, and a bank the size of screw needs
// a bigger one than leather does, so a device sized for one is not reusable for the other.
function teardown() {
  cancelAnimationFrame(L.fadeRaf);
  for (const b of [L.bankGPU, L.qGPU, L.minGPU, L.idxGPU, L.params, L.stageMin, L.stageIdx]) {
    if (b) { try { b.destroy(); } catch (e) { /* the device may already have taken it */ } }
  }
  if (L.device) { try { L.device.destroy(); } catch (e) { /* nothing left to lose */ } }
  L.device = null; L.pipe = null; L.bind = null;
  L.bankGPU = null; L.qGPU = null; L.minGPU = null; L.idxGPU = null;
  L.params = null; L.stageMin = null; L.stageIdx = null;
  L.bankU16 = null; L.nb9 = null; L.side = null; L.meta = null;
  L.current = null; L.cache.clear();
  L.N = 0; L.D = 0; L.Q = 0;
  const cv = $('liveCanvas');
  cv.getContext('2d').clearRect(0, 0, cv.width, cv.height);
  $('liveStrip').innerHTML = '';
  $('tblParity').innerHTML = '';
  $('liveReadouts').innerHTML = '';
  $('parityTol').textContent = '';
  $('stageTag').textContent = '';
}

// ---- deferred start ---------------------------------------------------------

// A phone should not spend 78 MB to find out it has no WebGPU.
function deferReason(set) {
  const size = set.bytes ? MB(set.bytes) : 'large';
  const conn = navigator.connection;
  if (conn && conn.saveData) {
    return 'Data saver is on in this browser, so the ' + size + ' bank download has not started.'
      + ' The heatmaps below were rendered offline by the pipeline and need no GPU.'
      + ' Running it live needs WebGPU and the download.';
  }
  if (window.matchMedia(NARROW).matches) {
    return 'This panel scores a part on your GPU, and it downloads a ' + size + ' memory bank'
      + ' before it can. It does not start by itself on a screen this narrow.'
      + ' The heatmaps below were rendered offline by the pipeline and need no GPU.'
      + ' Running it live needs a desktop GPU with WebGPU.';
  }
  return null;
}

async function defer(reason) {
  $('liveLoad').classList.add('hidden');
  $('liveBody').classList.add('hidden');
  $('liveFallback').classList.add('hidden');
  $('liveDefer').classList.remove('hidden');
  $('deferNote').textContent = reason;
  $('liveVerdict').textContent = 'not started';
  $('liveVerdict').className = 'pill off';
  $('liveAdapter').textContent = '';
  // the same offline heatmaps the panel near the bottom shows, from the same run index.
  // whichever run exported them, which is not necessarily the one driving the panels below.
  try {
    const { man } = await pageRuns();
    const run = man.runs.find((r) => r.heatmaps);
    if (!run) return;
    const list = await getJSON('data/' + run.heatmaps);
    $('deferStrip').innerHTML = list.map((e) =>
      '<figure><img loading="lazy" src="data/' + esc(e.png) + '" alt="'
      + esc(e.defect_type + ' ' + e.file) + '">'
      + '<figcaption><b>' + esc(e.defect_type) + '</b> ' + esc(e.file) + '</figcaption></figure>').join('');
  } catch (e) {
    console.warn('nazar: no offline heatmaps to show while the live panel waits', e);
  }
}

// ---- boot -------------------------------------------------------------------

async function loadSet(i) {
  if (L.switching) return;
  L.switching = true;
  L.gen++;
  const gen = L.gen;
  const t0 = performance.now();
  const set = L.sets[i];
  lockSwitch(true);
  teardown();
  $('liveDefer').classList.add('hidden');
  $('liveBody').classList.add('hidden');
  $('liveFallback').classList.add('hidden');
  $('liveLoad').classList.remove('hidden');
  $('liveVerdict').textContent = 'loading ' + set.category;
  $('liveVerdict').className = 'pill run';
  $('setStat').textContent = '';

  try {
    progress('reading ' + set.index, -1, '');
    L.meta = await getJSON(L.root + 'queries/' + set.index);
    if (gen !== L.gen) return;
    L.Q = L.meta.images[0].patches;
    L.D = L.meta.images[0].dims;

    // Colour scale is fixed across every part so a good part looks cold instead of being
    // stretched to look like a defect. The export carries d_star and not the map range, so
    // the top of the scale is the largest d_star in the set: the map is these same patch
    // distances upsampled and blurred, and neither step can lift a value above the grid
    // maximum, so no map in this set can exceed it.
    let hi = 0;
    for (const im of L.meta.images) if (im.reference.d_star > hi) hi = im.reference.d_star;
    L.scale = [0, hi];

    strip();

    const tag = L.meta.bank;
    L.side = await getJSON(L.root + 'sidecar_' + tag.replace(/^bank_/, '') + '.json');
    L.N = L.side.bank.rows;
    if (L.side.bank.dims !== L.D) {
      throw new Error('the bank is ' + L.side.bank.dims + ' dims and the queries are ' + L.D);
    }
    if (L.side.provenance.backbone !== L.meta.backbone) {
      throw new Error('the sidecar says backbone ' + L.side.provenance.backbone
        + ' and the queries say ' + L.meta.backbone + '. these banks are shape identical, refusing to mix them.');
    }

    // the GPU gets checked before the bank download, so a browser that cannot run the
    // kernel says so immediately instead of after a long wait
    progress('starting webgpu', -1, '');
    const device = await initGPU(L.N, L.D, L.Q);
    L.device = device;

    const bankBuf = await getBytes(L.root + tag + '.f16.bin', 'downloading the memory bank');
    if (bankBuf.byteLength !== L.N * L.D * 2) {
      throw new Error('bank is ' + bankBuf.byteLength + ' bytes, the sidecar says ' + (L.N * L.D * 2));
    }
    L.bankU16 = new Uint16Array(bankBuf);

    progress('reading the neighbour table', -1, '');
    const nbBuf = await fetch(L.root + tag + '.nb9.i32.bin').then((r) => r.arrayBuffer());
    L.nb9 = new Int32Array(nbBuf);
    if (L.nb9.length !== L.N * L.meta.b_nearest) {
      throw new Error('neighbour table is ' + L.nb9.length + ' entries, expected ' + (L.N * L.meta.b_nearest));
    }

    progress('unpacking the bank onto the gpu', -1,
      (L.N * L.D * 4 / 1e6).toFixed(0) + ' MB of float32, expanded from float16 on the way in');
    await new Promise((r) => setTimeout(r, 0));   // let the bar paint before the long loop
    L.bankGPU = uploadBank(device, L.bankU16, L.N, L.D);

    const wgsl = await fetch(KERNEL).then((r) => r.text());
    const mod = device.createShaderModule({ code: wgsl });
    const info = await mod.getCompilationInfo();
    for (const m of info.messages) {
      if (m.type === 'error') throw new Error('knn.wgsl: ' + m.message);
    }
    L.pipe = device.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: ENTRY } });

    const SU = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    L.qGPU = device.createBuffer({ size: L.Q * L.D * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    L.minGPU = device.createBuffer({ size: L.Q * 4, usage: SU });
    L.idxGPU = device.createBuffer({ size: L.Q * 4, usage: SU });
    L.params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(L.params, 0, new Uint32Array([L.N, L.Q, L.D, 0]));
    L.stageMin = device.createBuffer({ size: L.Q * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    L.stageIdx = device.createBuffer({ size: L.Q * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    L.bind = device.createBindGroup({
      layout: L.pipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: L.bankGPU } },
        { binding: 1, resource: { buffer: L.qGPU } },
        { binding: 2, resource: { buffer: L.minGPU } },
        { binding: 3, resource: { buffer: L.idxGPU } },
        { binding: 4, resource: { buffer: L.params } },
      ],
    });

    $('liveReadouts').innerHTML = [
      ro('score, w * d_star', 'roScore', 'this gpu, live'),
      ro('d_star', 'roD', 'worst patch, L2'),
      ro('w (eq.7)', 'roW', 'reweighting'),
      ro('kNN dispatch', 'roMs', 'ms on your gpu', 'hot'),
      ro('throughput', 'roGf', 'GFLOP/s, ' + WG_SIZE + ' threads per patch'),
      ro('distances', 'roRows', 'million, per part'),
      ro('worst patch', 'roPatch', 'of ' + L.Q + ' in the grid'),
      ro('nearest bank row', 'roNstar', 'of ' + L.N.toLocaleString()),
    ].join('');
    $('stageScale').textContent = 'colour scale fixed across all parts: '
      + f3(L.scale[0]) + ' to ' + f3(L.scale[1]) + ', the largest d_star in this set';
    $('srcLive').textContent = 'source: ' + L.root + 'queries/manifest.json, ' + L.root + 'queries/'
      + set.index + ', ' + L.root + tag
      + '.f16.bin (' + L.N.toLocaleString() + ' x ' + L.D + ', ' + (bankBuf.byteLength / 1e6).toFixed(1)
      + ' MB float16), ' + L.root + tag + '.nb9.i32.bin, kernel ' + KERNEL
      + '. Bank and images derived from MVTec AD, CC BY-NC-SA 4.0.'
      + ' Backbone ' + L.side.provenance.backbone + ', weights sha256 '
      + L.side.provenance.weights_sha256.slice(0, 12) + ', coreset ' + L.side.provenance.coreset_pct + '.';

    await scopeNote();

    // warm the pipeline so the first part the visitor picks is timed like the rest
    progress('warming the kernel', -1, '');
    await runKNN(new Float32Array(L.Q * L.D));
    if (gen !== L.gen) return;

    $('liveLoad').classList.add('hidden');
    $('liveBody').classList.remove('hidden');
    L.armed = true;
    $('setStat').textContent = set.category + ' bank ready in '
      + ((performance.now() - t0) / 1000).toFixed(1) + ' s';
    await pick(0);
  } catch (err) {
    if (gen === L.gen) {
      console.error(err);
      fallback(err.message);
    }
  } finally {
    if (gen === L.gen) {
      L.switching = false;
      lockSwitch(false);
    }
  }
}

async function boot() {
  progress('looking for the exported queries', -1, '');
  const found = await findRoot();
  if (!found) {
    fallback('queries/manifest.json is not being served, so there is nothing to run live yet.'
      + '\nlooked in: ' + ROOTS.map((r) => r + 'queries/manifest.json').join(', '));
    return;
  }
  L.root = found.root;
  const listed = (found.man && found.man.sets) || [];
  if (!listed.length) {
    fallback(L.root + 'queries/manifest.json lists no sets, so there is nothing to run live yet.');
    return;
  }

  progress('measuring the downloads', -1, '');
  L.sets = await Promise.all(listed.map(async (s) => Object.assign({}, s, await setBytes(s))));

  const def = L.sets.findIndex((s) => s.category === DEFAULT_CATEGORY);
  L.sel = def >= 0 ? def : 0;
  L.defaultCat = L.sets[L.sel].category;

  renderSwitch();
  markSet();
  wire();

  // the size is printed on every button already, so this line gives the range rather than
  // listing fifteen of them again
  const known = L.sets.map((s) => s.bytes).filter((b) => typeof b === 'number');
  const missing = L.sets.length - known.length;
  const fromServer = L.sets.every((s) => s.from === 'server');
  let cost = 'Each category is a separate memory bank and switching downloads it. ';
  if (known.length) {
    cost += 'The ' + L.sets.length + ' banks run from ' + MB(Math.min(...known)) + ' to '
      + MB(Math.max(...known)) + ' and the size is on each button. '
      + (fromServer
        ? 'Those sizes are what the server reported for the files when this page loaded.'
        : 'Those sizes are the byte counts in the sidecar files, because the server did not report a length.');
  } else {
    cost += 'The server did not report a length for any of them and no sidecar answered either.';
  }
  if (missing && known.length) cost += ' ' + missing + ' of them have no size on the button for the same reason.';
  $('setCost').textContent = cost;

  await categoryNote(L.sets[L.sel]);

  const why = deferReason(L.sets[L.sel]);
  if (why) {
    await defer(why);
    return;
  }
  L.armed = true;
  await loadSet(L.sel);
}

function wire() {
  $('setBtns').addEventListener('click', (ev) => {
    const b = ev.target.closest('button');
    if (b && !b.disabled) selectSet(Number(b.dataset.set));
  });
  $('deferGo').addEventListener('click', () => {
    L.armed = true;
    loadSet(L.sel);
  });
  $('liveStrip').addEventListener('click', (ev) => {
    const b = ev.target.closest('.thumb');
    if (b) pick(Number(b.dataset.i));
  });
  wireViews();
}

boot();
})();
