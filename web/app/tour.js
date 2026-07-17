// First run tour. Coach marks: one element lit at a time, everything else dimmed,
// a card next to it saying what the thing is.
//
// No storage of any kind, so it opens once per page load and that is the whole
// memory it has. Esc closes it, the ? button bottom right opens it again.
//
// The spotlight is one div sized to the target with a very large box shadow. The
// shadow is the dim, the div is the hole. pointer-events stays off it so the page
// underneath keeps working while the tour is up.

(function () {
'use strict';

// Every step names a real element. If one is missing or hidden the step is
// dropped rather than pointing the spotlight at nothing.
const STEPS = [
  {
    sel: '.bar-stats',
    title: 'What is loaded',
    body: 'The whole configuration, in the open: which part, which backbone, how big the memory is, and how many good and defective parts are in the test set. It all changes when you switch category.',
  },
  {
    sel: '#setBtns',
    title: 'Pick a part',
    body: 'Fifteen kinds of real part. Each is a separate memory bank and picking one downloads it, so the size is on the button. Every panel below follows what you pick here.',
  },
  {
    sel: '#liveStrip',
    title: 'The test parts',
    body: 'Click any of them. The label under each says what it actually is: orange for a defect, grey for a good one. None of these were in the memory.',
  },
  {
    sel: '.stage-wrap',
    title: 'Where it looks wrong',
    body: 'The red patch is where the surface sits furthest from anything in the memory of good parts. Nothing here was ever trained on a defect, so this is the method saying "I have not seen that before" rather than "that is a scratch".',
  },
  {
    sel: '#liveReadouts',
    title: 'Measured, not claimed',
    body: 'The score, the worst patch of the 784, and how long the nearest neighbour search took on your own GPU. These are timed while you watch, on your machine.',
  },
  {
    sel: '#tblParity',
    title: 'The page checks itself',
    body: 'Every part ships with the score the offline python pipeline computed for it. This table is the GPU answer against that number, live. If the kernel were wrong, this is where it would show.',
  },
  {
    sel: '#panel-threshold',
    title: 'The decision',
    body: 'One accuracy number decides nothing. Drag the line and watch a missed defect trade against a stopped line. Choosing where it sits is the actual product.',
  },
  {
    sel: '#panel-breaks',
    title: 'Where it fails',
    body: 'The defects that score below the worst good part, counted by type. No threshold catches them. This panel is here on purpose.',
  },
  {
    sel: '#panel-published',
    title: 'Against the paper',
    body: 'The reproduction measured against PatchCore\'s own published table, all fifteen categories, gaps and all.',
  },
];

const PAD = 6;          // breathing room around the lit element
const CARD_GAP = 12;    // between the hole and the card

let steps = [];
let at = 0;
let open = false;
let spot = null;
let card = null;
let opened = false;     // auto open is a once per load thing

const $ = (s) => document.querySelector(s);

function visible(el) {
  if (!el) return false;
  const r = el.getBoundingClientRect();
  return r.width > 1 && r.height > 1;
}

function build() {
  spot = document.createElement('div');
  spot.className = 'tour-spot';

  card = document.createElement('div');
  card.className = 'tour-card';
  card.innerHTML =
    '<div class="tour-head"><b class="tour-title"></b><span class="tour-count"></span></div>'
    + '<p class="tour-body"></p>'
    + '<div class="tour-btns">'
    + '<button type="button" class="tour-skip">Skip</button>'
    + '<button type="button" class="tour-next">Next</button>'
    + '</div>';

  card.querySelector('.tour-skip').addEventListener('click', close);
  card.querySelector('.tour-next').addEventListener('click', next);
}

// page coordinates, not viewport, so the hole stays put while the page scrolls.
// fast skips the tween: chasing a sticky header at 180ms looks broken.
function place(fast) {
  const step = steps[at];
  const el = $(step.sel);
  if (!el) return next();

  spot.style.transition = fast === true ? 'none' : '';
  const r = el.getBoundingClientRect();
  const top = r.top + window.scrollY - PAD;
  const left = r.left + window.scrollX - PAD;
  const w = r.width + PAD * 2;
  const h = r.height + PAD * 2;

  spot.style.top = top + 'px';
  spot.style.left = left + 'px';
  spot.style.width = w + 'px';
  spot.style.height = h + 'px';

  card.querySelector('.tour-title').textContent = step.title;
  card.querySelector('.tour-body').textContent = step.body;
  card.querySelector('.tour-count').textContent = (at + 1) + ' / ' + steps.length;
  card.querySelector('.tour-next').textContent = at === steps.length - 1 ? 'Done' : 'Next';

  // measure the card before deciding which side it goes on
  card.style.visibility = 'hidden';
  card.style.top = '0px';
  card.style.left = '0px';
  const cw = card.offsetWidth;
  const ch = card.offsetHeight;

  // under the hole by default, above it when that would fall off the bottom
  let ct = top + h + CARD_GAP;
  if (ct + ch > window.scrollY + window.innerHeight - 8) {
    const above = top - ch - CARD_GAP;
    if (above > window.scrollY + 8) ct = above;
  }
  // left aligned to the hole, pulled back when it would leave the viewport
  let cl = left;
  const maxL = window.scrollX + document.documentElement.clientWidth - cw - 8;
  if (cl > maxL) cl = maxL;
  if (cl < window.scrollX + 8) cl = window.scrollX + 8;

  card.style.top = ct + 'px';
  card.style.left = cl + 'px';
  card.style.visibility = '';
}

function show(i) {
  at = i;
  const el = $(steps[at].sel);
  if (el) {
    const r = el.getBoundingClientRect();
    // only scroll when the target is not already comfortably on screen. the sticky
    // header eats the top ~90px, so aim below it.
    if (r.top < 90 || r.bottom > window.innerHeight - 40) {
      window.scrollTo({ top: r.top + window.scrollY - 110, behavior: 'auto' });
    }
  }
  place();
}

function next() {
  if (at >= steps.length - 1) return close();
  show(at + 1);
}

function start() {
  steps = STEPS.filter((s) => visible($(s.sel)));
  if (!steps.length) return;
  open = true;
  opened = true;
  document.body.appendChild(spot);
  document.body.appendChild(card);
  document.addEventListener('keydown', onKey);
  window.addEventListener('resize', onMove);
  // the header is sticky, so its page position changes as you scroll and the hole
  // would be left behind. re-placing on scroll is a no-op for everything else.
  window.addEventListener('scroll', onMove, { passive: true });
  show(0);
}

function close() {
  if (!open) return;
  open = false;
  spot.remove();
  card.remove();
  document.removeEventListener('keydown', onKey);
  window.removeEventListener('resize', onMove);
  window.removeEventListener('scroll', onMove);
}

function onMove() {
  place(true);
}

function onKey(ev) {
  if (ev.key === 'Escape') close();
  else if (ev.key === 'ArrowRight' || ev.key === 'Enter') next();
}

function helpButton() {
  const b = document.createElement('button');
  b.type = 'button';
  b.className = 'tour-open';
  b.textContent = '?';
  b.title = 'What am I looking at';
  b.setAttribute('aria-label', 'Open the tour');
  b.addEventListener('click', () => (open ? close() : start()));
  document.body.appendChild(b);
}

// the live panel fills in after its manifest and bank arrive, and the tour points
// at things inside it, so wait for the switcher to exist before opening. give up
// after a few seconds and open anyway: the panels below the live one are static.
function whenReady(cb) {
  const t0 = Date.now();
  (function poll() {
    const ready = visible($('#setBtns')) && visible($('#liveStrip'));
    if (ready || Date.now() - t0 > 6000) return cb();
    setTimeout(poll, 200);
  })();
}

function boot() {
  build();
  helpButton();
  whenReady(() => { if (!opened) start(); });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
})();
