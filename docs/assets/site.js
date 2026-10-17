// trust-align project site: progressive enhancement over the static page. Every chart reads docs/data/*.json
// (written by scripts/06_site_data.py); nothing numeric is typed here. Needs window.d3 (D3 v7, loaded with an
// SRI hash before this module); without it, or without fetch, the page text stands on its own.

const root = document.documentElement;
root.classList.add('has-js');
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
const DUR = reduceMotion ? 0 : 700;

const $ = (s) => document.querySelector(s);
const css = (name) => getComputedStyle(root).getPropertyValue(name).trim();
const fmt3 = (x) => x.toFixed(3);
const epsLabel = (e) => `${Math.round(e * 100)} %`;
// JSON keys are Python's str(): ε = 0 is written '0.0'.
const epsKey = (e) => (Number.isInteger(e) ? e.toFixed(1) : String(e));
const key = (...parts) => parts.join('|');
const settingKey = (K, t, B, e) => key(K, t, B, epsKey(e));
const jitter = (i, n = 17) => ((i * 7919) % n) / (n - 1) - 0.5;

async function loadJSON(name) {
  const r = await fetch(`data/${name}.json`);
  if (!r.ok) throw new Error(`${name}.json: ${r.status}`);
  return r.json();
}

// ── theme, progress, reveal ─────────────────────────────────────────────────

function setupChrome() {
  const btn = $('#theme-btn');
  btn.addEventListener('click', () => {
    const dark = root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
    root.dataset.theme = dark ? 'light' : 'dark';
    try { localStorage.setItem('ta-theme', root.dataset.theme); } catch (e) { /* private mode */ }
    redrawAll();
  });
  const bar = $('#progress');
  const onScroll = () => {
    const h = document.documentElement.scrollHeight - innerHeight;
    bar.style.width = `${h > 0 ? (100 * scrollY) / h : 0}%`;
  };
  addEventListener('scroll', onScroll, { passive: true });
  onScroll();
  const revealObs = new IntersectionObserver((es) => es.forEach((e) => {
    if (e.isIntersecting) { e.target.classList.add('in'); revealObs.unobserve(e.target); }
  }), { threshold: 0.12 });
  document.querySelectorAll('.reveal').forEach((el) => revealObs.observe(el));
  const links = [...document.querySelectorAll('.nav-links a')];
  const secObs = new IntersectionObserver((es) => es.forEach((e) => {
    if (!e.isIntersecting) return;
    links.forEach((a) => a.classList.toggle('active', a.getAttribute('href') === `#${e.target.id}`));
  }), { rootMargin: '-40% 0px -55% 0px' });
  document.querySelectorAll('section.chapter').forEach((s) => secObs.observe(s));
}

function setupCounters(numbers) {
  const d3 = window.d3;
  const els = document.querySelectorAll('[data-count]');
  const obs = new IntersectionObserver((es) => es.forEach((e) => {
    if (!e.isIntersecting) return;
    obs.unobserve(e.target);
    const text = numbers[e.target.dataset.count];
    const target = Number(text.replace(/,/g, ''));
    if (!Number.isFinite(target)) { e.target.textContent = text; return; }
    d3.select(e.target).transition().duration(reduceMotion ? 0 : 1500).ease(d3.easeCubicOut)
      .tween('text', function () {
        const i = d3.interpolateNumber(0, target);
        return (t) => { this.textContent = t >= 1 ? text : Math.round(i(t)).toLocaleString('en'); };
      });
  }), { threshold: 0.4 });
  els.forEach((el) => obs.observe(el));
}

// ── shared chart helpers ────────────────────────────────────────────────────

function tooltip(container) {
  let tip = container.querySelector('.tip');
  if (!tip) {
    tip = document.createElement('div');
    tip.className = 'tip';
    container.appendChild(tip);
  }
  return {
    show(event, html) {
      tip.innerHTML = html;
      tip.classList.add('on');
      const box = container.getBoundingClientRect();
      let x = event.clientX - box.left + 12;
      let y = event.clientY - box.top + 12;
      if (x + tip.offsetWidth > box.width) x = Math.max(0, x - tip.offsetWidth - 24);
      if (y + tip.offsetHeight > box.height) y = Math.max(0, y - tip.offsetHeight - 24);
      tip.style.left = `${x}px`;
      tip.style.top = `${y}px`;
    },
    hide() { tip.classList.remove('on'); },
  };
}

function svgIn(container, width, height) {
  const d3 = window.d3;
  d3.select(container).selectAll('svg').remove();
  return d3.select(container).append('svg').attr('viewBox', `0 0 ${width} ${height}`).attr('width', width)
    .attr('height', height).attr('aria-hidden', 'true');
}

function legendIn(container) {
  // An HTML legend right after the chart (never clipped, wraps on phones).
  let el = container.nextElementSibling;
  if (!el || !el.classList.contains('legend')) {
    el = document.createElement('div');
    el.className = 'legend';
    container.after(el);
  }
  el.innerHTML = '';
  return el;
}

function legendItem(legend, color, text) {
  const s = document.createElement('span');
  const i = document.createElement('i'); i.style.background = color;
  s.append(i, document.createTextNode(text));
  legend.appendChild(s);
}

function quartiles(values) {
  const d3 = window.d3;
  const v = values.filter(Number.isFinite).sort(d3.ascending);
  return { q1: d3.quantileSorted(v, 0.25), med: d3.quantileSorted(v, 0.5), q3: d3.quantileSorted(v, 0.75),
    lo: v[0], hi: v[v.length - 1], n: v.length };
}

function batchColor(k, K) {
  return window.d3.interpolateViridis(K > 1 ? (k / (K - 1)) * 0.88 : 0);
}

function drawIn(path, dur = 1100) {
  // Stroke a path in from its start (no-op under reduced motion).
  if (reduceMotion || !path.node()) return;
  const L = path.node().getTotalLength();
  path.attr('stroke-dasharray', `${L} ${L}`).attr('stroke-dashoffset', L)
    .transition().duration(dur).ease(window.d3.easeCubicOut).attr('stroke-dashoffset', 0)
    .on('end', function () { window.d3.select(this).attr('stroke-dasharray', null); });
}

// Charts register a redraw; the first draw happens when they scroll into view, so animations are seen.
const redraws = [];
let drawTimer;
function redrawAll() { redraws.forEach((f) => f()); }
addEventListener('resize', () => { clearTimeout(drawTimer); drawTimer = setTimeout(redrawAll, 120); });
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', redrawAll);
const lazyObs = new IntersectionObserver((es) => es.forEach((e) => {
  if (!e.isIntersecting) return;
  lazyObs.unobserve(e.target);
  try { e.target.__draw(); } catch (err) { console.error(`chart ${e.target.id || e.target.className} failed:`, err); }
  redraws.push(e.target.__draw);
}), { rootMargin: '300px 0px 400px 0px' });
function register(container, fn) {
  container.__draw = fn;
  lazyObs.observe(container);
}

// ── hero: own clocks ↔ true frame ───────────────────────────────────────────

function setupHero(world) {
  const d3 = window.d3;
  const box = $('#hero-chart');
  const tip = tooltip(box);
  const shown = world.shown;
  const T = world.time_min;
  const K = world.K;
  // Observed time of latent τ in batch k, and the TIC value there, on the latent grid.
  const curves = shown.map((k) => {
    const tOf = T.map((tau, i) => tau + world.drift[k][i]);
    const v = tOf.map((t) => interp(T, world.tic[k], t));
    return { k, tOf, v };
  });
  const latentOf = (k, t) => interp(curves.find((c) => c.k === k).tOf, T, t, true);
  const anchors = world.anchors.a.map((a, i) => {
    const b = world.anchors.b[i], ta = world.anchors.ta_min[i], tb = world.anchors.tb_min[i];
    return { a, b, ta, tb, bad: world.anchors.bad[i], la: latentOf(a, ta), lb: latentOf(b, tb) };
  });
  let alpha = 0, playing = !reduceMotion, timer = null, tween = null;
  const state = $('#hero-state'), slider = $('#hero-alpha');
  const btns = { play: $('#hero-play'), raw: $('#hero-raw'), al: $('#hero-al') };
  let parts = null;

  function layout() {
    const W = Math.max(320, box.clientWidth), H = Math.min(360, Math.max(240, W * 0.52));
    const m = { t: 10, r: 14, b: 34, l: 14 };
    const rowH = (H - m.t - m.b) / shown.length;
    const x = d3.scaleLinear().domain([0, 45]).range([m.l, W - m.r]);
    const rowY = (ri) => m.t + ri * rowH;
    const yIn = d3.scaleLinear().domain([0, 1]).range([rowH * 0.98, rowH * 0.12]);
    return { W, H, m, rowH, x, rowY, yIn };
  }

  function draw() {
    const L = layout();
    const svg = svgIn(box, L.W, L.H).style('overflow', 'hidden');
    // Clip the drifted traces and anchors to the plot: on its own clock a batch can run past 0 or 45 min.
    const clipId = 'hero-clip';
    svg.append('clipPath').attr('id', clipId).append('rect')
      .attr('x', L.m.l).attr('y', 0).attr('width', L.W - L.m.l - L.m.r).attr('height', L.H - L.m.b);
    const g = svg.append('g').attr('clip-path', `url(#${clipId})`);
    svg.append('g').attr('class', 'axis').attr('transform', `translate(0,${L.H - L.m.b})`).call(d3.axisBottom(L.x).ticks(9));
    svg.append('text').attr('class', 'label').attr('x', (L.m.l + L.W - L.m.r) / 2).attr('y', L.H - 4).attr('text-anchor', 'middle')
      .text(L.W < 560 ? 'retention time (min)' : 'retention time (min): each batch on its own clock → the shared latent clock');
    const rows = g.selectAll('g.row').data(curves).join('g').attr('class', 'row')
      .attr('transform', (d, i) => `translate(0,${L.rowY(i)})`);
    rows.append('path').attr('fill', (d) => batchColor(d.k, K)).attr('fill-opacity', 0.12).attr('stroke', (d) => batchColor(d.k, K))
      .attr('stroke-width', 1.2);
    rows.append('text').attr('class', 'label').attr('x', L.W - L.m.r - 4).attr('y', 12).attr('text-anchor', 'end')
      .attr('fill', (d) => batchColor(d.k, K)).style('font-weight', 600).text((d) => `batch ${d.k} (${world.sizes[d.k]} samples)`);
    const aG = g.append('g');
    const aLines = aG.selectAll('line').data(anchors).join('line')
      .attr('stroke', (d) => (d.bad ? css('--bad') : css('--ink-2'))).attr('stroke-opacity', (d) => (d.bad ? 0.9 : 0.35))
      .attr('stroke-width', (d) => (d.bad ? 1.3 : 0.9)).attr('stroke-dasharray', (d) => (d.bad ? '4 3' : null))
      .style('pointer-events', 'stroke');
    aG.selectAll('line').on('pointerenter', (ev, d) => tip.show(ev, `${d.bad ? '<b>corrupted</b>' : 'clean'} anchor, batches ${d.a} → ${d.b}<br>`
      + `${d.ta.toFixed(2)} min → ${d.tb.toFixed(2)} min${d.bad ? `<br>latent times differ by ${Math.abs(d.la - d.lb).toFixed(2)} min` : ''}`))
      .on('pointerleave', tip.hide);
    const chain = g.append('g');
    const tri = chain.selectAll('path').data(shown).join('path').attr('d', d3.symbol(d3.symbolTriangle, 36)())
      .attr('fill', css('--ink')).attr('transform', 'rotate(180)');
    const chainLine = chain.append('path').attr('fill', 'none').attr('stroke', css('--ink')).attr('stroke-dasharray', '2 3').attr('stroke-width', 1);
    parts = { L, rows, aLines, tri, chainLine };
    render(alpha);
  }

  function render(a) {
    if (!parts) return;
    const { L, rows, aLines, tri, chainLine } = parts;
    const area = d3.area().x((d) => L.x(d.x)).y0(L.yIn(0)).y1((d) => L.yIn(d.v)).curve(d3.curveMonotoneX);
    rows.select('path').attr('d', (c) => area(T.map((tau, i) => ({ x: (1 - a) * c.tOf[i] + a * tau, v: c.v[i] }))));
    const rowOf = (k) => shown.indexOf(k);
    const yAt = (k, t, lat) => {
      const c = curves[rowOf(k)];
      const xObs = (1 - a) * t + a * lat;
      const v = interp(T, world.tic[k], t);
      return { x: L.x(xObs), y: L.rowY(rowOf(k)) + L.yIn(v) - 3 };
    };
    aLines.each(function (d) {
      const p = yAt(d.a, d.ta, d.la), q = yAt(d.b, d.tb, d.lb);
      d3.select(this).attr('x1', p.x).attr('y1', p.y).attr('x2', q.x).attr('y2', q.y);
    });
    const pts = shown.map((k) => {
      const t = world.chain.observed_min[k];
      const p = yAt(k, t, world.chain.latent_min);
      return { k, x: p.x, y: p.y - 6 };
    });
    tri.attr('transform', (k, i) => `translate(${pts[i].x},${pts[i].y}) rotate(180)`);
    chainLine.attr('d', d3.line().x((p) => p.x).y((p) => p.y)(pts));
    slider.value = String(Math.round(a * 100));
    const aligned = a > 0.5;
    state.classList.toggle('aligned', aligned);
    state.lastElementChild.textContent = aligned
      ? 'True frame: clean anchors stand upright, corrupted ones stay slanted'
      : 'Own clocks: the same compound, a different minute in every batch';
  }

  function go(target, dur) {
    if (tween) tween.stop();
    if (reduceMotion || dur === 0) { alpha = target; render(alpha); return; }
    const from = alpha;
    tween = d3.timer((t) => {
      const u = Math.min(1, t / dur);
      alpha = from + (target - from) * d3.easeCubicInOut(u);
      render(alpha);
      if (u >= 1) tween.stop();
    });
  }
  function setPlay(on) {
    playing = on;
    btns.play.setAttribute('aria-pressed', String(on));
    btns.raw.setAttribute('aria-pressed', 'false');
    btns.al.setAttribute('aria-pressed', 'false');
    if (timer) { clearInterval(timer); timer = null; }
    if (on) { timer = setInterval(() => go(alpha < 0.5 ? 1 : 0, 1600), 3200); }
  }
  btns.play.addEventListener('click', () => setPlay(!playing));
  btns.raw.addEventListener('click', () => { setPlay(false); btns.raw.setAttribute('aria-pressed', 'true'); go(0, 900); });
  btns.al.addEventListener('click', () => { setPlay(false); btns.al.setAttribute('aria-pressed', 'true'); go(1, 900); });
  slider.addEventListener('input', () => { setPlay(false); alpha = slider.value / 100; render(alpha); });
  draw();
  redraws.push(draw);
  if (playing) { setPlay(true); setTimeout(() => go(1, 1600), 900); }
}

function interp(xs, ys, x, monotoneX = false) {
  // Linear interpolation on a sorted grid (xs increasing). With monotoneX, xs may be any increasing array.
  if (x <= xs[0]) return ys[0];
  const n = xs.length;
  if (x >= xs[n - 1]) return ys[n - 1];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (xs[mid] <= x) lo = mid; else hi = mid; }
  const f = (x - xs[lo]) / (xs[hi] - xs[lo] || 1);
  return ys[lo] + f * (ys[hi] - ys[lo]);
}

// ── 01 one real run (fig 1 a) ───────────────────────────────────────────────

function setupSample(sample) {
  const d3 = window.d3;
  const box = $('#sample-chart');
  const T = sample.time_min, rows = sample.log_intensity.rows;
  const nT = T.length, nM = rows[0].length, mz0 = sample.mz_range[0];
  const { log10_min: lo, log10_max: hi, levels } = sample.log_intensity;
  // Spectrum of any bin, from the display matrix (undo the log and the 1e-4 floor; normalise to 100).
  const floor = 10 ** hi * 1e-4;
  const spectrum = (i) => {
    const v = rows[i].map((q) => Math.max(0, 10 ** (lo + (q / levels) * (hi - lo)) - floor));
    const m = Math.max(...v) || 1;
    return v.map((x) => (100 * x) / m);
  };
  let idx = sample.apex_index;
  let parts = null;

  function draw() {
    box.innerHTML = '';
    const wrap = document.createElement('div'); wrap.className = 'sample';
    const ticEl = document.createElement('div'); ticEl.className = 'tic';
    const heat = document.createElement('div'); heat.className = 'heat'; heat.tabIndex = 0;
    heat.setAttribute('role', 'slider'); heat.setAttribute('aria-label', 'Retention time');
    heat.setAttribute('aria-valuemin', String(T[0])); heat.setAttribute('aria-valuemax', String(T[nT - 1]));
    const canvas = document.createElement('canvas');
    const play = document.createElement('div'); play.className = 'playhead';
    heat.append(canvas, play);
    const ax = document.createElement('div'); ax.className = 'ax';
    ax.innerHTML = `<span>${T[0].toFixed(0)} min</span><span>retention time → · m/z ${mz0}–${mz0 + nM - 1} upwards</span><span>${T[nT - 1].toFixed(0)} min</span>`;
    const heatCol = document.createElement('div'); heatCol.style.gridColumn = '1'; heatCol.append(heat, ax);
    const specEl = document.createElement('div'); specEl.className = 'spec';
    wrap.append(ticEl, heatCol, specEl);
    box.appendChild(wrap);

    // Heatmap: one pixel per (bin, m/z), m/z upwards, ink on the page colour.
    canvas.width = nT; canvas.height = nM;
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(nT, nM);
    // Page colour → accent → ink, with a gamma that lifts the faint ions without flattening the peaks.
    const toAccent = d3.interpolateLab(css('--surface'), css('--accent')), toInk = d3.interpolateLab(css('--accent'), css('--ink'));
    const ramp = d3.range(levels + 1).map((q) => {
      const u = Math.pow(q / levels, 0.75);
      return d3.rgb(u < 0.6 ? toAccent(u / 0.6) : toInk((u - 0.6) / 0.4));
    });
    for (let i = 0; i < nT; i++) {
      for (let m = 0; m < nM; m++) {
        const c = ramp[rows[i][m]], p = 4 * ((nM - 1 - m) * nT + i);
        img.data[p] = c.r; img.data[p + 1] = c.g; img.data[p + 2] = c.b; img.data[p + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);

    // TIC above, aligned with the heatmap's width.
    const W = Math.max(200, heat.clientWidth), H = 70;
    const x = d3.scaleLinear().domain([0, nT - 1]).range([0, W]);
    const y = d3.scaleLinear().domain([0, 1]).range([H - 2, 6]);
    const tsvg = svgIn(ticEl, W, H);
    const tline = tsvg.append('path').datum(sample.tic).attr('fill', 'none').attr('stroke', css('--ink')).attr('stroke-width', 1.2)
      .attr('d', d3.line().x((d, i) => x(i)).y((d) => y(d)).curve(d3.curveMonotoneX));
    tsvg.append('path').datum(sample.tic).attr('fill', css('--accent')).attr('fill-opacity', 0.12)
      .attr('d', d3.area().x((d, i) => x(i)).y0(H - 2).y1((d) => y(d)).curve(d3.curveMonotoneX));
    drawIn(tline, 1400);
    const tmark = tsvg.append('circle').attr('r', 3.5).attr('fill', css('--bad'));
    tsvg.append('text').attr('class', 'label').attr('x', 2).attr('y', 11).text('TIC');

    // Spectrum panel.
    const SW = Math.max(200, specEl.clientWidth || 230), SH = 300;
    const ssvg = svgIn(specEl, SW, SH);
    const sm = { t: 22, r: 8, b: 30, l: 8 };
    const sx = d3.scaleLinear().domain([mz0, mz0 + nM - 1]).range([sm.l, SW - sm.r]);
    const sy = d3.scaleLinear().domain([0, 118]).range([SH - sm.b, sm.t]);
    ssvg.append('g').attr('class', 'axis').attr('transform', `translate(0,${SH - sm.b})`).call(d3.axisBottom(sx).ticks(5));
    ssvg.append('text').attr('class', 'label').attr('x', (sm.l + SW - sm.r) / 2).attr('y', SH - 2).attr('text-anchor', 'middle').text('m/z');
    const stitle = ssvg.append('text').attr('class', 'title').attr('x', sm.l).attr('y', 13);
    const bars = ssvg.append('g').selectAll('line').data(d3.range(nM)).join('line')
      .attr('x1', (m) => sx(mz0 + m)).attr('x2', (m) => sx(mz0 + m)).attr('y1', sy(0)).attr('y2', sy(0))
      .attr('stroke', css('--ink')).attr('stroke-width', 1);
    const labels = ssvg.append('g');
    parts = { W, x, y, play, tmark, stitle, bars, labels, sx, sy, heat };

    const setFromEvent = (ev) => {
      const r = heat.getBoundingClientRect();
      const cx = (ev.touches ? ev.touches[0].clientX : ev.clientX) - r.left;
      idx = Math.max(0, Math.min(nT - 1, Math.round((cx / r.width) * (nT - 1))));
      render(false);
    };
    heat.addEventListener('pointermove', setFromEvent);
    heat.addEventListener('pointerdown', setFromEvent);
    ticEl.addEventListener('pointermove', setFromEvent);
    heat.addEventListener('keydown', (ev) => {
      if (ev.key === 'ArrowRight' || ev.key === 'ArrowLeft') {
        ev.preventDefault();
        idx = Math.max(0, Math.min(nT - 1, idx + (ev.key === 'ArrowRight' ? 1 : -1)));
        render(false);
      }
    });
    render(true);
  }

  function render(animate) {
    if (!parts) return;
    const { W, x, y, play, tmark, stitle, bars, labels, sx, sy, heat } = parts;
    play.style.left = `${(idx / (nT - 1)) * 100}%`;
    tmark.attr('cx', x(idx)).attr('cy', y(sample.tic[idx]));
    heat.setAttribute('aria-valuenow', String(T[idx]));
    heat.setAttribute('aria-valuetext', `${T[idx].toFixed(2)} minutes`);
    const sp = spectrum(idx);
    stitle.text(`spectrum at ${T[idx].toFixed(2)} min`);
    bars.transition().duration(animate && !reduceMotion ? 700 : 0).attr('y2', (m) => sy(sp[m]));
    const top = d3.range(sp.length).sort((a, b) => sp[b] - sp[a]).slice(0, 3);
    labels.selectAll('text').data(top).join('text').attr('class', 'axis').attr('text-anchor', 'middle').style('font-size', '10px')
      .attr('x', (m) => sx(mz0 + m)).attr('y', (m) => sy(sp[m]) - 4).text((m) => mz0 + m);
    void W;
  }

  $('#sample-apex').addEventListener('click', () => { idx = sample.apex_index; render(true); });
  register(box, draw);
}

// ── 01 the graph zoo and the true drift ─────────────────────────────────────

function drawGraph(container, { K, edges, hub, hubColor, hubDashed, sizes, size, labels = true, animate = true }) {
  const d3 = window.d3;
  const W = size || Math.max(120, container.clientWidth), H = W;
  const svg = svgIn(container, W, H).attr('width', null).attr('height', null);
  const R = W / 2 - (labels && K <= 10 ? 20 : 10);
  const pos = (k) => {
    const ang = Math.PI / 2 - (2 * Math.PI * k) / K;
    return [W / 2 + R * Math.cos(ang), H / 2 - R * Math.sin(ang)];
  };
  const lines = svg.append('g').selectAll('line').data(edges).join('line')
    .attr('x1', (d) => pos(d[0])[0]).attr('y1', (d) => pos(d[0])[1])
    .attr('x2', (d) => pos(d[0])[0]).attr('y2', (d) => pos(d[0])[1])
    .attr('stroke', css('--muted')).attr('stroke-width', K > 10 ? 0.7 : 1.1).attr('stroke-opacity', 0.8);
  lines.transition().duration(animate ? DUR : 0).delay((d, i) => (animate && !reduceMotion ? i * (K > 10 ? 4 : 25) : 0))
    .attr('x2', (d) => pos(d[1])[0]).attr('y2', (d) => pos(d[1])[1]);
  if (hub !== null && hub !== undefined) {
    svg.append('circle').attr('cx', pos(hub)[0]).attr('cy', pos(hub)[1]).attr('r', (K > 10 ? 4.5 : 7) * 1.8)
      .attr('fill', 'none').attr('stroke', hubColor).attr('stroke-width', 2.2).attr('stroke-dasharray', hubDashed ? '3 2' : null);
  }
  const r = K > 10 ? 4.5 : 7;
  svg.append('g').selectAll('circle').data(d3.range(K)).join('circle')
    .attr('cx', (k) => pos(k)[0]).attr('cy', (k) => pos(k)[1]).attr('r', r)
    .attr('fill', (k) => batchColor(k, K)).attr('stroke', css('--ink')).attr('stroke-width', 0.6)
    .append('title').text((k) => `batch ${k}: ${sizes[k]} samples`);
  if (labels && K <= 10) {
    svg.append('g').selectAll('text').data(d3.range(K)).join('text')
      .attr('x', (k) => W / 2 + (R + 13) * Math.cos(Math.PI / 2 - (2 * Math.PI * k) / K))
      .attr('y', (k) => H / 2 - (R + 13) * Math.sin(Math.PI / 2 - (2 * Math.PI * k) / K) + 3.5)
      .attr('text-anchor', 'middle').attr('class', 'axis').style('font-size', '10px').text((k) => k);
  }
  return svg;
}

function setupZoo(meta, graphs) {
  const seg = $('#zoo-K'), zoo = $('#zoo');
  let K = 10;
  for (const k of meta.Ks) {
    const b = document.createElement('button');
    b.type = 'button'; b.textContent = `K = ${k}`; b.setAttribute('aria-pressed', String(k === K));
    b.addEventListener('click', () => { K = k; [...seg.children].forEach((c) => c.setAttribute('aria-pressed', String(c === b))); draw(); });
    seg.appendChild(b);
  }
  const order = ['complete', 'density0.4', 'ring', 'chain', 'star', 'weak_star'];
  function draw() {
    const g = graphs[String(K)];
    zoo.innerHTML = '';
    for (const t of order) {
      const panel = document.createElement('div'); panel.className = 'panel';
      const chart = document.createElement('div'); chart.className = 'chart';
      const name = document.createElement('div'); name.className = 'name'; name.textContent = meta.topology_labels[t];
      const edgesEl = document.createElement('div'); edgesEl.className = 'edges';
      panel.append(chart, name, edgesEl);
      zoo.appendChild(panel);
      let edges = t === 'density0.4' ? (g['density0.4'][key(30, '0.15')] || Object.values(g['density0.4'])[0]) : g.topologies[t];
      let ran = true;
      if (!edges) {               // K = 20 was not run on this topology: draw the graph anyway, dimmed
        ran = false;
        const hubOf = t === 'star' ? g.hub : g.weak_hub;
        edges = t === 'ring' ? [...g.topologies.chain, [0, K - 1]] : window.d3.range(K).filter((k) => k !== hubOf).map((k) => [Math.min(k, hubOf), Math.max(k, hubOf)]);
        chart.style.opacity = 0.4;
      }
      const hub = t === 'star' ? g.hub : t === 'weak_star' ? g.weak_hub : null;
      chart.setAttribute('role', 'img');
      chart.setAttribute('aria-label', `${meta.topology_labels[t]} graph of ${K} batches with ${edges.length} edges`);
      drawGraph(chart, { K, edges, hub, hubColor: t === 'star' ? meta.colors.Star : meta.colors['Star (weak hub)'], hubDashed: t !== 'star',
        sizes: g.sizes, size: 150, labels: false });
      edgesEl.textContent = (ran ? `${edges.length} edges` : 'not run at this K') + (hub !== null ? ` · hub ${hub} (${g.sizes[hub]} samples)` : '');
    }
  }
  register(zoo, draw);
}

function setupDrift(world) {
  const d3 = window.d3;
  const box = $('#drift-chart');
  const tip = tooltip(box);
  function draw() {
    const W = Math.max(280, box.clientWidth), H = Math.min(300, Math.max(220, W * 0.6));
    const m = { t: 10, r: 12, b: 36, l: 44 };
    const svg = svgIn(box, W, H);
    const x = d3.scaleLinear().domain([0, 45]).range([m.l, W - m.r]);
    const ext = d3.extent(world.drift.flat());
    const y = d3.scaleLinear().domain([Math.min(-3.5, ext[0] - 0.2), Math.max(3.5, ext[1] + 0.2)]).range([H - m.b, m.t]);
    svg.append('g').attr('class', 'grid').selectAll('line').data(y.ticks(7)).join('line')
      .attr('x1', m.l).attr('x2', W - m.r).attr('y1', (d) => y(d)).attr('y2', (d) => y(d));
    svg.append('g').attr('class', 'axis').attr('transform', `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(9));
    svg.append('g').attr('class', 'axis').attr('transform', `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(7));
    svg.append('text').attr('class', 'label').attr('x', (m.l + W - m.r) / 2).attr('y', H - 4).attr('text-anchor', 'middle').text('latent retention time τ (min)');
    svg.append('text').attr('class', 'label').attr('transform', `translate(12,${(m.t + H - m.b) / 2}) rotate(-90)`).attr('text-anchor', 'middle').text('true drift dₖ(τ), min');
    svg.append('line').attr('x1', m.l).attr('x2', W - m.r).attr('y1', y(0)).attr('y2', y(0)).attr('stroke', css('--axis'));
    const line = d3.line().x((d, i) => x(world.time_min[i])).y((d) => y(d));
    const paths = svg.append('g').selectAll('path').data(world.drift).join('path').attr('d', line).attr('fill', 'none')
      .attr('stroke', (d, k) => batchColor(k, world.K)).attr('stroke-width', 1.6).style('cursor', 'crosshair')
      .on('pointerenter', function (ev, d) {
        const k = world.drift.indexOf(d);
        paths.attr('stroke-opacity', (q) => (q === d ? 1 : 0.25)).attr('stroke-width', (q) => (q === d ? 2.6 : 1.4));
        tip.show(ev, `batch ${k} (${world.sizes[k]} samples)<br>drift ${d3.min(d).toFixed(2)} to ${d3.max(d).toFixed(2)} min`);
      })
      .on('pointerleave', () => { paths.attr('stroke-opacity', 1).attr('stroke-width', 1.6); tip.hide(); });
    paths.each(function (d, i) { drawIn(d3.select(this), 900 + i * 60); });
  }
  register(box, draw);
}

// ── 02 cycle closure and fig 4 (a) ──────────────────────────────────────────

function setupCycle(meta, graphs, fig) {
  const d3 = window.d3;
  const box = $('#cycle-chart'), readout = $('#cycle-readout');
  const K = fig.setting.K;
  const g = graphs[String(K)];
  const edges = g.topologies.chain;
  const [a, b] = fig.edge;
  let running = null;
  function draw() {
    const W = Math.max(240, Math.min(420, box.clientWidth)), H = 150;
    const svg = svgIn(box, W, H).attr('width', null).attr('height', null);
    // A chain laid out left to right, the broken edge highlighted.
    const xs = d3.scalePoint().domain(d3.range(K)).range([30, W - 30]);
    const yc = 72;
    svg.append('g').selectAll('line').data(edges).join('line')
      .attr('x1', (d) => xs(d[0])).attr('x2', (d) => xs(d[1])).attr('y1', yc).attr('y2', yc)
      .attr('stroke', (d) => (d[0] === a && d[1] === b ? css('--bad') : css('--muted'))).attr('stroke-width', (d) => (d[0] === a && d[1] === b ? 3 : 1.4));
    svg.append('text').attr('class', 'cutoff-label').attr('x', (xs(a) + xs(b)) / 2).attr('y', yc - 30).attr('text-anchor', 'middle')
      .text(`edge ${a}–${b}: ${fig.world_errors.joint_cv} min off`);
    svg.append('g').selectAll('circle').data(d3.range(K)).join('circle').attr('cx', (k) => xs(k)).attr('cy', yc).attr('r', 9)
      .attr('fill', (k) => batchColor(k, K)).attr('stroke', css('--ink')).attr('stroke-width', 0.7);
    svg.append('g').selectAll('text').data(d3.range(K)).join('text').attr('x', (k) => xs(k)).attr('y', yc + 28).attr('text-anchor', 'middle')
      .attr('class', 'axis').text((k) => `batch ${k}`);
    const route = [a, b, Math.min(K - 1, b + 1), b, a];
    const dot = svg.append('circle').attr('r', 5.5).attr('fill', css('--joint')).attr('stroke', '#fff').attr('stroke-width', 1.5)
      .attr('cx', xs(route[0])).attr('cy', yc - 14);
    const trail = svg.append('text').attr('class', 'label').attr('x', W / 2).attr('y', H - 6).attr('text-anchor', 'middle');
    function run() {
      if (running) running.stop?.();
      readout.innerHTML = `Round trip ${route.join(' → ')}: the joint maps compose to the identity, <strong>cycle error 0</strong> by construction. `
        + `Composed pairwise maps need not close, but here they are <strong>${fig.world_errors.pairwise} min</strong> from the truth; the joint fit is <strong>${fig.world_errors.joint_cv} min</strong> off.`;
      dot.interrupt().attr('cx', xs(route[0]));
      trail.text(`at batch ${route[0]}`);
      let t = dot;
      route.slice(1).forEach((k, i) => {
        t = t.transition().duration(reduceMotion ? 0 : 650).ease(d3.easeCubicInOut).attr('cx', xs(k))
          .on('start', () => trail.text(`→ batch ${k}`));
        if (i === route.length - 2) t.on('end', () => trail.text(`back at batch ${k}: exactly where it started`));
      });
    }
    run();
    $('#cycle-replay').onclick = run;
  }
  register(box, draw);
}

function setupFig4a(fig, meta) {
  const C = meta.colors;
  const series = [
    { id: 'joint_cv', label: `Joint (CV λ = ${fig.lams.joint_cv})`, color: C.Joint, dash: null, width: 2.2 },
    { id: 'joint_fixed', label: `Joint (λ = ${fig.lams.joint_fixed})`, color: C.Joint, dash: '2 3', width: 1.6 },
    { id: 'pairwise', label: `Pairwise (λ = ${fig.lams.pairwise})`, color: C['Composed pairwise'], dash: null, width: 1.6 },
    { id: 'truth', label: 'Truth', color: null, dash: '6 4', width: 1.4 },
    { id: 'anchors', label: 'Anchors', color: null },
  ];
  const on = new Set(series.map((s) => s.id));
  const box = $('#fig4a-toggles');
  for (const s of series) {
    const lab = document.createElement('label');
    const cb = document.createElement('input');
    cb.type = 'checkbox'; cb.checked = true; cb.dataset.id = s.id;
    cb.addEventListener('change', () => { cb.checked ? on.add(s.id) : on.delete(s.id); draw(false); });
    const sw = document.createElement('span');
    sw.className = 'swatch' + (s.dash ? ' dashed' : '') + (s.id === 'anchors' ? ' dot' : '');
    const col = s.color || css('--ink');
    sw.style.color = col;
    sw.style.background = s.dash ? '' : (s.id === 'anchors' ? C.bad : col);
    lab.append(cb, sw, document.createTextNode(s.label));
    box.appendChild(lab);
  }
  const container = $('#fig4a-chart');
  const tip = tooltip(container);
  const [a, b] = fig.edge;

  function draw(animate = true) {
    const d3 = window.d3;
    const W = Math.max(300, container.clientWidth);
    const H = Math.min(380, Math.max(240, W * 0.55));
    const m = { t: 12, r: 14, b: 40, l: 50 };
    const svg = svgIn(container, W, H);
    const allY = [...fig.curves.truth, ...fig.curves.joint_cv, ...fig.curves.joint_fixed, ...fig.curves.pairwise, ...fig.anchors.dt];
    const x = d3.scaleLinear().domain([0, 45]).range([m.l, W - m.r]);
    const y = d3.scaleLinear().domain([Math.min(...allY) - 0.5, Math.max(...allY) + 0.5]).nice().range([H - m.b, m.t]);
    svg.append('g').attr('class', 'grid').selectAll('line').data(y.ticks(6)).join('line')
      .attr('x1', m.l).attr('x2', W - m.r).attr('y1', (d) => y(d)).attr('y2', (d) => y(d));
    svg.append('g').attr('class', 'axis').attr('transform', `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(9));
    svg.append('g').attr('class', 'axis').attr('transform', `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(6));
    svg.append('text').attr('class', 'label').attr('x', (m.l + W - m.r) / 2).attr('y', H - 6).attr('text-anchor', 'middle')
      .text(`retention time in batch ${a} (min)`);
    svg.append('text').attr('class', 'label').attr('transform', `translate(12,${(m.t + H - m.b) / 2}) rotate(-90)`)
      .attr('text-anchor', 'middle').text(`drift g${a}→${b}(t) − t (min)`);
    const line = d3.line().x((d, i) => x(fig.t[i])).y((d) => y(d));
    if (on.has('anchors')) {
      const anchors = fig.anchors.ta.map((t, i) => ({ t, dt: fig.anchors.dt[i], bad: fig.anchors.bad[i] }));
      svg.append('g').selectAll('circle').data(anchors.filter((d) => !d.bad)).join('circle')
        .attr('cx', (d) => x(d.t)).attr('cy', (d) => y(d.dt)).attr('r', 3.4).attr('fill', 'none').attr('stroke', css('--ink-2')).attr('stroke-width', 1)
        .on('pointerenter', (e, d) => tip.show(e, `clean anchor<br>t = ${d.t.toFixed(2)} min, drift ${d.dt.toFixed(2)} min`)).on('pointerleave', tip.hide);
      svg.append('g').selectAll('path').data(anchors.filter((d) => d.bad)).join('path')
        .attr('d', 'M-3.8,-3.8L3.8,3.8M-3.8,3.8L3.8,-3.8').attr('transform', (d) => `translate(${x(d.t)},${y(d.dt)})`)
        .attr('stroke', C.bad).attr('stroke-width', 1.8).style('pointer-events', 'stroke')
        .on('pointerenter', (e, d) => tip.show(e, `<b>corrupted</b> anchor<br>t = ${d.t.toFixed(2)} min, drift ${d.dt.toFixed(2)} min`)).on('pointerleave', tip.hide);
    }
    for (const s of series) {
      if (s.id === 'anchors' || !on.has(s.id)) continue;
      const p = svg.append('path').datum(fig.curves[s.id]).attr('d', line).attr('fill', 'none')
        .attr('stroke', s.color || css('--ink')).attr('stroke-width', s.width).attr('stroke-dasharray', s.dash)
        .attr('class', s.id === 'truth' ? 'ink' : null);
      if (animate && !s.dash) drawIn(p, 1200);
    }
  }
  register(container, draw);
}

// ── 03 the explorer ─────────────────────────────────────────────────────────

function setupExplorer(meta, grid, graphs) {
  const d3 = window.d3;
  const settings = meta.settings.map(([K, t, B, e]) => ({ K, t, B, e }));
  const sel = { K: $('#sel-K'), t: $('#sel-topo'), B: $('#sel-B'), e: $('#sel-eps') };
  const labels = { K: (v) => String(v), t: (v) => meta.topology_labels[v], B: (v) => String(v), e: epsLabel };
  const values = { K: meta.Ks, t: meta.topologies, B: meta.budgets, e: meta.epsilons };
  for (const f of ['K', 't', 'B', 'e']) {
    for (const v of values[f]) {
      const o = document.createElement('option');
      o.value = String(v); o.textContent = labels[f](v);
      sel[f].appendChild(o);
    }
  }
  const cur = { K: 5, t: 'chain', B: 30, e: 0.3 };
  const parse = { K: Number, t: String, B: Number, e: Number };
  const exists = (c) => settings.some((s) => s.K === c.K && s.t === c.t && s.B === c.B && s.e === c.e);

  function resolve() {
    // Options are enabled when some setting has them together with the fields resolved so far (K, then
    // topology, then B, then ε); an impossible current choice falls back to the first possible one.
    const order = ['K', 't', 'B', 'e'];
    for (let i = 0; i < order.length; i++) {
      const f = order[i];
      const fixed = order.slice(0, i);
      const ok = new Set(settings.filter((s) => fixed.every((g) => s[g] === cur[g])).map((s) => s[f]));
      for (const o of sel[f].options) o.disabled = !ok.has(parse[f](o.value));
      if (!ok.has(cur[f])) cur[f] = values[f].find((v) => ok.has(v));
      sel[f].value = String(cur[f]);
    }
    if (!exists(cur)) throw new Error('no such setting');
    for (const c of chips.querySelectorAll('button')) {
      const [K, t, B, e] = JSON.parse(c.dataset.setting);
      c.setAttribute('aria-pressed', String(K === cur.K && t === cur.t && B === cur.B && e === cur.e));
    }
  }
  for (const f of ['K', 't', 'B', 'e']) {
    sel[f].addEventListener('change', () => { cur[f] = parse[f](sel[f].value); resolve(); draw(); });
  }
  const chips = $('#explore-chips');
  const tableI = [[5, 'complete', 10, 0.0], [5, 'complete', 10, 0.3], [5, 'complete', 30, 0.3], [5, 'chain', 30, 0.3], [5, 'ring', 30, 0.3]];
  const lead = document.createElement('span'); lead.className = 'muted'; lead.textContent = 'Key settings:';
  chips.appendChild(lead);
  for (const s of tableI) {
    const [K, t, B, e] = s;
    const btn = document.createElement('button');
    btn.type = 'button'; btn.dataset.setting = JSON.stringify(s);
    btn.textContent = `K = ${K}, ${meta.topology_labels[t].toLowerCase()}, B = ${B}, ε = ${epsLabel(e)}`;
    btn.addEventListener('click', () => { Object.assign(cur, { K, t, B, e }); resolve(); draw(); });
    chips.appendChild(btn);
  }

  const gBox = $('#explore-graph'), gNote = $('#explore-graph-note');
  const cBox = $('#explore-chart'), readout = $('#explore-readout');
  const tip = tooltip(cBox);
  let prev = null;

  function drawGraphPanel() {
    const g = graphs[String(cur.K)];
    const edges = cur.t === 'density0.4' ? g['density0.4'][key(cur.B, epsKey(cur.e))] : g.topologies[cur.t];
    const hub = cur.t === 'star' ? g.hub : cur.t === 'weak_star' ? g.weak_hub : null;
    drawGraph(gBox, { K: cur.K, edges, hub, hubColor: cur.t === 'star' ? meta.colors.Star : meta.colors['Star (weak hub)'],
      hubDashed: cur.t !== 'star', sizes: g.sizes, size: Math.min(260, Math.max(180, gBox.clientWidth)) });
    const e = grid[settingKey(cur.K, cur.t, cur.B, cur.e)];
    const hubNote = hub === null ? '' : `; hub batch ${hub} (${g.sizes[hub]} samples)`;
    gNote.textContent = `${meta.topology_labels[cur.t]}, K = ${cur.K}: ${edges.length} edges${hubNote}. `
      + `Anchors per connected pair on the tuning worlds: min ${e.anchors_per_pair[0]}, median ${e.anchors_per_pair[1]}.`;
  }

  function drawChart() {
    const e = grid[settingKey(cur.K, cur.t, cur.B, cur.e)];
    const methods = meta.methods;
    const W = Math.max(300, cBox.clientWidth), H = Math.min(360, Math.max(250, W * 0.5));
    const m = { t: 16, r: 12, b: 40, l: 52 };
    const svg = svgIn(cBox, W, H);
    const x = d3.scalePoint().domain(methods).range([m.l + 30, W - m.r - 30]);
    const all = methods.flatMap((mm) => e.worlds[mm]).filter(Number.isFinite);
    const y = d3.scaleLog().domain([Math.min(0.01, d3.min(all) * 0.8), Math.max(d3.max(all) * 1.4, 0.5)]).range([H - m.b, m.t]);
    svg.append('g').attr('class', 'grid').selectAll('line').data(y.ticks(6)).join('line')
      .attr('x1', m.l).attr('x2', W - m.r).attr('y1', (d) => y(d)).attr('y2', (d) => y(d));
    svg.append('g').attr('class', 'axis').attr('transform', `translate(${m.l},0)`)
      .call(d3.axisLeft(y).ticks(6, (d) => (d >= 0.01 ? String(+d.toPrecision(2)) : '')));
    svg.append('g').attr('class', 'axis').attr('transform', `translate(0,${H - m.b})`).call(d3.axisBottom(x).tickFormat((d) => meta.labels[d]));
    svg.append('text').attr('class', 'label').attr('transform', `translate(12,${(m.t + H - m.b) / 2}) rotate(-90)`)
      .attr('text-anchor', 'middle').text('median map error per world (min)');
    svg.append('line').attr('class', 'cutoff').attr('x1', m.l).attr('x2', W - m.r).attr('y1', y(meta.break_min)).attr('y2', y(meta.break_min));
    svg.append('text').attr('class', 'cutoff-label').attr('x', W - m.r).attr('y', y(meta.break_min) - 4).attr('text-anchor', 'end')
      .text(`broken-world line (${meta.break_min} min)`);
    const w = 28;
    const animate = prev !== null && !reduceMotion;
    for (const mm of methods) {
      const v = e.worlds[mm].filter(Number.isFinite);
      if (!v.length) continue;
      const q = quartiles(v);
      const cx = x(mm), col = meta.colors[mm];
      const p = prev ? quartiles(prev.worlds[mm].filter(Number.isFinite)) : q;
      const box = svg.append('g');
      const whisker = box.append('line').attr('x1', cx).attr('x2', cx).attr('stroke', col).attr('stroke-width', 1)
        .attr('y1', y(p.lo)).attr('y2', y(p.hi));
      const rect = box.append('rect').attr('x', cx - w / 2).attr('width', w).attr('fill', col).attr('fill-opacity', 0.25).attr('stroke', col)
        .attr('y', y(p.q3)).attr('height', Math.max(1, y(p.q1) - y(p.q3)));
      const med = box.append('line').attr('x1', cx - w / 2).attr('x2', cx + w / 2).attr('stroke', css('--ink')).attr('stroke-width', 1.8)
        .attr('y1', y(p.med)).attr('y2', y(p.med));
      whisker.transition().duration(animate ? DUR : 0).attr('y1', y(q.lo)).attr('y2', y(q.hi));
      rect.transition().duration(animate ? DUR : 0).attr('y', y(q.q3)).attr('height', Math.max(1, y(q.q1) - y(q.q3)));
      med.transition().duration(animate ? DUR : 0).attr('y1', y(q.med)).attr('y2', y(q.med));
      const pts = e.worlds[mm].map((d, i) => ({ d, seed: meta.test_seeds[i], from: prev ? prev.worlds[mm][i] : d })).filter((pt) => Number.isFinite(pt.d));
      svg.append('g').selectAll('circle').data(pts).join('circle')
        .attr('cx', (pt, i) => cx + jitter(i) * w * 0.85).attr('r', 3.2).attr('fill', col).attr('fill-opacity', 0.9)
        .attr('stroke', (pt) => (pt.d > meta.break_min ? css('--bad') : 'none')).attr('stroke-width', 1.2)
        .attr('cy', (pt) => y(Number.isFinite(pt.from) ? pt.from : pt.d))
        .on('pointerenter', (ev, pt) => tip.show(ev, `${meta.labels[mm]}, test world ${pt.seed}<br>${fmt3(pt.d)} min${pt.d > meta.break_min ? ' <b>(broken)</b>' : ''}`))
        .on('pointerleave', tip.hide)
        .transition().duration(animate ? DUR : 0).attr('cy', (pt) => y(pt.d));
    }
    readout.innerHTML = `<strong>K = ${cur.K}, ${meta.topology_labels[cur.t].toLowerCase()}, B = ${cur.B}, ε = ${epsLabel(cur.e)}.</strong> `
      + methods.map((mm) => `${meta.labels[mm]} ${e.median[mm]}`).join(' · ') + ` min (medians over ${meta.n_test_worlds} worlds)`;
    prev = e;
  }

  function drawTables() {
    const e = grid[settingKey(cur.K, cur.t, cur.B, cur.e)];
    $('#explore-caption').textContent = `Median over ${meta.n_test_worlds} test worlds of the per-world median map error (all scored pairs), `
      + `the λ tuned on the tuning worlds, worlds broken (> ${meta.break_min} min), and the median error on pairs without direct anchors.`;
    const tb = $('#explore-table tbody');
    tb.innerHTML = '';
    for (const mm of meta.methods) {
      const tr = document.createElement('tr');
      const lam = e.lam[mm] === null ? '—' : String(e.lam[mm]);
      const ind = e.median_indirect && e.median_indirect[mm] !== undefined ? e.median_indirect[mm] : '—';
      const broken = e.broken[mm];
      tr.innerHTML = `<td>${mm}</td><td>${e.median[mm]}</td><td>${lam}</td>`
        + `<td class="${broken ? 'broken' : ''}">${broken}/${meta.n_test_worlds}</td><td>${ind}</td>`;
      tb.appendChild(tr);
    }
    const tt = $('#explore-tests tbody');
    tt.innerHTML = '';
    if (!e.tests.length) {
      tt.innerHTML = '<tr><td colspan="5" class="muted">No paired test is reported for this setting (the compared fits coincide here, or the comparison cannot differ).</td></tr>';
    }
    for (const t of e.tests) {
      const tr = document.createElement('tr');
      const wins = t.wins ? t.wins : (t.ci ? `95 % CI [${t.ci[0]}, ${t.ci[1]}]` : '—');
      const holm = t.holm ? `${t.holm}${t.significant ? ' *' : ''}` : '—';
      tr.innerHTML = `<td>${t.comparison}</td><td>${t.median_delta}</td><td>${wins}</td><td>${t.p}</td><td>${holm}</td>`;
      tt.appendChild(tr);
    }
  }

  function draw() { drawGraphPanel(); drawChart(); drawTables(); }
  resolve();
  register(cBox, draw);
}

// ── 04 breakdown heatmap ────────────────────────────────────────────────────

function setupHeat(meta, breakdown) {
  const box = $('#heat');
  const entries = Object.entries(breakdown).map(([k, v]) => {
    const [K, t, B, e] = k.split('|');
    return { K: +K, t, B: +B, e: +e, v };
  });
  const d3 = window.d3;
  function draw() {
    const color = (p) => d3.interpolateLab(css('--surface-3'), css('--bad'))(p);
    $('#heat-legend').style.background = `linear-gradient(90deg, ${css('--surface-3')}, ${css('--bad')})`;
    box.innerHTML = '';
    let order = 0;
    for (const K of meta.Ks) {
      const rows = entries.filter((d) => d.K === K);
      if (!rows.length) continue;
      const epsilons = meta.epsilons.filter((e) => rows.some((d) => d.e === e));
      const h = document.createElement('div'); h.className = 'heat-k'; h.textContent = `K = ${K}`;
      box.appendChild(h);
      const gridEl = document.createElement('div'); gridEl.className = 'heat-grid';
      box.appendChild(gridEl);
      for (const t of meta.topologies) {
        const cells = rows.filter((d) => d.t === t);
        if (!cells.length) continue;
        const cell = document.createElement('div'); cell.className = 'heat-cell';
        const title = document.createElement('div'); title.className = 'small'; title.textContent = meta.topology_labels[t];
        const chart = document.createElement('div'); chart.className = 'chart';
        chart.setAttribute('role', 'img');
        chart.setAttribute('aria-label', `Share of broken worlds for ${meta.topology_labels[t]} at K = ${K}, joint and composed pairwise, by budget and corruption`);
        cell.append(title, chart);
        gridEl.appendChild(cell);
        const tip = tooltip(chart);
        const sw = 24, ch = 24, gap = 10, lab = 14, left = 30;
        const budgets = meta.budgets.filter((B) => cells.some((d) => d.B === B));
        const W = left + 2 * budgets.length * sw + gap + 2;
        const H = lab + epsilons.length * ch + 16;
        const svg = svgIn(chart, W, H).attr('width', null).style('max-width', `${W * 1.25}px`);
        ['Joint', 'Composed pairwise'].forEach((mm, mi) => {
          const x0 = left + mi * (budgets.length * sw + gap);
          svg.append('text').attr('class', 'axis').attr('x', x0).attr('y', 10).style('font-size', '9.5px').style('font-weight', 600).text(meta.labels[mm]);
          budgets.forEach((B, bi) => {
            epsilons.forEach((e, ei) => {
              const d = cells.find((c) => c.B === B && c.e === e);
              if (!d) return;
              const share = +d.v[mm].share;
              const r = svg.append('rect').attr('x', x0 + bi * sw).attr('y', lab + ei * ch).attr('width', sw - 2).attr('height', ch - 2)
                .attr('rx', 3).attr('fill', color(share)).attr('stroke', css('--ring')).attr('fill-opacity', reduceMotion ? 1 : 0)
                .on('pointerenter', (ev) => tip.show(ev, `K = ${K}, ${meta.topology_labels[t].toLowerCase()}, B = ${B}, ε = ${epsLabel(e)}<br>`
                  + `<b>${meta.labels[mm]}</b>: ${d.v[mm].share} broken (${d.v[mm].k}/${meta.n_test_worlds})<br>Wilson 95 % [${d.v[mm].ci[0]}, ${d.v[mm].ci[1]}]`))
                .on('pointerleave', tip.hide);
              r.transition().delay(reduceMotion ? 0 : order * 6 + (bi + ei) * 40).duration(reduceMotion ? 0 : 500).attr('fill-opacity', 1);
              svg.append('text').attr('x', x0 + bi * sw + sw / 2 - 1).attr('y', lab + ei * ch + ch / 2 + 3)
                .attr('text-anchor', 'middle').style('font-size', '8.5px').style('pointer-events', 'none')
                .attr('fill', share > 0.5 ? '#fff' : css('--ink-2')).text(d.v[mm].share.replace(/^0\./, '.'));
            });
            svg.append('text').attr('class', 'axis').attr('x', x0 + bi * sw + sw / 2 - 1).attr('y', H - 3).attr('text-anchor', 'middle').style('font-size', '8.5px').text(B);
          });
        });
        epsilons.forEach((e, ei) => {
          svg.append('text').attr('class', 'axis').attr('x', left - 5).attr('y', lab + ei * ch + ch / 2 + 3).attr('text-anchor', 'end').style('font-size', '8.5px').text(epsLabel(e));
        });
        order++;
      }
    }
  }
  register(box, draw);
}

// ── 05 λ rules ──────────────────────────────────────────────────────────────

function setupLambda(meta, lam) {
  const d3 = window.d3;
  const box = $('#lambda-chart'), readout = $('#lambda-readout');
  const topos = ['weak_star', 'chain', 'ring', 'density0.4'].filter((t) => lam.by_topology[key(t, 'cv')]);
  const rules = ['cv', 'fixed', 'oracle'];
  const colors = { cv: meta.colors.Joint, fixed: meta.colors.fixed, oracle: null };
  function draw() {
    const W = Math.max(300, box.clientWidth), H = Math.min(340, Math.max(240, W * 0.4));
    const m = { t: 40, r: 12, b: 36, l: 44 };
    const svg = svgIn(box, W, H);
    const x0 = d3.scaleBand().domain(topos).range([m.l, W - m.r]).paddingInner(0.25).paddingOuter(0.1);
    const x1 = d3.scaleBand().domain(rules).range([0, x0.bandwidth()]).padding(0.08);
    const ymax = d3.max(topos.flatMap((t) => rules.map((r) => lam.by_topology[key(t, r)].k)));
    const y = d3.scaleLinear().domain([0, ymax * 1.15]).nice().range([H - m.b, m.t]);
    svg.append('g').attr('class', 'grid').selectAll('line').data(y.ticks(5)).join('line')
      .attr('x1', m.l).attr('x2', W - m.r).attr('y1', (d) => y(d)).attr('y2', (d) => y(d));
    svg.append('g').attr('class', 'axis').attr('transform', `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5));
    svg.append('g').attr('class', 'axis').attr('transform', `translate(0,${H - m.b})`).call(d3.axisBottom(x0).tickFormat((d) => meta.topology_labels[d]));
    const n = lam.by_topology[key(topos[0], 'cv')].n;
    svg.append('text').attr('class', 'label').attr('transform', `translate(12,${(m.t + H - m.b) / 2}) rotate(-90)`).attr('text-anchor', 'middle')
      .text(`broken test worlds (of ${n})`);
    const legend = svg.append('g').attr('transform', `translate(${m.l},10)`);
    rules.forEach((r, i) => {
      const g = legend.append('g').attr('transform', `translate(${(i * (W - m.l - m.r)) / 3},0)`);
      g.append('rect').attr('width', 10).attr('height', 10).attr('rx', 2).attr('fill', colors[r] || css('--ink'));
      g.append('text').attr('x', 14).attr('y', 9).attr('class', 'axis').text(`${lam.rules[r]}: ${lam.totals[r]}/${lam.n_worlds}`);
    });
    for (const t of topos) {
      for (const r of rules) {
        const d = lam.by_topology[key(t, r)];
        const bx = x0(t) + x1(r);
        const bar = svg.append('rect').attr('x', bx).attr('width', x1.bandwidth()).attr('rx', 3).attr('y', y(0)).attr('height', 0)
          .attr('fill', colors[r] || css('--ink')).attr('tabindex', 0).attr('role', 'button').style('cursor', 'pointer')
          .on('click keydown', (ev) => {
            if (ev.type === 'keydown' && ev.key !== 'Enter' && ev.key !== ' ') return;
            ev.preventDefault();
            list(t, r);
          });
        bar.append('title').text(`${meta.topology_labels[t]}, ${lam.rules[r]}: ${d.k}/${d.n} broken`);
        bar.transition().duration(DUR).delay(reduceMotion ? 0 : topos.indexOf(t) * 80).ease(d3.easeCubicOut).attr('y', y(d.k)).attr('height', y(0) - y(d.k));
        svg.append('text').attr('x', bx + x1.bandwidth() / 2).attr('y', y(d.k) - 4).attr('text-anchor', 'middle').attr('class', 'axis')
          .style('font-size', '10.5px').style('opacity', 0).text(d.k).transition().delay(DUR).duration(300).style('opacity', 1);
      }
    }
  }
  function list(t, r) {
    const rows = Object.values(lam.settings).filter((s) => s.topology === t && s.broken[r].length)
      .sort((a, b) => a.K - b.K || a.budget - b.budget || a.eps - b.eps);
    const total = lam.by_topology[key(t, r)];
    const items = rows.map((s) => `K = ${s.K}, B = ${s.budget}, ε = ${epsLabel(s.eps)}: ${s.broken[r].length} (test worlds ${s.broken[r].join(', ')})`);
    readout.innerHTML = `<strong>${meta.topology_labels[t]}, ${lam.rules[r]}: ${total.k} of ${total.n} worlds broken.</strong>` + (items.length ? `<br>${items.join('<br>')}` : '');
  }
  register(box, draw);
}

// ── 06 scaling with K ───────────────────────────────────────────────────────

function setupScaling(meta, sc) {
  const d3 = window.d3;
  const box = $('#scaling-chart');
  const tip = tooltip(box);
  const topos = Object.keys(sc.fig3b);
  function draw() {
    const W = Math.max(300, box.clientWidth), H = Math.min(380, Math.max(260, W * 0.52));
    const m = { t: 14, r: 16, b: 44, l: 56 };
    const svg = svgIn(box, W, H);
    const lsx = sc.Ks.map((K) => Math.log2(K));
    const x = d3.scaleLinear().domain([lsx[0] - 0.3, lsx[lsx.length - 1] + 0.3]).range([m.l, W - m.r]);
    const ys = topos.flatMap((t) => Object.values(sc.fig3b[t].points).flat());
    const y = d3.scaleLinear().domain([d3.min(ys) - 0.003, d3.max(ys) + 0.003]).nice().range([H - m.b, m.t]);
    svg.append('g').attr('class', 'grid').selectAll('line').data(y.ticks(6)).join('line')
      .attr('x1', m.l).attr('x2', W - m.r).attr('y1', (d) => y(d)).attr('y2', (d) => y(d));
    svg.append('g').attr('class', 'axis').attr('transform', `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(6));
    svg.append('g').attr('class', 'axis').attr('transform', `translate(0,${H - m.b})`).call(d3.axisBottom(x).tickValues(lsx).tickFormat((d, i) => sc.Ks[i]));
    svg.append('text').attr('class', 'label').attr('x', (m.l + W - m.r) / 2).attr('y', H - 8).attr('text-anchor', 'middle').text('batches K (log₂ scale)');
    svg.append('text').attr('class', 'label').attr('transform', `translate(14,${(m.t + H - m.b) / 2}) rotate(-90)`).attr('text-anchor', 'middle')
      .text('error (joint) − error (pairwise), min');
    svg.append('line').attr('x1', m.l).attr('x2', W - m.r).attr('y1', y(0)).attr('y2', y(0)).attr('stroke', css('--ink')).attr('stroke-dasharray', '4 3').attr('stroke-width', 0.8);
    svg.append('text').attr('class', 'axis').attr('x', W - m.r).attr('y', y(0) + 12).attr('text-anchor', 'end').text('pairwise parity');
    const style = { 'density0.4': { c: meta.colors.Joint, dx: 0.07, sym: d3.symbolCircle }, complete: { c: css('--muted'), dx: -0.07, sym: d3.symbolDiamond } };
    const legend = legendIn(box);
    topos.forEach((t, ti) => {
      const f = sc.fig3b[t], s = style[t] || { c: css('--ink'), dx: 0, sym: d3.symbolSquare };
      const xs = f.band.x_log2;
      svg.append('path').datum(xs.map((xv, i) => [xv, f.band.lo[i], f.band.hi[i]]))
        .attr('d', d3.area().x((d) => x(d[0])).y0((d) => y(d[1])).y1((d) => y(d[2]))).attr('fill', s.c).attr('fill-opacity', 0)
        .transition().duration(DUR).delay(DUR).attr('fill-opacity', 0.16);
      const fit = svg.append('path').datum([xs[0], xs[xs.length - 1]])
        .attr('d', d3.line().x((d) => x(d)).y((d) => y(f.fit.intercept + f.fit.slope * d))).attr('stroke', s.c).attr('stroke-width', 1.8).attr('fill', 'none');
      drawIn(fit, 1000);
      sc.Ks.forEach((K, ki) => {
        const pts = f.points[String(K)];
        svg.append('g').selectAll('circle').data(pts.map((d, i) => ({ d, seed: meta.test_seeds[i] }))).join('circle')
          .attr('cx', (p, i) => x(lsx[ki] + s.dx + jitter(i, 13) * 0.08)).attr('cy', y(0)).attr('r', 2.4).attr('fill', s.c).attr('fill-opacity', 0.45)
          .on('pointerenter', (ev, p) => tip.show(ev, `${meta.topology_labels[t]}, K = ${K}, test world ${p.seed}<br>Δ = ${p.d.toFixed(4)} min`))
          .on('pointerleave', tip.hide)
          .transition().duration(DUR).delay((p, i) => (reduceMotion ? 0 : i * 15)).attr('cy', (p) => y(p.d));
        svg.append('path').attr('d', d3.symbol(s.sym, 64)()).attr('transform', `translate(${x(lsx[ki] + s.dx)},${y(+f.medians[String(K)])})`)
          .attr('fill', s.c).attr('stroke', css('--ink')).attr('stroke-width', 0.6)
          .append('title').text(`${meta.topology_labels[t]}, K = ${K}: median Δ ${f.medians[String(K)]} min`);
      });
      legendItem(legend, s.c, `${meta.topology_labels[t]}: slope ${f.slope} [${f.ci[0]}, ${f.ci[1]}] min per doubling of K`, ti);
    });
  }
  register(box, draw);
}

// ── 07 fish oil ─────────────────────────────────────────────────────────────

function setupFish(meta, fish) {
  const d3 = window.d3;
  const box = $('#fish-chart');
  const seg = $('#fish-topo');
  const tip = tooltip(box);
  const topos = [...new Set(Object.values(fish.settings).map((s) => s.topology))];
  let topo = 'complete';
  for (const t of topos) {
    const b = document.createElement('button');
    b.type = 'button'; b.textContent = meta.topology_labels[t]; b.dataset.topo = t; b.setAttribute('aria-pressed', String(t === topo));
    b.addEventListener('click', () => { topo = t; [...seg.children].forEach((c) => c.setAttribute('aria-pressed', String(c === b))); draw(); });
    seg.appendChild(b);
  }
  const budgets = [10, 100];
  const methods = ['Joint', 'Star', 'Composed pairwise'];
  const epsilons = [...new Set(Object.values(fish.settings).map((s) => s.eps))].sort((a, b) => a - b);
  let first = true;
  function draw() {
    const W = Math.max(300, box.clientWidth), H = Math.min(340, Math.max(240, W * 0.45));
    const m = { t: 30, r: 10, b: 40, l: 54 }, gap = 28;
    const pw = (W - m.l - m.r - gap) / budgets.length;
    const svg = svgIn(box, W, H);
    const vals = budgets.flatMap((B) => epsilons.flatMap((e) => methods.flatMap((mm) => fish.settings[key(topo, B, epsKey(e))].draws[mm]))).filter(Number.isFinite);
    const y = d3.scaleLog().domain([d3.min(vals) * 0.8, d3.max(vals) * 1.25]).range([H - m.b, m.t]);
    svg.append('g').attr('class', 'axis').attr('transform', `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5, (d) => String(+d.toPrecision(2))));
    svg.append('text').attr('class', 'label').attr('transform', `translate(14,${(m.t + H - m.b) / 2}) rotate(-90)`).attr('text-anchor', 'middle').text('held-out anchor error (min)');
    const marks = { Joint: d3.symbolCircle, Star: d3.symbolSquare, 'Composed pairwise': d3.symbolTriangle };
    budgets.forEach((B, bi) => {
      const x0 = m.l + bi * (pw + gap);
      const x = d3.scalePoint().domain(epsilons).range([x0 + 30, x0 + pw - 30]);
      svg.append('g').attr('class', 'grid').selectAll('line').data(y.ticks(5)).join('line').attr('x1', x0).attr('x2', x0 + pw).attr('y1', (d) => y(d)).attr('y2', (d) => y(d));
      svg.append('g').attr('class', 'axis').attr('transform', `translate(0,${H - m.b})`).call(d3.axisBottom(x).tickFormat(epsLabel));
      svg.append('text').attr('class', 'title').attr('x', x0).attr('y', 12).text(`B = ${B}`);
      svg.append('text').attr('class', 'label').attr('x', x0 + pw / 2).attr('y', H - 8).attr('text-anchor', 'middle').text('corruption ε');
      svg.append('line').attr('class', 'cutoff').attr('x1', x0).attr('x2', x0 + pw).attr('y1', y(meta.break_min)).attr('y2', y(meta.break_min));
      methods.forEach((mm, mi) => {
        epsilons.forEach((e) => {
          const s = fish.settings[key(topo, B, epsKey(e))];
          const q = quartiles(s.draws[mm]);
          const cx = x(e) + (mi - 1) * 14, col = meta.colors[mm];
          const g = svg.append('g').style('opacity', first && !reduceMotion ? 0 : 1)
            .on('pointerenter', (ev) => tip.show(ev, `${meta.topology_labels[topo].toLowerCase()}, B = ${B}, ε = ${epsLabel(e)}<br>`
              + `<b>${mm}</b>: median ${s.median[mm]} min (report), IQR ${fmt3(q.q1)}–${fmt3(q.q3)}<br>broken draws ${s.broken[mm]}/${fish.n_draws}`))
            .on('pointerleave', tip.hide);
          g.append('line').attr('x1', cx).attr('x2', cx).attr('y1', y(q.q1)).attr('y2', y(q.q3)).attr('stroke', col).attr('stroke-width', 1.6);
          g.append('path').attr('d', d3.symbol(marks[mm], 52)()).attr('transform', `translate(${cx},${y(q.med)})`).attr('fill', col).attr('stroke', css('--ink')).attr('stroke-width', 0.5);
          g.transition().duration(DUR).delay(reduceMotion ? 0 : mi * 120).style('opacity', 1);
        });
      });
    });
    const legend = legendIn(box);
    methods.forEach((mm, i) => legendItem(legend, meta.colors[mm], mm === 'Star' ? 'Star (strong hub)' : meta.labels[mm], i));
    if (y(meta.break_min) > m.t) {
      svg.append('text').attr('class', 'cutoff-label').attr('x', W - m.r).attr('y', y(meta.break_min) - 4).attr('text-anchor', 'end').text(`broken-draw line (${meta.break_min} min)`);
    }
    first = false;
  }
  register(box, draw);
}

// ── method legend ───────────────────────────────────────────────────────────

function setupLegend(meta) {
  const el = $('#method-legend');
  for (const mm of meta.methods) {
    const s = document.createElement('span');
    const i = document.createElement('i'); i.style.background = meta.colors[mm];
    s.append(i, document.createTextNode(mm));
    el.appendChild(s);
  }
  const s = document.createElement('span');
  const i = document.createElement('i'); i.style.background = meta.colors.bad;
  s.append(i, document.createTextNode('corrupted anchors, broken-world line'));
  el.appendChild(s);
}

// ── boot ────────────────────────────────────────────────────────────────────

async function main() {
  setupChrome();
  if (!window.d3) { root.classList.remove('has-js'); return; }   // CDN blocked: the static page stands
  const [meta, grid, graphs, breakdown, lam, scaling, fig4a, fish, world, sample, numbers] = await Promise.all(
    ['meta', 'grid', 'graphs', 'breakdown', 'lambda', 'scaling', 'fig4a', 'fish', 'world', 'sample', 'numbers'].map(loadJSON));
  setupCounters(numbers);
  setupHero(world);
  setupSample(sample);
  setupZoo(meta, graphs);
  setupDrift(world);
  setupLegend(meta);
  setupCycle(meta, graphs, fig4a);
  setupFig4a(fig4a, meta);
  setupExplorer(meta, grid, graphs);
  setupHeat(meta, breakdown);
  setupLambda(meta, lam);
  setupScaling(meta, scaling);
  setupFish(meta, fish);
}

main().catch((err) => {
  console.error(err);
  root.classList.remove('has-js');
});
