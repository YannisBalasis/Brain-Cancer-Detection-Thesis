/* Interactive components for the thesis page. No dependencies. */
(function () {
  'use strict';
  const D = window.DATA;
  const NS = 'http://www.w3.org/2000/svg';
  const root = document.documentElement;
  const $ = (s, c = document) => c.querySelector(s);

  /* ───────── i18n ───────── */
  const lang = () => root.dataset.lang || 'en';
  const T = (o) => (o == null ? '' : typeof o === 'string' ? o : o[lang()] ?? o.en);
  const UI = {
    acc: { en: 'Accuracy', el: 'Ακρίβεια' },
    f1: { en: 'Macro F1', el: 'Macro F1' },
    auc: { en: 'Macro AUC', el: 'Macro AUC' },
    params: { en: 'Parameters', el: 'Παράμετροι' },
    inf: { en: 'Inference', el: 'Χρόνος συμπερασμού' },
    rank: { en: 'Avg. rank', el: 'Μέση κατάταξη' },
    model: { en: 'Model', el: 'Μοντέλο' },
    ci: { en: '95% CI', el: 'ΔΕ 95%' },
    top: { en: 'Top tier', el: 'Κορυφαίο κλιμάκιο' },
    mid: { en: 'Intermediate', el: 'Ενδιάμεσο' },
    low: { en: 'Lowest tier', el: 'Χαμηλότερο κλιμάκιο' },
    na: { en: 'n/a', el: 'δ/υ' },
    trueC: { en: 'True class', el: 'Πραγματική κλάση' },
    predC: { en: 'Predicted', el: 'Πρόβλεψη' },
    params_l: { en: 'parameters', el: 'παράμετροι' },
    classes: { en: 'classes', el: 'κλάσεις' },
    on503: { en: 'accuracy (503)', el: 'ακρίβεια (503)' },
    sig: { en: 'significant', el: 'σημαντική' },
    nsig: { en: 'not significant', el: 'μη σημαντική' },
    better: { en: 'better', el: 'καλύτερο' },
    selectStage: { en: 'Select a stage above.', el: 'Επίλεξε ένα στάδιο παραπάνω.' },
    selectBlock: { en: 'Select a block in the diagram for details.', el: 'Επίλεξε ένα μπλοκ στο διάγραμμα για λεπτομέρειες.' },
    notApplicable: { en: '3-Class CNN has no "no tumour" class', el: 'Το 3-Class CNN δεν έχει κλάση «χωρίς όγκο»' },
    original: { en: 'Original MRI', el: 'Αρχική MRI' },
    medvitSample: { en: 'MedViT V2 maps come from its own median-confidence sample, so the slice differs from the other models.', el: 'Οι χάρτες του MedViT V2 προέρχονται από το δικό του δείγμα διάμεσης βεβαιότητας, οπότε η τομή διαφέρει από τα άλλα μοντέλα.' },
    satFlag: { en: 'gradient saturation', el: 'κορεσμός κλίσεων' },
    copied: { en: 'Copied.', el: 'Αντιγράφηκε.' },
    copyFail: { en: 'Select the text and copy manually.', el: 'Επίλεξε το κείμενο και αντέγραψέ το χειροκίνητα.' },
    darkOn: { en: 'Switch to light mode', el: 'Μετάβαση σε φωτεινό θέμα' },
    darkOff: { en: 'Switch to dark mode', el: 'Μετάβαση σε σκοτεινό θέμα' },
    pause: { en: 'Pause animations', el: 'Παύση κινήσεων' },
    play: { en: 'Play animations', el: 'Αναπαραγωγή κινήσεων' },
    textSize: { en: 'Text size', el: 'Μέγεθος κειμένου' },
    langBtn: { en: 'Ελληνικά (switch to Greek)', el: 'English (αλλαγή σε αγγλικά)' },
    order: { en: 'Sorted best first', el: 'Ταξινόμηση από το καλύτερο' },
    lowerBetter: { en: 'lower is better', el: 'μικρότερο είναι καλύτερο' },
    higherBetter: { en: 'higher is better', el: 'μεγαλύτερο είναι καλύτερο' },
    proposed: { en: 'proposed', el: 'προτεινόμενο' },
  };
  const u = (k) => T(UI[k]);
  const fmt = (v, d = 2) => {
    if (v == null || Number.isNaN(v)) return u('na');
    const s = Number(v).toFixed(d);
    return lang() === 'el' ? s.replace('.', ',') : s;
  };
  const fmtP = (p) => {
    if (p == null) return '';
    if (p >= 0.995) return fmt(1, 2);
    if (p >= 0.001) return fmt(p, p < 0.01 ? 4 : 3);
    const e = Math.floor(Math.log10(p)), m = p / Math.pow(10, e);
    return `${fmt(m, 1)}×10<sup>${e}</sup>`;
  };
  const fmtPshort = (p) => (p < 0.001 ? '<.001' : p >= 0.995 ? '1' : p.toFixed(3).replace(/^0/, '')).replace('.', lang() === 'el' ? ',' : '.');

  /* ───────── helpers ───────── */
  function S(tag, attrs = {}, kids = []) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) {
      const v = attrs[k];
      if (v == null || v === false) continue;
      if (k === 'text') e.textContent = v;
      else e.setAttribute(k, v);
    }
    kids.forEach((c) => c && e.appendChild(c));
    return e;
  }
  function H(tag, attrs = {}, html) {
    const e = document.createElement(tag);
    for (const k in attrs) if (attrs[k] != null) e.setAttribute(k, attrs[k]);
    if (html != null) e.innerHTML = html;
    return e;
  }
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const cls = (id) => D.CLASSES.find((c) => c.id === id);
  const cvar = (name) => `var(${name})`;

  /* ───────── tooltip ───────── */
  const tipEl = $('#tooltip');
  function showTip(html, x, y) {
    tipEl.innerHTML = html; tipEl.hidden = false;
    const r = tipEl.getBoundingClientRect();
    let L = x + 14, Tp = y + 14;
    if (L + r.width > innerWidth - 8) L = x - r.width - 14;
    if (Tp + r.height > innerHeight - 8) Tp = y - r.height - 14;
    tipEl.style.left = clamp(L, 8, innerWidth - r.width - 8) + 'px';
    tipEl.style.top = clamp(Tp, 8, innerHeight - r.height - 8) + 'px';
  }
  const hideTip = () => { tipEl.hidden = true; };
  function bindTip(el, fn) {
    el.addEventListener('pointerenter', (e) => showTip(fn(), e.clientX, e.clientY));
    el.addEventListener('pointermove', (e) => showTip(fn(), e.clientX, e.clientY));
    el.addEventListener('pointerleave', hideTip);
    el.addEventListener('focus', () => { const r = el.getBoundingClientRect(); showTip(fn(), r.right, r.top + r.height / 2); });
    el.addEventListener('blur', hideTip);
  }
  const row = (k, v) => `<div class="tt-row"><span class="tt-k">${k}</span><span>${v}</span></div>`;
  window.addEventListener('scroll', hideTip, { passive: true });

  /* ───────── motion state (shared with background) ───────── */
  const motion = {
    paused: () => (window.BrainBG ? window.BrainBG.isPaused() : window.matchMedia('(prefers-reduced-motion: reduce)').matches),
  };

  /* ───────── renderer registry: re-render on resize / language ───────── */
  const renderers = [];
  function register(fn) { renderers.push(fn); fn(); }
  let lastW = innerWidth, rT;
  window.addEventListener('resize', () => {
    clearTimeout(rT);
    rT = setTimeout(() => { if (Math.abs(innerWidth - lastW) > 4) { lastW = innerWidth; renderers.forEach((f) => f()); } }, 120);
  });

  /* ═════════ Flow diagram engine ═════════ */
  const KIND = {
    train: '--accent', eval: '--signal',
    in: '--b-in', conv: '--b-conv', attn: '--b-attn', dense: '--b-dense', fuse: '--b-fuse', out: '--b-out', note: '--ink-3',
  };
  const flowPulses = new Set();

  function wrapSub(s, max) {
    if (s.length <= max) return [s];
    const parts = s.split(' · ');
    const lines = []; let cur = '';
    for (const p of parts) {
      const next = cur ? cur + ' · ' + p : p;
      if (next.length > max && cur) { lines.push(cur); cur = p; } else cur = next;
    }
    if (cur) lines.push(cur);
    return lines.slice(0, 2);
  }

  function renderFlow(container, graph, opts) {
    const avail = Math.max(260, container.clientWidth);
    const NW = opts.nodeW || 140, NH = 64, GX = 26, GY = 30, pad = 10;
    const nodes = graph.nodes.map((n) => Object.assign({}, n));
    const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));
    const xs = nodes.map((n) => n.x), ys = nodes.map((n) => n.y);
    const maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
    const naturalW = 2 * pad + maxX * (NW + GX) + NW;
    let mode;
    if (graph.chain) mode = 'wrap';
    else mode = naturalW <= avail * 1.12 ? 'h' : 'v';

    let W, Hh;
    if (mode === 'wrap') {
      const k = Math.max(1, Math.floor((avail - 2 * pad + GX) / (NW + GX)));
      const rows = Math.ceil(nodes.length / k);
      const rowH = NH + GY + 8;
      nodes.forEach((n, i) => { n.px = pad + (i % k) * (NW + GX); n.py = pad + Math.floor(i / k) * rowH; n.row = Math.floor(i / k); });
      W = 2 * pad + Math.min(k, nodes.length) * (NW + GX) - GX; Hh = 2 * pad + rows * rowH - GY - 8;
    } else if (mode === 'h') {
      const laneH = NH + GY + 4;
      nodes.forEach((n) => { n.px = pad + n.x * (NW + GX); n.py = pad + (n.y - minY) * laneH; });
      W = naturalW; Hh = 2 * pad + (maxY - minY) * laneH + NH;
    } else {
      // Vertical: stack side notes (training phases) below the main flow so lanes stay narrow.
      const main = nodes.filter((n) => n.kind !== 'note'), notes = nodes.filter((n) => n.kind === 'note');
      if (notes.length) {
        const mx = Math.max(...main.map((n) => n.x));
        notes.forEach((n, k) => { n.x = mx + 1 + k; n.y = 0; });
      }
      const minYv = Math.min(...nodes.map((n) => n.y)), maxYv = Math.max(...nodes.map((n) => n.y)), maxXv = Math.max(...nodes.map((n) => n.x));
      const laneW = NW + 22, rowH = NH + GY + 10;
      nodes.forEach((n) => { n.px = pad + (n.y - minYv) * laneW; n.py = pad + n.x * rowH; });
      W = 2 * pad + (maxYv - minYv) * laneW + NW; Hh = 2 * pad + maxXv * rowH + NH;
    }

    const svg = S('svg', { viewBox: `0 0 ${W} ${Hh}`, role: 'group', 'aria-label': T(opts.label) });
    svg.style.maxWidth = W + 'px'; svg.style.margin = '0 auto';
    const gE = S('g'), gP = S('g', { 'aria-hidden': 'true' }), gN = S('g');
    svg.append(gE, gP, gN);

    const pulses = [];
    graph.edges.forEach(([a, b, style]) => {
      const A = byId[a], B = byId[b];
      let d;
      if (mode === 'h') {
        const x1 = A.px + NW, y1 = A.py + NH / 2, x2 = B.px, y2 = B.py + NH / 2, mx = (x1 + x2) / 2;
        d = `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`;
      } else if (mode === 'wrap' && A.row !== B.row) {
        const x1 = A.px + NW / 2, y1 = A.py + NH, x2 = B.px + NW / 2, y2 = B.py, my = (y1 + y2) / 2;
        d = `M${x1},${y1} L${x1},${my} L${x2},${my} L${x2},${y2}`;
      } else if (mode === 'wrap') {
        d = `M${A.px + NW},${A.py + NH / 2} L${B.px},${B.py + NH / 2}`;
      } else {
        const x1 = A.px + NW / 2, y1 = A.py + NH, x2 = B.px + NW / 2, y2 = B.py, my = (y1 + y2) / 2;
        d = `M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}`;
      }
      const p = S('path', { d, class: 'edge', 'data-a': a, 'data-b': b, 'stroke-dasharray': style === 'dash' ? '5 5' : null, 'marker-end': null });
      gE.appendChild(p);
      if (style !== 'dash') {
        const c = S('circle', { r: 3, class: 'pulse', opacity: 0 });
        gP.appendChild(c);
        pulses.push({ path: p, c, off: Math.random() });
      }
    });

    nodes.forEach((n) => {
      const label = T(n)[0] || n.en[0], sub = T(n)[1] || n.en[1] || '';
      const subLines = wrapSub(sub, Math.floor((NW - 20) / 5.9));
      const g = S('g', {
        class: 'node' + (opts.selected === n.id ? ' sel' : ''), tabindex: 0, role: 'button',
        'aria-pressed': opts.selected === n.id ? 'true' : 'false', 'aria-label': `${label}: ${sub}`, 'data-id': n.id,
        transform: `translate(${n.px},${n.py})`,
      });
      const isNote = n.kind === 'note';
      g.append(
        S('rect', { class: 'box', width: NW, height: NH, rx: 10, 'stroke-dasharray': isNote ? '4 4' : null }),
        isNote ? null : S('rect', { class: 'bar', x: 0, y: 0, width: 5, height: NH, rx: 2, style: `fill:${cvar(KIND[n.kind] || '--accent')}` }),
        S('text', { x: 14, y: subLines.length > 1 ? 21 : 27, 'font-weight': 600, text: label }),
      );
      subLines.forEach((s, i) => g.appendChild(S('text', { class: 'sub', x: 14, y: (subLines.length > 1 ? 38 : 45) + i * 14, text: s })));
      const choose = () => opts.onSelect(n.id);
      g.addEventListener('click', choose);
      g.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); choose(); } });
      g.addEventListener('pointerenter', () => highlight(n.id, true));
      g.addEventListener('pointerleave', () => highlight(n.id, false));
      gN.appendChild(g);
    });
    function highlight(id, on) {
      gE.querySelectorAll('.edge').forEach((p) => {
        if (p.dataset.a === id || p.dataset.b === id) p.classList.toggle('hot', on);
      });
    }
    if (opts.selected) highlight(opts.selected, true);

    container.replaceChildren(svg);
    // Register pulses for the shared animation loop.
    const entry = { pulses, svg };
    flowPulses.forEach((e) => { if (!e.svg.isConnected) flowPulses.delete(e); });
    flowPulses.add(entry);
    if (motion.paused()) pulses.forEach((p) => p.c.setAttribute('opacity', 0));
  }

  let fT = 0;
  function animateFlows(ts) {
    const dt = fT ? ts - fT : 16; fT = ts;
    if (!motion.paused()) {
      flowPulses.forEach((e) => {
        if (!e.svg.isConnected) { flowPulses.delete(e); return; }
        e.pulses.forEach((p) => {
          p.off = (p.off + dt / 2600) % 1;
          const L = p.path.getTotalLength();
          const pt = p.path.getPointAtLength(p.off * L);
          p.c.setAttribute('cx', pt.x); p.c.setAttribute('cy', pt.y);
          p.c.setAttribute('opacity', (Math.sin(p.off * Math.PI) * 0.9).toFixed(2));
        });
      });
    }
    requestAnimationFrame(animateFlows);
  }
  requestAnimationFrame(animateFlows);

  /* ═════════ Hero MRI tiles ═════════ */
  function renderHero() {
    const g = $('#hero-mri');
    g.replaceChildren(...D.CLASSES.map((c) => {
      const t = H('div', { class: 'mri-tile' });
      t.append(H('img', { src: `assets/xai/orig_${c.id}_medvit.jpg`, alt: `${T(c)} MRI`, width: 320, height: 320 }));
      const s = H('span', { style: `--dot:${cvar(c.color)}` }); s.textContent = T(c); t.append(s);
      return t;
    }));
  }

  /* ═════════ Pipeline ═════════ */
  let pipeSel = 'train';
  function renderPipeline() {
    renderFlow($('#pipeline-flow'), D.PIPELINE, {
      label: { en: 'Research pipeline', el: 'Ροή της έρευνας' }, selected: pipeSel, nodeW: 140,
      onSelect: (id) => { pipeSel = id; renderPipeline(); focusNode('#pipeline-flow', id); },
    });
    const n = D.PIPELINE.nodes.find((x) => x.id === pipeSel);
    const d = n.d[lang()] || n.d.en;
    $('#pipeline-detail').innerHTML = `<h3>${d[0]}<span class="chip">${n.kind === 'train' ? (lang() === 'el' ? 'δεδομένα & εκπαίδευση' : 'data & training') : (lang() === 'el' ? 'αξιολόγηση' : 'evaluation')}</span></h3><p>${d[1]}</p>` +
      (n.d.specs ? `<div class="specs">${n.d.specs.map((s) => `<span>${s}</span>`).join('')}</div>` : '');
  }
  function focusNode(sel, id) {
    const g = $(`${sel} .node[data-id="${id}"]`);
    if (g) g.focus({ preventScroll: true });
  }

  /* ═════════ Clinical app flow ═════════ */
  let appSel = 'cons';
  function renderAppFlow() {
    renderFlow($('#app-flow'), D.APP, {
      label: { en: 'Clinical decision support workflow', el: 'Ροή του συστήματος υποστήριξης αποφάσεων' }, selected: appSel, nodeW: 150,
      onSelect: (id) => { appSel = id; renderAppFlow(); focusNode('#app-flow', id); },
    });
    const n = D.APP.nodes.find((x) => x.id === appSel);
    const d = n.d[lang()] || n.d.en;
    $('#app-detail').innerHTML = `<h3>${d[0]}</h3><p>${d[1]}</p>` + `<div class="specs">${n.d.specs.map((s) => `<span>${s}</span>`).join('')}</div>`;
  }

  /* ═════════ Architectures ═════════ */
  let archIdx = 8, archSel = null;
  function renderArchTabs() {
    const tl = $('#arch-tabs');
    tl.replaceChildren(...D.ARCH.map((a, i) => {
      const m = D.MODELS[a.model - 1];
      const b = H('button', { type: 'button', role: 'tab', class: 'tab', id: `tab-${a.id}`, 'aria-controls': 'arch-panel', 'aria-selected': i === archIdx ? 'true' : 'false', tabindex: i === archIdx ? '0' : '-1' },
        `${m.name}<small>#${m.n} · ${fmt(m.params, 2)} M</small>`);
      b.addEventListener('click', () => selectArch(i));
      b.addEventListener('keydown', (e) => {
        let j = null;
        if (e.key === 'ArrowRight') j = (i + 1) % D.ARCH.length;
        if (e.key === 'ArrowLeft') j = (i - 1 + D.ARCH.length) % D.ARCH.length;
        if (e.key === 'Home') j = 0;
        if (e.key === 'End') j = D.ARCH.length - 1;
        if (j != null) { e.preventDefault(); selectArch(j); $(`#tab-${D.ARCH[j].id}`).focus(); }
      });
      return b;
    }));
    $('#arch-panel').setAttribute('aria-labelledby', `tab-${D.ARCH[archIdx].id}`);
  }
  function selectArch(i) {
    archIdx = i; archSel = null;
    D.ARCH.forEach((a, j) => { const t = $(`#tab-${a.id}`); t.setAttribute('aria-selected', j === i ? 'true' : 'false'); t.tabIndex = j === i ? 0 : -1; });
    $('#arch-panel').setAttribute('aria-labelledby', `tab-${D.ARCH[i].id}`);
    renderArch();
  }
  function renderArch() {
    const a = D.ARCH[archIdx], m = D.MODELS[a.model - 1];
    const nCls = { binary: 2, cnn3: 3, ensemble: 3 }[a.id] || 4;
    $('#arch-head').innerHTML = `<h3>${m.name}</h3>
      <div class="facts"><span>${T(a.tag)}</span><span><b>${fmt(m.params, 2)} M</b> ${u('params_l')}</span><span><b>${fmt(m.acc)}%</b> ${u('on503')}</span><span><b>${nCls}</b> ${u('classes')}</span></div>
      <p>${T(a.desc)}</p>`;
    const g = a.graph;
    const chain = g.edges.every(([x, y], i) => x === 'n' + i && y === 'n' + (i + 1)) && g.nodes.length === g.edges.length + 1;
    renderFlow($('#arch-flow'), Object.assign({}, g, { chain }), {
      label: { en: `${m.name} architecture`, el: `Αρχιτεκτονική ${m.name}` }, selected: archSel,
      onSelect: (id) => { archSel = id; renderArch(); focusNode('#arch-flow', id); },
    });
    const n = archSel && g.nodes.find((x) => x.id === archSel);
    $('#arch-detail').innerHTML = n
      ? `<h3>${T(n)[0]}<span class="chip">${T(n)[1]}</span></h3><p>${T(n.d)}</p>`
      : `<p>${u('selectBlock')}</p>`;
  }

  /* ═════════ Chart: performance (dot plot) ═════════ */
  const METRICS = {
    acc: { get: (m) => m.acc, lo: (m) => m.lo, hi: (m) => m.hi, dom: [70, 100], ticks: [70, 80, 90, 100], f: (v) => fmt(v) + '%', hiBetter: true,
      cap: { en: 'Accuracy with 95% Wilson score interval. Three tiers: five models at 97.6–98.6%, two near 93%, and the two ImageNet transfer-learning systems far behind.', el: 'Ακρίβεια με διάστημα Wilson 95%. Τρία κλιμάκια: πέντε μοντέλα στο 97,6–98,6%, δύο κοντά στο 93% και τα δύο συστήματα μεταφοράς μάθησης από το ImageNet πολύ πίσω.' } },
    f1: { get: (m) => m.f1, dom: [70, 100], ticks: [70, 80, 90, 100], f: (v) => fmt(v) + '%', hiBetter: true,
      cap: { en: 'Macro-averaged F1-score over the three tumour classes. It tracks accuracy closely because the subset is balanced.', el: 'Μέσο F1 (macro) στις τρεις κλάσεις όγκου. Ακολουθεί στενά την ακρίβεια, επειδή το υποσύνολο είναι ισορροπημένο.' } },
    auc: { get: (m) => m.auc, dom: [0.94, 1], ticks: [0.94, 0.96, 0.98, 1], f: (v) => fmt(v, v > 0.999 ? 4 : 3), hiBetter: true,
      cap: { en: 'Macro-averaged area under the ROC curve. MedViT V2 Dual reaches 0.9998, the best separation of any model.', el: 'Μέση επιφάνεια κάτω από την καμπύλη ROC (macro). Το MedViT V2 Dual φτάνει 0,9998, τον καλύτερο διαχωρισμό από όλα τα μοντέλα.' } },
    params: { get: (m) => m.params, dom: [1, 60], log: true, ticks: [1, 2, 5, 10, 20, 50], f: (v) => fmt(v, 2) + ' M', hiBetter: false,
      cap: { en: 'Trainable parameters, log scale. The custom CNNs are 20–40 times smaller than the transformers.', el: 'Εκπαιδεύσιμες παράμετροι, λογαριθμική κλίμακα. Τα custom CNN είναι 20 έως 40 φορές μικρότερα από τους transformers.' } },
    inf: { get: (m) => m.inf, lo: (m) => m.inf && m.inf - m.infSd, hi: (m) => m.inf && m.inf + m.infSd, dom: [200, 360], ticks: [200, 250, 300, 350], f: (v) => fmt(v, 1) + ' ms', hiBetter: false,
      cap: { en: 'GPU inference time per image, mean ± standard deviation. All models land within about 35 ms of each other, so size barely changes latency here. The Ensemble was not benchmarked.', el: 'Χρόνος συμπερασμού ανά εικόνα σε GPU, μέσος όρος ± τυπική απόκλιση. Όλα τα μοντέλα απέχουν περίπου 35 ms μεταξύ τους, άρα το μέγεθος σχεδόν δεν αλλάζει την καθυστέρηση εδώ. Το Ensemble δεν μετρήθηκε.' } },
  };
  const TIER = { binary: 'top', cnn4: 'top', cnn3: 'top', ensemble: 'top', medvitdual: 'top', multidual: 'mid', medvit: 'mid', efnet: 'low', resnet: 'low' };
  let perfMetric = 'acc';

  function segRadios(container, items, current, onPick, label) {
    container.replaceChildren(...items.map((it) => {
      const b = H('button', { type: 'button', role: 'radio', 'aria-checked': it.id === current ? 'true' : 'false', tabindex: it.id === current ? '0' : '-1' },
        (it.color ? `<span class="sw" style="background:${cvar(it.color)}"></span>` : '') + it.label);
      b.addEventListener('click', () => onPick(it.id));
      b.addEventListener('keydown', (e) => {
        const i = items.findIndex((x) => x.id === it.id);
        let j = null;
        if (e.key === 'ArrowRight' || e.key === 'ArrowDown') j = (i + 1) % items.length;
        if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') j = (i - 1 + items.length) % items.length;
        if (j != null) { e.preventDefault(); onPick(items[j].id); container.querySelectorAll('button')[j].focus(); }
      });
      return b;
    }));
    if (label) container.setAttribute('aria-label', label);
  }

  function renderPerf() {
    segRadios($('#perf-metric'), Object.keys(METRICS).map((k) => ({ id: k, label: u(k) })), perfMetric, (k) => { perfMetric = k; renderPerf(); }, lang() === 'el' ? 'Μετρική' : 'Metric');
    const M = METRICS[perfMetric];
    const host = $('#perf-chart');
    const w = Math.max(300, host.clientWidth);
    const narrow = w < 520;
    const lw = narrow ? 128 : 172, rw = narrow ? 62 : 84, top = 8, rowH = 34, gapT = perfMetric === 'acc' ? 22 : 0;
    const list = D.MODELS.filter((m) => M.get(m) != null).slice().sort((a, b) => (M.hiBetter ? M.get(b) - M.get(a) : M.get(a) - M.get(b)));
    const missing = D.MODELS.filter((m) => M.get(m) == null);
    // Row y positions, with a gap before each tier when showing accuracy.
    let y = top; const ys = []; let prevTier = null; const bands = [];
    list.forEach((m) => {
      if (perfMetric === 'acc' && TIER[m.id] !== prevTier) { y += gapT; bands.push({ tier: TIER[m.id], y0: y - gapT + 4 }); prevTier = TIER[m.id]; }
      ys.push(y + rowH / 2); y += rowH;
    });
    const plotH = y + missing.length * rowH;
    const Hh = plotH + 30;
    const x0 = lw, x1 = w - rw;
    const sx = M.log
      ? (v) => x0 + (Math.log(v) - Math.log(M.dom[0])) / (Math.log(M.dom[1]) - Math.log(M.dom[0])) * (x1 - x0)
      : (v) => x0 + (v - M.dom[0]) / (M.dom[1] - M.dom[0]) * (x1 - x0);
    const svg = S('svg', { viewBox: `0 0 ${w} ${Hh}`, role: 'img', 'aria-label': `${u(perfMetric)}: ${list.map((m) => `${m.name} ${M.f(M.get(m))}`).join(', ')}` });

    if (perfMetric === 'acc') {
      bands.forEach((b, i) => {
        const yEnd = (bands[i + 1] ? bands[i + 1].y0 : y + 4);
        if (b.tier === 'mid') svg.appendChild(S('rect', { class: 'band', x: 0, y: b.y0, width: w, height: yEnd - b.y0, rx: 8 }));
        svg.appendChild(S('text', { class: 'band-lbl', x: 4, y: b.y0 + 13, text: u(b.tier) }));
      });
    }
    M.ticks.forEach((t) => {
      svg.appendChild(S('line', { class: 'gridline', x1: sx(t), x2: sx(t), y1: top, y2: plotH }));
      svg.appendChild(S('text', { class: 'tick', x: sx(t), y: plotH + 18, 'text-anchor': 'middle', text: M.log ? t : fmt(t, M.dom[1] <= 1 ? 2 : 0) }));
    });
    svg.appendChild(S('text', { class: 'tick', x: x1, y: Hh - 1, 'text-anchor': 'end', text: `${u(perfMetric)} · ${M.hiBetter ? u('higherBetter') : u('lowerBetter')}` }));

    list.forEach((m, i) => {
      const cy = ys[i], v = M.get(m), hero = m.id === 'medvitdual';
      const g = S('g');
      g.appendChild(S('text', { class: 'lbl', x: lw - 12, y: cy + 4, 'text-anchor': 'end', 'font-weight': hero ? 600 : null, text: narrow ? m.name.replace('EfficientNet-B3', 'EffNet-B3') : m.name }));
      g.appendChild(S('line', { class: 'gridline', x1: x0, x2: sx(v), y1: cy, y2: cy, style: 'stroke-dasharray:2 3' }));
      if (M.lo) {
        const lo = M.lo(m), hi = M.hi(m);
        g.appendChild(S('line', { x1: sx(lo), x2: sx(hi), y1: cy, y2: cy, style: `stroke:${hero ? 'var(--c-hero)' : 'var(--ink-3)'};stroke-width:2;stroke-linecap:round` }));
        [lo, hi].forEach((e) => g.appendChild(S('line', { x1: sx(e), x2: sx(e), y1: cy - 5, y2: cy + 5, style: `stroke:${hero ? 'var(--c-hero)' : 'var(--ink-3)'};stroke-width:1.5` })));
      }
      g.appendChild(S('circle', { cx: sx(v), cy, r: hero ? 6 : 5, style: `fill:${hero ? 'var(--c-hero)' : 'var(--ink-2)'};stroke:var(--surface);stroke-width:2` }));
      g.appendChild(S('text', { class: 'val', x: w - 4, y: cy + 4, 'text-anchor': 'end', 'font-weight': hero ? 600 : null, text: M.f(v) }));
      const hit = S('rect', { class: 'hit', x: 0, y: cy - rowH / 2, width: w, height: rowH, tabindex: 0, role: 'img', 'aria-label': `${m.name}: ${M.f(v)}` });
      g.appendChild(hit);
      bindTip(hit, () => `<b>${m.name}</b>${hero ? ` · ${u('proposed')}` : ''}` +
        row(u('acc'), `${fmt(m.acc)}% (${fmt(m.lo, 1)}–${fmt(m.hi, 1)})`) + row(u('f1'), fmt(m.f1) + '%') + row(u('auc'), fmt(m.auc, m.auc > 0.999 ? 4 : 3)) +
        row(u('params'), fmt(m.params) + ' M') + row(u('inf'), m.inf ? `${fmt(m.inf, 1)} ± ${fmt(m.infSd, 1)} ms` : u('na')) + row(u('rank'), fmt(m.rank, 3)));
      svg.appendChild(g);
    });
    missing.forEach((m, i) => {
      const cy = y + i * rowH + rowH / 2;
      svg.appendChild(S('text', { class: 'lbl-2', x: lw - 12, y: cy + 4, 'text-anchor': 'end', text: m.name }));
      svg.appendChild(S('text', { class: 'val', x: x0, y: cy + 4, text: u('na') }));
    });
    host.replaceChildren(svg);
    $('#perf-cap').textContent = T(M.cap);

    // Table view
    const th = ['#', u('model'), u('acc') + ' (%)', u('ci'), u('f1') + ' (%)', u('auc'), u('params') + ' (M)', u('inf') + ' (ms)', u('rank')];
    $('#perf-table').innerHTML = `<div class="table-scroll"><table><thead><tr>${th.map((h) => `<th scope="col">${h}</th>`).join('')}</tr></thead><tbody>` +
      D.MODELS.map((m) => `<tr><td>${m.n}</td><th scope="row" style="text-align:left;font-weight:500;color:var(--ink)">${m.name}</th><td>${fmt(m.acc)}</td><td>${fmt(m.lo, 1)}–${fmt(m.hi, 1)}</td><td>${fmt(m.f1)}</td><td>${fmt(m.auc, m.auc > 0.999 ? 4 : 3)}</td><td>${fmt(m.params)}</td><td>${m.inf ? fmt(m.inf, 1) + ' ± ' + fmt(m.infSd, 1) : '—'}</td><td>${fmt(m.rank, 3)}</td></tr>`).join('') +
      '</tbody></table></div>';
  }

  /* ═════════ Chart: accuracy vs parameters ═════════ */
  function renderScatter() {
    const host = $('#scatter-chart');
    const w = Math.max(280, host.clientWidth), Hh = Math.round(clamp(w * 0.68, 250, 330));
    const L = 40, R = 14, Tp = 10, Bm = 40;
    const sx = (v) => L + (Math.log(v) - Math.log(1)) / (Math.log(70) - Math.log(1)) * (w - L - R);
    const sy = (v) => Tp + (100 - v) / (100 - 74) * (Hh - Tp - Bm);
    const svg = S('svg', { viewBox: `0 0 ${w} ${Hh}`, role: 'img', 'aria-label': D.MODELS.map((m) => `${m.name}: ${fmt(m.params)} M, ${fmt(m.acc)}%`).join('; ') });
    [75, 80, 85, 90, 95, 100].forEach((t) => {
      svg.appendChild(S('line', { class: 'gridline', x1: L, x2: w - R, y1: sy(t), y2: sy(t) }));
      svg.appendChild(S('text', { class: 'tick', x: L - 6, y: sy(t) + 4, 'text-anchor': 'end', text: t }));
    });
    [1, 2, 5, 10, 20, 50].forEach((t) => svg.appendChild(S('text', { class: 'tick', x: sx(t), y: Hh - Bm + 16, 'text-anchor': 'middle', text: t })));
    svg.appendChild(S('line', { class: 'axis', x1: L, x2: w - R, y1: Hh - Bm, y2: Hh - Bm }));
    svg.appendChild(S('text', { class: 'tick', x: w - R, y: Hh - 4, 'text-anchor': 'end', text: `${u('params')} (M, log)` }));
    svg.appendChild(S('text', { class: 'tick', x: L, y: Hh - 4, text: `${u('acc')} %` }));
    // Labels with hand-placed offsets to avoid collisions in the CNN cluster.
    const LAB = {
      binary: [10, -12, 'start'], cnn4: null, cnn3: null, ensemble: [10, 16, 'start'], multidual: [10, 16, 'start'],
      medvit: [-10, 16, 'end'], medvitdual: [-10, -10, 'end'], efnet: [-10, 4, 'end'], resnet: [-10, 4, 'end'],
    };
    D.MODELS.forEach((m) => {
      const hero = m.id === 'medvitdual';
      const cx = sx(m.params), cy = sy(m.acc);
      const c = S('circle', { cx, cy, r: hero ? 6.5 : 5, tabindex: 0, role: 'img', 'aria-label': `${m.name}: ${fmt(m.params)} M, ${fmt(m.acc)}%`,
        style: `fill:${hero ? 'var(--c-hero)' : 'var(--ink-2)'};stroke:var(--surface);stroke-width:2;cursor:pointer` });
      bindTip(c, () => `<b>${m.name}</b>` + row(u('params'), fmt(m.params) + ' M') + row(u('acc'), fmt(m.acc) + '%'));
      svg.appendChild(c);
      const l = LAB[m.id];
      if (l) svg.appendChild(S('text', { class: hero ? 'lbl' : 'lbl-2', x: cx + l[0], y: cy + l[1], 'text-anchor': l[2], 'font-weight': hero ? 600 : null, text: m.name.replace('EfficientNet-B3', 'EffNet-B3') }));
    });
    // Combined label for the overlapping 4-/3-Class CNN dots.
    const cx = sx(1.28), cy = sy(97.9);
    svg.appendChild(S('text', { class: 'lbl-2', x: cx + 12, y: cy + 4, text: '4- / 3-Class CNN' }));
    host.replaceChildren(svg);
  }

  /* ═════════ Chart: confusion matrix ═════════ */
  function seqColor(t) { const k = clamp(Math.round(t * 7), 0, 7); return { fill: `var(--seq-${k})`, dark: k >= 4 }; }
  function renderCM() {
    const host = $('#cm-chart');
    const w = Math.max(280, host.clientWidth);
    const lw = w < 420 ? 82 : 100, top = 34;
    const cell = Math.floor(Math.min(74, (w - lw - 4) / 4));
    const Hh = top + cell * 4 + 8;
    const svg = S('svg', { viewBox: `0 0 ${w} ${Hh}`, role: 'img', 'aria-label': 'Confusion matrix, MedViT V2 Dual' });
    const C = D.CLASSES;
    svg.appendChild(S('text', { class: 'tick', x: lw, y: 11, text: `${u('predC')} →` }));
    C.forEach((c, j) => svg.appendChild(S('text', { class: 'lbl-2', x: lw + j * cell + cell / 2, y: top - 8, 'text-anchor': 'middle', text: w < 420 ? T(c).slice(0, 5) + '.' : T(c) })));
    D.CM.forEach((r, i) => {
      const tot = r.reduce((a, b) => a + b, 0);
      svg.appendChild(S('text', { class: 'lbl-2', x: lw - 8, y: top + i * cell + cell / 2 + 4, 'text-anchor': 'end', text: T(C[i]) }));
      r.forEach((v, j) => {
        const pct = v / tot;
        const s = seqColor(Math.sqrt(pct));
        const x = lw + j * cell, y = top + i * cell;
        svg.appendChild(S('rect', { x: x + 1, y: y + 1, width: cell - 2, height: cell - 2, rx: 6, style: `fill:${s.fill}` + (i === j ? ';stroke:var(--ink);stroke-width:1.5' : '') }));
        svg.appendChild(S('text', { x: x + cell / 2, y: y + cell / 2 + 5, 'text-anchor': 'middle', 'font-size': 14, 'font-weight': i === j ? 600 : 400, style: `fill:${s.dark ? '#fff' : 'var(--ink)'}`, text: v }));
        const hit = S('rect', { class: 'hit', x, y, width: cell, height: cell, tabindex: 0, role: 'img', 'aria-label': `${T(C[i])} → ${T(C[j])}: ${v}` });
        bindTip(hit, () => `${u('trueC')}: <b>${T(C[i])}</b><br>${u('predC')}: <b>${T(C[j])}</b>` + row('n', v) + row('%', fmt(pct * 100, 1) + '%'));
        svg.appendChild(hit);
      });
    });
    host.replaceChildren(svg);
  }

  /* ═════════ Chart: per-class F1 (dot strip) ═════════ */
  function legend(items) {
    return `<div class="legend-row" style="margin:0 0 4px">${items.map((c) => `<span><span class="sw-inline" style="display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;background:${cvar(c.color)}"></span>${T(c)}</span>`).join('')}</div>`;
  }
  function renderF1() {
    const host = $('#f1-chart');
    const w = Math.max(280, host.clientWidth);
    const lw = w < 420 ? 104 : 122, R = 16, top = 10, rowH = 58;
    const dom = [86, 100];
    const sx = (v) => lw + (v - dom[0]) / (dom[1] - dom[0]) * (w - lw - R);
    const Hh = top + rowH * 3 + 30;
    const svg = S('svg', { viewBox: `0 0 ${w} ${Hh}`, role: 'img', 'aria-label': 'Per-class F1' });
    [86, 90, 94, 98].forEach((t) => {
      svg.appendChild(S('line', { class: 'gridline', x1: sx(t), x2: sx(t), y1: top, y2: top + rowH * 3 }));
      svg.appendChild(S('text', { class: 'tick', x: sx(t), y: top + rowH * 3 + 18, 'text-anchor': 'middle', text: t }));
    });
    svg.appendChild(S('text', { class: 'tick', x: w - R, y: Hh - 1, 'text-anchor': 'end', text: 'F1 (%)' }));
    const order = ['glioma', 'no_tumor', 'pituitary', 'meningioma'];
    const off = { glioma: -12, no_tumor: -4, pituitary: 4, meningioma: 12 };
    D.F1.models.forEach((name, i) => {
      const cy = top + i * rowH + rowH / 2;
      svg.appendChild(S('text', { class: 'lbl', x: lw - 12, y: cy + 4, 'text-anchor': 'end', text: name }));
      const vals = order.map((c) => D.F1.values[c][i]);
      svg.appendChild(S('line', { x1: sx(Math.min(...vals)), x2: sx(Math.max(...vals)), y1: cy, y2: cy, style: 'stroke:var(--axis);stroke-width:1' }));
      order.forEach((cid) => {
        const v = D.F1.values[cid][i], c = cls(cid), men = cid === 'meningioma';
        const dot = S('circle', { cx: sx(v), cy: cy + off[cid], r: men ? 6.5 : 5, tabindex: 0, role: 'img', 'aria-label': `${name}, ${T(c)}: ${fmt(v)}%`,
          style: `fill:${cvar(c.color)};stroke:var(--surface);stroke-width:2;cursor:pointer` });
        bindTip(dot, () => `<b>${name}</b>` + row(T(c), fmt(v) + '%'));
        svg.appendChild(dot);
        if (men) svg.appendChild(S('text', { class: 'val', x: sx(v), y: cy + off[cid] + 20, 'text-anchor': 'middle', text: fmt(v) }));
      });
    });
    host.innerHTML = legend(D.CLASSES);
    host.appendChild(svg);
    $('#f1-table').innerHTML = `<div class="table-scroll"><table><thead><tr><th scope="col">${u('model')}</th>${D.CLASSES.map((c) => `<th scope="col">${T(c)}</th>`).join('')}</tr></thead><tbody>` +
      D.F1.models.map((n, i) => `<tr><td>${n}</td>${D.CLASSES.map((c) => `<td>${fmt(D.F1.values[c.id][i])}</td>`).join('')}</tr>`).join('') + '</tbody></table></div>';
  }

  /* ═════════ Chart: bootstrap occlusion CI ═════════ */
  function renderCI() {
    const host = $('#ci-chart');
    const w = Math.max(280, host.clientWidth);
    const lw = w < 420 ? 104 : 140, R = 16, top = 8, rowH = 50;
    const sx = (v) => lw + v / 80 * (w - lw - R);
    const Hh = top + rowH * D.BOOT.length + 30;
    const svg = S('svg', { viewBox: `0 0 ${w} ${Hh}`, role: 'img', 'aria-label': 'Bootstrap confidence intervals for occlusion confidence drop' });
    [0, 20, 40, 60, 80].forEach((t) => {
      svg.appendChild(S('line', { class: 'gridline', x1: sx(t), x2: sx(t), y1: top, y2: top + rowH * D.BOOT.length }));
      svg.appendChild(S('text', { class: 'tick', x: sx(t), y: top + rowH * D.BOOT.length + 18, 'text-anchor': 'middle', text: t + '%' }));
    });
    const keys = [['g', 'glioma', -10], ['m', 'meningioma', 0], ['p', 'pituitary', 10]];
    D.BOOT.forEach((b, i) => {
      const cy = top + i * rowH + rowH / 2;
      svg.appendChild(S('text', { class: 'lbl', x: lw - 12, y: cy + 4, 'text-anchor': 'end', text: b.model }));
      keys.forEach(([k, cid, o]) => {
        const [mean, lo, hi, n] = b[k], c = cls(cid), yy = cy + o;
        svg.appendChild(S('line', { x1: sx(lo), x2: sx(hi), y1: yy, y2: yy, style: `stroke:${cvar(c.color)};stroke-width:2.5;stroke-linecap:round` }));
        const dot = S('circle', { cx: sx(mean), cy: yy, r: 4.5, tabindex: 0, role: 'img', 'aria-label': `${b.model}, ${T(c)}: ${fmt(mean, 1)}% [${fmt(lo, 1)}, ${fmt(hi, 1)}]`,
          style: `fill:${cvar(c.color)};stroke:var(--surface);stroke-width:2;cursor:pointer` });
        bindTip(dot, () => `<b>${b.model}</b> · ${T(c)}` + row(lang() === 'el' ? 'Μέση πτώση' : 'Mean drop', fmt(mean, 1) + '%') + row(u('ci'), `${fmt(lo, 1)}–${fmt(hi, 1)}%`) + row('n', n));
        svg.appendChild(dot);
      });
    });
    host.innerHTML = legend(D.CLASSES.filter((c) => c.id !== 'no_tumor'));
    host.appendChild(svg);
    const L = lang() === 'el';
    $('#ci-table').innerHTML = `<div class="table-scroll"><table><thead><tr><th scope="col">${u('model')}</th><th scope="col">${L ? 'Κλάση' : 'Class'}</th><th scope="col">${L ? 'Μέση' : 'Mean'} (%)</th><th scope="col">${u('ci')}</th><th scope="col">n</th></tr></thead><tbody>` +
      D.BOOT.map((b) => keys.map(([k, cid]) => `<tr><td>${b.model}</td><td>${T(cls(cid))}</td><td>${fmt(b[k][0], 1)}</td><td>${fmt(b[k][1], 1)}–${fmt(b[k][2], 1)}</td><td>${b[k][3]}</td></tr>`).join('')).join('') + '</tbody></table></div>';
  }

  /* ═════════ Statistics stepper ═════════ */
  const STEPS = [
    { k: '1', id: 'friedman', en: 'Friedman', el: 'Friedman', s: { en: 'Any difference at all?', el: 'Υπάρχει διαφορά;' } },
    { k: '2', id: 'nemenyi', en: 'Nemenyi CD', el: 'Nemenyi CD', s: { en: 'Which ranks are far apart?', el: 'Ποιες κατατάξεις απέχουν;' } },
    { k: '3', id: 'wilcoxon', en: 'Wilcoxon', el: 'Wilcoxon', s: { en: 'Which pairs differ?', el: 'Ποια ζεύγη διαφέρουν;' } },
    { k: '4', id: 'cohen', en: "Cohen's d", el: "Cohen's d", s: { en: 'Does it matter in practice?', el: 'Έχει πρακτική σημασία;' } },
    { k: '5', id: 'cv', en: '5-fold CV', el: '5-fold CV', s: { en: 'Is it stable?', el: 'Είναι σταθερό;' } },
  ];
  let stepIdx = 0, cdSel = null;
  function renderStepper() {
    const ol = $('#stat-steps');
    ol.replaceChildren(...STEPS.map((s, i) => {
      const li = H('li', { role: 'presentation' });
      const b = H('button', { type: 'button', role: 'tab', id: `st-${s.id}`, 'aria-selected': i === stepIdx ? 'true' : 'false', tabindex: i === stepIdx ? '0' : '-1', 'aria-controls': 'stat-body' },
        `<span class="k">${s.k}</span><b>${T(s)}</b>${T(s.s)}`);
      b.addEventListener('click', () => { stepIdx = i; renderStepper(); });
      b.addEventListener('keydown', (e) => {
        let j = null;
        if (e.key === 'ArrowRight') j = (i + 1) % STEPS.length;
        if (e.key === 'ArrowLeft') j = (i - 1 + STEPS.length) % STEPS.length;
        if (j != null) { e.preventDefault(); stepIdx = j; renderStepper(); $(`#st-${STEPS[j].id}`).focus(); }
      });
      li.append(b); return li;
    }));
    $('#stat-body').setAttribute('aria-labelledby', `st-${STEPS[stepIdx].id}`);
    renderStepBody();
  }

  function renderStepBody() {
    const body = $('#stat-body');
    const id = STEPS[stepIdx].id, L = lang() === 'el', St = D.STATS;
    body.replaceChildren();
    const text = (en, el) => body.appendChild(H('div', { class: 'step-text' }, L ? el : en));
    const pills = (arr) => body.appendChild(H('div', {}, arr.map((p) => `<span class="result-pill">${p}</span>`).join('')));
    const chart = () => { const c = H('div', { class: 'chart' }); body.appendChild(c); return c; };

    if (id === 'friedman') {
      text(`<p>The <strong>Friedman test</strong> ranks the nine models on every one of the 503 images and asks whether the average ranks differ more than chance would allow. It makes no assumption of normality, which accuracy scores rarely satisfy.</p><p style="margin-top:.6em">The null hypothesis of equal performance is decisively rejected. Lower average rank is better.</p>`,
        `<p>Ο <strong>έλεγχος Friedman</strong> κατατάσσει τα εννέα μοντέλα σε καθεμία από τις 503 εικόνες και ρωτά αν οι μέσες κατατάξεις διαφέρουν περισσότερο απ' όσο θα επέτρεπε η τύχη. Δεν υποθέτει κανονικότητα, την οποία οι τιμές ακρίβειας σπάνια ικανοποιούν.</p><p style="margin-top:.6em">Η μηδενική υπόθεση ίσης απόδοσης απορρίπτεται κατηγορηματικά. Μικρότερη μέση κατάταξη σημαίνει καλύτερο μοντέλο.</p>`);
      pills([`χ²<sub>F</sub>(${St.df}) = ${fmt(St.chi2)}`, `p = ${L ? St.p.replace('.', ',') : St.p}`, `N = 503`]);
      renderRanks(chart());
    } else if (id === 'nemenyi') {
      text(`<p>The <strong>Nemenyi post-hoc test</strong> gives a critical difference: two models whose average ranks differ by less than CD = ${fmt(St.cd, 3)} cannot be told apart. Bars join such groups. <strong>Select a model</strong> to see who lies within its CD.</p><p style="margin-top:.6em">Only Binary CNN and ResNet-50 Dual are separated by more than CD, so Nemenyi alone has low resolution here. The pairwise tests in step 3 are sharper.</p>`,
        `<p>Ο <strong>post-hoc έλεγχος Nemenyi</strong> δίνει μια κρίσιμη διαφορά: δύο μοντέλα των οποίων οι μέσες κατατάξεις απέχουν λιγότερο από CD = ${fmt(St.cd, 3)} δεν ξεχωρίζουν. Οι μπάρες ενώνουν τέτοιες ομάδες. <strong>Επίλεξε ένα μοντέλο</strong> για να δεις ποια βρίσκονται εντός της CD του.</p><p style="margin-top:.6em">Μόνο το Binary CNN και το ResNet-50 Dual απέχουν περισσότερο από CD, άρα ο Nemenyi μόνος του έχει χαμηλή διακριτική ικανότητα εδώ. Οι έλεγχοι ανά ζεύγη στο βήμα 3 είναι οξύτεροι.</p>`);
      pills([`CD = ${fmt(St.cd, 3)}`, `α = ${fmt(St.alpha)}`]);
      renderCD(chart());
    } else if (id === 'wilcoxon' || id === 'cohen') {
      const isP = id === 'wilcoxon';
      if (isP) text(`<p><strong>Wilcoxon signed-rank tests</strong> compare every pair of models on the same images: ${St.pairs} pairs. With so many tests, some would come out "significant" by luck, so each p-value is multiplied by ${St.pairs} (<strong>Bonferroni</strong>). Coloured cells are significant at α = 0.05; darker means stronger evidence.</p><p style="margin-top:.6em">MedViT V2 Dual is not distinguishable from the best model (p = 0.240), yet significantly beats the 4-Class CNN, the Ensemble, the Multi-Dual System and MedViT V2 Base. ResNet-50 Dual loses to every other model.</p>`,
        `<p>Οι <strong>έλεγχοι Wilcoxon</strong> συγκρίνουν κάθε ζεύγος μοντέλων στις ίδιες εικόνες: ${St.pairs} ζεύγη. Με τόσους ελέγχους, κάποιοι θα έβγαιναν «σημαντικοί» από τύχη, γι' αυτό κάθε p πολλαπλασιάζεται επί ${St.pairs} (<strong>Bonferroni</strong>). Τα χρωματιστά κελιά είναι σημαντικά για α = 0,05· πιο σκούρο σημαίνει ισχυρότερη ένδειξη.</p><p style="margin-top:.6em">Το MedViT V2 Dual δεν ξεχωρίζει από το καλύτερο μοντέλο (p = 0,240), αλλά υπερέχει σημαντικά του 4-Class CNN, του Ensemble, του Multi-Dual System και του MedViT V2 Base. Το ResNet-50 Dual χάνει από όλα τα υπόλοιπα.</p>`);
      else text(`<p>Significance says a difference exists; <strong>Cohen's d</strong> says how large it is. Below 0.2 is negligible, 0.2–0.5 small, 0.5–0.8 medium.</p><p style="margin-top:.6em">Every pair falls in the negligible-to-small range except one: Binary CNN versus ResNet-50 Dual (d = 0.528, medium). Inside the top group all effects are negligible: the leading models are practically equivalent.</p>`,
        `<p>Η σημαντικότητα λέει ότι μια διαφορά υπάρχει· το <strong>Cohen's d</strong> λέει πόσο μεγάλη είναι. Κάτω από 0,2 αμελητέα, 0,2–0,5 μικρή, 0,5–0,8 μέτρια.</p><p style="margin-top:.6em">Όλα τα ζεύγη είναι στο εύρος αμελητέο έως μικρό, εκτός από ένα: Binary CNN έναντι ResNet-50 Dual (d = 0,528, μέτριο). Μέσα στην κορυφαία ομάδα όλες οι επιδράσεις είναι αμελητέες: τα κορυφαία μοντέλα είναι πρακτικά ισοδύναμα.</p>`);
      pills(isP ? [`${St.pairs} ${L ? 'ζεύγη' : 'pairs'}`, `p<sub>Bonf</sub> = min(36·p, 1)`, `${L ? 'κατώφλι' : 'raw threshold'} ${fmt(St.bonf, 5)}`] : ['d < 0.2 ' + (L ? 'αμελητέο' : 'negligible'), '0.2–0.5 ' + (L ? 'μικρό' : 'small'), '≥ 0.5 ' + (L ? 'μέτριο' : 'medium')]);
      renderHeat(chart(), isP);
      const det = H('details', { class: 'table-view' }, `<summary>${L ? 'Προβολή ως πίνακας' : 'Show as table'}</summary>`);
      const M = D.MODELS;
      let rows = '';
      for (let i = 1; i < 9; i++) for (let j = 0; j < i; j++) {
        const p = D.PAIR.p[i][j], d = D.PAIR.d[i][j];
        rows += `<tr><td>${M[i].name} – ${M[j].name}</td><td>${fmtP(p)}</td><td>${fmt(d, 3)}</td><td>${p < 0.05 ? u('sig') : u('nsig')}</td></tr>`;
      }
      det.insertAdjacentHTML('beforeend', `<div class="table-scroll"><table><thead><tr><th scope="col">${L ? 'Ζεύγος' : 'Pair'}</th><th scope="col">p<sub>Bonf</sub></th><th scope="col">d</th><th scope="col"></th></tr></thead><tbody>${rows}</tbody></table></div>`);
      body.appendChild(det);
    } else if (id === 'cv') {
      text(`<p><strong>5-fold stratified cross-validation</strong> on the training and validation data (90% of the dataset) checks that a model's score is not an artefact of one lucky split. The frozen test set is never touched. Standard deviations stay below 0.6 points for all four models evaluated this way.</p>`,
        `<p>Το <strong>5-fold στρωματοποιημένο cross-validation</strong> στα δεδομένα εκπαίδευσης και επικύρωσης (90% του συνόλου) ελέγχει ότι η επίδοση ενός μοντέλου δεν οφείλεται σε έναν τυχερό διαχωρισμό. Το παγωμένο σύνολο ελέγχου δεν αγγίζεται ποτέ. Οι τυπικές αποκλίσεις μένουν κάτω από 0,6 μονάδες και για τα τέσσερα μοντέλα που αξιολογήθηκαν έτσι.</p>`);
      const C = D.CV;
      body.appendChild(H('div', { class: 'table-scroll' }, `<table class="cv-table"><caption class="cap" style="caption-side:bottom;text-align:left">${L ? 'Ακρίβεια επικύρωσης ανά fold (%).' : 'Validation accuracy per fold (%).'}</caption><thead><tr><th scope="col">Fold</th>${C.models.map((m) => `<th scope="col">${m}</th>`).join('')}</tr></thead><tbody>` +
        C.folds.map((f, i) => `<tr><td>${i + 1}</td>${f.map((v) => `<td>${fmt(v)}</td>`).join('')}</tr>`).join('') +
        `<tr><th scope="row" style="text-align:left">${L ? 'Μέσος ± ΤΑ' : 'Mean ± SD'}</th>${C.mean.map((v, i) => `<td><b>${fmt(v)}</b> ± ${fmt(C.sd[i])}</td>`).join('')}</tr></tbody></table>`));
    }
  }

  function renderRanks(host) {
    const w = Math.max(280, host.clientWidth);
    const lw = w < 420 ? 128 : 170, R = 52, rowH = 28;
    const list = D.MODELS.slice().sort((a, b) => a.rank - b.rank);
    const dom = [4.6, 5.4], sx = (v) => lw + (v - dom[0]) / (dom[1] - dom[0]) * (w - lw - R);
    const Hh = list.length * rowH + 30;
    const svg = S('svg', { viewBox: `0 0 ${w} ${Hh}`, role: 'img', 'aria-label': list.map((m) => `${m.name} ${fmt(m.rank, 3)}`).join(', ') });
    [4.6, 4.8, 5.0, 5.2, 5.4].forEach((t) => {
      svg.appendChild(S('line', { class: 'gridline', x1: sx(t), x2: sx(t), y1: 0, y2: list.length * rowH }));
      svg.appendChild(S('text', { class: 'tick', x: sx(t), y: list.length * rowH + 16, 'text-anchor': 'middle', text: fmt(t, 1) }));
    });
    svg.appendChild(S('text', { class: 'tick', x: w - 4, y: Hh - 1, 'text-anchor': 'end', text: `${u('rank')} · ${u('lowerBetter')}` }));
    list.forEach((m, i) => {
      const cy = i * rowH + rowH / 2, hero = m.id === 'medvitdual';
      svg.appendChild(S('text', { class: 'lbl', x: lw - 12, y: cy + 4, 'text-anchor': 'end', 'font-weight': hero ? 600 : null, text: m.name }));
      svg.appendChild(S('line', { x1: sx(dom[0]), x2: sx(m.rank), y1: cy, y2: cy, style: 'stroke:var(--grid);stroke-width:1' }));
      svg.appendChild(S('circle', { cx: sx(m.rank), cy, r: hero ? 6 : 5, style: `fill:${hero ? 'var(--c-hero)' : 'var(--signal)'};stroke:var(--surface);stroke-width:2` }));
      svg.appendChild(S('text', { class: 'val', x: w - 4, y: cy + 4, 'text-anchor': 'end', text: fmt(m.rank, 3) }));
    });
    host.replaceChildren(svg);
  }

  function renderCD(host) {
    const w = Math.max(300, host.clientWidth);
    const list = D.MODELS.slice().sort((a, b) => a.rank - b.rank);
    const narrow = w < 560;
    const side = narrow ? 112 : 170, axY = 54, rowH = narrow ? 24 : 26;
    const dom = [4.7, 5.4], x0 = side, x1 = w - side;
    const sx = (v) => x0 + (v - dom[0]) / (dom[1] - dom[0]) * (x1 - x0);
    const half = Math.ceil(list.length / 2);
    const Hh = axY + 52 + half * rowH + 8;
    const CD = D.STATS.cd;
    const svg = S('svg', { viewBox: `0 0 ${w} ${Hh}`, role: 'group', 'aria-label': 'Critical difference diagram' });

    // CD scale bar
    svg.appendChild(S('line', { x1: x0, x2: sx(dom[0] + CD), y1: 14, y2: 14, style: 'stroke:var(--ink);stroke-width:2' }));
    [x0, sx(dom[0] + CD)].forEach((x) => svg.appendChild(S('line', { x1: x, x2: x, y1: 9, y2: 19, style: 'stroke:var(--ink);stroke-width:1.5' })));
    svg.appendChild(S('text', { class: 'lbl', x: sx(dom[0] + CD) + 8, y: 18, text: `CD = ${fmt(CD, 3)}` }));

    // Highlight window
    const selM = list.find((m) => m.id === cdSel);
    if (selM) svg.appendChild(S('rect', { x: sx(Math.max(dom[0], selM.rank - CD)), y: axY - 10, width: sx(Math.min(dom[1], selM.rank + CD)) - sx(Math.max(dom[0], selM.rank - CD)), height: 20, rx: 4, style: 'fill:var(--signal-soft)' }));
    // Axis
    svg.appendChild(S('line', { class: 'axis', x1: x0, x2: x1, y1: axY, y2: axY, style: 'stroke:var(--ink-3);stroke-width:1.5' }));
    for (let t = 4.7; t <= 5.401; t += 0.1) {
      svg.appendChild(S('line', { x1: sx(t), x2: sx(t), y1: axY - 5, y2: axY, style: 'stroke:var(--ink-3)' }));
      svg.appendChild(S('text', { class: 'tick', x: sx(t), y: axY - 9, 'text-anchor': 'middle', text: fmt(t, 1) }));
    }
    // Cliques (maximal groups within CD)
    const cliques = [];
    for (let i = 0; i < list.length; i++) {
      let j = i; while (j + 1 < list.length && list[j + 1].rank - list[i].rank <= CD) j++;
      if (j > i && !cliques.some(([a, b]) => a <= i && b >= j)) cliques.push([i, j]);
    }
    cliques.forEach(([a, b], k) => svg.appendChild(S('line', { x1: sx(list[a].rank) - 3, x2: sx(list[b].rank) + 3, y1: axY + 12 + k * 8, y2: axY + 12 + k * 8, style: `stroke:${k ? 'var(--signal)' : 'var(--accent)'};stroke-width:4;stroke-linecap:round` })));

    // Labels with elbow leaders
    const baseY = axY + 12 + cliques.length * 8 + 14;
    list.forEach((m, i) => {
      const left = i < half, r = left ? i : list.length - 1 - i;
      const ly = baseY + r * rowH, x = sx(m.rank);
      const within = selM ? Math.abs(m.rank - selM.rank) <= CD : true;
      const g = S('g', { tabindex: 0, role: 'button', 'aria-pressed': cdSel === m.id ? 'true' : 'false', 'aria-label': `${m.name}, ${u('rank')} ${fmt(m.rank, 3)}`, style: 'cursor:pointer', class: within ? '' : 'dim' });
      g.appendChild(S('line', { x1: x, x2: x, y1: axY, y2: ly, style: 'stroke:var(--ink-3);stroke-width:1' }));
      g.appendChild(S('line', { x1: x, x2: left ? x0 - 6 : x1 + 6, y1: ly, y2: ly, style: 'stroke:var(--ink-3);stroke-width:1' }));
      g.appendChild(S('circle', { cx: x, cy: axY, r: 4, style: `fill:${cdSel === m.id ? 'var(--signal)' : 'var(--ink)'}` }));
      g.appendChild(S('text', { class: 'lbl', x: left ? x0 - 10 : x1 + 10, y: ly + 4, 'text-anchor': left ? 'end' : 'start', 'font-weight': cdSel === m.id ? 600 : null,
        text: (narrow ? m.name.replace('EfficientNet-B3', 'EffNet-B3').replace(' System', '') : m.name) }));
      g.appendChild(S('rect', { x: left ? 0 : x1, y: ly - rowH / 2, width: side, height: rowH, fill: 'transparent' }));
      const pick = () => { cdSel = cdSel === m.id ? null : m.id; renderCD(host); const n = host.querySelectorAll('g[role=button]')[i]; n && n.focus(); };
      g.addEventListener('click', pick);
      g.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); } });
      svg.appendChild(g);
    });
    host.replaceChildren(svg);
  }

  function renderHeat(host, isP) {
    const w = Math.max(300, host.clientWidth);
    const M = D.MODELS, n = M.length;
    const lw = w < 520 ? 30 : 160;
    const cell = Math.floor(Math.min(58, (w - lw - 8) / (n - 1)));
    const top = 30;
    const Hh = top + cell * (n - 1) + 40;
    const showText = cell >= 34;
    const svg = S('svg', { viewBox: `0 0 ${w} ${Hh}`, role: 'img', 'aria-label': isP ? 'Bonferroni-adjusted Wilcoxon p-values' : "Cohen's d effect sizes" });
    // Columns: models 1..8; rows: models 2..9 (lower triangle).
    for (let j = 0; j < n - 1; j++) svg.appendChild(S('text', { class: 'tick', x: lw + j * cell + cell / 2, y: top - 10, 'text-anchor': 'middle', text: '#' + M[j].n }));
    for (let i = 1; i < n; i++) {
      const y = top + (i - 1) * cell;
      svg.appendChild(S('text', { class: 'lbl-2', x: lw - 8, y: y + cell / 2 + 4, 'text-anchor': 'end', text: lw > 60 ? `#${M[i].n} · ${M[i].name}` : '#' + M[i].n }));
      for (let j = 0; j < i; j++) {
        const p = D.PAIR.p[i][j], d = D.PAIR.d[i][j], x = lw + j * cell;
        let fill, dark = false, label;
        if (isP) {
          if (p < 0.05) { const t = clamp((-Math.log10(p) - 1.3) / 12.7, 0, 1); const k = 2 + Math.round(t * 5); fill = `var(--seq-${k})`; dark = k >= 4; }
          else fill = 'var(--surface-2)';
          label = fmtPshort(p);
        } else {
          const k = d < 0.2 ? 1 : d < 0.5 ? 3 : 6; fill = `var(--seq-${k})`; dark = k >= 4; label = fmt(d, 2);
        }
        svg.appendChild(S('rect', { x: x + 1, y: y + 1, width: cell - 2, height: cell - 2, rx: 4, style: `fill:${fill}` }));
        if (showText) svg.appendChild(S('text', { x: x + cell / 2, y: y + cell / 2 + 4, 'text-anchor': 'middle', 'font-size': 10.5, style: `fill:${dark ? '#fff' : (isP && p >= 0.05 ? 'var(--ink-3)' : 'var(--ink)')};font-variant-numeric:tabular-nums`, text: label }));
        const hit = S('rect', { class: 'hit', x, y, width: cell, height: cell });
        const better = M[i].acc >= M[j].acc ? M[i] : M[j];
        bindTip(hit, () => `<b>${M[i].name}</b> vs <b>${M[j].name}</b>` + row('p<sub>Bonf</sub>', fmtP(p)) + row("Cohen's d", fmt(d, 3)) +
          row('', p < 0.05 ? `${u('sig')} · ${better.name} ${u('better')}` : u('nsig')));
        svg.appendChild(hit);
      }
    }
    // Model key for narrow layout
    if (lw <= 60) {
      const key = M.map((m) => `#${m.n} ${m.name}`).join(' · ');
      host.replaceChildren(svg, H('p', { class: 'cap' }, key));
    } else host.replaceChildren(svg);
    // Legend
    const L = lang() === 'el';
    const lg = isP
      ? `<span><i style="background:var(--surface-2)"></i>${L ? 'μη σημαντική (p ≥ 0,05)' : 'not significant (p ≥ 0.05)'}</span><span><i style="background:var(--seq-2)"></i>p &lt; 0.05</span><span><i style="background:var(--seq-5)"></i>p &lt; 10<sup>−6</sup></span><span><i style="background:var(--seq-7)"></i>p &lt; 10<sup>−12</sup></span>`
      : `<span><i style="background:var(--seq-1)"></i>${L ? 'αμελητέο' : 'negligible'} &lt; 0.2</span><span><i style="background:var(--seq-3)"></i>${L ? 'μικρό' : 'small'} 0.2–0.5</span><span><i style="background:var(--seq-6)"></i>${L ? 'μέτριο' : 'medium'} ≥ 0.5</span>`;
    host.insertAdjacentHTML('beforeend', `<div class="legend-row heat-legend">${lg}</div>`);
    host.querySelectorAll('.heat-legend i').forEach((i) => { i.style.cssText += ';display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-2px;border:1px solid var(--rule)'; });
  }

  /* ═════════ XAI explorer ═════════ */
  let xCls = 'meningioma', xMethod = 'gradcam', xModel = 'cnn4';
  function renderXAI() {
    segRadios($('#xai-class'), D.CLASSES.map((c) => ({ id: c.id, label: T(c), color: c.color })), xCls, (id) => { xCls = id; if (id === 'no_tumor' && xModel === 'cnn3') xModel = 'cnn4'; renderXAI(); });
    segRadios($('#xai-method'), D.XAI_METHODS.map((m) => ({ id: m.id, label: m.name })), xMethod, (id) => { xMethod = id; renderXAI(); });
    const meth = D.XAI_METHODS.find((m) => m.id === xMethod);
    $('#xai-info').innerHTML = T(meth);
    const grid = $('#xai-grid');
    const sat = (mid) => mid === 'medvit' && (xMethod === 'shap' || xMethod === 'ig');
    grid.replaceChildren(...D.XAI_MODELS.map((m) => {
      const na = m.id === 'cnn3' && xCls === 'no_tumor';
      const b = H('button', { type: 'button', class: 'xai-tile' + (sat(m.id) ? ' flag' : ''), 'aria-pressed': m.id === xModel ? 'true' : 'false', disabled: na ? '' : null,
        'aria-label': `${m.name}: ${meth.name}, ${T(cls(xCls))}${sat(m.id) ? ' (' + u('satFlag') + ')' : ''}` });
      const img = na
        ? `<div class="na">${u('notApplicable')}</div>`
        : `<img src="assets/xai/${xMethod}_${xCls}_${m.id}.jpg" alt="" width="320" height="320" loading="lazy">`;
      b.innerHTML = `<div class="img">${img}</div><span class="cap2">${m.name}${m.id === 'medvit' ? `<small>${lang() === 'el' ? 'άλλη τομή' : 'own sample'}</small>` : ''}</span>`;
      if (!na) b.addEventListener('click', () => { xModel = m.id; renderXAI(); grid.querySelector(`[aria-pressed="true"]`).focus(); });
      return b;
    }));
    // Focus panel
    const m = D.XAI_MODELS.find((x) => x.id === xModel);
    const c = cls(xCls);
    const notes = D.XAI_NOTES;
    const modelNote = xModel === 'medvit' ? notes.medvit[xMethod] : notes.family[xModel];
    const altMap = `${meth.name} ${lang() === 'el' ? 'χάρτης του' : 'map of'} ${m.name}, ${T(c)}`;
    let figs = `<figure><img src="assets/xai/${xMethod}_${xCls}_${xModel}.jpg" alt="${altMap}" width="320" height="320"><figcaption>${meth.name} · ${m.name}</figcaption></figure>`;
    if (xModel === 'medvit') figs += `<figure><img src="assets/xai/orig_${xCls}_medvit.jpg" alt="${u('original')}, ${T(c)}" width="320" height="320"><figcaption>${u('original')}</figcaption></figure>`;
    const f = $('#xai-focus');
    f.style.gridTemplateColumns = innerWidth <= 560 ? '1fr' : xModel === 'medvit' ? '' : 'minmax(0,300px) minmax(0,1fr)';
    f.innerHTML = figs + `<div class="txt"><h3>${m.name} · ${T(c)}</h3><p>${T(modelNote)}</p><p>${T(notes.cls[xCls])}</p>${xModel === 'medvit' ? `<p class="note">${u('medvitSample')}</p>` : ''}</div>`;
  }

  /* ═════════ Ablation ═════════ */
  const ABL = [
    { id: 'cnn4', name: '4-Class CNN', b: { en: 'CNN only', el: 'Μόνο CNN' }, acc: 98.01, on: ['cnn'] },
    { id: 'medvit', name: 'MedViT V2 Base', b: { en: 'Transformer only', el: 'Μόνο transformer' }, acc: 93.03, on: ['vit'] },
    { id: 'multidual', name: 'Multi-Dual System', b: { en: 'CNN + CNN + fusion', el: 'CNN + CNN + σύντηξη' }, acc: 93.41, on: ['cnn', 'cnn2', 'fuse'] },
    { id: 'medvitdual', name: 'MedViT V2 Dual', b: { en: 'Transformer + CNN + fusion', el: 'Transformer + CNN + σύντηξη' }, acc: 98.15, on: ['cnn', 'vit', 'fuse'] },
  ];
  let ablSel = 'medvitdual';
  function renderAblation() {
    const host = $('#ablation-card');
    const L = lang() === 'el';
    const sw = H('div', { class: 'abl-switches', role: 'radiogroup', 'aria-label': L ? 'Διαμόρφωση' : 'Configuration' });
    ABL.forEach((a, i) => {
      const b = H('button', { type: 'button', role: 'radio', class: 'abl-opt', 'aria-checked': a.id === ablSel ? 'true' : 'false', tabindex: a.id === ablSel ? '0' : '-1' },
        `<span class="radio" aria-hidden="true"></span><span><b>${a.name}</b><small>${T(a.b)}</small></span><span class="acc">${fmt(a.acc)}%</span>`);
      b.addEventListener('click', () => { ablSel = a.id; renderAblation(); host.querySelector('[aria-checked="true"]').focus(); });
      b.addEventListener('keydown', (e) => {
        let j = null;
        if (e.key === 'ArrowDown' || e.key === 'ArrowRight') j = (i + 1) % ABL.length;
        if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') j = (i - 1 + ABL.length) % ABL.length;
        if (j != null) { e.preventDefault(); ablSel = ABL[j].id; renderAblation(); host.querySelector('[aria-checked="true"]').focus(); }
      });
      sw.appendChild(b);
    });
    const a = ABL.find((x) => x.id === ablSel);
    const delta = a.acc - 98.01;
    const view = H('div', { class: 'abl-view', 'aria-live': 'polite' });
    // Schematic
    const W = 420, Hh = 170, on = (k) => a.on.includes(k);
    const svg = S('svg', { viewBox: `0 0 ${W} ${Hh}`, role: 'img', 'aria-label': `${a.name}: ${T(a.b)}` });
    const box = (x, y, w, h, label, k, kind) => {
      const act = on(k);
      svg.appendChild(S('rect', { x, y, width: w, height: h, rx: 10, style: `fill:${act ? `color-mix(in srgb, var(${kind}) 18%, var(--surface))` : 'var(--surface)'};stroke:${act ? `var(${kind})` : 'var(--rule)'};stroke-width:${act ? 2 : 1.2};stroke-dasharray:${act ? 'none' : '4 4'}` }));
      svg.appendChild(S('text', { x: x + w / 2, y: y + h / 2 + 4, 'text-anchor': 'middle', 'font-size': 12.5, 'font-weight': act ? 600 : 400, style: `fill:${act ? 'var(--ink)' : 'var(--ink-3)'}`, text: label }));
    };
    const line = (x1, y1, x2, y2, act) => svg.appendChild(S('path', { d: `M${x1},${y1} C${(x1 + x2) / 2},${y1} ${(x1 + x2) / 2},${y2} ${x2},${y2}`, style: `fill:none;stroke:${act ? 'var(--accent)' : 'var(--rule)'};stroke-width:${act ? 2 : 1.2}` }));
    box(4, 66, 64, 38, 'MRI', 'in', '--b-in');
    svg.lastChild.previousSibling.setAttribute('style', 'fill:var(--surface-2);stroke:var(--ink-3);stroke-width:1.5');
    svg.lastChild.setAttribute('style', 'fill:var(--ink)');
    const rows = [['vit', 'Transformer', '--b-attn', 14], ['cnn', 'CNN', '--b-conv', 66], ['cnn2', 'CNN #2', '--b-conv', 118]];
    rows.forEach(([k, lbl, kind, y]) => { line(68, 85, 120, y + 19, on(k)); box(120, y, 120, 38, lbl, k, kind); });
    const fuse = on('fuse');
    rows.forEach(([k, , , y]) => line(240, y + 19, 290, 85, on(k) && fuse));
    box(290, 66, 70, 38, L ? 'Σύντηξη' : 'Fusion', 'fuse', '--b-fuse');
    if (!fuse) { const single = rows.find(([k]) => on(k)); line(240, single[3] + 19, 376, 85, true); }
    else line(360, 85, 376, 85, true);
    svg.appendChild(S('circle', { cx: 388, cy: 85, r: 12, style: 'fill:var(--b-out)' }));
    svg.appendChild(S('text', { x: 388, y: 89, 'text-anchor': 'middle', 'font-size': 10, style: 'fill:var(--surface)', text: 'out' }));
    view.appendChild(svg);
    view.appendChild(H('div', { class: 'abl-big' }, `${fmt(a.acc)}%`));
    view.appendChild(H('div', { class: 'abl-delta ' + (delta < -0.01 ? 'neg' : delta > 0.01 ? 'pos' : '') }, delta === 0 ? (L ? 'γραμμή βάσης' : 'baseline') : `${delta > 0 ? '+' : '−'}${fmt(Math.abs(delta))} pp ${L ? 'έναντι 4-Class CNN' : 'vs 4-Class CNN'}`));
    const notes = {
      cnn4: { en: 'The baseline: one custom CNN, 1.28 M parameters.', el: 'Η γραμμή βάσης: ένα custom CNN, 1,28 M παράμετροι.' },
      medvit: { en: 'Transformer representations alone are not enough on a dataset of this size.', el: 'Οι αναπαραστάσεις του transformer μόνες τους δεν αρκούν σε σύνολο δεδομένων αυτού του μεγέθους.' },
      multidual: { en: 'Fusion by itself does not help: two CNN heads plus a masked loss make optimisation harder.', el: 'Η σύντηξη από μόνη της δεν βοηθά: δύο κεφαλές CNN και ένα masked loss δυσκολεύουν τη βελτιστοποίηση.' },
      medvitdual: { en: 'Averaging the two branches would predict about 95.5%. The hybrid reaches 98.15%: each branch covers the other\'s blind spots.', el: 'Ο μέσος όρος των δύο κλάδων θα έδινε περίπου 95,5%. Το υβριδικό φτάνει 98,15%: κάθε κλάδος καλύπτει τα τυφλά σημεία του άλλου.' },
    };
    view.appendChild(H('p', { class: 'abl-note' }, T(notes[a.id])));
    host.replaceChildren(sw, view);
  }

  /* ═════════ Settings: language, theme, text size, motion ═════════ */
  function setLang(l) {
    root.lang = l; root.dataset.lang = l;
    try { localStorage.setItem('mri-lang', l); } catch (e) {}
    const url = new URL(location.href);
    if (l === 'el') url.searchParams.set('lang', 'el'); else url.searchParams.delete('lang');
    history.replaceState(null, '', url);
    document.title = l === 'el' ? 'Όγκοι εγκεφάλου σε MRI · XAI' : 'Brain Tumour MRI · XAI';
    updateToolLabels();
    renderers.forEach((f) => f());
  }
  function isDark() {
    const t = root.dataset.theme;
    return t ? t === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  }
  const SCALES = ['1', '1.125', '1.25'];
  function updateToolLabels() {
    $('#btn-lang').setAttribute('aria-label', u('langBtn'));
    $('#btn-theme').setAttribute('aria-label', isDark() ? u('darkOn') : u('darkOff'));
    const paused = motion.paused();
    const bm = $('#btn-motion');
    bm.setAttribute('aria-pressed', paused ? 'true' : 'false');
    bm.setAttribute('aria-label', paused ? u('play') : u('pause'));
    const s = getComputedStyle(root).getPropertyValue('--scale').trim() || '1';
    $('#btn-text').setAttribute('aria-label', `${u('textSize')}: ${Math.round(parseFloat(s) * 100)}%`);
    ['#btn-lang', '#btn-theme', '#btn-motion', '#btn-text'].forEach((id) => $(id).title = $(id).getAttribute('aria-label'));
  }
  $('#btn-lang').addEventListener('click', () => setLang(lang() === 'en' ? 'el' : 'en'));
  $('#btn-theme').addEventListener('click', () => {
    const next = isDark() ? 'light' : 'dark';
    root.dataset.theme = next;
    try { localStorage.setItem('mri-theme', next); } catch (e) {}
    window.BrainBG && window.BrainBG.refresh();
    updateToolLabels();
  });
  $('#btn-text').addEventListener('click', () => {
    const cur = getComputedStyle(root).getPropertyValue('--scale').trim() || '1';
    const next = SCALES[(SCALES.indexOf(cur) + 1) % SCALES.length] || '1';
    root.style.setProperty('--scale', next);
    try { localStorage.setItem('mri-scale', next); } catch (e) {}
    updateToolLabels();
    renderers.forEach((f) => f());
  });
  $('#btn-motion').addEventListener('click', () => {
    const p = !motion.paused();
    window.BrainBG && window.BrainBG.setPaused(p);
    try { localStorage.setItem('mri-motion', p ? 'paused' : 'play'); } catch (e) {}
    if (p) document.querySelectorAll('.pulse').forEach((c) => c.setAttribute('opacity', 0));
    updateToolLabels();
  });
  matchMedia('(prefers-color-scheme: dark)').addEventListener?.('change', () => { window.BrainBG && window.BrainBG.refresh(); updateToolLabels(); });

  // Copy BibTeX
  $('#btn-copy').addEventListener('click', async () => {
    const st = $('#copy-status');
    try { await navigator.clipboard.writeText($('#bibtex').textContent); st.textContent = u('copied'); }
    catch (e) { st.textContent = u('copyFail'); }
    setTimeout(() => { st.textContent = ''; }, 2500);
  });

  // Active section in nav
  const links = [...document.querySelectorAll('.tb-nav a')];
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) links.forEach((a) => {
        const on = a.getAttribute('href') === '#' + e.target.id;
        a.classList.toggle('active', on);
        if (on) a.setAttribute('aria-current', 'true'); else a.removeAttribute('aria-current');
      });
    });
  }, { rootMargin: '-45% 0px -50% 0px' });
  document.querySelectorAll('main section[id]').forEach((s) => io.observe(s));

  /* ───────── deep links: ?model=1-9 &step=1-5 &cls=meningioma &method=shap &xmodel=medvit ───────── */
  (function () {
    const q = new URLSearchParams(location.search);
    const mi = parseInt(q.get('model'), 10); if (mi >= 1 && mi <= 9) archIdx = D.ARCH.findIndex((a) => a.model === mi);
    const st = parseInt(q.get('step'), 10); if (st >= 1 && st <= STEPS.length) stepIdx = st - 1;
    if (D.CLASSES.some((c) => c.id === q.get('cls'))) xCls = q.get('cls');
    if (D.XAI_METHODS.some((m) => m.id === q.get('method'))) xMethod = q.get('method');
    if (D.XAI_MODELS.some((m) => m.id === q.get('xmodel'))) xModel = q.get('xmodel');
    if (q.get('cd') && D.MODELS.some((m) => m.id === q.get('cd'))) cdSel = q.get('cd');
  })();

  /* ───────── boot ───────── */
  updateToolLabels();
  if (lang() === 'el') document.title = 'Όγκοι εγκεφάλου σε MRI · XAI';
  register(renderHero);
  register(renderPipeline);
  register(() => { renderArchTabs(); renderArch(); });
  register(renderPerf);
  register(renderScatter);
  register(renderCM);
  register(renderF1);
  register(renderCI);
  register(renderStepper);
  register(renderXAI);
  register(renderAblation);
  register(renderAppFlow);
})();
