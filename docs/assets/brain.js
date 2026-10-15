/* Animated background: a real sagittal MRI from the thesis (the meningioma case),
   redrawn as a halftone of particles. A sparse neural network links the particles and
   orange signals travel along it; the tumour itself pulses in the meningioma colour.
   Decorative only (aria-hidden). Honours prefers-reduced-motion and the pause toggle. */
(function () {
  const canvas = document.getElementById('brain-bg');
  if (!canvas || !canvas.getContext) return;
  const ctx = canvas.getContext('2d');

  let seed = 11;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646;

  const SRC = 'assets/xai/orig_meningioma_medvit.jpg';
  const TUMOUR = { x: -0.245, y: -0.41, r2: 0.011 }; // meningioma centre in normalised image coords
  const SAMPLE = 7000, NODES = 900;

  /* ── State ── */
  const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
  let paused = mq.matches;
  try { const s = localStorage.getItem('mri-motion'); if (s === 'paused') paused = true; if (s === 'play') paused = false; } catch (e) {}
  let W = 0, H = 0, dpr = 1, time = 0, last = 0, raf = 0;
  let mx = -1e4, my = -1e4, pmx = 0, pmy = 0, tpmx = 0, tpmy = 0;
  const col = { ink: '90,72,200', hot: '235,104,52' };

  let N = 0, px, py, pz, pw, ox, oy, sx, sy;
  let nodes = [], adj = [], edges = [];
  const pulses = [];

  function readColors() {
    const cs = getComputedStyle(document.documentElement);
    const hex = (v) => {
      v = v.trim(); if (!v.startsWith('#')) return null;
      const n = parseInt(v.slice(1), 16);
      return `${(n >> 16) & 255},${(n >> 8) & 255},${n & 255}`;
    };
    col.ink = hex(cs.getPropertyValue('--accent')) || col.ink;
    col.hot = hex(cs.getPropertyValue('--c-meningioma')) || col.hot;
  }

  /* ── Build particles from the MRI ── */
  function build(img) {
    const n = 220, c = document.createElement('canvas'); c.width = c.height = n;
    const g = c.getContext('2d'); g.drawImage(img, 0, 0, n, n);
    const d = g.getImageData(0, 0, n, n).data;
    const L = (x, y) => d[(y * n + x) * 4] / 255;
    const cand = [];
    for (let y = 4; y < n - 4; y++) for (let x = 4; x < n - 4; x++) {
      const v = L(x, y);
      if (v < 0.12) continue;
      // Sobel edge strength brings out gyri and sulci, not just the bright skull.
      const gx = L(x + 1, y - 1) + 2 * L(x + 1, y) + L(x + 1, y + 1) - L(x - 1, y - 1) - 2 * L(x - 1, y) - L(x - 1, y + 1);
      const gy = L(x - 1, y + 1) + 2 * L(x, y + 1) + L(x + 1, y + 1) - L(x - 1, y - 1) - 2 * L(x, y - 1) - L(x + 1, y - 1);
      cand.push(x, y, Math.min(1, 0.45 * v + 1.1 * Math.hypot(gx, gy)));
    }
    // Weighted sampling without replacement (Efraimidis–Spirakis keys).
    const m = cand.length / 3, keys = new Float32Array(m), idx = new Uint32Array(m);
    for (let i = 0; i < m; i++) { keys[i] = Math.pow(rnd(), 1 / Math.pow(cand[i * 3 + 2], 1.8)); idx[i] = i; }
    idx.sort((a, b) => keys[b] - keys[a]);
    N = Math.min(SAMPLE, m);
    px = new Float32Array(N); py = new Float32Array(N); pz = new Float32Array(N); pw = new Float32Array(N);
    ox = new Float32Array(N); oy = new Float32Array(N); sx = new Float32Array(N); sy = new Float32Array(N);
    for (let j = 0; j < N; j++) {
      const i = idx[j];
      px[j] = (cand[i * 3] + rnd() - 0.5 - n / 2) / (n / 2);
      py[j] = (cand[i * 3 + 1] + rnd() - 0.5 - n / 2) / (n / 2);
      pw[j] = cand[i * 3 + 2];
      pz[j] = (rnd() - 0.5) * 0.3 + (pw[j] - 0.5) * 0.35;
    }

    // Network: the strongest particles become neurons, each linked to its nearest neighbours.
    nodes = Array.from({ length: Math.min(NODES, N) }, (_, k) => Math.floor((k * N) / Math.min(NODES, N)));
    adj = nodes.map(() => []);
    const seen = new Set();
    for (let a = 0; a < nodes.length; a++) {
      const best = [];
      for (let b = 0; b < nodes.length; b++) {
        if (a === b) continue;
        const dx = px[nodes[a]] - px[nodes[b]], dy = py[nodes[a]] - py[nodes[b]], dd = dx * dx + dy * dy;
        if (dd > 0.012) continue;
        if (best.length < 3 || dd < best[best.length - 1][0]) { best.push([dd, b]); best.sort((u, v) => u[0] - v[0]); if (best.length > 3) best.pop(); }
      }
      for (const [, b] of best) {
        const key = a < b ? a * 100000 + b : b * 100000 + a;
        if (seen.has(key)) continue;
        seen.add(key); edges.push([a, b]); adj[a].push(b); adj[b].push(a);
      }
    }
    for (let i = 0; i < 10; i++) spawn();
  }

  function spawn() {
    const a = (rnd() * nodes.length) | 0;
    if (!adj[a] || !adj[a].length) return;
    pulses.push({ a, b: adj[a][(rnd() * adj[a].length) | 0], t: 0, hops: 8 + ((rnd() * 14) | 0), trail: [] });
  }

  // Where to draw: wraps the hero MRI grid at the top, settles on the right as you scroll.
  function frame() {
    const wide = W > 900;
    let cx = wide ? W * 0.73 : W * 0.5, cy = wide ? H * 0.5 : H * 0.45;
    let S = Math.min(W, H) * (wide ? 0.36 : 0.42);
    const prog = Math.min(1, (window.scrollY || 0) / (H * 0.8));
    if (wide && prog < 1) {
      const fig = document.getElementById('hero-mri');
      if (fig) {
        const r = fig.getBoundingClientRect();
        const k = 1 - prog, e = k * k * (3 - 2 * k);
        cx += (r.left + r.width / 2 - cx) * e;
        cy += (r.top + r.height / 2 - cy) * e;
        S *= 1 + 0.3 * e;
      }
    }
    return { cx, cy, R: S * 0.95 };
  }

  function draw(dt) {
    ctx.clearRect(0, 0, W, H);
    if (!N) return;
    const { cx, cy, R } = frame();
    pmx += (tpmx - pmx) * 0.05; pmy += (tpmy - pmy) * 0.05;
    const yaw = 0.22 * Math.sin(time * 0.00025) + pmx * 0.3, pitch = 0.08 * Math.sin(time * 0.00017) + pmy * 0.15;
    const cyw = Math.cos(yaw), syw = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    const glow = 0.5 + 0.5 * Math.sin(time * 0.0025);

    // Project particles (with a gentle push away from the pointer).
    for (let i = 0; i < N; i++) {
      const x = px[i], y = py[i], z = pz[i];
      const x1 = x * cyw + z * syw, z1 = -x * syw + z * cyw;
      const y2 = y * cp - z1 * sp, z2 = y * sp + z1 * cp;
      const f = 3 / (3 + z2);
      let X = cx + x1 * R * f, Y = cy + y2 * R * f;
      const dx = X + ox[i] - mx, dy = Y + oy[i] - my, d2 = dx * dx + dy * dy;
      if (d2 < 9000) { const k = (1 - d2 / 9000) * 2.2, d = Math.sqrt(d2) || 1; ox[i] += (dx / d) * k; oy[i] += (dy / d) * k; }
      ox[i] *= 0.93; oy[i] *= 0.93;
      sx[i] = X + ox[i]; sy[i] = Y + oy[i];
    }

    // Synapses.
    const net = new Path2D();
    for (const [a, b] of edges) { const i = nodes[a], j = nodes[b]; net.moveTo(sx[i], sy[i]); net.lineTo(sx[j], sy[j]); }
    ctx.strokeStyle = `rgba(${col.ink},0.22)`; ctx.lineWidth = 0.75; ctx.stroke(net);

    // Particles in four brightness buckets; tumour particles pulse in the hot colour.
    const buckets = [[], [], [], []], hot = [];
    for (let i = 0; i < N; i++) {
      const tx = px[i] - TUMOUR.x, ty = py[i] - TUMOUR.y, td = tx * tx + ty * ty;
      if (td < TUMOUR.r2) hot.push(i, 0.45 + 0.45 * glow * (1 - td / TUMOUR.r2));
      else buckets[Math.min(3, (pw[i] * 4) | 0)].push(i);
    }
    buckets.forEach((b, k) => {
      ctx.fillStyle = `rgba(${col.ink},${(0.3 + k * 0.22).toFixed(2)})`;
      const s = 1.1 + k * 0.3;
      for (const i of b) ctx.fillRect(sx[i] - s / 2, sy[i] - s / 2, s, s);
    });
    for (let j = 0; j < hot.length; j += 2) {
      const i = hot[j], k = hot[j + 1], s = 1.5 + 1.4 * k;
      ctx.fillStyle = `rgba(${col.hot},${(0.35 + 0.6 * k).toFixed(2)})`;
      ctx.fillRect(sx[i] - s / 2, sy[i] - s / 2, s, s);
    }

    // Signals travelling along the network.
    if (!paused) {
      if (pulses.length < 16 && rnd() < 0.1) spawn();
      for (let k = pulses.length - 1; k >= 0; k--) {
        const p = pulses[k];
        p.t += dt * 0.006;
        if (p.t >= 1) {
          p.trail.push([p.a, p.b]); if (p.trail.length > 5) p.trail.shift();
          if (--p.hops <= 0) { pulses.splice(k, 1); continue; }
          const nb = adj[p.b].filter((q) => q !== p.a);
          p.a = p.b; p.b = nb.length ? nb[(rnd() * nb.length) | 0] : adj[p.b][0]; p.t = 0;
        }
      }
    }
    for (const p of pulses) {
      ctx.lineWidth = 1.5;
      p.trail.forEach(([a, b], idx) => {
        ctx.strokeStyle = `rgba(${col.hot},${(0.12 + idx * 0.1).toFixed(2)})`;
        ctx.beginPath(); ctx.moveTo(sx[nodes[a]], sy[nodes[a]]); ctx.lineTo(sx[nodes[b]], sy[nodes[b]]); ctx.stroke();
      });
      const A = nodes[p.a], B = nodes[p.b];
      const x = sx[A] + (sx[B] - sx[A]) * p.t, y = sy[A] + (sy[B] - sy[A]) * p.t;
      const g = ctx.createRadialGradient(x, y, 0, x, y, 7);
      g.addColorStop(0, `rgba(${col.hot},0.9)`); g.addColorStop(1, `rgba(${col.hot},0)`);
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, 7, 0, Math.PI * 2); ctx.fill();
    }
  }

  function loop(ts) {
    const dt = last ? Math.min(ts - last, 50) : 16; last = ts;
    time += dt; draw(dt);
    raf = requestAnimationFrame(loop);
  }
  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = window.innerWidth; H = window.innerHeight;
    canvas.width = W * dpr; canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (paused) draw(0);
  }
  function setPaused(v) {
    paused = v; cancelAnimationFrame(raf); last = 0;
    if (paused) draw(0); else raf = requestAnimationFrame(loop);
  }

  window.addEventListener('resize', resize);
  window.addEventListener('pointermove', (e) => {
    mx = e.clientX; my = e.clientY;
    tpmx = e.clientX / window.innerWidth - 0.5; tpmy = e.clientY / window.innerHeight - 0.5;
  }, { passive: true });
  document.addEventListener('pointerleave', () => { mx = my = -1e4; });
  window.addEventListener('scroll', () => { if (paused) draw(0); }, { passive: true });
  mq.addEventListener && mq.addEventListener('change', (e) => setPaused(e.matches));

  readColors(); resize();
  const img = new Image();
  img.onload = () => { build(img); if (paused) draw(0); else raf = requestAnimationFrame(loop); };
  img.src = SRC;

  window.BrainBG = {
    isPaused: () => paused,
    setPaused,
    refresh() { readColors(); if (paused) draw(0); },
  };
})();
