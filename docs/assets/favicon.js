/* Animated favicon: a small neural-network brain rotating in the browser tab.
   Drawn on a 64×64 canvas and pushed to the <link rel="icon"> about 12 times a second.
   Falls back to the static SVG icon in browsers that ignore favicon updates (Safari).
   Stops when motion is paused or reduced. */
(function () {
  const link = document.getElementById('favicon');
  if (!link) return;
  const S = 64;
  const c = document.createElement('canvas');
  c.width = c.height = S;
  const g = c.getContext('2d');
  if (!g) return;

  // Nodes on two hemispheres plus a cerebellum, so the silhouette reads as a brain while it turns.
  let seed = 5;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646;
  const P = [];
  for (const side of [-1, 1]) {
    for (let i = 0; i < 15; i++) {
      const u = rnd() * 2 - 1, t = rnd() * Math.PI * 2, s = Math.sqrt(1 - u * u);
      let x = s * Math.cos(t), y = u, z = s * Math.sin(t);
      if (x * side < 0) x *= 0.25;                       // flat medial wall
      P.push([side * 0.36 + x * 0.36, y * 0.6 + 0.08, z * 0.9]);
    }
  }
  for (let i = 0; i < 4; i++) {
    const a = rnd() * Math.PI * 2;
    P.push([Math.cos(a) * 0.32, -0.55 + rnd() * 0.12, -0.5 + Math.sin(a) * 0.18]);
  }
  const E = [];
  P.forEach((p, i) => {
    const d = P.map((q, j) => [(p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2, j])
      .filter(([, j]) => j !== i).sort((a, b) => a[0] - b[0]);
    for (let k = 0; k < 2; k++) if (d[k][1] > i || !E.some(([a, b]) => a === d[k][1] && b === i)) E.push([i, d[k][1]]);
  });

  const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
  let angle = 0.6, timer = 0, pulse = { e: 0, t: 0 };

  function draw() {
    g.clearRect(0, 0, S, S);
    // Violet disc so the icon reads on both light and dark tab bars.
    g.fillStyle = '#5a48c8';
    g.beginPath(); g.arc(S / 2, S / 2, S / 2 - 1, 0, Math.PI * 2); g.fill();

    const ca = Math.cos(angle), sa = Math.sin(angle), tilt = -0.35, ct = Math.cos(tilt), st = Math.sin(tilt);
    const pr = P.map(([x, y, z]) => {
      const x1 = x * ca + z * sa, z1 = -x * sa + z * ca;
      const y2 = y * ct - z1 * st, z2 = y * st + z1 * ct;
      const f = 2.6 / (2.6 + z2);
      return [S / 2 + x1 * S * 0.43 * f, S / 2 - y2 * S * 0.43 * f, z2];
    });

    g.lineWidth = 2;
    E.forEach(([a, b]) => {
      const z = (pr[a][2] + pr[b][2]) / 2;
      g.strokeStyle = `rgba(255,255,255,${(0.35 + 0.4 * (0.5 - z / 2)).toFixed(2)})`;
      g.beginPath(); g.moveTo(pr[a][0], pr[a][1]); g.lineTo(pr[b][0], pr[b][1]); g.stroke();
    });
    pr.forEach(([x, y, z]) => {
      g.fillStyle = `rgba(255,255,255,${(0.6 + 0.4 * (0.5 - z / 2)).toFixed(2)})`;
      g.beginPath(); g.arc(x, y, 3, 0, Math.PI * 2); g.fill();
    });
    // One orange signal travelling along the network.
    const [a, b] = E[pulse.e];
    const x = pr[a][0] + (pr[b][0] - pr[a][0]) * pulse.t, y = pr[a][1] + (pr[b][1] - pr[a][1]) * pulse.t;
    g.fillStyle = '#ff8a4c';
    g.beginPath(); g.arc(x, y, 5, 0, Math.PI * 2); g.fill();

    link.type = 'image/png';
    link.href = c.toDataURL('image/png');
  }

  function tick() {
    angle += 0.14;
    pulse.t += 0.34;
    if (pulse.t >= 1) {
      const end = E[pulse.e][1];
      const next = E.map((e, i) => [e, i]).filter(([e]) => e[0] === end || e[1] === end);
      const pick = next[(rnd() * next.length) | 0];
      pulse = { e: pick ? pick[1] : (rnd() * E.length) | 0, t: 0 };
      if (pick && pick[0][1] === end) E[pulse.e] = [pick[0][1], pick[0][0]]; // travel away from the current node
    }
    draw();
  }

  const paused = () => mq.matches || (window.BrainBG && window.BrainBG.isPaused());
  function sync() {
    if (paused()) { clearInterval(timer); timer = 0; draw(); }
    else if (!timer) timer = setInterval(tick, 85);
  }
  sync();
  // Follow the page's pause button and the OS motion setting.
  document.getElementById('btn-motion')?.addEventListener('click', () => setTimeout(sync, 0));
  mq.addEventListener && mq.addEventListener('change', sync);
})();
