/* In-browser demo: the thesis 4-Class CNN running locally with TensorFlow.js,
   with Grad-CAM from the last convolutional layer (conv4_2).
   The model is rebuilt layer by layer from exported Keras weights (assets/model/cnn4.*).
   Nothing is uploaded: images are processed on the visitor's device. */
(function () {
  'use strict';
  const TFJS = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
  const CAM_LAYER = 'conv4_2';
  const $ = (s) => document.querySelector(s);
  const lang = () => document.documentElement.dataset.lang || 'en';
  const L = (en, el) => (lang() === 'el' ? el : en);
  const CLASSES = window.DATA.CLASSES; // order matches the model: glioma, meningioma, no tumour, pituitary

  const startBtn = $('#demo-start');
  if (!startBtn) return;
  const status = $('#demo-status');
  let features = null, head = null, last = null, busy = false;

  function loadScript(src) {
    return new Promise((res, rej) => {
      if (window.tf) return res();
      const s = document.createElement('script');
      s.src = src; s.onload = res; s.onerror = () => rej(new Error('TensorFlow.js failed to load'));
      document.head.appendChild(s);
    });
  }

  function makeLayer(e, inputShape, linear) {
    const tf = window.tf, o = inputShape ? { inputShape } : {};
    switch (e.cls) {
      case 'Conv2D': return tf.layers.conv2d(Object.assign(o, { name: e.name, filters: e.filters, kernelSize: e.kernel_size, strides: e.strides, padding: e.padding, activation: e.activation }));
      case 'BatchNormalization': return tf.layers.batchNormalization(Object.assign(o, { name: e.name, epsilon: e.epsilon }));
      case 'MaxPooling2D': return tf.layers.maxPooling2d(Object.assign(o, { name: e.name, poolSize: e.pool_size, strides: e.strides || e.pool_size }));
      case 'GlobalAveragePooling2D': return tf.layers.globalAveragePooling2d(Object.assign(o, { name: e.name }));
      case 'Dense': return tf.layers.dense(Object.assign(o, { name: e.name, units: e.units, activation: linear ? 'linear' : e.activation }));
      default: return null; // Dropout and InputLayer are identity at inference
    }
  }

  function build(spec, buf) {
    const tf = window.tf;
    const all = spec.layers.filter((e) => !['Dropout', 'InputLayer'].includes(e.cls));
    const cut = all.findIndex((e) => e.name === CAM_LAYER) + 1;
    const mk = (list, inShape, linearLast) => {
      const m = tf.sequential();
      list.forEach((e, i) => m.add(makeLayer(e, i === 0 ? inShape : null, linearLast && i === list.length - 1)));
      list.forEach((e, i) => {
        if (!e.weights.length) return;
        m.layers[i].setWeights(e.weights.map((w) => {
          const n = w.shape.reduce((a, b) => a * b, 1);
          return tf.tensor(buf.subarray(w.offset, w.offset + n), w.shape);
        }));
      });
      return m;
    };
    features = mk(all.slice(0, cut), spec.input, false);
    const fShape = features.outputs[0].shape.slice(1);
    head = mk(all.slice(cut), fShape, true); // logits; softmax applied separately
  }

  async function start() {
    startBtn.disabled = true;
    try {
      status.textContent = L('Loading TensorFlow.js…', 'Φόρτωση TensorFlow.js…');
      await loadScript(TFJS);
      status.textContent = L('Downloading model weights (5 MB)…', 'Λήψη βαρών μοντέλου (5 MB)…');
      const [spec, bin] = await Promise.all([
        fetch('assets/model/cnn4.json').then((r) => r.json()),
        fetch('assets/model/cnn4.bin').then((r) => r.arrayBuffer()),
      ]);
      build(spec, new Float32Array(bin));
      // Warm-up so the first real prediction is fast.
      window.tf.tidy(() => head.predict(features.predict(window.tf.zeros([1, 224, 224, 3]))));
      status.textContent = L(`Model ready · ${window.tf.getBackend()} backend`, `Το μοντέλο είναι έτοιμο · backend ${window.tf.getBackend()}`);
      $('#demo-body').hidden = false;
      startBtn.hidden = true;
      renderSamples();
      run('assets/xai/orig_meningioma_medvit.jpg', 'meningioma');
    } catch (err) {
      status.textContent = L('Could not start the demo: ', 'Δεν ήταν δυνατή η εκκίνηση: ') + err.message;
      startBtn.disabled = false;
    }
  }

  function renderSamples() {
    const box = $('#demo-samples');
    box.replaceChildren(...CLASSES.map((c) => {
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'demo-sample'; b.dataset.cls = c.id;
      b.setAttribute('aria-label', L(`Test image: ${c.en}`, `Εικόνα ελέγχου: ${c.el}`));
      b.innerHTML = `<img src="assets/xai/orig_${c.id}_medvit.jpg" alt="" width="72" height="72"><span><i style="background:var(${c.color})"></i>${c[lang()]}</span>`;
      b.addEventListener('click', () => run(`assets/xai/orig_${c.id}_medvit.jpg`, c.id));
      return b;
    }));
  }

  function loadImage(src) {
    return new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = src; });
  }

  // Jet-like colour map for the heatmap.
  function jet(t) {
    const r = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 3)));
    const g = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 2)));
    const b = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 1)));
    return [r * 255, g * 255, b * 255];
  }

  async function run(src, truth) {
    if (busy || !features) return;
    busy = true;
    document.querySelectorAll('.demo-sample').forEach((b) => b.setAttribute('aria-pressed', b.dataset.cls === truth ? 'true' : 'false'));
    const tf = window.tf;
    try {
      const img = await loadImage(src);
      const cIn = $('#demo-img'), gIn = cIn.getContext('2d');
      gIn.drawImage(img, 0, 0, 224, 224);
      const t0 = performance.now();
      const x = tf.tidy(() => tf.browser.fromPixels(cIn).toFloat().div(255).expandDims(0));
      const A = features.predict(x);
      const logits = head.predict(A);
      const probs = Array.from(await tf.softmax(logits).data());
      const k = probs.indexOf(Math.max(...probs));
      const ms = performance.now() - t0;
      // Grad-CAM: gradient of the predicted logit with respect to the conv4_2 activations.
      const cam = tf.tidy(() => {
        const grad = tf.grad((a) => head.apply(a).gather([k], 1).sum())(A);
        const w = grad.mean([1, 2]);                       // [1, C]
        const m = tf.relu(A.mul(w.reshape([1, 1, 1, -1])).sum(-1)); // [1, h, w]
        const up = tf.image.resizeBilinear(m.expandDims(-1), [224, 224]).squeeze();
        const mx = up.max();
        return up.div(tf.maximum(mx, 1e-8));
      });
      const camData = await cam.data();
      tf.dispose([x, A, logits, cam]);

      const cOut = $('#demo-cam'), g = cOut.getContext('2d');
      g.drawImage(img, 0, 0, 224, 224);
      const base = g.getImageData(0, 0, 224, 224);
      for (let i = 0; i < 224 * 224; i++) {
        const t = camData[i], [r, gg, b] = jet(t), a = 0.25 + 0.4 * t;
        base.data[i * 4] = base.data[i * 4] * (1 - a) + r * a;
        base.data[i * 4 + 1] = base.data[i * 4 + 1] * (1 - a) + gg * a;
        base.data[i * 4 + 2] = base.data[i * 4 + 2] * (1 - a) + b * a;
      }
      g.putImageData(base, 0, 0);
      last = { probs, k, ms, truth };
      renderResult();
    } catch (err) {
      $('#demo-result').textContent = L('Could not read this image.', 'Δεν ήταν δυνατή η ανάγνωση της εικόνας.');
    } finally { busy = false; }
  }

  function renderResult() {
    if (!last) return;
    const { probs, k, ms, truth } = last, c = CLASSES[k];
    const fmt = (v) => (v * 100).toFixed(v > 0.999 || v < 0.001 ? 2 : 1).replace('.', lang() === 'el' ? ',' : '.');
    const truthC = CLASSES.find((x) => x.id === truth);
    $('#demo-result').innerHTML = `
      <p class="ctl-label">${L('Prediction', 'Πρόβλεψη')}</p>
      <p class="demo-pred" style="color:var(${c.color})">${c[lang()]}</p>
      <p class="demo-conf">${fmt(probs[k])}% ${L('confidence', 'βεβαιότητα')} · ${ms.toFixed(0)} ms</p>
      ${truthC ? `<p class="demo-truth">${L('Ground truth', 'Πραγματική κλάση')}: <b>${truthC[lang()]}</b> ${truthC === c ? '✓' : '✗'}</p>` : ''}
      <ul class="demo-bars">${CLASSES.map((x, i) => `
        <li><span class="lbl">${x[lang()]}</span><span class="bar"><i style="width:${(probs[i] * 100).toFixed(1)}%;background:var(${x.color})"></i></span><span class="v">${fmt(probs[i])}%</span></li>`).join('')}
      </ul>`;
  }

  startBtn.addEventListener('click', start);
  $('#demo-file').addEventListener('change', (e) => {
    const f = e.target.files && e.target.files[0];
    if (!f) return;
    const url = URL.createObjectURL(f);
    run(url, null).finally(() => URL.revokeObjectURL(url));
  });
  // Drag and drop onto the demo card.
  const card = $('#demo');
  card.addEventListener('dragover', (e) => { if (features) { e.preventDefault(); card.classList.add('drop'); } });
  card.addEventListener('dragleave', () => card.classList.remove('drop'));
  card.addEventListener('drop', (e) => {
    card.classList.remove('drop');
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (!features || !f || !f.type.startsWith('image/')) return;
    e.preventDefault();
    const url = URL.createObjectURL(f);
    run(url, null).finally(() => URL.revokeObjectURL(url));
  });
  // Re-label when the language changes.
  new MutationObserver(() => { if (features) { renderSamples(); renderResult(); } })
    .observe(document.documentElement, { attributes: true, attributeFilter: ['data-lang'] });

  window.Demo = { run, ready: () => !!features, last: () => last };
})();
