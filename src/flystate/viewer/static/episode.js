import { predictionCards } from './identities.js';
import { el, heatColor, number, raw } from './charts.js';

const dialog = document.querySelector('#episode');
const content = document.querySelector('#episode-content');
let active, timer;
export function closeEpisode() {
  active?.abort();
  clearInterval(timer);
  timer = null;
  if (dialog.open) dialog.close();
}
dialog.addEventListener('close', () => {
  active?.abort();
  clearInterval(timer);
  timer = null;
});
async function request(path, signal) {
  const response = await fetch(`/api/${path}`, { signal, cache: 'no-store' });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail);
  return data;
}
export async function openEpisode(runId, evalId, sampleId, initialStep, identities) {
  closeEpisode();
  active = new AbortController();
  const { signal } = active;
  const close = el('button', { 'aria-label': 'Close episode', onclick: closeEpisode }, 'Close ×');
  const body = el('div', {}, el('div', { class: 'loading' }, 'Reading one verified trace chunk…'));
  content.replaceChildren(
    el(
      'div',
      { class: 'dialog-top' },
      el('div', {}, el('div', { class: 'eyebrow' }, 'EPISODE INSPECTOR'), el('h2', {}, sampleId)),
      close,
    ),
    body,
  );
  dialog.showModal();
  const base = `runs/${encodeURIComponent(runId)}`;
  try {
    const [data, predictions] = await Promise.all([
      request(`${base}/samples/${encodeURIComponent(sampleId)}`, signal),
      request(
        `${base}/evaluations/${encodeURIComponent(evalId)}/predictions?sample_id=${encodeURIComponent(sampleId)}&limit=200`,
        signal,
      ),
    ]);
    if (signal.aborted) return;
    const stimulus = el('canvas', {
      width: 256,
      height: 256,
      class: 'episode-image',
      role: 'img',
      'aria-label': 'Aligned image and observation windows',
    });
    const status = el('p', { class: 'readout-note' }, 'Loading aligned image…');
    const image = new Image();
    let imageReady = false;
    image.onload = () => {
      if (signal.aborted) return;
      imageReady = true;
      status.textContent =
        'The network sees one highlighted image window per step. The line shows the sequence of windows.';
      draw();
    };
    image.onerror = () => {
      if (!signal.aborted)
        status.textContent = 'Aligned image unavailable. Trace and trajectory remain available.';
    };
    image.src = `/api/${base}/samples/${encodeURIComponent(sampleId)}/image`;
    const slider = el('input', {
      type: 'range',
      min: 1,
      max: data.boxes.length,
      value: Math.min(initialStep, data.boxes.length),
      'aria-label': 'Episode observation',
    });
    const stepLabel = el('strong'),
      play = el('button', {}, 'Play'),
      probs = el('div', { class: 'top-probs' });
    const feature = el(
      'select',
      { 'aria-label': 'Neural feature' },
      ...Object.keys(data.activity).map((k) => el('option', { value: k }, k.replaceAll('_', ' '))),
    );
    const heatmap = el('canvas', {
      class: 'heatmap',
      width: data.activity[feature.value][0].length,
      height: data.boxes.length,
      role: 'img',
      'aria-label': 'Neural feature heatmap: rows are observations, columns are readout neurons',
    });
    const heatHint = el('p', { class: 'canvas-hint' });
    const spatial = el('canvas', {
      class: 'neuron-canvas',
      width: 700,
      height: 310,
      role: 'img',
      'aria-label': 'Rotatable 3D projection of readout neurons colored by current feature value',
    });
    const spatialHint = el(
      'p',
      { class: 'canvas-hint' },
      'Drag to rotate · scroll to zoom · hover to inspect a neuron.',
    );
    let yaw = 0.2,
      pitch = 0.1,
      zoom = 1,
      drag = null,
      projected = [];
    const positions = data.neurons.available ? data.neurons.positions : [];
    const center = [0, 1, 2].map(
      (axis) => positions.reduce((sum, p) => sum + p[axis], 0) / (positions.length || 1),
    );
    const extent = Math.max(
      1,
      ...positions.flatMap((p) => p.map((v, i) => Math.abs(v - center[i]))),
    );
    function range() {
      const values = data.activity[feature.value].flat();
      return [Math.min(...values), Math.max(...values)];
    }
    function draw() {
      const t = Number(slider.value) - 1,
        ctx = stimulus.getContext('2d');
      ctx.fillStyle = '#193642';
      ctx.fillRect(0, 0, 256, 256);
      if (imageReady) ctx.drawImage(image, 0, 0, 256, 256);
      const scale = 256 / data.image_size;
      ctx.strokeStyle = '#f8d277';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      data.boxes.slice(0, t + 1).forEach(([x0, y0, x1, y1], i) => {
        const x = ((x0 + x1) / 2) * scale,
          y = ((y0 + y1) / 2) * scale;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      const [x0, y0, x1, y1] = data.boxes[t];
      ctx.strokeStyle = '#7affd7';
      ctx.lineWidth = 3;
      ctx.strokeRect(x0 * scale, y0 * scale, (x1 - x0) * scale, (y1 - y0) * scale);
      stepLabel.textContent = `${t + 1} / ${data.boxes.length}`;
      const p = predictions.rows.find((row) => row.t === t + 1);
      probs.replaceChildren(
        p
          ? predictionCards(runId, p, identities, data.boxes.length)
          : el('p', {}, 'No prediction recorded at this observation.'),
      );
      drawNeurons();
    }
    function drawHeat() {
      const values = data.activity[feature.value],
        ctx = heatmap.getContext('2d'),
        [low, high] = range();
      values.forEach((row, t) =>
        row.forEach((v, n) => {
          ctx.fillStyle = heatColor((v - low) / (high - low || 1));
          ctx.fillRect(n, t, 1, 1);
        }),
      );
      heatHint.textContent = `${values[0].length} readout neurons × ${values.length} observations. Color range ${number(low, 4)}–${number(high, 4)} (${feature.value}, model units); fixed across time. Click a row to select it.`;
    }
    function drawNeurons() {
      const ctx = spatial.getContext('2d'),
        t = Number(slider.value) - 1,
        [low, high] = range();
      ctx.fillStyle = '#152e3b';
      ctx.fillRect(0, 0, 700, 310);
      if (!positions.length) {
        ctx.fillStyle = '#c6dedf';
        ctx.font = '14px sans-serif';
        ctx.fillText('Matching neuron geometry unavailable', 25, 155);
        return;
      }
      projected = positions
        .map((p, i) => {
          const [x, y, z] = p.map((v, a) => (v - center[a]) / extent);
          const rx = x * Math.cos(yaw) + z * Math.sin(yaw),
            rz = -x * Math.sin(yaw) + z * Math.cos(yaw);
          const ry = y * Math.cos(pitch) - rz * Math.sin(pitch),
            depth = y * Math.sin(pitch) + rz * Math.cos(pitch);
          return {
            x: 350 + rx * 130 * zoom,
            y: 155 + ry * 130 * zoom,
            z: depth,
            i: data.neurons.feature_indices[i],
          };
        })
        .sort((a, b) => a.z - b.z);
      projected.forEach((p) => {
        ctx.beginPath();
        ctx.fillStyle = heatColor((data.activity[feature.value][t][p.i] - low) / (high - low || 1));
        ctx.arc(p.x, p.y, 2.4, 0, Math.PI * 2);
        ctx.fill();
      });
      ctx.fillStyle = '#b4d2da';
      ctx.font = '11px sans-serif';
      ctx.fillText(
        `${data.neurons.population} · ${positions.length} neurons · observation ${t + 1}`,
        16,
        24,
      );
    }
    heatmap.onpointermove = (event) => {
      const rect = heatmap.getBoundingClientRect();
      const t = Math.min(
        data.boxes.length - 1,
        Math.floor(((event.clientY - rect.top) / rect.height) * data.boxes.length),
      );
      const n = Math.min(
        heatmap.width - 1,
        Math.floor(((event.clientX - rect.left) / rect.width) * heatmap.width),
      );
      heatHint.textContent = `Observation ${t + 1}, neuron ${data.neurons.ids?.[n] || `index ${n}`}: ${number(data.activity[feature.value][t][n], 6)} (${feature.value})`;
    };
    heatmap.onclick = (event) => {
      const rect = heatmap.getBoundingClientRect();
      slider.value = String(
        Math.min(
          data.boxes.length,
          1 + Math.floor(((event.clientY - rect.top) / rect.height) * data.boxes.length),
        ),
      );
      draw();
    };
    spatial.onpointerdown = (e) => {
      drag = [e.clientX, e.clientY];
      spatial.setPointerCapture(e.pointerId);
    };
    spatial.onpointerup = () => {
      drag = null;
    };
    spatial.onpointercancel = () => {
      drag = null;
    };
    spatial.onpointermove = (e) => {
      if (drag) {
        yaw += (e.clientX - drag[0]) * 0.008;
        pitch += (e.clientY - drag[1]) * 0.008;
        drag = [e.clientX, e.clientY];
        drawNeurons();
        return;
      }
      const rect = spatial.getBoundingClientRect(),
        x = ((e.clientX - rect.left) / rect.width) * 700,
        y = ((e.clientY - rect.top) / rect.height) * 310;
      const nearest = projected.reduce(
        (best, p) =>
          Math.hypot(p.x - x, p.y - y) < (best?.distance ?? 12)
            ? { ...p, distance: Math.hypot(p.x - x, p.y - y) }
            : best,
        null,
      );
      if (nearest)
        spatialHint.textContent = `Neuron ${data.neurons.ids[nearest.i]} · ${feature.value}: ${number(data.activity[feature.value][Number(slider.value) - 1][nearest.i], 6)}`;
    };
    spatial.addEventListener(
      'wheel',
      (e) => {
        e.preventDefault();
        zoom = Math.max(0.5, Math.min(4, zoom * Math.exp(-e.deltaY * 0.001)));
        drawNeurons();
      },
      { passive: false },
    );
    slider.oninput = draw;
    feature.onchange = () => {
      drawHeat();
      draw();
    };
    play.onclick = () => {
      if (timer) {
        clearInterval(timer);
        timer = null;
        play.textContent = 'Play';
      } else {
        play.textContent = 'Pause';
        timer = setInterval(() => {
          slider.value = String((Number(slider.value) % data.boxes.length) + 1);
          draw();
        }, 550);
      }
    };
    body.replaceChildren(
      el(
        'div',
        { class: 'episode-grid' },
        el(
          'section',
          {},
          el('h3', {}, 'Source image'),
          stimulus,
          status,
          el('div', { class: 'episode-controls' }, play, slider, stepLabel),
        ),
        probs,
      ),
      el(
        'section',
        {},
        el('div', { class: 'toolbar' }, el('h3', {}, 'Readout activity'), feature),
        heatmap,
        heatHint,
        spatial,
        spatialHint,
        el(
          'p',
          { class: 'readout-note' },
          'The spatial view shows cached readout neurons, not whole-brain activity. Geometry is shown only when its brain-file hash matches the run.',
        ),
        data.neurons.available
          ? el(
              'p',
              { class: 'readout-note' },
              `${positions.length} neurons positioned; ${data.neurons.missing_positions} missing coordinates. All feature columns remain in the heatmap.`,
            )
          : el('p', { class: 'readout-note' }, data.neurons.reason),
      ),
      raw(
        {
          sample: data.sample,
          boxes: data.boxes,
          summary: data.summary,
          brain_ms: data.brain_ms,
        },
        'Episode metadata & spike summaries',
      ),
    );
    drawHeat();
    draw();
  } catch (error) {
    if (error.name !== 'AbortError')
      body.replaceChildren(el('div', { class: 'error', role: 'alert' }, error.message));
  }
}
