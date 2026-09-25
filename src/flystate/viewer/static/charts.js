// Dependency-free chart primitives. Artifact strings only enter text nodes.
export const colors = ['#087f79', '#c97538', '#527cc0', '#9668ae', '#7a9745', '#b15367'];
export const modeColors = {
  persistent: colors[0],
  reset: colors[1],
  reset_concat: colors[2],
};
export const pct = (value) => (value == null ? '—' : `${(value * 100).toFixed(2)}%`);
export const number = (value, digits = 2) =>
  value == null ? '—' : Number(value).toLocaleString('en-US', { maximumFractionDigits: digits });
export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else if (value != null) node.setAttribute(key, value);
  }
  for (const child of children.flat())
    if (child != null)
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  return node;
}
function svg(tag, attrs = {}, text = '') {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  node.textContent = text;
  return node;
}
export function download(name, content, type) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = el('a', { href: url, download: name });
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export function raw(data, label = 'Artifact data & provenance') {
  const text = JSON.stringify(data, null, 2);
  return el(
    'details',
    {},
    el('summary', {}, label),
    el('pre', {}, text),
    el(
      'button',
      {
        onclick: () => download('flystate-artifact.json', text, 'application/json'),
      },
      'Download JSON',
    ),
  );
}
export function table(headers, rows) {
  return el(
    'div',
    { class: 'table-wrap' },
    el(
      'table',
      {},
      el(
        'thead',
        {},
        el(
          'tr',
          {},
          // A header is plain content, or {content, attrs} when the cell needs attributes.
          headers.map((h) =>
            h?.content === undefined
              ? el('th', { scope: 'col' }, h)
              : el('th', { scope: 'col', ...h.attrs }, h.content),
          ),
        ),
      ),
      el(
        'tbody',
        {},
        rows.map((row) =>
          el(
            'tr',
            {},
            row.map((cell) => el('td', {}, cell)),
          ),
        ),
      ),
    ),
  );
}
export function card(title, subtitle, ...body) {
  return el(
    'section',
    { class: 'card' },
    el(
      'div',
      { class: 'card-head' },
      el('div', {}, el('h2', {}, title), subtitle ? el('p', {}, subtitle) : null),
    ),
    ...body,
  );
}
export function stats(items) {
  return el(
    'div',
    { class: 'stats' },
    items.map(([name, value, note]) =>
      el(
        'div',
        { class: 'stat' },
        el('div', { class: 'stat-label' }, name),
        el('div', { class: 'stat-value metric-number' }, value),
        el('div', { class: 'stat-note' }, note),
      ),
    ),
  );
}
export function bars(rows, format = number) {
  const max = Math.max(0, ...rows.map((r) => r.value)) || 1;
  return el(
    'div',
    {},
    rows.map((row, i) =>
      el(
        'div',
        { class: 'bar-row' },
        el('span', {}, row.label),
        el(
          'div',
          { class: 'bar-track' },
          el('div', {
            class: 'bar-fill',
            style: `width:${Math.max(0, (row.value / max) * 100)}%;background:${colors[i % colors.length]}`,
          }),
        ),
        el('strong', { class: 'metric-number' }, format(row.value)),
      ),
    ),
  );
}
// Values are in the displayed units; bounds share the same scale as the curve.
export function lineChart(
  series,
  { label = 'Accuracy (%)', baseline = null, xLabel = 'Observation', unit = '%' } = {},
) {
  const usable = series.filter((s) => s.points.length);
  if (!usable.length) return el('div', { class: 'empty' }, 'No recorded curve is available.');
  const points = usable.flatMap((s) => s.points);
  const minX = Math.min(...points.map((p) => p.x)),
    maxX = Math.max(...points.map((p) => p.x));
  const values = points.flatMap((p) => [p.y, p.low ?? p.y, p.high ?? p.y]);
  const minY = Math.min(0, ...values, baseline ?? 0);
  let maxY = Math.max(...values, baseline ?? 0);
  maxY = maxY > minY ? maxY + (maxY - minY) * 0.12 : minY + 1;
  const x = (value) => 58 + ((value - minX) / (maxX - minX || 1)) * 700;
  const y = (value) => 264 - ((value - minY) / (maxY - minY)) * 216;
  const plot = svg('svg', {
    viewBox: '0 0 790 310',
    class: 'chart',
    role: 'img',
    'aria-label': label,
  });
  plot.append(
    svg('title', {}, `${label}. Interactive point tooltips; exact values in the table below.`),
  );
  for (let i = 0; i <= 4; i++) {
    const v = minY + ((maxY - minY) * i) / 4;
    plot.append(svg('line', { x1: 58, x2: 758, y1: y(v), y2: y(v), stroke: '#e8eeee' }));
    plot.append(
      svg('text', { x: 47, y: y(v) + 4, 'text-anchor': 'end' }, `${number(v, 2)}${unit}`),
    );
  }
  const ticks = [...new Set(points.map((p) => p.x))].sort((a, b) => a - b);
  ticks.forEach((v, i) => {
    if (ticks.length <= 18 || i % Math.ceil(ticks.length / 12) === 0 || i === ticks.length - 1)
      plot.append(svg('text', { x: x(v), y: 286, 'text-anchor': 'middle' }, number(v)));
  });
  plot.append(
    svg('text', { x: 58, y: 22 }, label),
    svg('text', { x: 758, y: 308, 'text-anchor': 'end' }, xLabel),
  );
  if (baseline !== null)
    plot.append(
      svg('line', {
        x1: 58,
        x2: 758,
        y1: y(baseline),
        y2: y(baseline),
        stroke: '#81939b',
        'stroke-dasharray': '5 5',
      }),
    );
  usable.forEach((s, i) => {
    const color = s.color || colors[i % colors.length];
    const ordered = [...s.points].sort((a, b) => a.x - b.x);
    if (ordered.every((p) => p.low != null && p.high != null)) {
      plot.append(
        svg('polygon', {
          points: [
            ...ordered.map((p) => `${x(p.x)},${y(p.low)}`),
            ...[...ordered].reverse().map((p) => `${x(p.x)},${y(p.high)}`),
          ].join(' '),
          fill: color,
          opacity: '.10',
        }),
      );
    }
    plot.append(
      svg('polyline', {
        points: ordered.map((p) => `${x(p.x)},${y(p.y)}`).join(' '),
        fill: 'none',
        stroke: color,
        'stroke-width': '2.4',
        'stroke-linejoin': 'round',
      }),
    );
    for (const p of ordered) {
      const point = svg('circle', {
        cx: x(p.x),
        cy: y(p.y),
        r: 3.8,
        fill: color,
        stroke: 'white',
        'stroke-width': 1.3,
        tabindex: 0,
      });
      const detail = `${s.name} · ${xLabel} ${p.x}: ${number(p.y, 4)}${unit}${p.low != null ? ` [${number(p.low, 4)}, ${number(p.high, 4)}]` : ''}`;
      point.append(svg('title', {}, detail));
      point.setAttribute('aria-label', detail);
      plot.append(point);
    }
  });
  const legend = el(
    'div',
    { class: 'legend' },
    usable.map((s, i) =>
      el(
        'span',
        {},
        el('i', {
          style: `background:${s.color || colors[i % colors.length]}`,
        }),
        s.name,
      ),
    ),
  );
  if (baseline !== null)
    legend.append(el('span', {}, `Dashed reference: ${number(baseline)}${unit}`));
  legend.append(
    el(
      'button',
      {
        class: 'chart-export',
        onclick: () =>
          download(
            'flystate-chart.svg',
            new XMLSerializer().serializeToString(plot),
            'image/svg+xml',
          ),
      },
      'Export SVG',
    ),
  );
  const data = table(
    ['Series', xLabel, label, 'Interval'],
    usable.flatMap((s) =>
      s.points.map((p) => [
        s.name,
        number(p.x),
        number(p.y, 5),
        p.low == null ? '—' : `${number(p.low, 5)} to ${number(p.high, 5)}`,
      ]),
    ),
  );
  return el(
    'div',
    {},
    plot,
    legend,
    el('details', {}, el('summary', {}, 'Exact chart values'), data),
  );
}
export function accuracySeries(name, rows, color, top5 = false, interval = true) {
  return {
    name,
    color,
    points: rows.map((r) => ({
      x: r.t,
      y: 100 * (top5 ? r.top5_accuracy : r.accuracy),
      low: interval && !top5 && r.ci_low != null ? 100 * r.ci_low : null,
      high: interval && !top5 && r.ci_high != null ? 100 * r.ci_high : null,
    })),
  };
}
export function confusion(rows, classes) {
  const matrix = Array.from({ length: classes }, () => new Array(classes).fill(0));
  rows.forEach((r) => {
    if (matrix[r.y_true]) matrix[r.y_true][r.y_pred] = r.count;
  });
  const maximum = Math.max(1, ...rows.map((r) => r.count));
  const canvas = el('canvas', {
    width: 500,
    height: 500,
    class: 'matrix',
    role: 'img',
    'aria-label':
      'Final observation confusion matrix. True labels in rows; predicted labels in columns.',
  });
  const ctx = canvas.getContext('2d');
  const size = 460 / classes;
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, 500, 500);
  for (let row = 0; row < classes; row++)
    for (let col = 0; col < classes; col++) {
      ctx.fillStyle = heatColor(matrix[row][col] / maximum);
      ctx.fillRect(28 + col * size, 12 + row * size, size, size);
    }
  ctx.fillStyle = '#647580';
  ctx.font = '11px sans-serif';
  ctx.fillText('0', 25, 488);
  ctx.fillText(String(classes - 1), 468, 488);
  ctx.fillText('Predicted label →', 175, 498);
  ctx.save();
  ctx.translate(12, 310);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('True label →', 0, 0);
  ctx.restore();
  const hint = el(
    'div',
    { class: 'canvas-hint' },
    `Counts: 0–${maximum}. Hover over a cell to inspect. Labels are zero-based.`,
  );
  canvas.onpointermove = (event) => {
    const rect = canvas.getBoundingClientRect();
    const col = Math.floor((((event.clientX - rect.left) / rect.width) * 500 - 28) / size);
    const row = Math.floor((((event.clientY - rect.top) / rect.height) * 500 - 12) / size);
    if (row >= 0 && row < classes && col >= 0 && col < classes)
      hint.textContent = `True ${row} → predicted ${col}: ${matrix[row][col]} image(s)`;
  };
  return el(
    'div',
    {},
    canvas,
    hint,
    el(
      'details',
      {},
      el('summary', {}, 'Nonzero confusion counts'),
      table(
        ['True label', 'Predicted label', 'Count'],
        rows.map((r) => [r.y_true, r.y_pred, r.count]),
      ),
    ),
  );
}
export function heatColor(value) {
  const t = Math.min(1, Math.max(0, value));
  return `rgb(${Math.round(236 - 225 * t)},${Math.round(246 - 115 * t)},${Math.round(244 - 119 * t)})`;
}
// Google's Turbo colormap (polynomial fit), skipping its darkest blue so points stay visible on
// dark backgrounds; t in [0, 1].
function turbo(t) {
  const x = 0.08 + 0.92 * Math.min(1, Math.max(0, t));
  const poly = (c) => c.reduceRight((acc, k) => acc * x + k, 0);
  const channel = (c) => Math.round(255 * Math.min(1, Math.max(0, poly(c))));
  return `rgb(${channel([0.1357, 4.6154, -42.6603, 132.1311, -152.9424, 59.2864])},${channel([
    0.0914, 2.1942, 4.843, -14.185, 4.2773, 2.8296,
  ])},${channel([0.1067, 12.6419, -60.582, 110.3628, -89.9031, 27.3482])})`;
}
// ColorBrewer RdBu end points with a light neutral center; s in [-1, 1].
const DIVERGING = [
  [33, 102, 172],
  [247, 247, 247],
  [178, 24, 43],
];
function diverging(s) {
  const u = Math.min(1, Math.max(-1, s));
  const [from, to, w] = u < 0 ? [DIVERGING[1], DIVERGING[0], -u] : [DIVERGING[1], DIVERGING[2], u];
  return `rgb(${from.map((c, i) => Math.round(c + (to[i] - c) * w)).join(',')})`;
}
/**
 * Build a color scale for neural values that resists outliers: the range spans the 1st to 99th
 * percentile. Values of both signs get a diverging map centered on zero, with a signed square
 * root so the many near-zero values stay distinguishable; other values get a linear Turbo map.
 * @param {number[]} values Every value the scale must cover.
 * @returns {{low: number, high: number, diverging: boolean, color: function(number): string,
 *   gradient: string, ticks: {value: number, position: number}[]}} The clipped range, the map
 *   kind, a value-to-color function, a CSS gradient for a legend, and legend ticks placed at
 *   their fraction of the gradient.
 */
export function colorScale(values) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  const pick = (q) => sorted[Math.min(sorted.length - 1, Math.floor(q * (sorted.length - 1)))];
  let low = sorted.length ? pick(0.01) : 0,
    high = sorted.length ? pick(0.99) : 1;
  if (high <= low) [low, high] = [low - 1e-9, low + 1e-9];
  const bound = Math.max(Math.abs(low), Math.abs(high));
  const isDiverging = low < 0 && high > 0 && Math.min(-low, high) >= 0.1 * bound;
  // Legend position in [0, 1] of a value, and the value at a legend position.
  const position = isDiverging
    ? (v) => (1 + Math.sign(v) * Math.sqrt(Math.min(1, Math.abs(v) / bound))) / 2
    : (v) => Math.min(1, Math.max(0, (v - low) / (high - low)));
  const valueAt = isDiverging
    ? (q) => Math.sign(2 * q - 1) * (2 * q - 1) ** 2 * bound
    : (q) => low + q * (high - low);
  const paint = isDiverging ? (q) => diverging(2 * q - 1) : turbo;
  const stops = Array.from({ length: 17 }, (_, i) => paint(i / 16));
  const positions = [0, 0.25, 0.5, 0.75, 1];
  return {
    low: isDiverging ? -bound : low,
    high: isDiverging ? bound : high,
    diverging: isDiverging,
    color: (v) => paint(position(v)),
    gradient: `linear-gradient(to right, ${stops.join(', ')})`,
    ticks: positions.map((q) => ({ value: valueAt(q), position: q })),
  };
}
export function histogram(values, title) {
  if (!values.length) return el('p', { class: 'muted' }, 'No timing records.');
  const min = Math.min(...values),
    max = Math.max(...values),
    width = (max - min) / 12 || 1;
  const bins = new Array(12).fill(0);
  values.forEach((v) => bins[Math.min(11, Math.floor((v - min) / width))]++);
  return el(
    'div',
    {},
    lineChart(
      [
        {
          name: title,
          points: bins.map((v, i) => ({ x: min + width * (i + 0.5), y: v })),
        },
      ],
      {
        label: 'Image count per equal-width bin',
        xLabel: 'Time (ms)',
        unit: '',
      },
    ),
    el(
      'p',
      { class: 'chart-summary' },
      `Mean ${number(values.reduce((a, b) => a + b, 0) / values.length)} ms · range ${number(min)}–${number(max)} ms · ${values.length} images`,
    ),
  );
}
