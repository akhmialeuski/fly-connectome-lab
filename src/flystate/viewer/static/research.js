import { interpretation, interpretationCard } from './interpretation.js';
import { bars, card, colors, el, lineChart, number, pct, raw, stats, table } from './charts.js';
import { face } from './identities.js';

const pathValue = encodeURIComponent;
const status = (value) =>
  el(
    'span',
    {
      class: `badge ${value === 'failed' ? 'failed' : value === 'completed' ? 'completed' : ''}`,
    },
    value || 'unknown',
  );
// Register columns in display order, each with the value it sorts by.
const COLUMNS = [
  { key: 'name', label: 'Experiment / study', value: (row) => row.name },
  { key: 'kind', label: 'Kind / representation', value: (row) => row.kind },
  { key: 'classes', label: 'Classes', value: (row) => row.classes },
  { key: 'validation', label: 'Validation', value: (row) => row.scores?.validation?.accuracy },
  { key: 'status', label: 'Status', value: (row) => row.status },
  { key: 'created', label: 'Created', value: (row) => row.created_utc },
  { key: 'result', label: 'Idea and results', value: (row) => interpretation(row).result },
];
const RECORD_FIELDS = [
  ['study', 'Study'],
  ['mode', 'Memory mode'],
  ['config_name', 'Configuration'],
  ['id', 'Path'],
];
const MAX_FIELD_DEPTH = 3;
// Newest attempts first unless the viewer chooses otherwise.
export const DEFAULT_SORT = { key: 'created', descending: true };

function scalarLeaves(value, prefix, depth, found) {
  if (value == null || depth > MAX_FIELD_DEPTH) return;
  if (['number', 'string', 'boolean'].includes(typeof value)) found.add(prefix);
  else if (typeof value === 'object' && !Array.isArray(value))
    for (const [key, child] of Object.entries(value))
      scalarLeaves(child, `${prefix}.${key}`, depth + 1, found);
}

/**
 * List every field the register can be sorted by: its columns, record fields, and each scalar
 * parameter or score path that occurs in at least one entry.
 * @param {object[]} entries Catalog rows.
 * @returns {{key: string, label: string, value: function(object): *}[]} Sortable fields.
 */
export function sortFields(entries) {
  const found = new Set();
  for (const row of entries) {
    scalarLeaves(row.parameters, 'parameters', 1, found);
    scalarLeaves(row.scores, 'scores', 1, found);
  }
  const lookup = (path) => (row) =>
    path.split('.').reduce((value, part) => (value == null ? undefined : value[part]), row);
  return [
    ...COLUMNS,
    ...RECORD_FIELDS.map(([key, label]) => ({ key, label, value: (row) => row[key] })),
    ...[...found].sort().map((path) => ({ key: path, label: path, value: lookup(path) })),
  ];
}

/**
 * Return the rows ordered by one field. Missing values always come last, numbers compare
 * numerically, text compares naturally (s2 before s16), and ties keep a stable order by path.
 * @param {object[]} entries Catalog rows.
 * @param {{key: string, value: function(object): *}[]} fields Fields from sortFields.
 * @param {{key: string, descending: boolean}} sort Selected field and direction.
 * @returns {object[]} A sorted copy of the rows.
 */
export function sortEntries(entries, fields, sort) {
  const field =
    fields.find((item) => item.key === sort.key) ||
    COLUMNS.find((item) => item.key === DEFAULT_SORT.key);
  const direction = sort.descending ? -1 : 1;
  const text = new Intl.Collator('en', { numeric: true, sensitivity: 'base' });
  return entries
    .map((row) => ({ row, value: field.value(row) }))
    .sort((a, b) => {
      const aMissing = a.value == null || a.value === '';
      const bMissing = b.value == null || b.value === '';
      if (aMissing || bMissing) return aMissing - bMissing || text.compare(a.row.id, b.row.id);
      const order =
        typeof a.value === 'number' && typeof b.value === 'number'
          ? a.value - b.value
          : text.compare(String(a.value), String(b.value));
      return direction * order || text.compare(a.row.id, b.row.id);
    })
    .map((item) => item.row);
}

export function experimentTable(entries, sort = null, onSort = null) {
  const headers = COLUMNS.map((column) => {
    if (!onSort) return column.label;
    const active = sort?.key === column.key;
    const arrow = active ? (sort.descending ? ' ↓' : ' ↑') : '';
    return {
      content: el(
        'button',
        { class: 'sort-header', type: 'button', onclick: () => onSort(column.key) },
        `${column.label}${arrow}`,
      ),
      attrs: {
        'aria-sort': active ? (sort.descending ? 'descending' : 'ascending') : 'none',
      },
    };
  });
  return table(
    headers,
    entries.map((row) => [
      el(
        'div',
        { class: 'experiment-path' },
        el(
          'a',
          {
            class: 'run-link',
            href: row.legacy_run_id
              ? `#run/${pathValue(row.legacy_run_id)}`
              : `#experiment/${pathValue(row.id)}`,
          },
          row.name,
        ),
        el('small', { class: 'mono' }, row.study === '.' ? 'Training runs' : row.study),
        el('small', {}, row.config_name || ''),
      ),
      el(
        'div',
        {},
        String(row.kind).replaceAll('_', ' '),
        el(
          'small',
          {},
          [row.parameters?.representation, row.parameters?.history, row.mode]
            .filter(Boolean)
            .join(' · '),
        ),
        row.parameters?.train_per_class
          ? el(
              'small',
              {},
              `${row.parameters.train_per_class} photos / identity · subset seed ${row.parameters.subset_seed}`,
            )
          : null,
      ),
      row.classes ?? '—',
      pct(row.scores?.validation?.accuracy),
      el(
        'div',
        {},
        status(row.status),
        row.warnings?.length ? el('small', {}, 'Some evidence is unreadable') : null,
      ),
      row.created_utc
        ? el(
            'time',
            { datetime: row.created_utc, class: 'mono' },
            `${row.created_utc.slice(0, 16).replace('T', ' ')} UTC`,
          )
        : '—',
      el(
        'div',
        {},
        el('p', {}, interpretation(row).idea),
        el('p', {}, interpretation(row).result),
      ),
    ]),
  );
}

export async function researchPage(path, api, signal) {
  let detail,
    selectedFile = '',
    offset = 0,
    requestId = 0,
    selectedByUser = false;
  const page = el('div');
  const summary = el('div'),
    measurements = el('div'),
    preview = el('div');
  const picker = el('select', { 'aria-label': 'Experiment evidence' });
  const notes = el('div');
  const base = `experiments/detail?path=${pathValue(path)}`;
  const imageUrl = (sampleId) =>
    `/api/experiments/image?path=${pathValue(path)}&sample_id=${pathValue(sampleId)}`;
  const portrait = (sampleId, caption, small = false) =>
    face('', sampleId, caption, { small, imageUrl: imageUrl(sampleId) });
  const identity = (label) => {
    const ref = detail.identities.find((row) => row.label === label);
    return ref
      ? portrait(ref.sample_id, `Identity ${ref.identity}`, true)
      : el('span', {}, `Class ${label} · reference photo unavailable`);
  };
  function predictionPanel(row) {
    const classes = detail.documents['report.json']?.classes;
    if (!Array.isArray(classes) || classes.length !== row.probabilities?.length)
      return raw(row, 'Recorded prediction');
    const ranked = row.probabilities
      .map((probability, i) => ({ probability, label: classes[i] }))
      .sort((a, b) => b.probability - a.probability)
      .slice(0, 5);
    return card(
      'Who does the model think this is?',
      row.y_true === row.y_pred
        ? 'The first choice matches the actual person.'
        : 'The first choice is a different person.',
      el(
        'div',
        { class: 'candidate-grid' },
        el('div', {}, el('h3', {}, 'Input photograph'), portrait(row.sample_id, row.sample_id)),
        el('div', {}, el('h3', {}, 'Actual person'), identity(row.y_true)),
      ),
      el(
        'div',
        { class: 'candidate-grid' },
        ...ranked.map((item, i) =>
          el(
            'div',
            {
              class: `candidate${i === 0 ? ' first-choice' : ''}${item.label === row.y_true ? ' correct-identity' : ''}`,
              'data-label': item.label,
            },
            el(
              'div',
              { class: 'candidate-rank' },
              i === 0 ? "Model's first choice" : `Choice ${i + 1}`,
            ),
            identity(item.label),
            el('strong', { class: 'candidate-probability' }, pct(item.probability)),
            el('small', {}, 'Model probability'),
          ),
        ),
      ),
      el(
        'p',
        { class: 'readout-note' },
        'Predicted people are shown using photographs from this attempt’s training set. These are identity choices, not generated faces. Pixel and encoded-input probes are diagnostic controls; only neural probes use recorded fly activity.',
      ),
    );
  }
  async function showEvidence() {
    const current = ++requestId;
    if (!picker.value) {
      preview.replaceChildren(el('p', {}, 'Evidence is not available yet.'));
      return;
    }
    selectedFile = picker.value;
    preview.replaceChildren(el('p', {}, 'Reading evidence…'));
    try {
      const data = await api(
        `experiments/evidence?path=${pathValue(path)}&name=${pathValue(selectedFile)}&offset=${offset}&limit=10`,
        signal,
      );
      if (signal.aborted || current !== requestId) return;
      if (data.text !== undefined) {
        let text = data.text;
        if (data.name.endsWith('.json')) {
          try {
            text = JSON.stringify(JSON.parse(text), null, 2);
          } catch {
            /* Preserve damaged evidence as text. */
          }
        }
        preview.replaceChildren(el('pre', { class: 'report-text' }, text));
        return;
      }
      const panel = el('div');
      const predictions =
        data.rows.length &&
        Array.isArray(data.rows[0].probabilities) &&
        'y_pred' in data.rows[0];
      const columns = predictions
        ? ['Sample', 'Actual label', 'Predicted label', 'Result', '']
        : data.columns;
      const rows = predictions
        ? data.rows.map((row) => [
            row.sample_id,
            row.y_true,
            row.y_pred,
            row.y_true === row.y_pred ? 'Correct' : 'Incorrect',
            el(
              'button',
              { onclick: () => panel.replaceChildren(predictionPanel(row)) },
              'Inspect prediction',
            ),
          ])
        : data.rows.map((row) =>
            data.columns.map((key) =>
              typeof row[key] === 'object' ? JSON.stringify(row[key]) : (row[key] ?? '—'),
            ),
          );
      const previous = el(
        'button',
        {
          onclick: () => {
            offset = Math.max(0, offset - 10);
            showEvidence();
          },
        },
        'Previous rows',
      );
      const next = el(
        'button',
        {
          onclick: () => {
            offset += 10;
            showEvidence();
          },
        },
        'Next rows',
      );
      previous.disabled = offset === 0;
      next.disabled = offset + data.rows.length >= data.total;
      preview.replaceChildren(
        el(
          'div',
          { class: 'toolbar' },
          previous,
          el(
            'span',
            {},
            `${data.rows.length ? offset + 1 : 0}–${offset + data.rows.length} of ${data.total} rows`,
          ),
          next,
        ),
        table(columns, rows),
        panel,
      );
      if (predictions) panel.append(predictionPanel(data.rows[0]));
    } catch (error) {
      if (!signal.aborted && current === requestId)
        preview.replaceChildren(el('div', { class: 'error', role: 'alert' }, error.message));
    }
  }
  async function refresh() {
    const next = await api(base, signal);
    if (signal.aborted) return;
    detail = next;
    const manifest = detail.documents['manifest.json'] || {};
    const parameters = manifest.parameters || {};
    const report = detail.documents['report.json'] || {};
    const scores = Object.entries(report.scores || {}).filter(([, value]) =>
      Number.isFinite(value?.accuracy),
    );
    summary.replaceChildren(
      ...[
        el(
          'div',
          { class: 'page-head' },
          el('h1', { class: 'experiment-path' }, path.split('/').at(-1)),
          el('p', { class: 'experiment-path mono' }, path),
        ),
        interpretationCard({
          kind: parameters.kind || manifest.kind,
          parameters,
          status: manifest.status,
          error: manifest.error,
          scores: report.scores,
          report,
          classes: detail.documents['config.json']?.dataset?.subset?.n_identities,
        }),
        stats([
          [
            'Status',
            manifest.status || 'Unreadable',
            String(parameters.kind || manifest.kind || 'Experiment').replaceAll('_', ' '),
          ],
          [
            'Representation',
            parameters.representation || 'Not recorded',
            parameters.history || '',
          ],
          [
            'Elapsed',
            manifest.elapsed_seconds == null
              ? 'Running / unavailable'
              : `${number(manifest.elapsed_seconds)} s`,
            'Recorded wall time',
          ],
          [
            'Validation',
            pct(report.scores?.validation?.accuracy),
            report.scores?.validation
              ? 'Exploratory held-out validation'
              : 'No recognition score recorded',
          ],
        ]),
        manifest.error
          ? el(
              'div',
              { class: 'error', role: 'alert' },
              `${manifest.error_type || 'Failure'}: ${manifest.error}`,
            )
          : null,
        detail.warnings.length
          ? card(
              'Read warnings',
              'Other available evidence remains accessible.',
              ...detail.warnings.map((text) => el('p', {}, text)),
            )
          : null,
      ].filter(Boolean),
    );
    measurements.replaceChildren();
    if (scores.length)
      measurements.append(
        card(
          'Recorded recognition scores',
          'Training scores measure fitting; validation scores are exploratory. No new test result is implied.',
          bars(
            scores.map(([split, score]) => ({
              label: split.replaceAll('_', ' '),
              value: score.accuracy,
            })),
            pct,
          ),
          table(
            ['Split', 'Correct / total', 'Top-1', 'Top-5', 'Log loss'],
            scores.map(([split, score]) => [
              split,
              `${Math.round(score.accuracy * score.n)} / ${score.n}`,
              pct(score.accuracy),
              pct(score.top5_accuracy),
              number(score.log_loss),
            ]),
          ),
        ),
      );
    if (report.budget_measurements?.length)
      measurements.append(
        card(
          'Optimizer convergence',
          'Training-fold numerical diagnosis. These fits do not estimate recognition accuracy.',
          lineChart(
            [
              {
                name: 'Objective',
                color: colors[0],
                points: report.budget_measurements.map((row) => ({
                  x: row.max_iterations,
                  y: row.objective,
                })),
              },
            ],
            { xLabel: 'Iteration cap', label: 'Training objective', unit: '' },
          ),
          table(
            ['Budget', 'Actual iterations', 'Objective', 'Gradient norm', 'Converged'],
            report.budget_measurements.map((row) => [
              row.max_iterations,
              row.iterations.join(', '),
              number(row.objective, 8),
              Number(row.gradient_infinity_norm).toExponential(3),
              row.converged ? 'Yes' : 'No',
            ]),
          ),
        ),
      );
    if (report.cohort_counts)
      measurements.append(
        card(
          'Cohort audit',
          'Recorded membership counts; no classifier score.',
          bars(
            Object.entries(report.cohort_counts).map(([label, value]) => ({ label, value })),
            number,
          ),
          raw(report, 'Audit measurements'),
        ),
      );
    notes.replaceChildren(
      raw(parameters, 'Experiment parameters'),
      raw(detail.documents, 'Complete recorded metadata'),
    );
    const files = detail.files.filter((file) => file.preview);
    picker.replaceChildren(
      ...files.map((file) =>
        el('option', { value: file.name }, `${file.name} (${number(file.bytes / 1024)} KiB)`),
      ),
    );
    const preferred =
      !selectedByUser && files.some((file) => file.name === 'validation-predictions.parquet')
        ? 'validation-predictions.parquet'
        : selectedFile;
    picker.value = files.some((file) => file.name === preferred)
      ? preferred
      : files[0]?.name || '';
    if (selectedFile !== picker.value) offset = 0;
    await showEvidence();
  }
  picker.onchange = () => {
    selectedByUser = true;
    offset = 0;
    showEvidence();
  };
  page.append(
    summary,
    measurements,
    card(
      'Saved evidence',
      'Files are discovered from this attempt. Tables are paginated; large numeric arrays remain on disk.',
      picker,
      preview,
    ),
    card('Reproducibility record', 'Recorded parameters and metadata.', notes),
  );
  await refresh();
  return { page, refresh };
}
