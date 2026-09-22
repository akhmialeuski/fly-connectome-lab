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
export function experimentTable(entries) {
  return table(
    [
      'Experiment / study',
      'Kind / representation',
      'Classes',
      'Validation',
      'Status',
      'Idea and results',
    ],
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
