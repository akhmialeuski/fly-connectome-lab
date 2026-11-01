import { isSequential, interpretationCard } from './interpretation.js';
import {
  DEFAULT_SORT,
  experimentTable,
  researchPage,
  sortEntries,
  sortFields,
} from './research.js';
import { face, identityFace } from './identities.js';
import {
  accuracySeries,
  bars,
  card,
  colors,
  confusion,
  el,
  histogram,
  lineChart,
  modeColors,
  number,
  pct,
  raw,
  stats,
  table,
} from './charts.js';
import { openEpisode, closeEpisode } from './episode.js';

const content = document.querySelector('#content');
let inventory,
  generation = 0,
  controller,
  refreshView = null,
  catalogPolling = false;
const titles = {
  overview: 'Overview',
  runs: 'Experiments',
  run: 'Experiments',
  experiment: 'Diagnostics',
  diagnostics: 'Diagnostics',
  comparisons: 'Comparisons',
  evidence: 'Evidence library',
  caches: 'Trace caches',
};
export async function api(path, signal) {
  const response = await fetch(`/api/${path}`, { signal, cache: 'no-store' });
  const data = await response.json();
  if (!response.ok)
    throw new Error(
      typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail),
    );
  return data;
}
const escapePath = encodeURIComponent;
const badge = (value) =>
  el(
    'span',
    {
      class: `badge ${value === 'completed' ? 'completed' : value === 'failed' ? 'failed' : ''}`,
    },
    value,
  );
const mode = (value) =>
  el(
    'span',
    { class: 'mode', style: `--mode-color:${modeColors[value] || colors[0]}` },
    value.replaceAll('_', ' '),
  );
const head = (title, description) =>
  el(
    'div',
    { class: 'page-head' },
    el('div', { class: 'eyebrow' }, 'SEQUENTIAL VISUAL MEMORY'),
    el('h1', {}, title),
    el('p', {}, description),
  );
const errorBox = (error) =>
  el('div', { class: 'error', role: 'alert' }, error.message || String(error));
const loading = () => el('div', { class: 'loading' }, 'Reading local artifacts…');
const empty = (message) => el('div', { class: 'empty' }, message);
function option(value, text) {
  return el('option', { value }, text);
}
function warnings(items) {
  return items.length
    ? el(
        'div',
        { class: 'notice' },
        'Some artifacts could not be read.',
        raw(items, 'Read warnings'),
      )
    : document.createDocumentFragment();
}
async function overview(signal) {
  const runs = inventory.runs;
  const classes = [...new Set(runs.map((r) => r.classes))].sort((a, b) => b - a);
  content.append(
    head(
      'A window into memory.',
      'Follow every observation, compare memory policies, and inspect the evidence behind each result.',
    ),
    stats([
      [
        'Experiments',
        number(inventory.experiments.filter(isSequential).length),
        'Sequential training runs',
      ],
      [
        'Completed',
        number(runs.filter((r) => r.status === 'completed').length),
        'Training artifacts available',
      ],
      [
        'Comparisons',
        number(inventory.reports.filter((r) => r.kind === 'comparisons').length),
        'Paired statistical reports',
      ],
      ['Trace caches', number(inventory.caches.length), 'Reusable neural recordings'],
    ]),
    warnings(inventory.warnings),
  );
  if (!runs.length) {
    const register = el('div', {}, experimentTable(inventory.experiments.filter(isSequential)));
    content.append(
      card(
        'Experiment register',
        'Sequential image-patch experiments. Diagnostic controls are in Diagnostics.',
        register,
      ),
    );
    refreshView = () =>
      register.replaceChildren(experimentTable(inventory.experiments.filter(isSequential)));
    if (!inventory.experiments.some(isSequential))
      content.append(
        empty('No experiments found in FLYSTATE_HOME. New attempts appear automatically.'),
      );
    return;
  }
  const cohort = el(
    'select',
    { 'aria-label': 'Dataset cohort' },
    classes.map((n) => option(n, `${n} identities`)),
  );
  const split = el(
    'select',
    { 'aria-label': 'Evaluation split' },
    option('test', 'Test'),
    option('val', 'Validation'),
  );
  const intervals = el('input', {
    type: 'checkbox',
    checked: '',
    'aria-label': 'Show confidence intervals',
  });
  const plot = el('div', {}, loading());
  const curveCard = card(
    'Accuracy across observations',
    'Latest completed evaluation per run, within the selected class count.',
    el(
      'div',
      { class: 'toolbar' },
      el('label', {}, 'Cohort', cohort),
      el('label', {}, 'Split', split),
      el('label', {}, intervals, '95% Wilson intervals'),
    ),
    plot,
  );
  const register = el('div', {}, experimentTable(inventory.experiments.filter(isSequential)));
  refreshView = () =>
    register.replaceChildren(experimentTable(inventory.experiments.filter(isSequential)));
  content.append(
    curveCard,
    card(
      'Experiment register',
      'Open a run to inspect predictions, episodes, and provenance.',
      register,
    ),
  );
  let updateId = 0;
  async function update() {
    const current = ++updateId;
    plot.replaceChildren(loading());
    const selected = runs.filter((r) => r.classes === Number(cohort.value));
    const results = await Promise.all(
      selected.map(async (r) => {
        try {
          const detail = await api(`runs/${escapePath(r.run_id)}`, signal);
          const evaluation = detail.evaluations.find(
            (e) => e.status === 'completed' && e.split === split.value,
          );
          if (!evaluation) return { run: r, missing: true };
          return {
            run: r,
            data: await api(
              `runs/${escapePath(r.run_id)}/evaluations/${escapePath(evaluation.eval_id)}`,
              signal,
            ),
          };
        } catch (error) {
          if (error.name === 'AbortError') throw error;
          return { run: r, error };
        }
      }),
    );
    if (current !== updateId || signal.aborted) return;
    const usable = results.filter((r) => r.data);
    const series = usable.map(({ run: r, data }) =>
      accuracySeries(
        `${r.mode.replaceAll('_', ' ')} · ${r.run_id.slice(9, 15)}`,
        data.metrics,
        modeColors[r.mode],
        false,
        intervals.checked,
      ),
    );
    plot.replaceChildren(
      lineChart(series, { baseline: 100 / Number(cohort.value) }),
      el(
        'p',
        { class: 'chart-summary' },
        `Chance accuracy: ${pct(1 / Number(cohort.value))}. Class count is a display filter; formal compatibility is checked in saved paired comparisons.`,
      ),
    );
    if (usable.length)
      plot.append(
        table(
          ['Run', 'Split', 'Final accuracy', '95% interval', 'Images'],
          usable.map(({ run: r, data: d }) => {
            const last = d.metrics.at(-1);
            return [
              el('a', { href: `#run/${escapePath(r.run_id)}` }, r.name),
              d.meta.split,
              pct(last.accuracy),
              `${pct(last.ci_low)}–${pct(last.ci_high)}`,
              last.n,
            ];
          }),
        ),
      );
    for (const item of results.filter((r) => !r.data))
      plot.append(
        el(
          'p',
          { class: 'muted' },
          `${item.run.name}: ${item.error?.message || 'no completed evaluation for this split'}`,
        ),
      );
  }
  const change = () =>
    update().catch((error) => {
      if (error.name !== 'AbortError') plot.replaceChildren(errorBox(error));
    });
  cohort.onchange = change;
  split.onchange = change;
  intervals.onchange = change;
  await update();
}
function experiments(diagnostics = false) {
  content.append(
    head(
      diagnostics ? 'Diagnostic register' : 'Experiment register',
      diagnostics
        ? 'Probes, audits, and numerical checks. These are not sequential-memory training runs.'
        : 'Sequential image-patch experiments. Open a run to inspect episodes and memory state.',
    ),
  );
  const search = el('input', {
    type: 'search',
    placeholder: 'Filter by name, path, or parameters',
    'aria-label': 'Search experiments',
  });
  const study = el('select', { 'aria-label': 'Experiment study' });
  const kind = el('select', { 'aria-label': 'Experiment kind' });
  const policy = el('select', { 'aria-label': 'Memory mode' });
  const sortBy = el('select', { 'aria-label': 'Sort by' });
  const direction = el('button', { type: 'button', 'aria-label': 'Sort direction' });
  const list = el('div'),
    count = el('p', { class: 'muted' });
  // The chosen order survives refreshes and visits; storage may be unavailable in private mode.
  const storageKey = `flystate.sort.${diagnostics ? 'diagnostics' : 'runs'}`;
  let sort = { ...DEFAULT_SORT };
  try {
    sort = { ...sort, ...JSON.parse(localStorage.getItem(storageKey) || '{}') };
  } catch {
    // Keep the default order.
  }
  function setSort(key, descending = sort.key === key ? !sort.descending : key === 'created') {
    sort = { key, descending };
    try {
      localStorage.setItem(storageKey, JSON.stringify(sort));
    } catch {
      // The order still applies for this page view.
    }
    update();
  }
  function choices(select, values, label) {
    const selected = select.value;
    select.replaceChildren(
      option('', label),
      ...[...new Set(values.filter(Boolean))]
        .sort()
        .map((v) => option(v, v === '.' ? 'Training runs' : v.replaceAll('_', ' '))),
    );
    select.value = [...select.options].some((item) => item.value === selected) ? selected : '';
  }
  const update = () => {
    const entries = inventory.experiments.filter((row) => isSequential(row) !== diagnostics);
    choices(
      study,
      entries.map((row) => row.study),
      'All studies',
    );
    choices(
      kind,
      entries.map((row) => row.kind),
      'All experiment kinds',
    );
    choices(
      policy,
      entries.map((row) => row.mode),
      'All memory modes',
    );
    const filtered = entries.filter(
      (row) =>
        (!study.value || row.study === study.value) &&
        (!kind.value || row.kind === kind.value) &&
        (!policy.value || row.mode === policy.value) &&
        `${row.name} ${row.id} ${row.config_name || ''} ${JSON.stringify(row.parameters)}`
          .toLowerCase()
          .includes(search.value.toLowerCase()),
    );
    const fields = sortFields(entries);
    sortBy.replaceChildren(...fields.map((field) => option(field.key, `Sort: ${field.label}`)));
    sortBy.value = fields.some((field) => field.key === sort.key) ? sort.key : DEFAULT_SORT.key;
    direction.textContent = sort.descending ? '↓ Descending' : '↑ Ascending';
    list.replaceChildren(
      filtered.length
        ? experimentTable(sortEntries(filtered, fields, sort), sort, (key) => setSort(key))
        : empty('No matching experiments.'),
    );
    count.textContent = `${filtered.length} of ${entries.length} ${diagnostics ? 'diagnostics' : 'experiments'} · updates automatically every 5 seconds`;
  };
  search.oninput = update;
  for (const select of [study, kind, policy]) select.onchange = update;
  sortBy.onchange = () => setSort(sortBy.value, sortBy.value === 'created');
  direction.onclick = () => setSort(sort.key, !sort.descending);
  refreshView = update;
  update();
  content.append(
    card(
      diagnostics ? 'Diagnostic attempts' : 'Sequential experiments',
      'New studies require no viewer configuration.',
      el('div', { class: 'toolbar' }, search, study, kind, policy, sortBy, direction),
      count,
      list,
    ),
  );
}

async function runPage(runId, signal) {
  const detail = await api(`runs/${escapePath(runId)}`, signal);
  if (signal.aborted) return;
  const cfg = detail.config,
    manifest = detail.manifest;
  content.append(
    head(cfg.name, runId),
    interpretationCard({
      kind: 'training',
      mode: cfg.memory.mode,
      status: manifest.status,
      classes: cfg.dataset.subset.n_identities,
      scores: { validation: { accuracy: detail.summary?.final_val_accuracy } },
    }),
    stats([
      ['Memory policy', cfg.memory.mode.replaceAll('_', ' '), 'Recorded configuration'],
      ['Classes', cfg.dataset.subset.n_identities, 'Identity classification'],
      ['Observations', cfg.episodes.steps, `${cfg.episodes.window} px stimulus window`],
      ['Final validation', pct(detail.summary?.final_val_accuracy), manifest.status],
    ]),
    warnings(detail.warnings),
  );
  const selector = el(
    'select',
    { 'aria-label': 'Evaluation' },
    option('', 'Training validation curve'),
    ...detail.evaluations.map((e) =>
      option(
        e.status === 'completed' ? e.eval_id : '',
        `${e.split} · ${e.created_utc.slice(0, 19)} · ${e.status}`,
      ),
    ),
  );
  const target = el('div');
  content.append(
    el('div', { class: 'toolbar' }, el('label', {}, 'Result set', selector)),
    target,
    card(
      'Reproducibility record',
      'Configuration, manifest, environment, and fitted readout metadata.',
      raw(detail),
    ),
  );
  const initial =
    detail.evaluations.find((e) => e.status === 'completed' && e.split === 'test') ||
    detail.evaluations.find((e) => e.status === 'completed');
  selector.value = initial?.eval_id || '';
  let updateId = 0;
  async function update() {
    const current = ++updateId;
    target.replaceChildren(loading());
    if (!selector.value) {
      target.replaceChildren(
        card(
          'Training validation',
          'This curve is from model selection; no held-out test inference is implied.',
          lineChart([accuracySeries('Validation', detail.validation, colors[0])], {
            baseline: 100 / cfg.dataset.subset.n_identities,
          }),
        ),
      );
      return;
    }
    const evalId = selector.value;
    try {
      const data = await api(
        `runs/${escapePath(runId)}/evaluations/${escapePath(evalId)}`,
        signal,
      );
      if (current !== updateId || signal.aborted) return;
      target.replaceChildren(
        card(
          'Classification accuracy',
          `${data.meta.split} split · ${data.meta.n_samples} images · shaded band: 95% Wilson interval for top-1`,
          lineChart(
            [
              accuracySeries('Top-1', data.metrics, colors[0]),
              accuracySeries('Top-5', data.metrics, colors[2], true),
            ],
            { baseline: 100 / data.meta.n_classes },
          ),
        ),
        el(
          'div',
          { class: 'grid-two' },
          card(
            'Confusion matrix',
            'Final observation · row = true identity · column = prediction',
            confusion(data.confusion, data.meta.n_classes),
          ),
          card(
            'Inference timing',
            'Brain simulation per image, in milliseconds.',
            histogram(
              data.timings.map((r) => r.brain_ms),
              'Brain',
            ),
            el(
              'details',
              {},
              el('summary', {}, 'Readout timing'),
              histogram(
                data.timings.map((r) => r.readout_ms),
                'Readout',
              ),
            ),
          ),
        ),
        predictionsPanel(runId, evalId, data.meta, detail.identities, signal),
      );
    } catch (error) {
      if (error.name !== 'AbortError' && current === updateId)
        target.replaceChildren(errorBox(error));
    }
  }
  selector.onchange = update;
  await update();
}
function predictionsPanel(runId, evalId, meta, identities, signal) {
  const step = el(
    'select',
    { 'aria-label': 'Prediction observation' },
    ...Array.from({ length: meta.timesteps }, (_, i) => option(i + 1, `Observation ${i + 1}`)),
  );
  step.value = String(meta.timesteps);
  const filter = el(
    'select',
    { 'aria-label': 'Prediction correctness' },
    option('', 'All predictions'),
    option('false', 'Incorrect only'),
    option('true', 'Correct only'),
  );
  const search = el('input', {
    type: 'search',
    placeholder: 'Exact sample ID',
    'aria-label': 'Sample ID',
  });
  const apply = el('button', {}, 'Find sample');
  const list = el('div'),
    previous = el('button', {}, '← Previous'),
    next = el('button', {}, 'Next →'),
    count = el('span');
  let offset = 0,
    queryId = 0;
  const query = async () => {
    const current = ++queryId;
    list.replaceChildren(loading());
    previous.disabled = next.disabled = true;
    const params = new URLSearchParams({ t: step.value, offset, limit: 50 });
    if (filter.value) params.set('correct', filter.value);
    if (search.value.trim()) params.set('sample_id', search.value.trim());
    try {
      const data = await api(
        `runs/${escapePath(runId)}/evaluations/${escapePath(evalId)}/predictions?${params}`,
        signal,
      );
      if (current !== queryId || signal.aborted) return;
      list.replaceChildren(
        data.rows.length
          ? table(
              [
                'Image shown',
                'Actual person',
                'Model’s answer',
                'Probability',
                'Result',
                'Inspect',
              ],
              data.rows.map((r) => [
                face(runId, r.sample_id, r.sample_id, { small: true }),
                identityFace(runId, r.y_true, identities, true),
                identityFace(runId, r.y_pred, identities, true),
                pct(r.p_pred),
                badge(r.correct ? 'Same person' : 'Different person'),
                el(
                  'button',
                  {
                    class: 'text-button',
                    onclick: () =>
                      openEpisode(runId, evalId, r.sample_id, Number(step.value), identities),
                  },
                  'View episode →',
                ),
              ]),
            )
          : empty('No predictions match these filters.'),
      );
      count.textContent = data.total
        ? `${offset + 1}–${Math.min(offset + 50, data.total)} of ${data.total} predictions`
        : '0 predictions';
      previous.disabled = offset === 0;
      next.disabled = offset + 50 >= data.total;
    } catch (error) {
      if (error.name !== 'AbortError' && current === queryId)
        list.replaceChildren(errorBox(error));
    }
  };
  const reset = () => {
    offset = 0;
    query();
  };
  step.onchange = reset;
  filter.onchange = reset;
  apply.onclick = reset;
  search.onkeydown = (e) => {
    if (e.key === 'Enter') reset();
  };
  previous.onclick = () => {
    offset = Math.max(0, offset - 50);
    query();
  };
  next.onclick = () => {
    offset += 50;
    query();
  };
  query();
  return card(
    'Prediction explorer',
    'Compare the input photo with the person selected by the model. Actual-person and answer photos are training examples; open an episode to see all five candidates and how the answer changes.',
    el('div', { class: 'toolbar' }, step, filter, search, apply),
    list,
    el('div', { class: 'pagination' }, count, el('div', {}, previous, next)),
  );
}
async function reportsPage(comparisons, signal) {
  content.append(
    head(
      comparisons ? 'Paired comparisons' : 'Evidence library',
      comparisons
        ? 'Inspect the recorded effect sizes and uncertainty for compatible experiments.'
        : 'Calibration, CPU benchmarks, pixel controls, and research reports.',
    ),
  );
  const kinds = comparisons
    ? ['comparisons']
    : ['reports', 'calibrations', 'benchmarks', 'design-checks'];
  const category = el(
    'select',
    { 'aria-label': 'Report category' },
    ...kinds.map((k) => option(k, k.replaceAll('-', ' '))),
  );
  const selector = el('select', {
      class: 'report-picker',
      'aria-label': 'Report',
    }),
    target = el('div');
  content.append(
    el('div', { class: 'toolbar' }, el('label', {}, 'Category', category)),
    selector,
    target,
  );
  let updateId = 0;
  async function show() {
    const current = ++updateId;
    if (!selector.value) {
      target.replaceChildren(empty('No saved reports in this category.'));
      return;
    }
    target.replaceChildren(loading());
    try {
      const data = await api(
        `reports/${escapePath(category.value)}/${escapePath(selector.value)}`,
        signal,
      );
      if (current !== updateId || signal.aborted) return;
      target.replaceChildren(renderReport(category.value, data));
    } catch (error) {
      if (error.name !== 'AbortError' && current === updateId)
        target.replaceChildren(errorBox(error));
    }
  }
  function options() {
    selector.replaceChildren(
      ...inventory.reports
        .filter((r) => r.kind === category.value)
        .map((r) => option(r.id, r.id)),
    );
    return show();
  }
  category.onchange = options;
  selector.onchange = show;
  await options();
}
function renderReport(kind, d) {
  if (d.text !== undefined)
    return card(
      'Research report',
      'Saved Markdown, displayed as plain text.',
      el('pre', { class: 'report-text' }, d.text),
    );
  const output = el('div');
  if (kind === 'comparisons' && d.rows?.length) {
    const last = d.rows.at(-1);
    output.append(
      stats([
        ['Final difference', `${number(last.diff_pp)} pp`, 'Run A minus run B'],
        [
          '95% bootstrap interval',
          `${number(last.ci_low_pp)} / ${number(last.ci_high_pp)}`,
          'Percentage points',
        ],
        ['McNemar p', number(last.p, 5), 'Final observation, unadjusted'],
        [
          'Final memory effect',
          d.delta_mem_pp == null ? 'Not applicable' : `${number(d.delta_mem_pp.diff_pp)} pp`,
          'Persistent minus reset only',
        ],
      ]),
      card(
        'Difference across observations',
        `${d.split} split · ${d.bootstrap} paired bootstrap resamples · per-observation intervals`,
        el('p', { class: 'mono' }, `A: ${d.run_a}\nB: ${d.run_b}`),
        lineChart(
          [
            {
              name: 'A − B',
              points: d.rows.map((r) => ({
                x: r.t,
                y: r.diff_pp,
                low: r.ci_low_pp,
                high: r.ci_high_pp,
              })),
            },
          ],
          {
            label: 'Accuracy difference (percentage points)',
            baseline: 0,
            unit: ' pp',
          },
        ),
        table(
          ['Observation', 'A accuracy', 'B accuracy', 'Difference (pp)', 'McNemar p'],
          d.rows.map((r) => [
            r.t,
            pct(r.acc_a),
            pct(r.acc_b),
            number(r.diff_pp),
            number(r.p, 5),
          ]),
        ),
      ),
    );
  } else if (kind === 'calibrations' && d.amplitudes) {
    output.append(
      card(
        'Signal propagation',
        'Recorded response distance d(t) by injection amplitude.',
        lineChart(
          d.amplitudes.map((a) => ({
            name: `Amplitude ${a.amplitude}`,
            points: a.d.map((y, i) => ({ x: i + 1, y })),
          })),
          {
            label: 'Response distance d(t)',
            xLabel: 'Simulation step',
            unit: '',
          },
        ),
      ),
      card(
        'Firing rates & latency',
        'Rates in Hz; latency in simulation steps.',
        table(
          [
            'Amplitude',
            'All neurons (Hz)',
            'Input (Hz)',
            'Readout (Hz)',
            'First response',
            'Half response',
          ],
          d.amplitudes.map((a) => [
            a.amplitude,
            number(a.rate_all_hz),
            number(a.rate_input_hz),
            number(a.rate_readout_hz),
            a.latency_first ?? '—',
            a.latency_half ?? '—',
          ]),
        ),
      ),
    );
    if (d.recommendation)
      output.append(
        el(
          'div',
          { class: 'notice' },
          `Recorded recommendation: amplitude ${d.recommendation.amplitude}, ${d.recommendation.steps_per_observation} simulation steps per observation.`,
        ),
      );
  } else if (kind === 'benchmarks' && d.grid) {
    const batches = [...new Set(d.grid.map((r) => r.batch))];
    output.append(
      card(
        'CPU scaling',
        'Lower is faster. Batch size is the number of concurrently simulated flies.',
        lineChart(
          batches.map((batch) => ({
            name: `Batch ${batch}`,
            points: d.grid
              .filter((r) => r.batch === batch)
              .map((r) => ({ x: r.threads, y: r.ms_per_episode_step })),
          })),
          {
            label: 'Milliseconds / episode step',
            xLabel: 'Threads',
            unit: ' ms',
          },
        ),
      ),
      card(
        'Benchmark measurements',
        d.host?.cpu_model,
        table(
          ['Threads', 'Batch', 'ms / episode step', 'CPU %', 'RSS (MB)'],
          d.grid.map((r) => [
            r.threads,
            r.batch,
            number(r.ms_per_episode_step, 3),
            number(r.cpu_percent),
            number(r.rss_mb),
          ]),
        ),
      ),
    );
    if (d.sustained)
      output.append(
        card(
          'Sustained load',
          'Recorded performance drift; the measurements do not establish its cause.',
          bars(
            [
              {
                label: 'First 30 seconds',
                value: d.sustained.ms_per_episode_step_first_30s,
              },
              {
                label: 'Last 30 seconds',
                value: d.sustained.ms_per_episode_step_last_30s,
              },
            ],
            (v) => `${number(v)} ms`,
          ),
          el(
            'p',
            { class: 'muted' },
            `End/start ratio: ${number(d.sustained.throttle_ratio, 3)}`,
          ),
        ),
      );
  } else if (kind === 'design-checks' && d.baselines) {
    output.append(
      card(
        'Pixel controls',
        `${d.evaluation_split} split · ${d.n_classes} classes · ${d.n_eval} images`,
        bars(
          Object.entries(d.baselines).map(([label, row]) => ({
            label: label.replaceAll('_', ' '),
            value: row.accuracy,
          })),
          pct,
        ),
        el(
          'p',
          { class: 'notice' },
          `Recorded memory gap: ${number(d.gap_pp)} pp. ${d.message}`,
        ),
      ),
    );
    if (d.window_baselines)
      output.append(
        card(
          'Individual stimulus windows',
          'Independent pixel classification per window; 95% intervals.',
          lineChart(
            [
              accuracySeries(
                'Window accuracy',
                d.window_baselines.map((r, i) => ({ ...r, t: i + 1 })),
                colors[0],
              ),
            ],
            { baseline: 100 / d.n_classes },
          ),
        ),
      );
  }
  output.append(
    card(
      'Source artifact',
      'Complete recorded values, including metadata and provenance.',
      raw(d, 'View complete JSON'),
    ),
  );
  return output;
}
function caches() {
  content.append(
    head(
      'Trace caches',
      'Inspect reusable simulation output without rebuilding or changing it.',
    ),
    card(
      'Stored neural recordings',
      'N = images · T = observations · F = readout features',
      inventory.caches.length
        ? table(
            ['Cache key', 'Status', 'N', 'T', 'F', 'Threads', 'Batch'],
            inventory.caches.map((c) => [
              el('span', { class: 'mono' }, c.key),
              badge(c.status),
              c.N,
              c.T,
              c.F,
              c.threads,
              c.batch_size,
            ]),
          )
        : empty('No trace caches found.'),
    ),
  );
}
async function navigate(refresh = false) {
  const current = ++generation;
  controller?.abort();
  controller = new AbortController();
  const { signal } = controller;
  closeEpisode();
  refreshView = null;
  const [page = 'overview', id] = (location.hash.slice(1) || 'overview').split('/');
  document.querySelector('#section-label').textContent = titles[page] || 'Overview';
  document.querySelectorAll('nav a').forEach((a) => {
    const active =
      a.hash === `#${page === 'run' ? 'runs' : page === 'experiment' ? 'diagnostics' : page}`;
    a.classList.toggle('active', active);
    if (active) a.setAttribute('aria-current', 'page');
    else a.removeAttribute('aria-current');
  });
  content.replaceChildren(loading());
  try {
    if (!inventory || refresh) inventory = await api('catalog', signal);
    if (current !== generation) return;
    content.replaceChildren();
    if (page === 'overview') await overview(signal);
    else if (page === 'runs' || page === 'diagnostics') experiments(page === 'diagnostics');
    else if (page === 'experiment' && id) {
      const view = await researchPage(decodeURIComponent(id), api, signal);
      if (current !== generation) return;
      content.append(view.page);
      refreshView = view.refresh;
    } else if (page === 'run' && id) await runPage(decodeURIComponent(id), signal);
    else if (page === 'comparisons' || page === 'evidence')
      await reportsPage(page === 'comparisons', signal);
    else if (page === 'caches') caches();
    else content.append(empty('Page not found. Choose a section in the navigation.'));
  } catch (error) {
    if (error.name !== 'AbortError' && current === generation)
      content.replaceChildren(
        head('Unable to read results', 'Check the local server and storage installation.'),
        errorBox(error),
        el('button', { onclick: () => navigate(true) }, 'Retry'),
      );
  }
}
window.addEventListener('hashchange', () => navigate());
document.querySelector('#refresh').onclick = () => navigate(true);
navigate();

async function pollCatalog() {
  if (document.hidden || catalogPolling || !inventory) return;
  catalogPolling = true;
  const current = generation;
  try {
    const latest = await api('catalog');
    if (current !== generation) return;
    if (JSON.stringify(latest) !== JSON.stringify(inventory)) {
      inventory = latest;
      await refreshView?.();
    }
  } catch (error) {
    document.querySelector('#refresh').title =
      `Automatic refresh failed: ${error.message}. Click to retry.`;
  } finally {
    catalogPolling = false;
  }
}
setInterval(pollCatalog, 5000);
window.addEventListener('focus', pollCatalog);
document.addEventListener('visibilitychange', pollCatalog);
