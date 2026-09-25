import { card, el, pct } from './charts.js';

// Classification follows the recorded run contract, never a study-name allowlist.
export const isSequential = (entry) => entry.kind === 'training';

const words = (value) => String(value).replaceAll('_', ' ').replaceAll('-', ' ');
const number = (value) => (Number.isFinite(value) ? Number(value.toPrecision(4)) : 'not recorded');
const stateRule = (p) =>
  p.reset_each_window ? 'state reset before every glimpse' : 'state kept across glimpses';
const GRAPHS = {
  fly: 'the MaleCNS graph',
  degree: 'a degree-preserving shuffle',
  random_target: 'a random-target null graph',
  feedforward: 'the feedforward-only graph (synapses from driven neurons only)',
};

// Reader-facing ideas of diagnostic kinds that do not record their own hypothesis.
const KIND_IDEAS = {
  drive_sweep_record: (p) =>
    `Record spiking fly responses with the image drive scaled ${number(p.amplitude_scale)} times ` +
    `(at most ${number(p.max_kick_v_per_step)} V per step)${p.noise_enabled === false ? ', noise off' : ''}. ` +
    'A separate decode scores them.',
  drive_sweep_decode: () =>
    'Decode identity from each recorded spiking population with the readout used for the encoded input, to locate where identity is lost.',
  rate_access_record: (p) =>
    `Run the same fly graph with graded, non-spiking units (leak ${number(p.leak)}, gain ${number(p.gain)}, ${stateRule(p)}). A separate decode scores them.`,
  rate_access_memory: () =>
    'Test memory: compare the final state carried across glimpses with the reset control on the same photographs, with a Bonferroni-corrected gate.',
  rate_access_memory_curve: () =>
    'Measure, without labels, how well the final state recalls each earlier glimpse input.',
  confirmation_record: (p) =>
    `Record the final graded state of every photograph of the untouched confirmation cohort on ${p.graph === 'fly' ? GRAPHS.fly : GRAPHS.degree}, ${stateRule(p)}.`,
  confirmation_evaluate: () =>
    'Fit the standard readout on training photographs only, then score the untouched held-out photographs once.',
  wiring_record: (p) =>
    `Record the final graded state on ${GRAPHS[p.family] || 'a recorded graph'}` +
    `${p.seed == null ? '' : ` (seed ${p.seed})`} at gain ${number(p.gain)}` +
    `${p.alpha == null ? '' : `, which is ${number(p.alpha)} over its spectral radius ${number(p.giant_component_radius)}`}` +
    `, ${stateRule(p)}.`,
  wiring_select: () =>
    "Freeze each graph family's operating point by training-only cross-validation on the development cohort.",
  wiring_analyze: () =>
    'Pool the untouched cohorts and decide whether the MaleCNS wiring beats null graphs at matched operating points, with a 5-point margin.',
  matched_neural_access: () =>
    "Compare identity decoding from the fly's sampled state with the encoded input on identical photographs, folds and readout.",
  input_access: () =>
    'Measure how much identity the image pixels and encoded currents carry, on training photographs only.',
  input_loss_selection: () =>
    'Choose the input-control regularization by fit-fold log loss, then apply the prespecified gate.',
};

function heldOutSummary(scores) {
  const scored = Object.entries(scores || {}).filter(
    ([, value]) => Number.isFinite(value?.accuracy) && Number.isFinite(value?.held_out),
  );
  if (!scored.length) return null;
  const [best, value] = scored.reduce((top, row) => (row[1].accuracy > top[1].accuracy ? row : top));
  const chance = Number.isFinite(value.chance) ? `, chance ${pct(value.chance)}` : '';
  return `Best held-out accuracy: ${pct(value.accuracy)} (${words(best)}, ${value.held_out} photographs${chance}) across ${scored.length} separately fitted readouts.`;
}

export function interpretation(entry) {
  const p = entry.parameters || {};
  let idea;
  if (typeof p.hypothesis === 'string' && p.hypothesis.trim()) {
    idea = p.hypothesis;
  } else if (isSequential(entry)) {
    const policy =
      {
        persistent: 'Keep neural state between successive image patches.',
        reset: 'Reset neural state between image patches as a memory control.',
        reset_concat:
          'Reset neural state between patches and combine their recorded features in the readout.',
      }[entry.mode] || 'Inspect the recorded memory policy in the configuration.';
    idea = `Recognize identity from sequential image patches. ${policy}`;
  } else if (entry.kind === 'identity_probe') {
    const input = {
      pixels: 'Test identity information in image pixels, without simulating neural memory.',
      encoded:
        'Test identity information in encoded input currents, without simulating neural memory.',
      neural: `Test identity information in saved neural activity using ${p.history === 'last' ? 'the final observation' : 'combined observations'}. This fits a diagnostic readout, not neural synapses.`,
    };
    idea =
      input[p.representation] || 'Test identity information in the recorded representation.';
    if (p.representation === 'neural') {
      if (p.features === 'both') idea += ' Read both saved neural feature blocks.';
      else if (typeof p.features === 'string')
        idea += ` Read only the ${p.features.replaceAll('_', ' ')} block.`;
      if (Object.hasOwn(p, 'pca_components'))
        idea += p.pca_components == null
          ? ' Fit the regularized readout without PCA.'
          : ` Fit PCA with a cap of ${p.pca_components} components.`;
    }
    if (p.train_per_class)
      idea += ` Train on ${p.train_per_class} photographs per identity (subset seed ${p.subset_seed ?? 'not recorded'}).`;
    if (p.label_mode && p.label_mode !== 'true') idea += ` Label control: ${p.label_mode}.`;
  } else if (entry.kind === 'convergence_diagnostic') {
    idea =
      'Check whether the training optimizer converges when its iteration budget increases; this does not measure recognition.';
  } else if (entry.kind === 'cohort_audit') {
    idea =
      'Audit dataset membership and split integrity before interpreting recognition results.';
  } else if (Object.hasOwn(KIND_IDEAS, entry.kind)) {
    idea = KIND_IDEAS[entry.kind](p);
  } else if (typeof entry.kind === 'string' && entry.kind.endsWith('_analysis')) {
    const topic = entry.kind.slice(0, -'_analysis'.length).replaceAll('_', ' ');
    idea = `Compare the recorded ${topic} attempts and apply their saved decision rule.`;
  } else {
    idea =
      'No specific hypothesis is recorded for this experiment kind. Inspect its configuration and evidence.';
  }
  const accuracy = entry.scores?.validation?.accuracy;
  const measurements = entry.report?.budget_measurements;
  const reserve = entry.report?.reserve_count;
  const conclusion = entry.report?.conclusion ?? entry.result_summary;
  const gate = entry.report?.gate ?? entry.gate;
  let result;
  if (entry.status === 'failed')
    result = `Attempt failed. ${entry.error || 'Inspect the recorded error.'} No successful result is implied.`;
  else if (entry.status === 'completed' && typeof conclusion === 'string' && conclusion.trim())
    result = conclusion;
  else if (entry.status === 'completed' && typeof gate === 'string' && gate.trim()) {
    const count = entry.case_count ?? Object.keys(entry.report?.fits || {}).length;
    result = `Recorded decision: ${gate.replaceAll('_', ' ')}.`;
    if (count) result += ` ${count} recorded cases.`;
    const scored = Object.entries(entry.report?.fits || {})
      .map(([name, fit]) => [name, fit?.scores?.validation?.accuracy])
      .filter(([, accuracy]) => Number.isFinite(accuracy));
    if (scored.length) {
      const [bestCase, bestAccuracy] = scored.reduce((best, row) =>
        row[1] > best[1] ? row : best,
      );
      result += ` Best recorded validation: ${pct(bestAccuracy)} (${bestCase}); this is exploratory selection, not a test score.`;
    }
  }
  else if (entry.status === 'completed' && Array.isArray(entry.decisions) && entry.decisions.length)
    result = entry.decisions
      .map((row) => {
        const [low, high] = Array.isArray(row.interval_95_pp) ? row.interval_95_pp : [];
        const sign = row.difference_pp > 0 ? '+' : '';
        return `${words(row.name)}: ${row.decision} (${sign}${number(row.difference_pp)} points, 95% interval [${number(low)}, ${number(high)}]).`;
      })
      .join(' ');
  else if (entry.status === 'completed' && entry.selection && typeof entry.selection === 'object')
    result = `Frozen operating points: ${Object.entries(entry.selection)
      .map(([family, alpha]) => `${words(family)} α ${number(alpha)}`)
      .join(', ')}.`;
  else if (entry.status === 'completed' && heldOutSummary(entry.scores))
    result = heldOutSummary(entry.scores);
  else if (entry.status === 'completed' && Number.isFinite(entry.episodes))
    result = `Recorded the final states of ${entry.episodes} photographs. A separate evaluation scores them.`;
  else if (Array.isArray(measurements) && measurements.length)
    result = `Optimizer convergence: ${measurements.filter((m) => m.converged).length} of ${measurements.length} recorded budget checks converged. These are numerical checks, not recognition scores.`;
  else if (Number.isFinite(reserve))
    result = `Audit recorded ${reserve} reserved photographs. Inspect the membership and exclusion evidence below; reserve photographs were not scored by this audit.`;
  else if (Number.isFinite(accuracy)) {
    result = `Validation accuracy: ${pct(accuracy)}.`;
    if (entry.classes > 0)
      result += ` Uniform-choice reference: ${pct(1 / entry.classes)} (${entry.classes} classes).`;
    result += ' This score alone does not establish a memory benefit.';
  } else if (entry.status === 'completed')
    result =
      'Completed. No validation recognition score is recorded; inspect the reported audit or numerical evidence below.';
  else
    result = `Status: ${entry.status || 'unknown'}. No completed recognition result is available.`;
  return { idea, result };
}

export function interpretationCard(entry) {
  const { idea, result } = interpretation(entry);
  return card('Idea and results', '', el('p', {}, idea), el('p', {}, result));
}
