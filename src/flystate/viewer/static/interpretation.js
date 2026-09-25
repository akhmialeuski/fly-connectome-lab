import { card, el, pct } from './charts.js';

// Classification follows the recorded attempt kind, never a study-name allowlist.
export const isTrainingRun = (entry) => entry.kind === 'training';

// Kinds written by the sequential-patch studies that compare memory policies (T32 to T35).
// Each builds the idea text from the attempt's own recorded parameters.
const SEQUENTIAL_IDEAS = {
  drive_sweep_record: (p) =>
    `Show 16 sequential patches to the spiking fly model at ${p.amplitude_scale ?? '?'}× encoder drive, ${p.noise_enabled === false ? 'without' : 'with'} background noise, and record every population's spikes.`,
  drive_sweep_decode: () =>
    'Decode identity from each recorded spiking population with the fit-only readout, against the encoded input reference.',
  rate_access_record: (p) =>
    `Run the fly wiring with graded units (gain ${p.gain ?? '?'}, leak ${p.leak ?? '?'}${p.driven_leak != null && p.driven_leak !== p.leak ? `, input-neuron leak ${p.driven_leak}` : ''}) and ${p.reset_each_window ? 'reset the state before every patch' : 'keep the state across patches'}.`,
  rate_access_memory: () =>
    'Test whether keeping the network state across patches identifies people better than resetting it, paired by photograph.',
  rate_access_memory_curve: () =>
    'Measure, without identity labels, how much of each earlier patch the final network state still holds.',
  confirmation_record: (p) =>
    `Record the frozen graded fly model on never-used identities (${p.graph === 'degree_preserving_shuffle' ? 'degree-preserving shuffled graph' : 'MaleCNS graph'}, ${p.reset_each_window ? 'state reset before every patch' : 'state kept across patches'}).`,
  confirmation_evaluate: () =>
    'Score every held-out photograph once and compare memory against the reset and shuffled-graph controls.',
};
export const isSequential = (entry) =>
  isTrainingRun(entry) || Object.hasOwn(SEQUENTIAL_IDEAS, entry.kind);
// Held-out cases summarized for a confirmation evaluation, in reading order.
const CONFIRMATION_CASES = [
  ['persistent/central_brain', 'central brain with memory'],
  ['reset/central_brain', 'reset before every patch'],
  ['shuffled/central_brain', 'shuffled graph with memory'],
];

export function interpretation(entry) {
  const p = entry.parameters || {};
  let idea;
  if (typeof p.hypothesis === 'string' && p.hypothesis.trim()) {
    idea = p.hypothesis;
  } else if (Object.hasOwn(SEQUENTIAL_IDEAS, entry.kind)) {
    idea = SEQUENTIAL_IDEAS[entry.kind](p);
  } else if (isTrainingRun(entry)) {
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
  else if (entry.status === 'completed' && entry.kind === 'confirmation_evaluate') {
    const cases = CONFIRMATION_CASES.filter(([key]) => entry.scores?.[key]?.held_out);
    const chance = entry.scores?.[cases[0]?.[0]]?.chance;
    result = cases.length
      ? `Held-out photographs: ${cases
          .map(([key, label]) => {
            const score = entry.scores[key];
            return `${label} ${score.held_out_correct}/${score.held_out} (${pct(score.accuracy)})`;
          })
          .join(', ')}.${Number.isFinite(chance) ? ` Chance ${pct(chance)}.` : ''}`
      : 'Completed. Inspect the recorded held-out scores below.';
  }
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
