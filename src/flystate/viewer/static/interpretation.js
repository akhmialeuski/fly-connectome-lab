import { card, el, pct } from './charts.js';

// Classification follows the recorded run contract, never a study-name allowlist.
export const isSequential = (entry) => entry.kind === 'training';

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
