import { el, pct } from './charts.js';

// A class prediction identifies a person. Its photograph is a training example,
// never a generated face or a retrieved test image.
export function face(runId, sampleId, caption, { small = false } = {}) {
  const fallback = el('span', { class: 'face-unavailable', hidden: '' }, 'Photo unavailable');
  const image = el('img', {
    src: `/api/runs/${encodeURIComponent(runId)}/samples/${encodeURIComponent(sampleId)}/image`,
    alt: caption,
    loading: 'lazy',
    width: 128,
    height: 128,
    onerror: () => {
      image.hidden = true;
      fallback.hidden = false;
    },
  });
  return el(
    'figure',
    { class: small ? 'face face-small' : 'face' },
    el('div', { class: 'face-picture' }, image, fallback),
    el('figcaption', {}, caption),
  );
}
export function identityFace(runId, label, identities, small = false) {
  const identity = identities.find((item) => item.label === label);
  if (!identity)
    return el(
      'div',
      { class: 'face-missing' },
      `Identity for class ${label}`,
      el('small', {}, 'Reference photo unavailable'),
    );
  return face(runId, identity.sample_id, `Identity ${identity.identity}`, { small });
}
export function predictionCards(runId, prediction, identities, observationCount) {
  const cards = prediction.top5_labels.map((label, i) =>
    el(
      'div',
      {
        class: `candidate${i === 0 ? ' first-choice' : ''}${label === prediction.y_true ? ' correct-identity' : ''}`,
        'data-label': label,
      },
      el('div', { class: 'candidate-rank' }, i === 0 ? "Model's first choice" : `Choice ${i + 1}`),
      identityFace(runId, label, identities),
      el('strong', { class: 'candidate-probability' }, pct(prediction.top5_probs[i])),
      el('small', {}, 'Model probability'),
      label === prediction.y_true
        ? el('span', { class: 'identity-match' }, 'Matches the actual person')
        : null,
    ),
  );
  return el(
    'section',
    { class: 'visual-predictions' },
    el('h3', {}, `Who does the model think this is?`),
    el(
      'p',
      { class: 'prediction-explanation' },
      `After ${prediction.t} of ${observationCount} image windows, ${prediction.correct ? 'its first choice matches the actual person.' : 'its first choice is a different person.'}`,
    ),
    el('div', { class: 'candidate-grid' }, cards),
    el(
      'p',
      { class: 'readout-note' },
      'These are example photos of the predicted people from the training set. The model selects an identity; it does not generate a face. Percentages are model probabilities for this step, not overall accuracy.',
    ),
    el(
      'details',
      {},
      el('summary', {}, 'Technical class labels'),
      el(
        'p',
        { class: 'mono' },
        `True label ${prediction.y_true}; predicted label ${prediction.y_pred}.`,
      ),
    ),
  );
}
