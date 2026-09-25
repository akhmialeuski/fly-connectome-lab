# T38 protocol: the confirmed graded model at 100 identities, over encoder seeds and blank delays

Owning issue: [#75](https://github.com/akhmialeuski/fly-connectome-lab/issues/75). Research plan: [#39](https://github.com/akhmialeuski/fly-connectome-lab/issues/39).
This protocol and `run.sh` are committed before any T38 recording.

## Questions

T35 ([#70](https://github.com/akhmialeuski/fly-connectome-lab/issues/70)) confirmed on 20 untouched identities that the graded MaleCNS network recognizes faces from 16 sequential patches and holds the evidence in its own state (52.5% against 10.0% with reset, chance 5%). T38 tests the same frozen model on three open P6 items of #39:

1. **Scale.** Recognition and memory on the original PoC task of 100 identities, where chance is 1%.
2. **Encoder robustness.** The same measurements for five realizations of the random sparse encoder and its driven-neuron sample.
3. **Forgetting.** How much identity remains in the state after 1 to 16 blank windows that follow the last glimpse.

## Fixed model

This is T35's confirmed operating point, unchanged:

$$x_{t+1} = (1 - \lambda)\, x_t + \lambda \tanh\!\left(g\, W x_t + u_t\right)$$

- The graph `W` is `flybrain`'s effective MaleCNS matrix, unshuffled and untrained.
- Gain `g = 1.0`. Leak `λ = 0.02` for the non-driven neurons and 1.0 for the driven neurons.
- The input `u` is the sparse encoder current × 20. There are 4 updates per window, no noise, and the rest state is zero.

T37's development selection (α = 0.25) is not adopted. It rests on a flat 20-identity CV landscape and was not confirmed.

**Readout.** Standard `fit_classifier`: fold-local standard scaling, PCA 60 and logistic regression. C is chosen from {0.01, 0.1, 1, 10} by 5-fold CV accuracy on the 1,400 training photographs only, then the model is refitted and applied once to the 600 held-out photographs (validation and test splits). Tolerance 1e-6, 50,000 iterations. Fitted readouts are archived in their full scaler, PCA and logistic form (`export_classifier`). With 100 classes, that form is smaller than the affine map used in T37.

## Cohorts

- **Development.** `configs/celeba-persistent.yaml`: 100 identities, selection seed 0, 14 training, 3 validation and 3 test photographs per identity. These photographs were inspected with the spiking model on 2026-09-20 and have never been used with the graded model. All development numbers are reported as development only.
- **Confirmation.** The same configuration with `dataset.subset.selection_seed=92`. The seed was found by an identity-only search, `analysis/select_confirmation_cohort.py`, whose output is in `analysis/select_confirmation_cohort.json`. The search looked for the first selection seeds ≥ 1 whose 100 identities overlap none of the 340 identities used by any earlier configuration or cohort. Those are seed 0 with 100 and with 20 identities, seed 2, and the ten T37 cohorts 4, 5, 6, 8, 10, 11, 15, 22, 24 and 25. The first clean seeds were 92, 232 and 383, and 92 is used. The confirmation cohort is recorded and evaluated once.

## Recordings

| Cohort | Encoder seed | State | Blank delays kept, windows |
| --- | --- | --- | --- |
| Development | 0, 1, 2, 3, 4 | persistent | 0, 1, 2, 4, 8, 16 |
| Development | 0, 1, 2, 3, 4 | reset before every window | 0 |
| Confirmation | 0 | persistent | 0, 1, 2, 4, 8, 16 |
| Confirmation | 0 | reset before every window | 0 |

- A blank window has zero encoder current and the same 4 updates, and the state carries through it. The reset control has no delays, because a reset before a blank window empties the state by construction.
- Photographs are simulated in batches of 250. Batching changes no value, which a test checks bit for bit.
- Stored states are the final central-brain (28,100 cells) and descending-neuron (1,314 cells) states, float32, outside Git.

## Scored cases

Every evaluation scores:

- the central brain at each delay and the descending neurons at delay 0 for the persistent recording
- the central brain and descending neurons for the reset recording
- persistent minus reset for both populations as paired comparisons, with a 10,000-draw identity-cluster bootstrap and McNemar's test.

Development seed 0 and the confirmation also score the three input references: encoded current of all windows, of the last window, and pixels of all windows.

## Decision rules

- **S1, recognition at scale (confirmation, primary).** The persistent central-brain state at delay 0 scores above the 1% chance level on the 600 held-out photographs, by a one-sided exact binomial test with p < 0.001.
- **S2, memory at scale (confirmation).** Persistent minus reset on the central brain is at least 10 percentage points, and its 95% identity-cluster interval excludes 0.
- **E, encoder robustness (development).** Every encoder seed satisfies the S2 criterion on its development cohort. The mean, standard deviation and range of accuracy and memory gain across seeds are reported.
- **F, forgetting (development and confirmation, descriptive).** Accuracy is reported at each blank delay. The half-retention delay is the delay at which accuracy above chance falls to half its delay-0 value, found by linear interpolation. If that never happens within 16 windows, it is reported as more than 16.

**Stated prediction for F.** The linearized slowest recurrent mode decays by $(0.98 + 0.02 \cdot 0.737)^4 \approx 0.979$ per window, which is a time constant of about 47 windows. Faster modes carry most of the identity signal, so observed half-retention is expected to be well below $47 \ln 2 \approx 33$ windows. A half-retention above that bound would contradict the linear picture.

Every score is reported, including failures. S1 and S2 decide the scale claim, and E decides robustness.

## Resources and records

- Code on `work/scale`, stacked on T37 (`work/wiring`). Recording starts only after T37's Phase B and archive are complete, so two simulations never run at once. `CODE_COMMIT` and `git_dirty` are recorded in every attempt manifest.
- Numba threads: 12. Estimated time: about 25 minutes per persistent recording with delays, about 13 minutes per reset recording, and roughly 5 to 6 hours in all.
- `run.sh` stops if C: has less than 1 GB free. States are deleted only after evaluation, independent replay and archiving, and their SHA-256 values stay in each attempt inventory.
