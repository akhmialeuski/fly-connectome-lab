# T37 protocol: wiring specificity at matched operating points

Owning issue: [#74](https://github.com/akhmialeuski/fly-connectome-lab/issues/74). Research plan: [#39](https://github.com/akhmialeuski/fly-connectome-lab/issues/39).
This protocol is committed before any T37 recording. The Phase A selection is committed before any Phase B recording.

## Question

T35 ([#70](https://github.com/akhmialeuski/fly-connectome-lab/issues/70)) confirmed that the graded MaleCNS network recognizes faces from 16 sequential patches with memory held in its own state (52.5% against 10.0% with reset, chance 5%). It could not decide whether the specific MaleCNS wiring matters. MaleCNS beat one degree-preserving shuffle by +5.8 points, with interval [−0.8, 12.5] and McNemar p = 0.19.

T37 asks whether MaleCNS outperforms null graphs that keep its degree statistics, signs, weights and row normalization. The comparison is made at the same dynamical operating point, with enough photographs to resolve a 5-point difference.

## Measurements that motivate the design

Measured on `flybrain`'s effective matrix (`sensory_input=False`) with the committed functions of this study:

| Graph | Edges | Self-loops | Reciprocated edges | Giant-component spectral radius |
| --- | ---: | ---: | ---: | ---: |
| MaleCNS | 25,088,107 | 65 | 28.4% | 0.73681 |
| Degree-preserving shuffle, seed 0 | 25,000,270 | 357 | 0.51% | 0.25673 |
| Random-target null, seed 0 | 25,050,783 | 158 | not measured | recorded by Phase A |

- The full MaleCNS matrix also has eigenvalues +1 and −1. They come from one isolated pair of ENS neurons, bodies 41064 and 111705, each of which is the other's only input. A directed graph's spectrum is the union of the spectra of its strongly connected components, so this pair does not interact with the network. The operating radius is therefore that of the giant strongly connected component (147,789 neurons).
- At equal gain, the T35 shuffle was about three times more contracting than MaleCNS. Its +5.8-point gap therefore mixes wiring with operating point.
- `giant_component_radius` uses ARPACK with a fixed start vector, six requested eigenvalues and one BLAS thread. With one requested eigenvalue ARPACK settled on a close neighbour (0.2361 instead of 0.2365 on the synthetic test graph), and with a random start the value varied in the 14th digit. Both defects were found by tests and fixed before any recording. The MaleCNS and shuffle radii in the table were recomputed with the fixed function and match the exploratory values to five digits. The random-target radius is left to the Phase A attempts, because the only earlier value came from the unreliable single-eigenvalue solve.
- Each radius is cached under `FLYSTATE_HOME/cache/spectral-radius/`, keyed by the solver version and the SHA-256 of the exact CSR arrays. Every recording states whether its radius was computed or read from that cache, so all recordings of one graph use exactly the same gain.

## Related work

- conn2res (Suárez et al., [Nat. Commun. 2024](https://www.nature.com/articles/s41467-024-44900-4), with [code](https://github.com/netneurolab/conn2res) whose `Conn.normalize` divides by the largest eigenvalue modulus) scales every empirical or null matrix to a common spectral radius α. It compares against 500 degree-preserving rewired nulls, and the human connectome's memory-capacity advantage appears near α = 1.
- [Topological Sensitivity in Connectome-Constrained Neural Networks (arXiv 2604.04033)](https://arxiv.org/abs/2604.04033) reports that topology advantages of Drosophila-connectome networks largely disappear under degree-preserving nulls with fair initialization.
- [Morra and Daley, Using Connectome Features to Constrain Echo State Networks (arXiv 2206.02094)](https://arxiv.org/abs/2206.02094) compared a fly-connectome ESN with sparsity-matched random wiring.

## Fixed model

Identical to T35: graded leaky-tanh units `x ← (1 − leak) x + leak · tanh(gain · W x + u)`, with leak 0.02 for non-driven neurons and 1.0 for the driven neurons. The input is the T35 sparse encoder current × 20, 4 updates per window, 16 raster windows, no noise, and zero rest state. No fly parameter is trained.

**Readout**, identical to T35: `flystate.readouts.fitting.fit_classifier`, with fold-local standard scaling, PCA 60 and multinomial logistic regression. C comes from {0.01, 0.1, 1, 10} by 5-fold CV accuracy on the 280 training photographs, then the model is refitted on all 280 and applied once to the 120 held-out photographs. Tolerance 1e-6, 50,000 iterations.

**Populations.** Primary: central brain (`cb_intrinsic` without Kenyon cells or driven neurons, 28,100 cells). Secondary: descending neurons (1,314 cells). Only these two final states are stored.

**Learned readouts** are archived as their exact affine form, `weights = coef · components / scale` with the matching bias (`export_linear_readout`), about 4.5 MB per central-brain case instead of 13.5 MB for the PCA basis. An independent replay recomputes every held-out prediction from these arrays and the recorded states before the states are deleted.

## Graphs

| Name | Construction | Kept | Destroyed |
| --- | --- | --- | --- |
| `fly` | Effective MaleCNS graph | Everything | Nothing |
| `degree<s>` | `degree_preserving_shuffle(seed=s)`: permute all edge targets, merge parallel edges, restore each row's absolute sum | In- and out-degree, presynaptic sign and weights, row normalization | Partners, reciprocity, cell-type blocks |
| `random<s>` | `random_target_shuffle(seed=s)`: draw every target uniformly from neurons with inputs, merge, restore row sums | Out-degree, presynaptic sign and weights, row normalization | In-degree distribution as well |
| `feedforward-g1` | `feedforward_only`: keep only synapses whose presynaptic neuron is driven, weights unchanged, no renormalization | The direct drive every neuron receives from the driven neurons | Every synapse from a non-driven neuron |

Gain is `alpha / radius` of each graph's own giant component, except where a gain is stated.

## Phase A: operating point on the development cohort

- Cohort: `configs/celeba-confirm.yaml` with `selection_seed: 0`, which is the 20-identity development cohort already used by T32 to T34. It has 280 training and 120 held-out photographs.
- Grid: α ∈ {0.25, 0.5, 0.75, 1.0, 1.25} for `fly`, `degree0` and `random0`, 15 recordings in total.
- **Selection rule.** For each family, take the α with the highest best-C mean 5-fold CV accuracy of the central-brain readout on the 280 training photographs. Ties go to the smaller α. Held-out development accuracy is reported but not used.
- The selection file is copied to `selection.json` in this directory and committed before Phase B.

## Phase B: ten untouched cohorts

- Cohorts: `selection_seed` ∈ {4, 5, 6, 8, 10, 11, 15, 22, 24, 25}. These are the first ten seeds ≥ 3 whose 20 identities overlap neither the 140 identities of every earlier configuration (seed 0 with 100 identities, seed 0 with 20, seed 2 with 20) nor each other. The check covered all archived configurations. Rejected seeds: 3, 7, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21, 23. The result is 200 identities, 2,800 training and 1,200 held-out photographs.
- Recordings per cohort:
  - `fly` at the selected fly α
  - `degree0` to `degree4` at the selected degree α
  - `random0` and `random1` at the selected random-target α
  - `fly-g1` and `fly-g1-reset` at gain 1.0, persistent and reset
  - `degree0-g1` at gain 1.0
  - `feedforward-g1` at gain 1.0.
- `run.sh phase-b` records, evaluates each cohort and pools all cohorts.

## Decision rules

Differences are pooled over all 1,200 held-out photographs, in percentage points. The 95% interval comes from 10,000 two-level bootstrap draws. Each draw resamples identities with replacement within every cohort and, independently, the ensemble's null seeds with replacement. The point estimate is the pooled accuracy of the tested graph minus the mean pooled accuracy of the ensemble.

The smallest effect of interest is **5 percentage points**. Each interval is read one way only:

| Interval | Reading |
| --- | --- |
| Entirely above 0 | Advantage |
| Entirely below 0 | Disadvantage |
| Inside (−5, +5) and containing 0 | No practically relevant difference |
| Otherwise | Inconclusive |

| Rule | Contrast (central brain first, descending neurons as secondary) | Question |
| --- | --- | --- |
| **W1 (primary)** | `fly` minus mean of `degree0`…`degree4` | Does the specific wiring beat degree-preserving nulls at matched operating points? |
| W2 | `fly` minus mean of `random0`, `random1` | Does it beat nulls that also lose the in-degree distribution? |
| M1 | `fly-g1` minus `fly-g1-reset` | Does the T35 memory effect replicate? It must be ≥ 10 points with the interval above 0. |
| M2 | `fly-g1` minus `feedforward-g1` | Does reverberation among non-driven neurons add to per-neuron leaky integration? |
| R | `fly-g1` minus `degree0-g1` | T35's exact wiring comparison, now with ten times the photographs |

Single-null contrasts also report exact McNemar tests. Every cohort's value and every failure are reported.

**Power.** In T35, MaleCNS and the shuffle disagreed on 21 of 120 photographs. At that rate the standard error of the pooled difference over 1,200 photographs is about 1.2 points, so the 95% half-width is about 2.5 to 3 points after identity clustering and seed resampling. A true difference of +5.8 points would then give an interval above 0 with probability of about 0.95. A true difference near 0 would give an interval inside ±5 points.

## Resources and records

- Code: branch `work/wiring`, stacked on `feat/separate-diagnostics-view`. `CODE_COMMIT` is recorded in every attempt manifest, and a dirty tree is visible there.
- Numba threads: 12. The propagation kernel sums each row in one thread in edge order, so states do not depend on the thread count. A unit test checks this.
- The work runs in a separate clone on the Linux file system. Writing a 200 MB probe there did not grow the WSL disk image on C:. `run.sh` stops if C: has less than 1 GB free.
- Stored states are the central-brain and descending final states, float32, about 47 MB per recording, outside Git under `FLYSTATE_HOME`. Their SHA-256 values stay in each attempt inventory. They are deleted only after evaluation, independent replay and archiving.
- The archive holds every attempt's metadata, predictions, learned affine readouts, the selection, the analysis and `inventory.sha256`. It is verified from a fresh clone before the study is called archived.
