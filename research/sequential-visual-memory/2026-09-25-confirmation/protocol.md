# Objective

This is the single confirmation of the research programme in #39. T33 (#65) and T34 (#66) established on development photographs that the MaleCNS wiring, run as a quiet graded network, carries the full identity information of the glimpses, and that with slow units the network's own state integrates identity across glimpses. Every development photograph has been looked at repeatedly. T35 therefore tests the frozen model **once**, on **20 identities never used before**, and reports whatever it finds.

The question is the one this PoC started with. After seeing a face only as 16 small patches in sequence, can the fly network's own state at the end identify the person, and does that depend on memory carried across patches?

# Frozen cohort

- `configs/celeba-confirm.yaml`: identical to the development configuration except `selection_seed: 2`. The seed was fixed by a rule written before any confirmation data was touched: the smallest seed ≥ 1 whose 20 identities overlap none of the 120 identities used by any earlier cohort. Seed 1 shares one identity and was rejected.
- 20 identities, 20 photographs each: 280 training photographs (14 per identity) and 120 held-out photographs (6 per identity), using the configuration's own per-identity split. No held-out photograph is used for fitting, scaling, PCA or choosing C.
- Same face alignment, 16-window raster, sparse encoder and seed as all earlier studies. Dataset fingerprint and cache key are recorded in every attempt.

# Frozen model and controls

The operating point comes from T34's preregistered rule: gain **1.0**, leak **0.02**, driven-neuron leak **1.0 (fast)**, input scale 20, 4 updates per window, no noise, `flybrain`'s effective graph. Four recordings of all 400 photographs:

| Recording | Graph | State between windows | Role |
| --- | --- | --- | --- |
| `persistent` | MaleCNS | persistent | The model under test |
| `reset` | MaleCNS | reset before every window | Memory control: the final state sees only the last glimpse |
| `shuffled` | degree-preserving shuffle (seed 0) | persistent | Wiring control: same in-degree, out-degree and presynaptic sign for every neuron, row normalization restored |
| `shuffled-reset` | degree-preserving shuffle (seed 0) | reset | Completes the 2 × 2 design |

Input references on the same photographs: encoded current of all 16 windows, encoded current of the last window only, and pixels of all windows.

**Readout.** The project's standard readout `flystate.readouts.fitting.fit_classifier`: fold-local scaling and PCA(60), logistic regression, C chosen from {0.01, 0.1, 1, 10} by 5-fold CV accuracy on the training photographs only, then refitted on all 280 and applied once to the 120 held-out photographs. Tolerance 1e-6, 50,000 iterations.

**Populations.** Primary: central brain (28,100 cells, excluding Kenyon cells). Secondary: descending neurons (1,314 cells), the original PoC readout. All seven populations are reported.

# Decision rules (fixed before any confirmation recording)

1. **Recognition from the fly's own final state (R1).** The persistent central-brain state scores above chance on the 120 held-out photographs: one-sided exact binomial test against 1/20 with p < 0.001.
2. **Memory (R2).** Persistent minus reset, central brain, is at least **10 percentage points**, with a 95% identity-cluster bootstrap interval (10,000 draws over the 20 identities) that excludes zero.
3. **Specific wiring (R3).** Persistent MaleCNS minus persistent shuffled, central brain, with the same bootstrap. A fly-wiring advantage is claimed only if the interval excludes zero. Otherwise the result is that the memory effect does not depend on the specific MaleCNS wiring beyond its degree statistics and signs.
4. Descending neurons are tested with rules 1 to 3 as a secondary result. McNemar's test is reported for every paired comparison. Every score is reported, including failures.

The overall claim "the fly network recognizes faces from sequential patches using its own memory" is made only if R1 and R2 both hold for the central brain. R3 decides whether that claim is specific to the fly wiring.

# Records

Code on `work/confirmation`, rebased onto T34 as `CODE_COMMIT`. This protocol is committed before the first confirmation recording. Recordings store only final states (about 100 MB each). The archive keeps every attempt's metadata, the held-out predictions and the paired comparisons, and is verified from a fresh clone.
