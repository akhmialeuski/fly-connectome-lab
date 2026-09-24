# Objective

T31 (#62) established a large, matched information gap. The encoded input carries identity information (92/200 fit-only OOF top-1, 20 identities), but the original MaleCNS descending-neuron state decodes at chance (10 and 11 of 200). T32 locates **where** and **why** that information disappears, using mechanism-level measurements instead of a larger readout search. It stays on the **fixed-connectome track** of #39: the graph, synaptic weights, neuron model, `dt`, and warmup are not changed. The only manipulated quantity is the amplitude of the existing encoder's current, plus episode noise as a diagnostic.

# Prior work that shapes this study

A literature and open-source review was done before designing this study. Full notes are in the first comment below. The points that decide the design:

| Source | What they did | Lesson for this study |
| --- | --- | --- |
| [Shiu et al. 2024, Nature](https://www.nature.com/articles/s41586-024-07763-9) ([code](https://github.com/philshiu/Drosophila_brain_model)) | Whole-brain FlyWire LIF model, with sensory neurons driven by suprathreshold Poisson input at 100 to 150 Hz, with raw synapse counts | Signals propagate through the fly graph when each input spike is strong and inputs are driven hard |
| `flybrain` 0.1.0 own eye encoder (`eyes.py`, installed source) | Injects up to **0.8 V per step** into identified visual projection types checked to reach descending neurons | Our encoder's cap of **0.05 V per step is 16× lower** than the simulator author's own visual drive |
| [Eon Systems embodied fly](https://eon.systems/updates/embodied-brain-emulation) | Shiu LIF plus flyvis visual activity | Report that visual activity is "somewhat decorative" and barely influences outputs: the same failure we see |
| [Seeslab connectome](https://github.com/eudald-seeslab/connectome), [BPU larva, arXiv 2507.10951](https://arxiv.org/abs/2507.10951) | Fly connectomes classify images only in noise-free rate or message-passing models, with a Kenyon-cell or trained-input readout, often with trained per-synapse gains | Spiking with noise and a fixed random input has no published success, so readout population and delivery matter |
| [Suárez et al. 2021](https://www.nature.com/articles/s42256-021-00376-1), [conn2res 2024](https://www.nature.com/articles/s41467-024-44900-4) | Connectome reservoirs perform best near the edge of stability | Operating regime, not only wiring, determines usable memory and separability |
| [Ganguly et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11228034/), [Li et al. 2020](https://www.cell.com/cell-reports/fulltext/S2211-1247(20)31127-X) | Visual input reaches ~8% of Kenyon cells via ~49 visual projection neurons | Random drive into visual projection neurons mostly misses the visual mushroom-body route |

No prior project was found that decodes face identity with a spiking whole-fly connectome. This study therefore cannot copy a working recipe. It tests the specific failure mechanisms those projects point to.

# Preliminary observations (physiology only, no labels used)

Measured on the unchanged model with the original configuration (`configs/celeba-smoke.yaml`), training photographs only, with the same episode-noise streams for paired stimulus and blank runs:

1. **The image drive is subthreshold.** Rest voltage of a silent neuron is ≈0.772 and threshold 1.0. A constant kick `k` settles at `(0.14 + k) / (1 - exp(-0.2))`. At `k = 0.05` that is 1.048, reached only after ~9 of the 10 steps. Only 6.8% of encoded kicks exceed the 0.0413 needed to cross threshold without a noise kick. The image adds only **about 150 to 240 extra spikes per window among the 3,872 driven neurons, against ~77,000 spikes per window network-wide**.
2. **Evoked activity dies within one or two synapses.** Over four windows at the original amplitude and paired noise, the image changed descending-neuron spikes by **+6 against a background of 1,198**. At 4× and 10× amplitude, evoked spikes reach central-brain intrinsic neurons (+4,700 and +11,600) and descending neurons (+148 and +328).
3. **The mushroom body is saturated at rest.** With original noise, **all 4,064 Kenyon cells fire every step (49.98 Hz, the 50 Hz ceiling) and all 97 MBONs fire at 46 Hz**, before any image is shown. They produce about half of all network spikes. Real Kenyon cells fire sparsely. Without noise the whole network is silent at rest.
4. **Synaptic input is normalized per neuron.** Each neuron's absolute incoming weights sum to 1.0 (`weights.npz`), so with gain 3 a neuron needs roughly 8% of its weighted inputs to fire in the same step to cross threshold from rest. A sparse, weak input cannot recruit that coincidence.

These observations motivate the hypotheses. They are not yet a decoding result.

# Hypotheses

- **H1, input strength.** The information loss starts at injection: the original 0.05 V/step drive is too weak to convert image values into spikes that propagate. Raising only the encoder amplitude makes identity decodable from non-input populations.
- **H2, population access.** Decodability falls with synaptic distance from the injected neurons, and descending neurons, the T31 readout, are among the least accessible populations.
- **H3, operating regime.** Background noise and the saturated mushroom body mask evoked activity, so with noise off the same drive is more decodable.
- **H4, memory.** Separately from H1 to H3, this study records whether the **final** network state, after all 16 windows, still carries identity information about the whole sequence. This is measured here only descriptively, because a matched persistent/reset memory test belongs to a later study.

# Method

- **Cohort and readout (identical to T30/T31).** These are the exact 200 fit photographs of 20 identities from T29 (`research/sequential-visual-memory/2026-09-24-input-access/cohort.json`), the same five stratified folds (`StratifiedKFold(5, shuffle=True, random_state=0)`), with fold-local `StandardScaler` + PCA(60), `LogisticRegression` with C grid {0.01, 0.1, 1, 10}, tolerance 1e-6, 50,000 iterations, and C selected by **pooled fit-only OOF log loss**. The 80 previously inspected queries, the 60 validation photographs, the historical test set, and the reserve photographs are **not used**. Reference, reproduced by the same code before any neural fit: encoded current **92/200**, log loss 2.3765, and pixels 88/200, log loss 2.5068.
- **Simulation.** Original graph, sensory mask, LIF dynamics, 25-step noise-enabled warmup (seed 0), 16 raster windows × 10 steps, persistent state across windows, episode noise from `SeedSequence([seed, stable_int(sample_id)])` as in the runtime. Numba 4 threads per process.
- **Amplitude.** Encoder amplitude multiplied by `s ∈ {1, 2, 4, 8, 16}`, where `s = 16` equals `flybrain`'s own 0.8 V/step visual cap. Noise on (original, episode seed 0) for all conditions, and noise off for `s ∈ {1, 4, 16}` as the H3 diagnostic.
- **Populations** are disjoint: the 3,872 directly driven neurons (positive control for spike conversion, not fly processing), the other 5,329 visual projection neurons, central-brain intrinsic neurons excluding Kenyon cells (28,003), Kenyon cells (4,064), descending neurons (1,314), a fixed random sample of 8,000 optic-lobe intrinsic neurons, and VNC intrinsic neurons (13,161).
- **Representations.** (a) Per-window spike counts of all 16 windows, concatenated: what the population encodes about each glimpse, with memory supplied by the readout. (b) Spike counts of the last window only. (c) Voltage of the final step: the fly's own state after the whole sequence (H4).
- **Controls.** Per-population blank runs quantify evoked-minus-blank activity. Constant features are dropped before scaling. PCA(60) caps every population at the same readout dimension, but raw population sizes differ, which is stated alongside each result.

# Decision rules (fixed before any neural decoding)

- **Delivery restored** for a non-input population at amplitude `s`: its all-window count representation reaches **≥ 46/200 OOF top-1 (half of the input reference) and OOF log loss < ln 20 = 2.996** with original noise.
- **H1 supported** if at least one non-input central population (central-brain intrinsic, Kenyon cells, or descending neurons) meets "delivery restored" at some `s > 1` while it fails at `s = 1`. **H1 rejected** if no non-input population meets it at any `s ≤ 16`.
- **Operating amplitude for follow-up studies:** the smallest `s` at which central-brain intrinsic neurons meet "delivery restored". It is chosen from fit-only OOF results, then **replicated on episode-noise seed 1** before being adopted. If no amplitude qualifies, report that and do not adopt one.
- **H2** is assessed descriptively by ranking populations at each `s`. **H3** is supported for a population if noise-off at the same `s` improves OOF top-1 by ≥ 10 points. **H4** is reported descriptively for the final-voltage representation, and no memory claim is made here.

# Resources and records

Working data under `$FLYSTATE_HOME/runs/diagnostics/2026-09-25-drive-sweep/` on the Linux disk. Spike counts are stored as uint8 (about 230 MB per condition, compressed). They are deleted after the study once the summary tables and the analysis code are archived, since they are exactly regenerable from the recorded configuration. Every condition, failure, timing, and peak memory is reported as a comment here and cross-linked from #39.
