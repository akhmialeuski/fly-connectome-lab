# T27 stage-0 results: image delivery through the fixed fly connectome

The preregistered pilot in [issue #54](https://github.com/akhmialeuski/fly-connectome-lab/issues/54) completed on 2026-09-24. It used eight **training** photographs from four identities, three isolated raster windows per photograph, and 1,314 fixed cells in each of four anatomical populations. These 24 image/window units were replayed independently for seven conditions. The original C0/C6 blank attempts and the two corrected blank attempts remain in the archive; the correction and its exact overlap verification are documented in [correction-2026-09-24.md](correction-2026-09-24.md).

All nine source inventories passed. C1's terminal state and external descending-neuron trace matched an independent replay through the original runtime exactly. Corrected blank trajectories match the original blanks at every common checkpoint, including all saved numeric arrays. C0R and C1-C4 used exactly matched episode-noise streams over shared physical steps; C5/C6R had zero episode-noise kicks. The analysis wrote 4,512 per-image/window/checkpoint/population rows to `snapshot/analysis/per-unit.parquet` and its complete machine-readable summary to `snapshot/analysis/report.json`.

The table reports the median across 24 paired units of each population's endpoint RMS voltage difference from its time-matched blank, in volts. Each RMS is computed over the same 1,314 selected neurons, using float64 accumulation on saved float32 voltages. This measures signal delivery, not identity decoding.

| Condition | Stimulus time | Visual projection | Optic-lobe intrinsic | Central-brain intrinsic | Descending neuron |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1: original current, noise on | 0.20 s | 0.1540 | 0.0567 | 0.1468 | 0.1101 |
| C2: original current, twice as long | 0.40 s | 0.1773 | 0.1133 | 0.2497 | 0.1929 |
| C3: half current, twice as long | 0.40 s | 0.1503 | 0.0956 | 0.2382 | 0.1838 |
| C4: double current | 0.20 s | 0.2087 | 0.0790 | 0.1745 | 0.1329 |
| C5: original current, episode noise off | 0.20 s | 0.1011 | 0.0052 | 0.0964 | 0.0286 |

The predefined stage gate passes: the original C1 stimulus has a finite, nonzero median evoked endpoint response in **4/4** observed populations. The necessary next experiment is a separately frozen, training-only expansion across the 20 original identities. The pilot cannot identify a best population or conclude that a representation preserves identity.

The response does not simply disappear when the input stops. At 0.20 s after C1 offset, the corresponding median RMS differences are 0.1383, 0.1117, 0.2441, and 0.1915 V in the table's population order. Because the circuit is recurrent, has tonic drive, and the blank is also active, these late paired differences are observations of changed trajectories; they are **not** evidence of usable sequential memory. The per-unit artifact includes all predeclared checkpoints, signed and absolute voltage contrasts, spike counts, near-threshold fractions, silent/high-rate fractions, and physical times. In C1, mean silent fractions during the stimulus were 85.2% visual projection, 88.6% optic-lobe intrinsic, 53.7% central-brain intrinsic, and 75.8% descending neuron. These are descriptive, selected-mask measurements, not population-wide estimates.

This small training-only cohort cannot estimate recognition accuracy, within- versus between-identity separability, or generalization. C1-C5 change current, duration, or episode noise without retraining, so the table is a physiology map rather than a model comparison. Historical validation/test/reserve images and scores were not used for this decision. No recurrent weights, plasticity rule, fly equation, encoder topology, or specialist vision model changed.

The `snapshot/` directory is an exact byte-for-byte copy of the ten immutable working attempts (nine response cases plus analysis): 69 files, 42,327,958 bytes before Git LFS pointer substitution. All nine `responses.npz` files use Git LFS. [Source-copy verification](provenance/source-copy-verification.json) records every relative filename, size, SHA-256, and the canonical inventory hash. External CelebA images and MaleCNS files are identified by hashes in `cohort.json` and `population-masks.json`; they are not committed.

An independent [fresh-clone verification](provenance/remote-lfs-verification.json) restored all nine LFS arrays from GitHub and checked all 69 file hashes, the 4,512-row Parquet analysis, the nine-attempt control gate, and the recorded stage decision at remote commit `36bb51a425546cdb35a0f1b33336e6fedf10fa3a`.
