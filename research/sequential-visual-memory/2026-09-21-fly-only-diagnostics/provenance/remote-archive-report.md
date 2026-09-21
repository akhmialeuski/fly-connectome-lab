### Smoke research archive is preserved remotely and verified

Archive commit: [`d12c494172c6a6c148d6c8cfaabc43e0fb160a0e`](https://github.com/akhmialeuski/fly-connectome-lab/commit/d12c494172c6a6c148d6c8cfaabc43e0fb160a0e).
Study: [`research/sequential-visual-memory/2026-09-21-fly-only-diagnostics/`](https://github.com/akhmialeuski/fly-connectome-lab/tree/d12c494172c6a6c148d6c8cfaabc43e0fb160a0e/research/sequential-visual-memory/2026-09-21-fly-only-diagnostics).

- 345 tracked files, including the 344-entry study SHA-256 inventory, two audits, all 18 smoke attempts, 16 fitted readouts, prediction probabilities, CV/feature statistics, failed-attempt logs, and source/environment provenance.
- Approximately 100 MiB of evidence. Git LFS contains 16 learned NPZ payloads totaling **102,603,828 bytes**.
- After pushing, all 16 payloads were fetched from GitHub into a separate empty LFS storage directory. Every fetched byte and SHA-256 matched the preserved model file. This verification did not reuse the checkout's LFS object cache.
- Independent coefficient replay reproduces all **4,800 prediction rows**, with maximum absolute probability difference below **2e-15**. The historical smoke anchor also reproduces its saved parameter arrays exactly.
- All 20 per-attempt issue comments were fetched back and matched to their submitted bodies.
- Source photographs, transformed image arrays, input feature matrices, and downloaded connectome files are excluded; pinned external-input records and the parent-study restoration instructions are retained.

A pre-publication tracked-file audit caught the repository's old log exclusion applying outside its original acquisition subdirectory. The rule now explicitly permits research archive logs, and all 345 inventory files are tracked. The private source-note review remains excluded.

This is the completed P0/smoke snapshot, not a claim that the entire roadmap is done. Main anchors are executing and will be preserved alongside it. Full smoke results and interpretation: [table](https://github.com/akhmialeuski/fly-connectome-lab/issues/40#issuecomment-5763070286), [conclusions](https://github.com/akhmialeuski/fly-connectome-lab/issues/40#issuecomment-5763070651).
