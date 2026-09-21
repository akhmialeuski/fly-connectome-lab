# Sequential visual memory

Research question: does persistent activity in the frozen MaleCNS reservoir improve identity classification over reset-between-window controls when a face is revealed as a sequence of partial observations?

Studies are independent, immutable siblings:

- [2026-09-20-celeba-poc1](2026-09-20-celeba-poc1/README.md): first frozen CelebA protocol, smoke and main runs, persistent/reset/reset-concat comparisons, and the preceding engineering evidence.

The external connectome is fixed. Learning occurs in the per-observation readout: standardization, PCA, and logistic classification. Consequently, archived `snapshot/runs/<run-id>/model/weights.npz` files are the learned experiment coefficients, while upstream `brain/weights.npz` is the separately acquired fixed connectome. These are different artifacts despite sharing a basename.
