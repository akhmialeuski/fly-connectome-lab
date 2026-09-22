### P0 follow-up: effective graph identity

The runtime graph has now been explicitly reconstructed using the same sensory-row masking and CSC conversion as the installed flybrain 0.1.0 implementation. No model parameter or graph policy was changed.

- Source: 25,582,938 stored connections.
- `sensory_input=false`: incoming rows of 17,937 neurons whose superclass contains `sensory` are masked.
- Effective graph: 25,088,107 connections (494,831 removed), comprising 15,431,166 positive and 9,656,941 negative entries.
- Exact float32 CSC data, int32 indices, and int32 indptr digests are recorded; both cohorts produce identical graph identities.

This establishes which graph the original experiments actually simulated. It does not establish that sensory masking caused poor recognition. Any future masking comparison must be a separately declared intervention.

Source-file and effective-array hashes: [P0 evidence](https://github.com/akhmialeuski/fly-connectome-lab/issues/40#issuecomment-5762869013). Working audits are preserved under the new diagnostics study, separately from the original archive.
