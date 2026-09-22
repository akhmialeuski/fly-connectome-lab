## Independent restore verification: main diagnostic payloads

The four main-cohort Git LFS payloads (two fitted readout NPZ files and two training-prediction Parquet tables; 25,486,298 bytes total) were fetched from origin at `f821470ac263a584eb2a42e05108030d7fbb2630` into a separate object store. All SHA-256 digests match the study inventory and the local archived bytes.

Together with the earlier smoke verification, this verifies remote availability of the original 18 fitted diagnostic models. Independent coefficient replay already reproduced all 8,200 stored training/validation prediction rows; the main subset's maximum probability difference was 7.22e-15. This verification does not yet cover the later convergence or retry attempts.

The machine-readable report is preserved as `diagnostics-main-remote.json` under the working verification directory and will be copied into this study's provenance on the next archive update. Research artifacts remain in `research/sequential-visual-memory/2026-09-21-fly-only-diagnostics/`; external photographs and downloaded connectome inputs remain excluded.
