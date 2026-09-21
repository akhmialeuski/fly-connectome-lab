# Versioned research results

This directory preserves computed evidence, not just instructions for rerunning an experiment. Learned model coefficients, neural traces, predictions, metrics, fitted preprocessing metadata, and research reports are part of the Git history. Large numeric payloads use Git LFS; readable metadata remains ordinary Git content.

## Study index

| Experiment family | Study | Description |
| --- | --- | --- |
| Sequential visual memory | [2026-09-20-celeba-poc1](sequential-visual-memory/2026-09-20-celeba-poc1/README.md) | Five trained CelebA runs, six trace caches, calibration, controls, and first frozen comparisons |
| Sequential visual memory | [2026-09-21-fly-only-diagnostics](sequential-visual-memory/2026-09-21-fly-only-diagnostics/README.md) | Cohort audits, 22 smoke/main diagnostic attempts, 18 fitted models, numerical failures, and prediction replay; causal follow-up pending |

## Storage contract for future experiments

```text
research/
  <experiment-family>/
    <YYYY-MM-DD-study-name>/
      README.md                 # hypothesis, protocol, results, limitations, restoration
      experiment.json           # run IDs, roles, cache links, code and data identities
      checksums.sha256          # every archived file except this inventory itself
      external-inputs.json      # exact external URLs, versions, sizes, and SHA-256
      external-inputs.sha256    # checks for separately acquired source files
      snapshot/
        runs/<run-id>/          # immutable original run files and trained coefficients
        runs/<report-kind>/    # comparisons, calibration, benchmarks, design checks, reports
        cache/features/<key>/  # computed neural recordings and their commit digests
      preprocessing/<key>/    # fitted templates, sample indexes, metadata; no image pixels
      provenance/             # protocol freeze and supporting measurement records
      environment/            # dependency lock and project metadata from the training commit
```

Create a new sibling study directory for every new research campaign. A run remains identified by its original run ID, effective configuration hash, dataset fingerprint, cache key, and model digest. Label engineering probes separately from frozen-protocol evidence. Do not overwrite older study directories or rewrite historical run files; record later corrections or additional evaluations in a new study with a reference to the original.

Every study must explain what was trained, what was measured, and how to restore its results without retraining. Record the full source commit, seed, dependency versions, thread count, train/validation/test membership, and external input checksums. All copied artifacts must retain their original bytes. The SHA-256 inventory is the integrity boundary; exclude only the inventory itself to avoid a circular digest.

External source datasets, downloaded connectome files, image archives, and original or transformed photographs are not stored here. They have separate source and checksum records. Derived image caches can be regenerated from the pinned inputs and fitted preprocessing procedure. External availability can change: an unavailable or mismatched source is an explicit reproducibility failure, not permission to use a newer or different dataset silently.

## Git LFS is required for complete results

Install [Git LFS](https://git-lfs.com/) before retrieving numeric artifacts. A normal Git clone without LFS payloads contains small pointer files, not the learned coefficients. GitHub source ZIP downloads are not the supported restoration method.

```bash
git lfs install
git lfs pull --include='research/**'
git lfs fsck
```

To fetch only one study, restrict `--include` to its directory. After fetching, run the study's `sha256sum --check checksums.sha256` before using any model. Never point `FLYSTATE_HOME` at the tracked archive: restore into a new working directory, preserving the archive as immutable evidence.

Git LFS objects must be pushed and fetched successfully before a study is considered archived remotely. Source hashes, Git pointer files, or a local-only commit are not substitutes for preserving the actual learned payloads.
