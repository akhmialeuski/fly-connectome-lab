# Frozen-trace neural readout ablation

This dated study belongs to [issue #52](https://github.com/akhmialeuski/fly-connectome-lab/issues/52), under the [fly-only research roadmap](https://github.com/akhmialeuski/fly-connectome-lab/issues/39). The [protocol](protocol.md) and [machine-readable schedule](schedule.json) were frozen before the first new classifier fit. The original 20-identity training and validation membership is recorded in the [preceding study](../2026-09-23-noise-recognition/membership.json).

The source trace is the noise-enabled episode-seed-zero native float32 trace from [issue #50](https://github.com/akhmialeuski/fly-connectome-lab/issues/50). Its independent remote-restoration record is in the [source study](../2026-09-23-noise-recognition/provenance/remote-lfs-verification.json). This study references that immutable source rather than duplicating its neural trace. Images, downloaded connectome files, historical test photographs, and reserve photographs are not stored or scored here.

To reproduce a case, provide the same registered CelebA and verified flybrain files described in the [source study](../2026-09-23-noise-recognition/README.md), set `FLYSTATE_HOME` to the data home, and run from the repository root with Python 3.12 and `uv sync --frozen`. For example, case B1 is:

```bash
uv run flystate diagnose run \
  "$FLYSTATE_HOME/runs/20260920-071344-celeba-smoke-9406ee/config.yaml" \
  --output runs/diagnostics/2026-09-23-readout-ablation/B1 \
  --trace-source runs/diagnostics/2026-09-23-noise-recognition/traces/seed0 \
  --trace-precision float32 \
  --representation neural --history last --features voltage \
  --components 60 --label-mode true --max-iterations 50000 \
  --train-per-class 14 --subset-seed 0 --json
```

Use each scheduled feature block, history, and component setting for B2-B7. `--components 0` denotes B6 without PCA. Working attempts are created only under `FLYSTATE_HOME/runs/diagnostics/2026-09-23-readout-ablation/` and may never overwrite a completed or failed attempt. Each attempt writes a manifest, source provenance, split sample IDs, fold-selection scores, feature statistics, per-sample probabilities, fitted coefficients, and a checksum inventory.

The [results](results.md) and [analysis report](snapshot/analysis/report.json) show all eight cases, their paired uncertainty, independent numeric coefficient replay, and the preregistered decision. `snapshot/` preserves B1-B7 and the analysis byte-for-byte; B0 and native source traces remain in the preceding study. The exact archive member hashes and byte counts are recorded in [local verification](provenance/local-archive-verification.json). Its canonical tree digest is SHA-256 of compact, UTF-8 JSON for the lexically ordered list of `[relative_path, file_sha256]` pairs, with JSON keys sorted and `ensure_ascii=False`.
