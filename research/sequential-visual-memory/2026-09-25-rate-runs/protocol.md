# Objective

T33 to T35 (#65, #66, #70) ran the graded MaleCNS model through dedicated diagnostic commands. Their attempts are not run directories, so the results viewer cannot open them in the episode inspector like the original POC runs. T36 makes the graded model a first-class backend of the standard pipeline (`flystate trace build`, `train`, `evaluate`, `compare`) and produces **standard persistent, reset and reset-concat runs** of the frozen T35 model on the confirmation cohort. Every observation, top-five prediction and neural state can then be inspected episode by episode, exactly like the POC 1 runs.

# Method

- **Backend.** A new configuration value `brain.backend: rate` with a `brain.rate` section (gain, leak, driven-neuron leak, input scale). `RateEpisodeBrain` implements the trace builder's existing `EpisodeBrain` interface on top of the unchanged T33 `RateReservoir`. No other pipeline code changes. `EpisodeBrain` now refuses non-spiking configurations, so no spiking-only command can silently simulate a graded one. The rate model has no spikes, and its graded state is recorded as the `voltage` block. Validation requires a noise-free configuration without warmup.
- **Compatibility.** The `rate` section is omitted from the serialization of spiking configurations. Configuration hashes and trace cache keys of every existing run and study are therefore unchanged. This was checked for all committed configurations; for example, `celeba-smoke` still hashes to `9406ee…`, the suffix of the original 2026-09-20 run.
- **Equivalence.** An offline test shows that the trace cache written by the standard builder equals the T33 direct simulation, rounded to the cache's float16, for persistent and reset modes.
- **Runs.** `configs/celeba-confirm-rate.yaml`: the T35 confirmation cohort (20 never-before-used identities; 280 train, 60 validation, 60 test photographs) and the frozen T35 operating point (gain 1.0, leak 0.02, fast driven neurons, input scale 20, 4 updates per patch). The readout is the original POC readout: descending neurons (1,314 cells), standard per-observation logistic readout with fold-local PCA(60), and C chosen by training-only CV. Reset and reset-concat runs use `--set memory.mode=…`, as the POC 1 script did.
- **Comparisons.** `flystate compare` for persistent against reset, and persistent against reset-concat, as in POC 1.

# Status of the evidence

The model and cohort are frozen from T34 and T35, so no parameter is chosen here. The 120 held-out photographs were already scored once in T35 with a different readout. These standard runs are therefore a replication of the T35 result in the standard pipeline and a way to inspect it, not a new independent confirmation. Results are reported whatever they are.
