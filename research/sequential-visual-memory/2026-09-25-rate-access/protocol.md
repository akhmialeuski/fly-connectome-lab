# Objective

T32 (#64) measured where identity information is lost in the original spiking model. At the original drive, the formal decode shows the driven neurons' own spikes at 51/200 against 92/200 for the current injected into them, and every population one or more synapses away at 6 to 14/200, the range of the blank control (chance is 10/200). The remaining T32 amplitudes are still being decoded and are reported in #64. T33 asks the decisive follow-up question, which does not depend on those remaining numbers. **Is the loss caused by the MaleCNS wiring, or by the spiking operating regime that `flybrain` runs on it?** The same effective graph is run as a deterministic graded (rate) network, and the T32 cohort, encoder, windows, populations and readout are reused unchanged. If identity survives in populations the input does not reach directly, the wiring can carry it, and the spiking regime is the bottleneck. If it does not survive even there, the wiring and its row normalization are the bottleneck.

T33 also makes the first direct test of **memory held by the fly network itself**, the original question of this PoC: after the 16th glimpse, does the network state still carry information from earlier glimpses, compared with a control whose state is reset before every glimpse?

# Why a graded model is a legitimate test

Every published success of a connectome used as a computing substrate runs graded dynamics. [wetware](https://github.com/Roxx0x/wetware) (larval connectome, echo-state regime, only the readout learns) reads 8×8 digits at about 94%, which is the level of a linear classifier on the pixels themselves. The [Drosophila ESN study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12109256/) scales FlyWire weights to spectral radius 0.99. fly-self-driving, Fly Dino and ConnecTorch use leaky `tanh` updates on MaleCNS. [Lappalainen et al. 2024](https://www.nature.com/articles/s41586-024-07939-3) model the fly visual system with graded, non-spiking neurons, and many fly neurons (for example in the lamina) are non-spiking. The spiking whole-brain attempts, fly-craftax, the Eon Systems embodied fly and the eye pathway of `flybrain` itself, all report that visual input fails to reach downstream neurons. This is a modelling choice about neuron dynamics, and it is declared as a separate model, not as the unchanged `flybrain` simulator.

# Model

`x[t+1] = (1 - leak) * x[t] + leak * tanh(gain * W_eff @ x[t] + u[t])`

- `W_eff` is the exact effective matrix of `flybrain` with `sensory_input=False`: 25,088,107 signed edges, each neuron's absolute incoming weights summing to 1. It is taken from a `FlyBrain` instance, so the sensory mask is identical. Its spectral radius is **1.000** (next eigenvalues 0.737, 0.730), measured with ARPACK.
- `u[t]` is the original sparse encoder current into the same 3,872 driven neurons, multiplied by `input_scale = 20`, so that it spans [-1, 1], the input range used by Fly Dino and fly-self-driving.
- No noise and no tonic drive. The rest state is `x = 0`, so every result is a deterministic function of the photograph.
- **4 synaptic updates per window**, following fly-self-driving (4 updates per decision) and Fly Dino (3). Central-brain neurons are one to three hops from the driven neurons.

# Frozen grid

| Factor | Values | Reason |
| --- | --- | --- |
| `gain` | 0.5, 1.0, 1.5 | Subcritical, critical, supercritical relative to spectral radius 1, following conn2res and Suárez et al. |
| `leak` | 1.0, 0.25 | No leak, as in wetware-style reservoirs, and slow units that can hold state across windows |
| State between windows | persistent, reset before each window | The memory control of the original PoC design |

This gives 6 dynamics settings × 2 state conditions = 12 recordings. Nothing else is varied.

# Measurements

For each recording, the T32 decoder (the unchanged T30 fit-only OOF procedure on the exact 200 T29 photographs and folds) reads each of the seven T32 populations in two representations:

- **All-window state**: the population state at the end of each of the 16 windows, concatenated. With the reset control, this is the readout-memory ceiling (the readout holds the history).
- **Last-window state**: the state at the end of the 16th window only. With the persistent condition, this is the **memory held by the fly network**. With the reset control, it contains only the last glimpse.

References recomputed on the same photographs: encoded current 92/200 over all windows, and the encoded current of the last window alone.

# Decision rules (fixed before any T33 recording)

1. **Wiring carries identity (G1).** Supported if, in at least one of the six dynamics settings, a population the input does not reach directly (other visual projection neurons, central brain, Kenyon cells, or descending neurons) reaches **≥ 46/200 OOF top-1 with OOF log loss < ln 20** in the persistent all-window state. This is the T32 "delivery restored" threshold, so the two neuron models are judged by the same rule.
2. **Memory held by the fly network (G2).** In each of the six dynamics settings, the persistent last-window central-brain state is compared with the reset last-window state of the same setting, paired on the same 200 photographs. G2 is supported if, in at least one setting, the persistent state wins by **≥ 10 percentage points** and the identity-cluster bootstrap interval (10,000 draws, 20 identity clusters) at the Bonferroni-adjusted level 1 − 0.05/6 excludes zero. Testing every setting with a correction avoids choosing a setting by the same score it is then judged on.
3. Descending neurons, the original PoC readout, are reported with the same paired test, as a secondary result.
4. All 12 recordings and all population and representation scores are reported, whatever the outcome. This is development evidence on the 200 fit photographs. Any positive result requires confirmation on a new, untouched cohort in a separate frozen study, with a degree-preserving shuffled-graph control.

# Resources and records

About 1 GB of float32 states per recording under `$FLYSTATE_HOME/runs/diagnostics/2026-09-25-rate-access/`, deleted after archiving the decoded tables and the analysis code, since the recordings are exactly regenerable. Code, tests and this protocol are committed before the first T33 recording.
