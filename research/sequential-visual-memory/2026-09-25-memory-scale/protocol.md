# Objective

T33 (#65) showed that the MaleCNS wiring, run with graded dynamics, carries the full input identity information, and that with slow units (leak 0.25) the network's own state after 16 glimpses identifies the person far better than a state reset before every glimpse. In the central brain the persistent state scores about 60/200 against 19/200 for the reset control, which passes the Bonferroni-corrected gate. A label-free memory curve shows why the score stays well below the all-window readout (87 to 94/200): the network keeps a fading trace of only about the last eight glimpses.

T34 asks two questions on the same development photographs:

1. **Memory time scale.** Can slower units extend the memory to the whole 16-glimpse sequence, and raise identity recognition from the final state?
2. **Where the memory lives.** Is it held by the recurrent fly network, or merely by the driven input neurons integrating their own input? A control gives the driven neurons fast dynamics (leak 1.0) while the rest of the network stays slow.

The operating point chosen here is then frozen for a single confirmation on new identities in T35.

# Grounding

In a leaky echo-state network the state is `x <- (1 - a) x + a tanh(...)`, and the linear memory time scale grows roughly as `1 / (a (1 - g ρ))` ([Jaeger et al. 2007, Neural Networks 20(3)](https://doi.org/10.1016/j.neunet.2007.04.016), and [Lukoševičius 2012, A Practical Guide to Applying Echo State Networks](https://doi.org/10.1007/978-3-642-35289-8_36)). The leak rate is the standard knob for matching reservoir memory to the input's time scale. The T33 memory curve measured about 8 windows (32 updates) at leak 0.25. Covering 64 updates therefore calls for leaks around 0.1 or below. Too slow a leak can blur successive glimpses together, which is why several values are tested, not one.

# Frozen grid

Everything is as in T33: the effective `flybrain` graph with sensory mask, `tanh` units, no noise, input scale 20, 4 updates per window, the same 280 training photographs, encoder and populations. Gain is fixed at **1.0**, because in T33 gain changed the leak-0.25 memory score by at most 2 photographs.

| Factor | Values |
| --- | --- |
| Network leak | 0.25 (T33 anchor), 0.1, 0.05, 0.02 |
| Driven-neuron leak | same as network, 1.0 (fast input neurons) |
| State between windows | persistent, reset before each window |

This gives 4 × 2 × 2 = 16 recordings. The anchor `leak 0.25, same, persistent/reset` repeats T33's `g1.0-l0.25` recordings with the updated code and must reproduce their `responses.npz` byte for byte.

Decoding uses the unchanged T30 fit-only OOF readout on the exact 200 T29 photographs and folds, restricted to the **last-window state** of every population (the memory representation), with the input references always included.

# Decision rules (fixed before any T34 recording)

1. **Code parity.** The anchor recordings must match T33 byte for byte, or the study stops for diagnosis.
2. **Memory test.** For each of the 8 persistent settings, the persistent last-window central-brain state is compared with the reset state of the same setting, using the T33 paired identity-cluster bootstrap (10,000 draws, 20 clusters) at the Bonferroni level 1 − 0.05/8. Descending neurons are reported with the same test.
3. **Network, not input, memory.** Memory is attributed to the recurrent network only if a setting with **fast driven neurons** (driven leak 1.0) passes the memory test in the central brain.
4. **Operating point for T35.** Among the settings that pass rule 3, choose the one with the highest persistent last-window central-brain OOF top-1. Ties go to lower OOF log loss, then to the larger leak. If no setting passes rule 3, the best setting that passes rule 2 is chosen and T35 reports that its memory may partly reside in the input neurons.
5. All 16 recordings, every population's score, and failures are reported. The label-free memory curve (held-out R² of each window's input from the final state, with no identity labels) is reported for every persistent setting as descriptive evidence.

# Resources and records

About 1 GB of float32 states per recording under `$FLYSTATE_HOME/runs/diagnostics/2026-09-25-memory-scale/`, deleted after archiving the decoded tables, memory reports and scripts. Code, tests and this protocol are committed before the first T34 recording.
