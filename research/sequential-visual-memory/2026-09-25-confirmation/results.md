## T35 result: confirmed once on untouched identities

All four recordings and the single evaluation ran from the committed protocol (`ce5fd5e`) on the new cohort: 20 identities that no earlier study used, 280 training photographs, and **120 held-out photographs that nothing had touched before this run**. The readout (`fit_classifier`: fold-local scaling and PCA 60, logistic regression, C chosen by 5-fold CV on the training photographs only) was fitted once per case and applied once to the held-out photographs. Chance is 1/20 = 5% (6/120).

### Held-out identification

| Case | Correct of 120 | Accuracy | Wilson 95% | One-sided binomial p against chance |
| --- | ---: | ---: | --- | ---: |
| **MaleCNS, persistent: central brain (primary)** | **63** | **52.5%** | [43.6, 61.2] | 5.1e-49 |
| MaleCNS, persistent: descending neurons | 58 | 48.3% | [39.6, 57.2] | 1.4e-42 |
| MaleCNS, persistent: optic lobe sample | 59 | 49.2% | [40.4, 58.0] | 7.6e-44 |
| MaleCNS, persistent: VNC intrinsic | 56 | 46.7% | [38.0, 55.6] | 4.1e-40 |
| MaleCNS, persistent: other visual projection | 53 | 44.2% | [35.6, 53.1] | 1.6e-36 |
| MaleCNS, persistent: Kenyon cells | 51 | 42.5% | [34.0, 51.4] | 3.5e-34 |
| MaleCNS, persistent: driven neurons (fast) | 15 | 12.5% | [7.7, 19.6] | 9.9e-4 |
| MaleCNS, reset: central brain | 12 | 10.0% | [5.8, 16.7] | 0.017 |
| MaleCNS, reset: descending neurons | 15 | 12.5% | [7.7, 19.6] | 9.9e-4 |
| Shuffled graph, persistent: central brain | 56 | 46.7% | [38.0, 55.6] | 4.1e-40 |
| Shuffled graph, persistent: descending neurons | 54 | 45.0% | [36.4, 53.9] | 1.1e-37 |
| Shuffled graph, reset: central brain | 15 | 12.5% | [7.7, 19.6] | 9.9e-4 |
| Input reference: encoded current, all 16 windows | 61 | 50.8% | [42.0, 59.6] | 2.1e-46 |
| Input reference: encoded current, last window | 15 | 12.5% | [7.7, 19.6] | 9.9e-4 |
| Input reference: pixels, all 16 windows | 66 | 55.0% | [46.1, 63.6] | 4.7e-53 |

The remaining population scores of the reset and shuffled recordings are in the report. Every reset population scores 10.0 to 16.7%, the level of a single last glimpse.

### Paired comparisons on the same 120 photographs

| Comparison | Difference, points | Identity-cluster 95% interval | McNemar (discordant pairs, p) |
| --- | ---: | --- | --- |
| **Central brain: persistent − reset (memory)** | **+42.5** | **[31.7, 53.3]** | 52 against 1, p = 6.5e-12 |
| Descending: persistent − reset | +35.8 | [24.2, 46.7] | 47 against 4, p = 4.1e-9 |
| **Central brain: MaleCNS − shuffled graph (wiring)** | **+5.8** | **[−0.8, 12.5]** | 14 against 7, p = 0.19 |
| Descending: MaleCNS − shuffled graph | +3.3 | [−2.5, 10.0] | 14 against 10, p = 0.54 |
| Shuffled graph: persistent − reset | +34.2 | [25.8, 42.5] | 45 against 4, p = 1.1e-8 |
| Encoded current (all windows) − MaleCNS central brain | −1.7 | [−12.5, 9.2] | 18 against 20, p = 0.87 |

### Decisions under the preregistered rules

- **R1, recognition from the network's own final state: confirmed.** After seeing each face only as 16 sequential 32×32 patches, the central-brain state at the end identifies the person in 63 of 120 new photographs (52.5%, chance 5%, p = 5e-49). Descending neurons, the original PoC readout, reach 48.3%.
- **R2, memory: confirmed.** Resetting the state before each glimpse drops the same readout to 10.0%. The gain from carrying state across glimpses is +42.5 points (interval [31.7, 53.3], McNemar p = 6.5e-12). In descending neurons it is +35.8 points. The fast driven neurons themselves hold nothing (12.5%, the level of the last glimpse), so the memory is held by the recurrent network.
- **R3, specific fly wiring: not established.** A degree-preserving shuffle of the same graph, keeping every neuron's in-degree, out-degree and presynaptic sign and the row normalization, reaches 46.7% with memory (+34.2 points over its own reset). The MaleCNS advantage of +5.8 points has an interval that includes zero (p = 0.19). What this test supports is that a network with the MaleCNS degree distribution, sign structure and normalization, run with slow graded units, can do this. It gives no evidence that the exact MaleCNS wiring is required.
- **The overall claim is confirmed.** Both R1 and R2 hold for the primary population, so the fly-network model recognizes faces from sequential patches using memory held in its own state. The network's single final state is as informative as a linear readout of all 16 raw encoded glimpses (63 against 61/120, difference not significant).

### What "the fly network" means here

The confirmed model is the MaleCNS v1.0 effective graph from `flybrain` (166,700 neurons, 25,088,107 signed, row-normalized edges, sensory inputs masked) with **graded leaky-tanh units** (leak 0.02, gain 1.0, fast input neurons, no noise). No synapse, gain or time constant was trained. The only trained component is the linear readout of the final state. The original `flybrain` spiking regime fails this task (#62, #64). That failure comes from its operating point: subthreshold input, noise, reset to zero, a 100 ms membrane time constant against a 3.2 s sequence, and a saturated mushroom body. The wiring is not the cause.

These are held-out results from a single, preregistered run. Archive and remote verification follow.
