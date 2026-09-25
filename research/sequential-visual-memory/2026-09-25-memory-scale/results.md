## T34 results: memory lives in the recurrent network and slower units spread it across the whole sequence

All 16 recordings ran from clean commits (`299c1be` for the anchor, `73bb697` for the rest; `src/` is identical in both), after the provenance and parity corrections reported above. Each setting's persistent and reset recordings were decoded together by the unchanged T30 readout on the exact 200 T29 photographs and folds, restricted to last-window states. The input references were recomputed in every decode and matched T33 exactly: encoded current 92/200 over all windows and 22/200 for the last window. The anchor setting reproduced T33 exactly (central brain 60 against 19, descending 48 against 29, and the same memory curve to two decimals). No validation, historical test, reserve or query photograph was used.

### Final-state identity: persistent against reset, OOF top-1 of 200

`d-same` gives the driven neurons the network's leak. `d1.0` makes them fast, so they cannot hold memory themselves.

| Setting | Central brain | Descending | Driven neurons | Other visual projection | Optic lobe sample | Kenyon cells | VNC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| leak 0.25, d-same | 60 / 19 | 48 / 29 | 28 / 21 | 48 / 22 | 56 / 21 | 42 / 22 | 50 / 23 |
| leak 0.25, d1.0 | 53 / 19 | 43 / 28 | 21 / 21 | 27 / 22 | 50 / 21 | 42 / 22 | 50 / 23 |
| leak 0.1, d-same | 63 / 19 | 55 / 28 | 39 / 21 | 55 / 22 | 63 / 21 | 40 / 22 | 49 / 23 |
| leak 0.1, d1.0 | 54 / 19 | 56 / 28 | 21 / 21 | 48 / 22 | 52 / 21 | 43 / 22 | 53 / 23 |
| leak 0.05, d-same | 64 / 19 | 57 / 28 | 50 / 21 | 61 / 22 | 68 / 21 | 46 / 22 | 52 / 23 |
| leak 0.05, d1.0 | 60 / 20 | 56 / 28 | 21 / 21 | 59 / 22 | 65 / 21 | 48 / 22 | 54 / 23 |
| leak 0.02, d-same | 58 / 19 | 49 / 28 | 57 / 21 | 67 / 22 | 63 / 21 | 50 / 22 | 52 / 23 |
| **leak 0.02, d1.0** | **64 / 21** | **57 / 28** | 21 / 21 | 60 / 22 | 67 / 21 | 46 / 22 | 51 / 23 |

Each cell reads persistent / reset. Chance is 10/200. Reset values are nearly constant because the reset state sees only the last glimpse.

### Preregistered memory test (Bonferroni level 1 − 0.05/8)

| Setting | Central brain gain, points | Bonferroni interval | Descending gain, points | Bonferroni interval |
| --- | ---: | --- | ---: | --- |
| leak 0.25, d-same | +20.5 | [11.0, 30.5] | +9.5 | [−1.0, 19.0] |
| leak 0.25, d1.0 | +17.0 | [9.5, 26.9] | +7.5 | [−1.0, 15.5] |
| leak 0.1, d-same | +22.0 | [12.0, 32.0] | +13.5 | [2.5, 24.4] |
| leak 0.1, d1.0 | +17.5 | [10.0, 26.0] | +14.0 | [1.5, 26.0] |
| leak 0.05, d-same | +22.5 | [11.5, 34.0] | +14.5 | [4.0, 27.5] |
| leak 0.05, d1.0 | +20.0 | [11.5, 29.5] | +14.0 | [2.0, 26.0] |
| leak 0.02, d-same | +19.5 | [11.0, 28.5] | +10.5 | [1.5, 20.5] |
| leak 0.02, d1.0 | +21.5 | [13.0, 30.5] | +14.5 | [3.0, 27.0] |

The central brain passes in all 8 settings. Descending neurons pass in the 6 settings with leak ≤ 0.1. At leak 0.25 their gain stays below 10 points.

### Label-free memory curve, central brain

Held-out R² of each window's input predicted from the final persistent state (committed `rate-memory-curve`, no identity labels):

| Setting | Windows 1 to 4 | 5 to 8 | 9 to 12 | 13 to 16 | Mean |
| --- | --- | --- | --- | --- | ---: |
| leak 0.25, d-same | −0.26 −0.27 −0.25 −0.08 | −0.04 −0.10 0.04 0.27 | 0.27 0.16 0.36 0.62 | 0.54 0.50 0.69 0.83 | 0.20 |
| leak 0.1, d-same | 0.24 0.18 0.17 0.27 | 0.29 0.21 0.13 0.28 | 0.35 0.32 0.23 0.44 | 0.32 0.29 0.38 0.19 | 0.27 |
| leak 0.05, d-same | 0.51 0.31 0.28 0.36 | 0.35 0.22 0.18 0.32 | 0.26 0.26 0.14 0.33 | 0.17 0.08 0.04 0.06 | 0.24 |
| leak 0.02, d-same | 0.59 0.41 0.30 0.35 | 0.34 0.18 0.16 0.33 | 0.21 0.15 0.06 0.28 | −0.03 −0.08 −0.14 −0.13 | 0.19 |
| **leak 0.02, d1.0** | 0.51 0.31 0.23 0.33 | 0.33 0.20 0.14 0.26 | 0.22 0.20 0.14 0.40 | 0.07 0.11 0.22 0.41 | 0.26 |

Leak 0.25 keeps only the last eight glimpses. Leak 0.1 spreads the memory evenly over all 16. Slower leaks with slow input neurons favour the **first** glimpses, because the slow input layer has not yet absorbed the last ones. Fast input neurons restore the late glimpses while the slow network keeps the early ones.

### Decisions under the preregistered rules

1. **Code parity: passed** with the corrected element-wise criterion (see above).
2. **Memory test: passed** in every setting for the central brain, and at leak ≤ 0.1 for descending neurons.
3. **Network, not input, memory: supported.** With fast driven neurons, which score exactly the reset value themselves (21/200), the central brain still holds 53 to 64/200 against 19 to 21, and every such setting passes the corrected test. The memory is held by recurrent MaleCNS dynamics downstream of the injection site.
4. **Operating point for T35: leak 0.02 with fast driven neurons (d1.0), gain 1.0.** Among the four settings that satisfy rule 3, it has the highest persistent central-brain score, 64/200 (32%, chance 5%). It is also the setting with the largest descending-neuron gain (+14.5 points, tied with leak 0.05 d-same).

### Limits

These are development photographs used by every earlier study, and the operating point was chosen on them, so its score is optimistic. The final state reaches about 64/200, against 92/200 when a readout sees every window separately. The network's memory superposes glimpses rather than storing each separately, which costs information. T35 now tests the frozen operating point once on 20 identities never used before, with reset and degree-preserving shuffled-graph controls.
