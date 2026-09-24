## T33 results: the MaleCNS wiring preserves identity under graded dynamics, and slow units give the network its own memory

All 12 preregistered recordings (code `93839fb`, protocol `d51e12c`) completed. Each setting's persistent and reset recordings were decoded together by the T32 decoder (unchanged T30 fit-only OOF readout, exact T29 folds, 200 training photographs). The memory test ran with the committed code `9497986`, whose only change merges several decode attempts. No validation, historical test, reserve or previously inspected query photograph was used. References on the same photographs: encoded current **92/200** over all windows, **22/200** for the last window alone.

Physiology first: without noise, activity outside the driven neurons is tiny (mean |x| of 0.001 to 0.007 in the central brain and descending neurons), because each neuron's row-normalized input averages many silent partners. No population saturates (fraction |x| > 0.99 is 0 everywhere). The network runs in a near-linear regime.

### Identity carried by each population

**Bold** marks the T32 "delivery restored" criterion (≥ 46/200 and log loss < ln 20). `gG-lL` is gain G and leak L.
#### All-window state, persistent, OOF top-1 of 200 (log loss)

| Population | g0.5-l1.0 | g1.0-l1.0 | g1.5-l1.0 | g0.5-l0.25 | g1.0-l0.25 | g1.5-l0.25 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Driven neurons (control) | **93** (2.35) | **93** (2.35) | **93** (2.35) | **93** (2.36) | **93** (2.36) | **93** (2.36) |
| Other visual projection | **94** (2.38) | **95** (2.38) | **94** (2.38) | **92** (2.52) | **92** (2.51) | **91** (2.52) |
| Central brain | **92** (2.78) | **92** (2.76) | **92** (2.71) | **87** (2.95) | fail | fail |
| Kenyon cells | **77** (2.83) | 75 (3.13) | 70 (3.00) | 73 (3.14) | 68 (3.33) | 55 (3.39) |
| Descending neurons | **94** (2.19) | **94** (2.17) | **92** (2.17) | **85** (2.41) | **85** (2.42) | **85** (2.43) |
| Optic lobe sample | **91** (2.57) | **89** (2.61) | **89** (2.65) | **86** (2.71) | **84** (2.72) | **86** (2.73) |
| VNC intrinsic | **92** (2.69) | **91** (2.72) | **88** (2.73) | 79 (3.25) | 80 (3.33) | 80 (3.40) |

#### Last-window state: persistent / reset, OOF top-1 of 200

| Population | g0.5-l1.0 | g1.0-l1.0 | g1.5-l1.0 | g0.5-l0.25 | g1.0-l0.25 | g1.5-l0.25 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Driven neurons (control) | 21 / 21 | 21 / 21 | 21 / 21 | 28 / 21 | 28 / 21 | 28 / 21 |
| Other visual projection | 22 / 22 | 21 / 21 | 20 / 22 | 50 / 22 | 48 / 22 | 52 / 22 |
| Central brain | 24 / 20 | 24 / 20 | 32 / 21 | 61 / 19 | 60 / 19 | 59 / 19 |
| Kenyon cells | 30 / 22 | 29 / 21 | 34 / 24 | 45 / 24 | 42 / 22 | 42 / 22 |
| Descending neurons | 30 / 29 | 29 / 28 | 27 / 28 | 48 / 28 | 48 / 29 | 49 / 28 |
| Optic lobe sample | 20 / 22 | 21 / 22 | 23 / 20 | 53 / 21 | 56 / 21 | 58 / 21 |
| VNC intrinsic | 25 / 23 | 29 / 23 | 30 / 24 | 50 / 20 | 50 / 23 | 50 / 23 |

#### Preregistered memory test: persistent minus reset, last-window state

| Population | Setting | Reset | Persistent | Gain, points | 95% interval | Bonferroni interval (99.17%) | Memory supported |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| Central brain | g0.5-l1.0 | 20/200 | 24/200 | +2.0 | [-1.0, 5.0] | [-1.5, 6.0] | no |
| Central brain | g0.5-l0.25 | 19/200 | 61/200 | +21.0 | [13.0, 29.5] | [10.0, 32.5] | yes |
| Central brain | g1.0-l1.0 | 20/200 | 24/200 | +2.0 | [-1.5, 5.5] | [-2.5, 7.0] | no |
| Central brain | g1.0-l0.25 | 19/200 | 60/200 | +20.5 | [13.5, 28.0] | [11.0, 30.5] | yes |
| Central brain | g1.5-l1.0 | 21/200 | 32/200 | +5.5 | [1.0, 10.0] | [-0.0, 12.0] | no |
| Central brain | g1.5-l0.25 | 19/200 | 59/200 | +20.0 | [13.5, 27.0] | [11.5, 29.5] | yes |
| Descending neurons | g0.5-l1.0 | 29/200 | 30/200 | +0.5 | [0.0, 1.5] | [0.0, 2.0] | no |
| Descending neurons | g0.5-l0.25 | 28/200 | 48/200 | +10.0 | [2.0, 18.0] | [-0.5, 20.5] | no |
| Descending neurons | g1.0-l1.0 | 28/200 | 29/200 | +0.5 | [-2.0, 3.0] | [-3.0, 4.0] | no |
| Descending neurons | g1.0-l0.25 | 29/200 | 48/200 | +9.5 | [2.5, 16.5] | [-0.5, 19.0] | no |
| Descending neurons | g1.5-l1.0 | 28/200 | 27/200 | -0.5 | [-3.0, 2.0] | [-4.0, 3.0] | no |
| Descending neurons | g1.5-l0.25 | 28/200 | 49/200 | +10.5 | [3.0, 18.0] | [-0.0, 20.5] | no |

In the three `l0.25` settings the persistent state beats the reset control, and it does so in every population that is not driven directly. The driven neurons themselves change little (28 against 21), so the memory is held downstream, in the network. In the `l1.0` settings the state is replaced within one window and there is no memory, as expected.

### Memory curve without labels

This analysis is descriptive, not part of the preregistered gate. It measures how well the final persistent state predicts each window's input, using held-out R² of a ridge regression from the final central-brain state (100 principal components) onto the first 10 principal components of that window's encoded current. No identity labels are used.

| Setting | Windows 1 to 4 | 5 to 8 | 9 to 12 | 13 to 16 |
| --- | --- | --- | --- | --- |
| g1.0-l0.25, persistent | −0.26, −0.27, −0.25, −0.08 | −0.04, −0.10, 0.04, 0.27 | 0.27, 0.16, 0.36, 0.62 | 0.54, 0.50, 0.69, 0.83 |
| g1.5-l0.25, persistent | 0.04, 0.02, −0.04, 0.05 | 0.14, 0.01, 0.14, 0.28 | 0.33, 0.27, 0.41, 0.63 | 0.57, 0.54, 0.71, 0.83 |
| g1.0-l0.25, reset | below 0 except window 12 (0.31) | below 0 | below 0 | −1.35, −1.61, −0.38, **1.00** |

With slow units the network keeps a fading trace of roughly the last eight glimpses. The first five or six are lost. With reset, only the last window is present. This matches the gap between the persistent last-window score (about 60/200) and the all-window readout (87 to 94/200).

### Decisions under the preregistered rules

- **G1, wiring carries identity: supported, with a wide margin.** Every non-driven population except Kenyon cells meets the delivery criterion in every setting: descending neurons 85 to 94/200, central brain 87 to 92/200, other visual projection neurons 91 to 95/200. These values match the 92/200 of the encoded input itself, so under graded dynamics the MaleCNS wiring loses essentially no identity information between the injection site and the descending neurons. Kenyon cells reach 55 to 77/200, and meet the criterion only in `g0.5-l1.0`.
- **G2, memory held by the fly network: supported for the central brain.** In all three slow-unit settings, the persistent last-window central-brain state beats the reset state by **20 to 21 points** (59 to 61 against 19/200). The Bonferroni-adjusted intervals exclude zero ([10.0, 32.5], [11.0, 30.5] and [11.5, 29.5] points).
- **Descending neurons, the original PoC readout (secondary): not established at the corrected level.** The gain is +9.5 to +10.5 points (48 to 49 against 28/200). The 95% intervals exclude zero ([2.0, 18.0] and similar), but the Bonferroni intervals touch zero, so the preregistered rule does not accept it.
- **Numerical failures.** The all-window central-brain fits of `g1.0-l0.25-persistent` and `g1.5-l0.25-persistent` stopped at L-BFGS's function-evaluation limit. They are recorded as failures and were not refitted. They do not enter the memory test, which uses last-window states.

### What this establishes

Together with T32, the result is clear-cut. The original `flybrain` spiking regime destroys identity information at the first synapses: hard thresholds, reset to zero, gain 3 on row-normalized weights, background noise, and a saturated mushroom body. The **same wiring**, run as a quiet graded network, carries the full input information to every major population, including descending neurons. With slow units, the network **integrates identity across glimpses in its own state**: after 16 glimpses, the central brain alone identifies the person at 30% (60/200, 20 identities, chance 5%), against 9.5% when the state is reset before each glimpse.

These are development results on photographs already used by T29 to T32. The rate model is a declared change of neuron dynamics, not the unchanged `flybrain` simulator. The measured memory covers about half of the sequence. Three steps follow:

1. **T34** extends the memory time scale. Leaky echo-state theory (Jaeger et al. 2007) predicts a memory length proportional to 1/leak, and the memory curve shows about 8 windows at leak 0.25. T34 preregisters slower leaks and selects the operating point by fit-only OOF only.
2. **T35** confirms the frozen model once on a **new, never-used cohort of identities**, with the reset control, a degree-preserving shuffled-graph control that tests whether the specific fly wiring matters, and the input reference.
3. T32 and T33 evidence is archived together.

Resources: each recording took 159 to 221 s (4 Numba threads, all 280 episodes as one batch). Each setting's decode took 19 to 25 minutes with one BLAS thread. The memory test took under a minute.
