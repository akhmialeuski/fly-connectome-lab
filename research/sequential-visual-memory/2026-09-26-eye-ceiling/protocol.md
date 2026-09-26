# T39 protocol: face identity that survives Drosophila compound-eye sampling

Owning issue: [#78](https://github.com/akhmialeuski/fly-connectome-lab/issues/78). Research plan: [#39](https://github.com/akhmialeuski/fly-connectome-lab/issues/39). This protocol, the code and `run.sh` are committed before any T39 computation.

## Question

Our encoder feeds full-resolution 32×32 RGB patches into randomly chosen visual projection neurons, so the network receives far more input information than a fly's eye can deliver. This stage measures the information ceiling of eye-limited input. That is the best held-out identity accuracy of the standard linear readout on the eye's samples alone, with no brain simulation. From it, a frozen rule decides whether faces remain a target for an eye-constrained model.

## Eye model

- **Lattice.** A regular hexagonal lattice with a 4.5° interommatidial angle. The Drosophila eye has about 750 to 900 ommatidia at about 4.5° spacing, and MaleCNS v1.0 annotates 892 right-eye and 879 left-eye columns.
- **Acceptance.** Each ommatidium averages the image under a Gaussian with a full width at half maximum of 8.6° (primary), and 7.7° and 9.5° for sensitivity, which spans the 7.7° to 9.5° range of the literature. The Gaussian is truncated at 3 standard deviations and normalized.
- **Channel.** Luminance with ITU-R BT.601 weights stands in for broadband R1 to R6. An RGB variant is an upper bound for R7 and R8 colour, because the camera has no UV channel.
- **Face size.** The aligned 128×128 face is assumed to span θ ∈ {30°, 60°, 90°, 120°} of visual angle, and the lattice covers exactly that square.

## Cases (24 per cohort, plus the reference)

For every θ:

- `hex_luma_w<θ>`, the primary case
- `hex_luma_w<θ>_a7p7` and `hex_luma_w<θ>_a9p5`, the acceptance bounds
- `hex_rgb_w<θ>`, colour
- `square_luma_w<θ>`, a square lattice with the same number of samples, which separates resolution from geometry
- `hex_point_w<θ>`, point samples without acceptance blur.

`pixels_full` is the full-resolution 128×128×3 image. The T35 evaluation also adds its three input references: encoded current of all windows, of the last window, and pixels of all 16 windows.

## Cohorts and readout

- **Cohorts.** The 100-identity development cohort `configs/celeba-persistent.yaml` has 1,400 training and 600 held-out photographs, and chance is 1%. The 20-identity development cohort `configs/celeba-smoke.yaml` has 280 and 120, and chance is 5%. No untouched cohort is used.
- **Readout.** `fit_classifier`: fold-local standard scaling, PCA up to 60 components, multinomial logistic regression, and C from {0.01, 0.1, 1, 10} by 5-fold CV on training photographs only. It is refitted and scored once on the held-out photographs. The tolerance is 1e-6, and the iteration budget is 500,000 with continuation (the T38 budget).

## Decision rule (frozen)

On the 100-identity cohort, R(θ) = (A_eye(θ) − chance) / (A_pixels_full − chance), where A_eye is the accuracy of `hex_luma_w<θ>`.

| R(90°) | Decision |
| --- | --- |
| ≥ 0.50 | **viable**: faces remain a target for an eye-constrained model, and B3 (retinotopic input through MaleCNS columns) follows |
| 0.25 to 0.50 | **limited**: faces continue only on 20-identity tasks, and B5 (fly-relevant visual tasks) becomes the main line |
| < 0.25 | **not viable**: faces stop being a biological target and stay only as a machine-learning benchmark, and the PoC pivots to B5 |

If the pixel reference is not above chance, R is undefined and the stage reports that instead of a decision. The full R(θ) curve, the 20-identity results and every control are reported.

## Records

The archive holds every attempt's metadata, the lattice coordinates (`lattices.json`), the predictions, the fitted readouts and the decisions, and it is verified from a fresh clone. The sample arrays (`responses.npz`) regenerate deterministically and stay outside Git, with their hashes kept.
