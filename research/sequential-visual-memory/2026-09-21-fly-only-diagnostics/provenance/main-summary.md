### Completed four preregistered main-cohort anchors

All four attempts are now reported individually. Cohort: 100 identities, 1,400 training photographs, 300 unchanged validation photographs, still 14 training photographs per identity. Chance top-1 is 1%. No historical-test or reserve predictions were scored.

| Anchor | Training top-1 | Validation top-1 | Outcome |
| --- | ---: | ---: | --- |
| Pixel-all, PCA60 | Not published | Not published | Failed: L-BFGS exhausted 5,000 iterations at tolerance 1e-6 |
| Encoded-all, PCA60 | Not published | Not published | Same stopping failure |
| Persistent neural-last, PCA60 | 795/1,400 (56.79%) | 1/300 (0.33%) | Completed |
| Reset neural-all, PCA60 | 744/1,400 (53.14%) | 2/300 (0.67%) | Completed |

Attempt reports: [pixel-all](https://github.com/akhmialeuski/fly-connectome-lab/issues/40#issuecomment-5763209476), [encoded-all](https://github.com/akhmialeuski/fly-connectome-lab/issues/40#issuecomment-5763238353), [persistent neural-last](https://github.com/akhmialeuski/fly-connectome-lab/issues/40#issuecomment-5763260816), [reset neural-all](https://github.com/akhmialeuski/fly-connectome-lab/issues/40#issuecomment-5763321313).

The neural anchors remain near/below chance and do not establish useful recognition or a memory advantage. The full-history stage comparison remains numerically incomplete: failed input controls must not be treated as zero accuracy or compared as if they converged. No extra neural variant was promoted from smoke.

Next is the separately preregistered training-only convergence diagnosis. It will reconstruct the first failing fold/C, preserve its coefficients and objective/gradient measurements, and test only a larger iteration cap on that same problem. Existing failed attempts will remain unchanged. Main evidence and replay verification will be added alongside the already archived smoke evidence.
