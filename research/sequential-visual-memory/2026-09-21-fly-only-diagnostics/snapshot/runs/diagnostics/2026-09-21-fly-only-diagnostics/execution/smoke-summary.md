### Completed preregistered smoke queue

All 18 attempts completed or failed explicitly; none was silently retried. Validation uses the same 60 photographs (three per identity); chance top-1 is 5%. Training-only memorization is not generalization.

| # | Diagnostic | Training top-1 | Validation top-1 | Best training CV | Evidence |
| ---: | --- | ---: | --- | ---: | --- |
| 01 | pixels; persistent; last; both; PCA 60; true | 60.71% | 10/60 (16.67%) | 11.43% | [attempt](https://github.com/akhmialeuski/fly-connectome-lab/issues/40#issuecomment-5762879086) |
| 02 | pixels; persistent; all; both; PCA 60; true | Failed: ConvergenceWarning | — | — | [attempt](https://github.com/akhmialeuski/fly-connectome-lab/issues/40#issuecomment-5762887728) |
| 03 | encoded; persistent; last; both; PCA 60; true | 69.29% | 11/60 (18.33%) | 11.43% | [attempt](https://github.com/akhmialeuski/fly-connectome-lab/issues/40#issuecomment-5762898053) |
| 04 | encoded; persistent; all; both; PCA 60; true | Failed: ConvergenceWarning | — | — | [attempt](https://github.com/akhmialeuski/fly-connectome-lab/issues/40#issuecomment-5762913493) |
| 05 | neural; persistent; last; both; PCA 60; true | 100.00% | 4/60 (6.67%) | 6.07% | [attempt](https://github.com/akhmialeuski/fly-connectome-lab/issues/40#issuecomment-5762922548) |
| 06 | neural; reset; last; both; PCA 60; true | 100.00% | 2/60 (3.33%) | 7.14% | [attempt](https://github.com/akhmialeuski/fly-connectome-lab/issues/40#issuecomment-5762931967) |
| 07 | neural; reset; all; both; PCA 60; true | 99.64% | 4/60 (6.67%) | 4.64% | [attempt](https://github.com/akhmialeuski/fly-connectome-lab/issues/40#issuecomment-5762945554) |
| 08 | neural; persistent; all; both; PCA 60; true | 100.00% | 4/60 (6.67%) | 5.36% | [attempt](https://github.com/akhmialeuski/fly-connectome-lab/issues/40#issuecomment-5762958354) |
| 09 | neural; persistent; last; both; PCA 20; true | 28.57% | 1/60 (1.67%) | 6.07% | [attempt](https://github.com/akhmialeuski/fly-connectome-lab/issues/40#issuecomment-5762967048) |
| 10 | neural; persistent; last; both; PCA 120; true | 100.00% | 2/60 (3.33%) | 4.64% | [attempt](https://github.com/akhmialeuski/fly-connectome-lab/issues/40#issuecomment-5762975206) |
| 11 | neural; persistent; last; both; PCA 240; true | 100.00% | 2/60 (3.33%) | 5.36% | [attempt](https://github.com/akhmialeuski/fly-connectome-lab/issues/40#issuecomment-5762983540) |
| 12 | neural; persistent; last; both; PCA none; true | 100.00% | 3/60 (5.00%) | 5.36% | [attempt](https://github.com/akhmialeuski/fly-connectome-lab/issues/40#issuecomment-5762992540) |
| 13 | neural; persistent; last; spike_trace; PCA 60; true | 74.64% | 5/60 (8.33%) | 4.64% | [attempt](https://github.com/akhmialeuski/fly-connectome-lab/issues/40#issuecomment-5763001152) |
| 14 | neural; persistent; last; voltage; PCA 60; true | 100.00% | 5/60 (8.33%) | 6.07% | [attempt](https://github.com/akhmialeuski/fly-connectome-lab/issues/40#issuecomment-5763009403) |
| 15 | encoded; persistent; all; both; PCA 60; permuted | 100.00% | 4/60 (6.67%) | 6.07% | [attempt](https://github.com/akhmialeuski/fly-connectome-lab/issues/40#issuecomment-5763026562) |
| 16 | neural; persistent; last; both; PCA 60; permuted | 98.21% | 3/60 (5.00%) | 5.00% | [attempt](https://github.com/akhmialeuski/fly-connectome-lab/issues/40#issuecomment-5763036478) |
| 17 | encoded; persistent; all; both; PCA 60; memorization | 100.00% | Not scored (training-only control) | 15.00% | [attempt](https://github.com/akhmialeuski/fly-connectome-lab/issues/40#issuecomment-5763043990) |
| 18 | neural; persistent; last; both; PCA 60; memorization | 100.00% | Not scored (training-only control) | 35.00% | [attempt](https://github.com/akhmialeuski/fly-connectome-lab/issues/40#issuecomment-5763052133) |

Promotion decision under the preregistered rule:

```json
{
  "anchor_accuracy": 0.06666666666666667,
  "qualifying_candidates": [],
  "selected_candidate": null
}
```

No reserve or historical-test predictions were scored. Failed controls are missing measurements, not zero accuracy. The four mandatory main-cohort anchors follow this report; numerical stopping failures will be investigated separately with training-only diagnostics.
