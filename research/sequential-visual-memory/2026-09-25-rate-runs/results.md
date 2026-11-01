# T36 results: standard runs of the graded MaleCNS model

Three standard runs of the frozen T35 model (gain 1.0, leak 0.02, fast driven neurons, input scale 20, 4 updates per patch) on the confirmation cohort, produced by `flystate trace build`, `train`, `evaluate` and `compare` exactly as the POC 1 runs were. The readout is the original POC readout: descending neurons (1,314 cells), graded state recorded as the `voltage` block, and a per-observation logistic readout with fold-local PCA(60) and C chosen by training-only CV. Test split: 60 photographs of 20 identities; chance 5%.

| Run | Test accuracy after patch 1 | After patch 8 | After patch 16 | Final validation |
| --- | ---: | ---: | ---: | ---: |
| Persistent state | 28.3% | 51.7% | **53.3%** (32/60) | 43.3% |
| Reset before every patch | 28.3% | 35.0% | 13.3% (8/60) | 11.7% |
| Reset, readout concatenates all patches | 28.3% | 56.7% | 56.7% (34/60) | 46.7% |

Test accuracy of the persistent run grows as patches accumulate (28, 27, 35, 32, 30, 35, 47, 52, 42, 50, 60, 57, 58, 57, 55, 53% after patches 1 to 16). The reset run only ever knows the current patch.

| Paired comparison, test split, final patch | Difference | Bootstrap 95% interval | McNemar |
| --- | ---: | --- | --- |
| Persistent − reset | **+40.0 points** | [25.0, 55.0] | 27 against 3 discordant, p = 2.7e-5 |
| Persistent − reset-concat | −3.3 points | [−16.7, 10.0] | 7 against 9 discordant, p = 0.80 |

The network's own final state identifies the person about as well as a readout that stores every patch separately, and far better than a state reset before each patch. This agrees with T35: descending neurons scored 48.3% on all 120 held-out photographs with the single-shot readout. These runs replicate that result in the standard pipeline and make it inspectable episode by episode. The model and cohort were frozen before these runs, but the test photographs were already scored once in T35, so this is not a second independent confirmation.
