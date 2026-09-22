### P0 follow-up: cohort integrity and unused-image reserve

Execution: [#40](https://github.com/akhmialeuski/fly-connectome-lab/issues/40). The original split membership is unchanged; these audits do not score predictions.

| Cohort | Train / validation / historical test | Reserved unused photographs | Identities with three reserved photographs | Exact cross-split duplicates | Perceptual review pairs |
| --- | --- | ---: | ---: | ---: | ---: |
| Smoke, 20 identities | 280 / 60 / 60 | 37 | 11 | 0 | 2 |
| Main, 100 identities | 1,400 / 300 / 300 | 224 | 67 | 0 | 8 |

Reserve screening uses source-file/RGB equality and 64-bit grayscale dHash distance <=3 against historical images and previously accepted reserve candidates. Perceptual matches are conservative review candidates, not proven duplicates. Coverage after screening is lower than the preliminary availability-only count (main: 67 rather than 69 identities with three photos). The smoke and main reserves are cohort-specific lists and may overlap; their counts must not be added as independent photographs.

A balanced new three-image confirmation set for all original identities is unavailable. Reserve images remain excluded from all diagnostic fitting/scoring, and the original inspected test is not reused as fresh confirmation. The first campaign remains exploratory. Full per-image signatures, candidate pairs, rejected reserves, accepted reserves, and provenance are retained for archival.

Raw audit comments: [smoke](https://github.com/akhmialeuski/fly-connectome-lab/issues/40#issuecomment-5762860009), [main](https://github.com/akhmialeuski/fly-connectome-lab/issues/40#issuecomment-5762869013).
