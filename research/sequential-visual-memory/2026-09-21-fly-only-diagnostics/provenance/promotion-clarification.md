### Clarification of the preregistered promotion rule, before candidates 08–14 finish

The optional additional main-cohort neural variant is selected from smoke candidates **08–14** (persistent history/readout/PCA variants). Their corresponding reference is candidate **05**, persistent-last/both/PCA60. Candidates 05–07 are the fixed neural anchors/controls; reset-all is already in the four mandatory main anchors. Permutation and memorization controls are not promotion candidates.

A variant must improve validation accuracy over candidate 05 by at least ten percentage points. Among qualifying completed variants, select by highest mean training CV accuracy, then fewer effective PCA dimensions (no PCA retains the full input dimension), then preregistered candidate order. Failed attempts cannot qualify. At most one additional variant is promoted. This makes the reference unambiguous before reading those variant results and preserves the original bounded selection rule.

Candidate 05 has also reproduced **all seven saved parameter arrays exactly** against observation 16 of historical run `20260920-071344-celeba-smoke-9406ee`, including scaler, PCA, class labels, coefficients, and intercept. The numerical audit is retained alongside execution evidence.
