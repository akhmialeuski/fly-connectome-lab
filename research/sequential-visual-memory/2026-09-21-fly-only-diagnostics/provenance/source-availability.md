### Source-data ceiling for a larger-data follow-up

A read-only audit of the installed, validated CelebA identity annotations found:

- 202,599 photographs across 10,177 identities.
- Maximum: **35 photographs for any one identity**.
- 6,348 identities have at least 20 photographs; 3,661 have at least 25; 2,360 have at least 30; only three have 35.
- No identity has 40, 50, or 100 photographs in this source.

Thus substantially increasing independent examples per identity cannot be achieved merely by changing the current sample-count setting. A newly declared cohort can support somewhat more training photographs while retaining separate validation and confirmation photographs, but any such cohort is a new experiment. The existing 14-per-identity training set, historical splits, and reserved photographs remain unchanged.

This availability audit does not select a new cohort or score any predictions. Exact counts are retained in `execution/source-availability.json` for the new study. The proposed nested learning curve on the current cohort remains the first controlled test of the small-data hypothesis; augmentation, if investigated later, must be grouped by original photograph and reported as correlated views rather than new independent examples.
