### Sequential execution: `p0-main-audit`

Exit code: 0. Completed.

```json
{
  "cohort_counts": {
    "test": 300,
    "train": 1400,
    "val": 300
  },
  "reserve_count": 224,
  "balanced_full_confirmation_available": false,
  "cross_split_duplicate_candidates": 8,
  "exact_duplicate_pairs": 0,
  "graph": {
    "csc_arrays": {
      "data": {
        "dtype": "float32",
        "sha256": "809ef60873eba5df0869eeb692e0154f41a42b2d731bb0603c516e2630ae6020"
      },
      "indices": {
        "dtype": "int32",
        "sha256": "f6f92c104fb517ebbd4c541715a503ecead3aeb19f827744604835dedaafcf7a"
      },
      "indptr": {
        "dtype": "int32",
        "sha256": "fb2275b0936352de59b5da4f7f295c89f8ae268b5c8abf55c8ec6be49f34a2a1"
      }
    },
    "effective_stored_edges": 25088107,
    "negative_edges": 9656941,
    "positive_edges": 15431166,
    "sensory_input": false,
    "sensory_neurons": 17937,
    "source_files": {
      "brain.npz": "cc9bd1ecd00bd703a6fa648bc6ad145c93c7c1ee53debdcc9ce0d1f4305e6aca",
      "weights.npz": "c29919aa44069a271b1ee978abe05fa9bf6e45e4ba3e436e92b624ef1b5be40c"
    },
    "source_stored_edges": 25582938
  },
  "decision": "exploratory development; reserved images must not be fitted or scored",
  "identities_with_three_reserved": 67
}
```

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-21-fly-only-diagnostics/p0-main-audit/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
