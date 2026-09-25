## T37 result: the specific MaleCNS wiring gives no advantage at matched operating points

Owning issue: [#74](https://github.com/akhmialeuski/fly-connectome-lab/issues/74). Protocol: [protocol.md](protocol.md), frozen in `5e8ee78` before any recording. Phase A selection: [selection.json](selection.json), frozen in `bedd1cf` before Phase B.

Phase B recorded 12 graphs on 10 cohorts that no earlier study had touched: 200 identities, 2,800 training and 1,200 held-out photographs. All 120 recordings are clean attempts from `bedd1cf`. Each cohort was evaluated once with the standard readout: fold-local scaling, PCA 60 and logistic regression, with C chosen by 5-fold CV on training photographs only. Cohorts s4, s5 and s6 were evaluated from `bedd1cf`. The other seven were evaluated from `e5b812c`, after an evaluation failure described below. The pooled analysis ran once. Every number here was reproduced independently by [analysis/verify.py](analysis/verify.py): 32,400 predictions were replayed from the archived affine readouts on the recorded states. See [provenance/verify-with-states.json](provenance/verify-with-states.json).

### Decisions under the preregistered rules

Differences are in percentage points on the 1,200 held-out photographs. Intervals come from 10,000 two-level bootstrap draws, resampling identities within cohorts and null seeds. The pre-results note in [#74](https://github.com/akhmialeuski/fly-connectome-lab/issues/74#issuecomment-5835114408) adds a magnitude qualifier to every interval that lies inside ±5 points.

| Rule | Central brain | 95% interval | Frozen reading |
| --- | ---: | --- | --- |
| **W1**: MaleCNS minus mean of 5 degree-preserving shuffles | **−2.65** | [−4.63, −0.53] | **disadvantage**, smaller than the 5-point margin |
| **W2**: MaleCNS minus mean of 2 random-target nulls | −1.58 | [−3.67, +0.58] | no practically relevant difference |
| **M1**: memory, MaleCNS persistent minus reset (gain 1) | **+28.50** | [+25.17, +31.83] | advantage, McNemar 401 against 59, p = 6.4e-57 |
| **M2**: recurrence, MaleCNS minus feedforward-only (gain 1) | +4.50 | [+2.08, +6.92] | advantage, p = 3.5e-4 |
| **R**: T35 replication, MaleCNS minus shuffle seed 0 (gain 1) | +0.42 | [−1.83, +2.67] | no practically relevant difference, p = 0.77 |

The same rules for **descending neurons**, the secondary population:

| Rule | Descending neurons | 95% interval | Frozen reading |
| --- | ---: | --- | --- |
| W1 | −4.27 | [−6.45, −2.15] | disadvantage |
| W2 | −3.62 | [−6.08, −1.25] | disadvantage |
| M1 | +23.33 | [+20.33, +26.50] | advantage, p = 5.4e-43 |
| M2 | +6.58 | [+4.08, +9.08] | advantage, p = 5.9e-8 |
| R | −0.83 | [−3.17, +1.50] | no practically relevant difference |

### Pooled held-out accuracy

Chance is 5%, since every cohort is a 20-identity task.

| Graph and state | Operating point | Central brain | Descending |
| --- | --- | ---: | ---: |
| MaleCNS, persistent | α 0.25 (gain 0.339) | 494 (41.2%) | 456 (38.0%) |
| Degree-preserving shuffles 0 to 4, persistent | α 0.75 | 501 to 541 (41.8 to 45.1%) | 487 to 527 (40.6 to 43.9%) |
| Random-target nulls 0 and 1, persistent | α 0.5 | 520 and 506 (43.3%, 42.2%) | 492 and 507 (41.0%, 42.3%) |
| MaleCNS, persistent | gain 1 | 500 (41.7%) | 473 (39.4%) |
| MaleCNS, reset before every window | gain 1 | 158 (13.2%) | 193 (16.1%) |
| Degree-preserving shuffle 0, persistent | gain 1 | 495 (41.3%) | 483 (40.3%) |
| Feedforward-only, persistent | gain 1 | 446 (37.2%) | 394 (32.8%) |

### What this establishes

1. **The specific MaleCNS wiring confers no recognition-with-memory advantage.** At matched operating points MaleCNS is 2.65 points below the degree-preserving ensemble, with an interval entirely below 0 and entirely inside the ±5-point margin. It is not distinguishable from the random-target nulls, which lose the in-degree distribution as well. For descending neurons, the original PoC readout, the nulls are better by 3.6 to 4.3 points. The claim "the fly wiring helps" is ruled out at the preregistered margin. What remains is a small, well-resolved advantage for the shuffled graphs at the selected operating points.
2. **T35's +5.8 points was sampling variation.** The exact T35 comparison (gain 1, shuffle seed 0) gives +0.42 [−1.83, +2.67] on ten times as many photographs, with McNemar 98 against 93.
3. **Memory replicates on 200 new identities.** Carrying the state across glimpses adds 28.5 points in the central brain, and every cohort gains between 20.0 and 35.0 points.
4. **Most of the memory needs no recurrence among non-driven neurons.** Keeping only the synapses that leave the driven neurons gives 37.2% against 13.2% with reset, which is 24.0 of the 28.5 points. Each non-driven neuron then integrates its direct drive with its own slow leak. Recurrence adds a significant 4.5 points [+2.1, +6.9] in the central brain and 6.6 points in descending neurons. The central-brain interval crosses +5, so whether this gain is smaller than the 5-point margin is not resolved.
5. **The operating point hardly matters.** Phase A found at most 2.5 points of CV range across α within each family, and gain 1 against α 0.25 changes MaleCNS by 0.5 points on the untouched cohorts.

The mechanism is therefore a large, sparse, signed, row-normalized network of slow leaky units that receive direct projections from the input neurons. The MaleCNS connectome is one such network, but not a privileged one for this task. This agrees with [arXiv 2604.04033](https://arxiv.org/abs/2604.04033), where Drosophila-connectome advantages largely disappear under degree-preserving nulls with fair controls.

### Heterogeneity

Per-cohort W1 differences range from −9.8 (s5) to +1.8 (s11) in the central brain, and 9 of 10 cohorts are negative. The memory gain M1 is positive in every cohort (20.0 to 35.0). M2 is positive in 8 of 10 cohorts. The per-cohort values are in `snapshot/phase-b/analyze/report.json`.

### Failures and corrections, all recorded in #74

- **Phase A evaluation.** The first attempt stopped before creating an attempt, because case names containing a dot are not allowed. `09d5764` renamed the cases. [Comment](https://github.com/akhmialeuski/fly-connectome-lab/issues/74#issuecomment-5832277913).
- **Solver budget.** The cohort s8 evaluation failed when `fly-g1-reset/central_brain` stopped at SciPy's L-BFGS function-evaluation cap of 15,000. That is below the declared budget of 50,000 iterations. `e5b812c` continues such fits with `warm_start` within the same budget. Refits of cohort s4 gave identical predictions and CV scores, so s4 to s6 were kept. The failed attempt is preserved in `snapshot/phase-b/evaluate-failed/`. Three fits were continued: two in s8, after 14,029 and 13,860 iterations, and one in s25, after 14,129 iterations. The logs name no case. By iteration count, the s25 fit is `fly-g1-reset/central_brain`. [Comment](https://github.com/akhmialeuski/fly-connectome-lab/issues/74#issuecomment-5839048959).
- **Spectral radius solver.** Two ARPACK defects were fixed before any recording: a random start vector and a single requested eigenvalue.
- **conn2res normalization.** It uses `eigh`, so it is exact only for symmetric matrices. [Comment](https://github.com/akhmialeuski/fly-connectome-lab/issues/74#issuecomment-5832311871).
- **Verifier order.** The first verifier run failed on W1's interval. The analysis report stores its cohorts as canonical JSON with sorted keys, while the bootstrap drew them in `run.sh` order. The verifier now restores `run.sh` order, and every interval then reproduced exactly.

### Corrections made after the first version of this file (2026-09-25)

- **M2 qualifier.** M2 was first described as "smaller than the 5-point margin". Its interval reaches +6.92, so under the pre-results note the qualifier does not apply. The frozen reading, advantage, is unchanged.
- **McNemar method.** The protocol promised exact McNemar tests, but `flystate.evaluation.stats.mcnemar` switches to the continuity-corrected chi-square at 25 or more discordant pairs. That applies to every contrast here. Exact two-sided binomial p-values from the same counts:

  | Contrast | Only first correct | Only second correct | Exact p |
  | --- | ---: | ---: | ---: |
  | M1, central brain | 401 | 59 | 1.5e-63 |
  | M2, central brain | 137 | 83 | 3.3e-4 |
  | R, central brain | 98 | 93 | 0.77 |
  | M1, descending | 346 | 66 | 6.7e-47 |
  | M2, descending | 143 | 64 | 4.1e-8 |
  | R, descending | 91 | 101 | 0.52 |

  No reading changes.
- **Continued fits.** The count of continued L-BFGS fits was first given as two. It is three, and the s25 fit was added above.

The three discrepancies were found while writing the T37 report for the project knowledge base.
