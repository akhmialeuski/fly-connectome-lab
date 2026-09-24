# Corrected blank checkpoint coverage

The first C0 and C6 blank attempts were completed before any paired scientific analysis. They recorded global steps 0, 1, 2, 5, 10, 20, 21, 22, 25, and 30. The ten-step driven conditions C1, C4, and C5 also recorded recovery at global steps 11, 12, and 15. A matched blank voltage vector for those three times was therefore absent. Per-step spike totals and noise counts do not reconstruct voltages; interpolation would change the measurement.

The original C0 and C6 attempts remain immutable and valid at their saved checkpoints. New C0R and C6R attempts repeat the **same** 20-step blank plus ten further blank steps, with identical config, cohort, masks, and per-(sample,window) noise seeds. They add checkpoints 11, 12, and 15, producing the union `(0, 1, 2, 5, 10, 11, 12, 15, 20, 21, 22, 25, 30)`. C0R retains episode noise; C6R has no episode noise. No stimulus or fly equation changes.

Before using either correction, require exact equality with C0/C6 for all overlapping voltage, cumulative spike, readout-trace, per-step total/active, and noise arrays, as applicable. If an overlap fails, invalidate the paired recovery analysis and investigate the recorder. This correction was declared in [issue #54](https://github.com/akhmialeuski/fly-connectome-lab/issues/54) before the new attempts were run and before response contrasts were inspected.
