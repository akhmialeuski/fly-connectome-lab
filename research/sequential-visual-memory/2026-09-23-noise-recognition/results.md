# Results: isolated episode noise and storage precision

The frozen T25 protocol in [issue #50](https://github.com/akhmialeuski/fly-connectome-lab/issues/50)
was executed at source commit `47bbf1eaa672e7d7269607f7dbd20212aa98478b`.
All three trace attempts, six readout fits, and the paired analysis completed.
No failed or nonconverged readout was excluded. The 280 training and 60
validation photographs were fixed before execution; no historical test or
reserve photograph was newly simulated or scored.

The seed-zero native float32 trace reproduced all 340 original float16 cached
rows exactly after casting. All three runs used all 16 raster image patches and
recorded two original feature blocks from 1,314 descending neurons, giving
native shape `(340, 16, 2628)`. Their warmed `rest.npz` files were byte-identical
(SHA-256 `704a18c1d0d76def0778eedfabaaf3060fa0a27dd7566a50ea4bd1d0535b0ce6`).
Only episode noise changed. The seed-zero float16 fit also exactly reproduced
every exported model array and all train/validation predictions of the earlier
full-data neural diagnostic.

| Episode condition | Trace treatment | Chosen C | Final solver iterations | Mean CV accuracy | Training accuracy | Validation accuracy | Validation log loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Noise seed 0 | float32 | 0.01 | 56 | 5.71% | 81.07% | 3.33% (2/60) | 3.7166 |
| Noise seed 0 | float16 | 10.0 | 201 | 6.07% | 100.00% | 6.67% (4/60) | 8.4530 |
| Noise seed 1 | float32 | 0.01 | 61 | 5.71% | 80.00% | 10.00% (6/60) | 3.3013 |
| Noise seed 1 | float16 | 0.01 | 58 | 5.71% | 80.36% | 10.00% (6/60) | 3.3012 |
| Noise off | float32 | 10.0 | 419 | 6.07% | 100.00% | 3.33% (2/60) | 20.9637 |
| Noise off | float16 | 0.01 | 76 | 6.07% | 73.21% | 3.33% (2/60) | 5.8949 |

Uniform choice among 20 identities is 5%. Every result in this table is an
exploratory development-set result. Cross-validation preprocessing was fitted
inside each training fold; validation labels did not enter fitting or C
selection. The same final-observation readout was used in every fit.

The paired differences below are `noise off − noisy`, in percentage points.
Intervals are 95% percentile intervals from 2,000 seed-zero bootstrap draws
over the 20 identities (three paired validation photographs per identity).
The noise-off endpoint is shared and was not treated as two independent runs.

| Trace treatment | Versus noise seed 0 | Versus noise seed 1 | Versus mean of two noisy seeds |
| --- | ---: | ---: | ---: |
| float32 | 0.00 [−6.67, 6.67] | −6.67 [−15.00, 1.67] | **−3.33 [−10.00, 3.33]** |
| float16 | −3.33 [−13.33, 5.00] | −6.67 [−15.00, 0.00] | **−5.00 [−12.50, 1.67]** |

The preregistered promotion gate required an improvement of at least ten
percentage points and a positive lower interval bound **at both precisions**.
It failed. Disabling episode noise in this operating regime did not rescue
recognition and is not promoted to a larger recognition study. With only two
declared noisy seeds and 60 fixed validation photographs, these intervals do
not establish a general population effect or prove that noise is beneficial.

Storage precision affected **readout selection** differently across traces.
The seed-zero float32 and float16 fits chose C=0.01 and C=10.0 respectively,
with 23/60 validation predictions disagreeing and a maximum class-probability
difference of 0.9924. Noise-off chose C=10.0 versus C=0.01, with 25/60
disagreements and a maximum difference of 1.0. Noise seed 1 chose C=0.01 at
both precisions, had zero prediction disagreements, and a maximum probability
difference of 0.0002883. Thus the small **trace rounding error** measured in
the prior study does not guarantee stable model selection or predictions on
this small cohort. The poor validation scores and the noise-off float32
training/validation gap are compatible with overfitting or weak class signal;
this study does not identify a unique causal mechanism.

Every one of the 2,040 saved train and validation probability vectors was
independently recalculated from the exported scaler, PCA and linear
coefficients. The largest absolute probability discrepancy was
`2.44e-15`; labels, top-one predictions and reported accuracies also matched.
See [numeric replay](provenance/numeric-replay.json). The original completed
attempts under `FLYSTATE_HOME` were copied byte-for-byte into `snapshot/`:
116 files, 183,124,833 bytes, ten verified checksum inventories, and canonical
file-digest tree SHA-256
`ac1245f138b72410358dc0b7e594a10720e4f9f851abdce06277512f42078c24`.
See [local archive verification](provenance/local-archive-verification.json).
An independent fresh GitHub clone subsequently restored all 12 Git LFS NPZ
payloads and verified all 116 files, ten attempt inventories, and the same
file-digest tree; see [remote LFS verification](provenance/remote-lfs-verification.json).

The manifests record the frozen source commit. Their `git_dirty` field is null
because Git status exceeded the manifest helper's five-second timeout on the
Windows-mounted checkout; a separate explicit status check completed cleanly.
No existing completed run or cache was changed. This experiment fits only a
readout and cannot establish a persistent-state memory benefit. The next
research step should examine operating regime, accessible neuron populations,
and temporal signal with a new preregistered protocol before changing brain
dynamics or enabling additional plasticity.
