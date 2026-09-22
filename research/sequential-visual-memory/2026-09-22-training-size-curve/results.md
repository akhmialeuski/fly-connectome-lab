# Fixed-cohort training-size results

All 48 preregistered attempts completed successfully with clean source commit
`c11b847ea3f5ef6704d5b94ff2812517b04bc092`. Twenty identities and the same
60 validation photographs were held fixed. At 2, 4, and 8 photographs per identity,
five nested training-subset draws were fitted; the 14-photograph endpoint is one
shared fit per representation, not five independent replicates. The neural probe
uses the final saved persistent-state observation. Only diagnostic readout
parameters were trained; this study did not train fly synapses or change dynamics.

| Training photos / identity | Pixel control | Encoded-current control | Neural final-state probe |
| --- | --- | --- | --- |
| 2 | 24.33% (16.67–38.33%) | 28.33% (21.67–38.33%) | 6.00% (3.33–10.00%) |
| 4 | 37.33% (25.00–48.33%) | 37.67% (26.67–43.33%) | 3.67% (0.00–6.67%) |
| 8 | 46.00% (40.00–51.67%) | 45.33% (40.00–50.00%) | 4.67% (1.67–8.33%) |
| 14 | 51.67% | 51.67% | 6.67% |

Entries are mean validation accuracy, with the range across training subsets in
parentheses. Uniform-choice reference is 5%. The mean paired change from 2 to 14
photographs is +27.33 percentage points for pixels (95% identity-cluster bootstrap
interval +15.99 to +39.67), +23.33 for encoded currents (+12.00 to +35.00), and
+0.67 for neural activity (-5.68 to +8.67). These exploratory intervals use 2,000
identity-cluster resamples and are conditional on the fixed cohort and declared
subsets; they do not measure uncertainty over new populations or new neural seeds.

More training photographs clearly help the input controls in this cohort. The
measured neural learning curve stays near chance and does not demonstrate a
memory benefit. This does not establish that more data can never help the fly
model. It motivates testing signal preservation and dynamics before expanding
the training set further. Pixel and encoded controls are diagnostic tools, not
replacement vision models or the intended sequential-memory system.

No historical test or reserved photographs were scored. All 48 per-attempt reports
were posted to issue #42 during execution; execution records retain their URLs.
Full parameters, subset memberships, cross-validation details, coefficients,
prediction rows, elapsed time, and execution logs are preserved in the snapshot.

Independent verification reproduced all 7,920 saved prediction rows from the 48
exported coefficient sets. Maximum absolute probability difference was
1.5543122344752192e-15; predicted labels matched exactly. All seven exported arrays
for each full-data endpoint were exactly equal to its prior diagnostic anchor
(scaler, PCA, classes, coefficients, and intercept). This verification required
no refitting. Detailed records are in `execution/replay-verification.json` and
`execution/full-endpoint-parity.json` within the snapshot.
