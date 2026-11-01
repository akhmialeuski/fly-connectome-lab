# Training-only signal stability results

The preregistered attempt `paired-responses` completed all 163 episodes under
clean source `5652dc6112aaa329a508c8d7fd49910783d09bd5`. Native float32 responses
rounded to float16 exactly matched the archived baseline for all 40 selected
training photographs and all 16 observations. No validation, historical test,
or reserved photographs were scored. No classifier or neural parameters were fitted.

## Storage precision

Native-versus-float16 RMS error across the measured feature vector was
9.93e-5 at observation 1 and 1.04e-4 at observation 16. Its maximum ratio to
between-identity response RMS under matched noise was 0.00081262 (0.0813%)
over all observations. Maximum absolute rounding error was 0.00097513 in the
spike filter and 0.00024414 in voltage.

This makes gross storage distortion an unlikely explanation for the measured
response geometry. It does not prove classifier predictions would be unchanged
if fitted directly on float32 features; that requires its own controlled comparison.

## Noise versus image differences

The following are mean squared distances in the combined recorded feature vector,
not accuracy scores. Every comparison uses the same warmed rest state and original
image geometry. Pairwise distances are dependent descriptive measurements, not
independent statistical replicates.

| Comparison | Pairs | Observation 1 | Observation 16 |
| --- | --- | --- | --- |
| Same image, different common-noise seed | 40 | 0.0668262 | 0.0919163 |
| Different identities, same noise | 760 | 0.0149261 | 0.0570459 |
| Same identity, different image, same noise | 20 | 0.0149294 | 0.0588132 |

Noise-induced mean squared change was 4.48 times the between-identity image
change at observation 1, and 1.61 times at observation 16. Within-identity image
pairs were not more tightly grouped in this raw distance measure. Voltage and
spike-filter blocks separately showed the same qualitative pattern; both have
full per-observation distributions in the saved report.

The network responds to image input even with episode noise disabled: final
stimulus-minus-zero-current mean squared response was 0.0230837 for the spike
filter and 0.0125089 for voltage. These amplitudes are not evidence of identity
decodability. Noise-off episodes intentionally start from the same noise-enabled
baseline warmed state, so this intervention isolates episode noise rather than
changing both rest and stimulation.

## Interpretation and next decision

Noise is a plausible contributor to poor decoding in the measured operating
regime. This two-seed, training-only geometry study does not establish a causal
recognition improvement from removing noise, nor a sequential-memory benefit.
Do not replace the original model or its noise level based on these distances.
The next justified experiment is a preregistered matched recognition comparison
with independent episode-noise control, keeping input encoding, timing, rest,
readout fitting, membership and train-only preprocessing fixed. Float32-versus-
float16 readouts should be paired on the same native responses. Do not change
amplitude, population, or synapses simultaneously with noise.

## Verification and resources

All seven response cases and their SHA-256 digests were checked. Combined-feature
pair-distance means were independently recomputed using Gram matrices rather than
the implementation's pair loop. Native replay passed before intervention cases.
Wall time was 98.98 seconds; peak recorded RSS was 1,200,648,192 bytes (1.12 GiB).
Arrays include native generated responses, spike counts, warmed rest and neuron
indices; original photographs and encoded image arrays are excluded.

Working evidence is under
`$FLYSTATE_HOME/runs/diagnostics/2026-09-22-signal-stability/`. The sibling study
archive retains the frozen protocol, membership, complete attempt, execution logs,
and independent verification. All eight generated NPZ payloads were independently restored from origin and
verified against their SHA-256 digests; see the provenance verification record.
