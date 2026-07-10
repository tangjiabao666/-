# RR diagnosis and fix

## Root cause

The rejection score was not batch-invariant. On the fixed competition dev split, the same checkpoint changed from 68.25% RR at batch size 1 to 41.27% at batch size 16.

The rejection head pooled Mel features with `mel.mean(dim=2)` and the fusion gate averaged padded waveforms without using the real audio lengths. Consequently, zero padding changed the rejection score and encouraged a duration shortcut.

Sample-level evidence from 275 dev samples:

- Positive audio averaged 2.56 seconds; negative audio averaged 3.88 seconds.
- Audio duration alone had AUC 0.8152, higher than the model rejection margin AUC of 0.7890.
- Enrollment-to-mixture speaker cosine had AUC 0.5188, close to random.
- The rejection margin correlated with duration at -0.544.

## Fix

`WaveformFusionGate` now computes masked waveform statistics. `RejectionHead` now pools only valid Mel frames whose STFT windows are inside the real waveform. The model passes `mixed_lengths` through both paths.

The TSE-stage trainer also keeps frozen modules in evaluation mode, records training arguments and seeds in checkpoints, and supports deterministic rejection-head calibration.

## Validation

The component padding-invariance tests pass locally and on the CUDA server. After the fix, the baseline RR variation across batch sizes dropped from 26.98 percentage points to about 1.6 percentage points, with the remaining difference limited to samples near the decision threshold.

The selected external-data checkpoint is `checkpoints_final_external_acc/latest.pt`. With batch size 8 and rejection threshold `-0.10` on the fixed dev split:

- CER: 4.10%
- RR: 60.32% (38/63)
- Positive accept rate: 80.66% (171/212)
- Rejection accuracy: 76.00%
- Positive exact match: 90.09%

Evaluation must keep the batch size and threshold fixed when comparing results.
