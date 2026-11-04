# NF4 Muse conditioning diagnostic

The 3,000-update random-expert fit on the 150-frame training episode
`episode_0000` failed its 200-action rollout from that same initialization.
This is an in-sample development investigation, not a generalization study.
The original rollout used fresh NF4 Muse features, live point-prompted SAM,
and the first action of each sampled chunk. Its seven-dimensional action trace
shows 91 clipped actions, all in the gripper channel.

The demo has 55 gripper commands of -1 and 95 of +1, with transitions after
frames 46 and 141. Train-only normalization and rollout denormalization use
the same saved statistics. The first rollout RGB JPEG has the same SHA-256 as
the first training RGB. Those checks do not prove subsequent trajectory parity.
The flow matching and Euler direction match the pinned official implementation.

## Matched-noise diagnostic

For eight evenly spaced training frames and four fixed noise seeds each,
fresh NF4 image features were sampled with the saved 3,000-update checkpoint.
For the first seed, predictions matched the fit's saved diagnostic to within
`7e-8` in raw-action units. On these 32 samples, only one gripper prediction
was positive and 15/32 signs matched the demonstration; a constant-sign
predictor would score 16/32 on this balanced eight-frame selection. Zeroing
normalized robot state produced identical sampled actions and identical
prefix K/V tensors at the stored precision. Masking SAM changed sampled
gripper commands by about 0.002 absolute units on average; zeroing Muse changed
them by about 0.67. These out-of-distribution ablations establish sensitivity
on the chosen frames, not causal effects on rollout success.

Inspecting the actual fusion attention found that the mean state-token
attention was at most `6.6e-16` on those eight frames in the 3,000-update
checkpoint. The resampled visual-query norm averaged about 1,150. The earlier
20-update preflight had query norms about 13 and state attention about 8–28%
on the same frames. The separately trained 1,200-update subset checkpoint had
query norms about 580; it was not a checkpoint on the 3,000-update training
trajectory, so it is supporting evidence rather than a time-series measurement.

## Controlled intervention

An opt-in, non-parametric layer normalization was applied to resampled visual
queries before state/SAM fusion. It defaults off for old checkpoints. A fresh
run used the same full episode, seeds, 3,000 updates, optimizer and diagnostic
seeds as the baseline. It held the fusion-query norm at approximately
sqrt(512) = 22.63. On the same 32 in-sample samples, gripper sign accuracy
increased from 15/32 to 24/32, raw gripper MAE fell from 1.22 to 0.71, and
zeroing state changed prefix K/V by 1.37–6.16 maximum absolute units across
the eight frames. State attention averaged 12.5–25%; these averages are
consistent with individual heads still concentrating strongly on one token.
Sampled first-action normalized MAE over the original eight fixed seeds fell
from 0.84 to 0.77. The intervention is encouraging but does not establish
successful control; no rollout of this checkpoint has been performed.

Artifacts on Modal:

* `/eval/quantized_muse_full_episode_v1/` — baseline fit and weights.
* `/eval/quantized_muse_full_diagnostic_v3/report.json` — baseline diagnostics.
* `/eval/quantized_muse_preflight20_diagnostic_v2/report.json` — 20-update comparison.
* `/eval/quantized_muse_subset32_diagnostic_v1/report.json` — independent subset comparison.
* `/eval/quantized_muse_full_prenorm_v1/` — query-normalized fit and weights.
* `/eval/quantized_muse_prenorm_diagnostic_v1/report.json` — intervention diagnostics.

Reproduce the controlled fit with `scripts.modal_quantized_muse_preflight`
using `--frames 0 --updates 3000 --max-seconds 420 --normalize-queries` and a
fresh `--name`. Run `scripts.modal_quantized_muse_diagnostic` with a fresh
`--name` and the fit's `--fit-name`. Both jobs use an H200 and existing assets.

## Query-normalized training-initialization rollout

The 3,000-update query-normalized checkpoint was run from the same
`episode_0000` initial state for 200 actions, with live SAM and NF4 Muse.
There was no additional training and the dynamics objective remained off.
The initial-state hash and first rendered JPEG hash matched the baseline. The
policy used the baseline's recorded CUDA Torch seed schedule `7 + step`, on
the same H200/PyTorch policy path. The baseline recorded seeds but not the
noise arrays or hashes, so byte-for-byte noise equality cannot be independently
verified; the new trace records hashes for all 200 generated noise arrays.

The task did not succeed within 200 actions. The query-normalized rollout
clipped 46 actions, all in the gripper channel, versus 91 in the baseline.
Its executed gripper command changed sign 74 times, versus 10 in the baseline;
the recorded demonstration changes sign twice (at frames 47 and 142). Comparing
step-aligned commands after trajectories have diverged is descriptive, not a
matched-state accuracy measurement. Its first sampled action differed from the
demonstrated action already at step 0; by frame 5 the rendered robot pose was
visibly different, and by frame 30 the demonstration had moved toward the milk
while the rollout arm remained near the image center. At nominal frame 47,
the demonstration gripper was near the milk while the rollout remained away.
This locates divergence in the approach, before a demonstrated grasp; it does
not isolate the cause or establish a SAM benefit.

The video and trace are in
`/eval/muse_smolvla_live_episode_0000_prenormmatchedv1/rollout.mp4` and
`report.json` on the Modal volume. The baseline is in
`/eval/muse_smolvla_live_episode_0000_quantizedfullepisodev1/`.
