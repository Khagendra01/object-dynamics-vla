# Query-normalized NF4 Muse/SAM exposure curve

## Scope

This experiment continues the existing 3,000-update query-normalized Muse/SAM
checkpoint on the same 150-frame `episode_0000` training demonstration. The
frozen NF4 Muse backbone, pooled-prefix adapter, causal cached SAM measurements,
fresh official action expert, normalization, 50-step action chunks, batch-one
frame order, AdamW hyperparameters and disabled dynamics objective are the same
as the source run. No robot-policy checkpoint or teacher was loaded. The source
checkpoint did **not** contain optimizer or training RNG state: AdamW moments
were restarted at update 3,000 with the same hyperparameters and Torch RNG was
reseeded to 8008. This is a weight continuation with an optimizer/RNG
discontinuity, not an exact uninterrupted 10,000-update trajectory.

All 150 RGB hashes and the train-only normalization statistics were checked
against the source and native exposure run. Frozen Muse NF4 features were
re-extracted for each frame before training. The H200 training job used 4 CPU
and 64 GiB host RAM and completed 7,000 additional updates in about 414 seconds
of training, 495 seconds function time; requested GPU/CPU/RAM cost arithmetic
was approximately $0.72, excluding startup, builds and storage.

## Independent all-frame evaluation

Checkpoints were saved and volume-committed before any scoring. Each milestone
was scored in a separate H200 function with its own 360-second timeout. Each
uses the same 150 saved float32 `[50,32]` noise tensors as the native exposure
curve (SHA-256 `6ed5cab3f9e605ac8b5ea8b1091a9c6817620e718c68d5cb8552d44eba4428da`).
Metrics are first-action errors in **raw controller-command units**, with one
fixed noise draw per training observation, and are not closed-loop results.

| Updates | Arm MAE channels 0–5 | Gripper MAE | Negative sign correct (55) | Positive sign correct (95) |
|---:|---|---:|---:|---:|
| 3,000 | .188, .342, .391, .028, .039, .034 | .672 | 39/55 | 84/95 |
| 5,000 | .178, .291, .303, .018, .032, .036 | .921 | 47/55 | 53/95 |
| 7,500 | .090, .163, .201, .010, .023, .014 | .237 | 51/55 | 94/95 |
| 10,000 | .108, .163, .155, .013, .021, .020 | .392 | 55/55 | 85/95 |

The 7,500 checkpoint is the candidate on the predeclared balance of motion and
both gripper classes. The 10,000 checkpoint has lower z-channel error and
perfect negative-class agreement, but larger positive-class error and worse
positive-class agreement. Checkpoint selection and scoring use the same
training demonstration, so these numbers are not an independent test.

## Additional saved noise draws for the 7,500 candidate

Three new float32 `[150,50,32]` noise banks were generated with NumPy
`default_rng(923)`, saved and committed **before** evaluation. The file is
`/eval/quantized_exposure_eval7500_extra3_v1/additional_noise.npy`, SHA-256
`77fbe98982a595b5459eeb13adf512f0e34ee26c30f32ac358d5d919ec4ff2ef`.
Across 450 sampled first actions, arm MAE was
`[.085, .171, .183, .011, .024, .015]`, gripper MAE `.250`, negative sign
agreement `155/165`, and positive sign agreement `281/285`. Per-draw gripper
sign agreements were 51/55 and 94/95; 53/55 and 93/95; 51/55 and 94/95.
Samples share the same 150 observations, so they are not 450 independent
episodes. These results justify a candidate control test, not a success claim.

## Checkpoints, audit and reproducibility

The source 3,000-update checkpoint remains unchanged. Milestone checkpoints in
`/eval/quantized_muse_exposure_v1/` are `checkpoint_5000.pt`,
`checkpoint_7500.pt`, and `checkpoint_10000.pt`; each includes prefix and
trainable policy weights, AdamW state, Torch CPU/CUDA RNG state, next-frame
index, backbone revisions, dataset fingerprint, fixed evaluation noise hash,
and the optimizer-restart disclosure. `stats.npz` and `report.json` are in the
same directory. A separate CPU audit strictly loaded the prefix and policy
parameters, restored AdamW state and checked saved RNG metadata and hashes for
all three checkpoints. It did not execute a numerical next-update replay, and
there is not yet a dedicated command that resumes these new checkpoints.

Per-frame prediction reports are on the volume under
`/eval/quantized_exposure_eval5000_v1/`, `quantized_exposure_eval7500_v1/`,
`quantized_exposure_eval10000_v1/`, and `quantized_exposure_eval7500_extra3_v1/`.
The three milestone evaluations estimated $0.142, $0.144 and $0.150 respectively
in requested resources; the additional-draw evaluation estimated $0.208.
Startup/build/storage and other charges are excluded. No simulator rollout was
run in this experiment.
