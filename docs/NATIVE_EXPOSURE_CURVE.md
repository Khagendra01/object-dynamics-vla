# Single-demonstration exposure curve

## Alignment prerequisite

The existing `/eval/milk_recorded_action_replay.json` replays the source HDF5
`demo_0` actions from its original saved simulator state. It reported success
at action index 138 of 150, final reward 1, and mean/max simulator-state MAE
0.00160/0.03931 against recorded subsequent states. This is open-loop action
replay, not a learned-policy result, and is evidence for this specific
controller/initialization rather than a complete audit of all observation paths.

## Matched all-frame evaluation

Both 3,000-update native scratch and query-normalized Muse/SAM checkpoints were
evaluated on all 150 training observations. Each frame used one shared saved
`[50,32]` float32 Gaussian tensor from `default_rng(921)`; the saved bank
`/eval/native_exposure_curve_v1/noise.npy` has SHA-256
`6ed5cab3f9e605ac8b5ea8b1091a9c6817620e718c68d5cb8552d44eba4428da`.
The measurement is sampled first-action error, in raw controller-command
units. It is in-sample and uses only one noise draw per observation.

| Model / updates | Arm MAE by channel 0–5 | Gripper MAE | Negative sign correct (55) | Positive sign correct (95) |
|---|---|---:|---:|---:|
| Native / 3,000 | .169, .483, .206, .017, .025, .040 | 1.099 | 39/55 | 45/95 |
| Muse/SAM / 3,000 | .188, .342, .391, .028, .039, .034 | .672 | 39/55 | 84/95 |
| Native / 5,000 | .113, .195, .183, .010, .043, .021 | .391 | 50/55 | 93/95 |
| Native / 9,018 | .068, .184, .138, .011, .017, .015 | .265 | 47/55 | 94/95 |

The positive-class sampled error fell substantially with exposure, although
negative-class sign agreement declined from 5,000 to 9,018. Native and Muse
have different architectures, pretrained backbones and Transformers versions;
the table alone cannot isolate any one of those differences. More seeds and
other episodes are needed before generalization claims.

## Continuation and budget

The original 3,000-update native checkpoint did not save optimizer or RNG state.
The continuation loaded its exact trainable weights and **restarted AdamW** at
update 3,000 with the same hyperparameters and frame order; its random seed was
reset to 8008. Thus this is a continued-weight learning curve with an optimizer
discontinuity, not a bit-identical resume of the original optimization run.
The 5,000 and 9,018 checkpoints save optimizer and Torch CPU/CUDA RNG state,
the next frame index and the noise-bank hash, along with the trainable weights;
normalization statistics and experiment metadata are alongside them.

The first continuation used H200/2 CPU/16 GiB, with a 690-second hard timeout
and a 600-second update cutoff to preserve evaluation and checkpoint time. It
stopped at 9,018 updates after 630.1 function seconds, estimated at $0.834 for
requested GPU/CPU/RAM at listed rates. An H100/2 CPU/16 GiB follow-up restored
that optimizer and RNG state, saved a resumable checkpoint at 9,770, and timed
out during evaluation before reaching 10,000. Its final recorded function
time was 116.2 seconds, estimated at $0.135 (125 seconds at requested rates is
about $0.145). The 9,770 checkpoint was committed before the timeout but is
not evaluated. Combined requested-resource estimates are about $0.968–0.979;
startup, image builds, storage, and other charges are outside those estimates.
The separate all-frame Muse evaluation took 112 seconds and estimated $0.160
in requested resources; it is not included in the continuation estimate.

## Artifacts

* `/eval/native_exposure_curve_v1/`: shared noise, stats, report,
  `evaluation_3000.json`, `evaluation_5000.json`, `evaluation_9018.json`,
  and resumable checkpoints at 5,000 and 9,018.
* `/eval/native_exposure_finish_v1/`: timeout report and resumable checkpoint
  at 9,770; no completed evaluation.
* `/eval/quantized_muse_allframes_v1/report.json`: Muse/SAM 3,000-update
  evaluation with the same saved noise and per-frame predictions.

No new simulator rollout was run in this experiment. The falling in-sample
sampled-action errors support testing a longer, budgeted training schedule for
Muse/SAM, without claiming the native model has converged or can complete the
task closed-loop.
