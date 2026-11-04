# Hackathon submission draft: Object Dynamics VLA

## Project

Object Dynamics VLA trains a robot action policy conditioned on frozen,
quantized Muse visual-language features and SAM object tracking. The combined
policy completed LIBERO's milk-placement task in **166 actions** from a
demonstrated training initialization. Across three sampling seeds on that same
initialization and frozen checkpoint, the simulator reported **1/3 successes**.

The experimental policy combines frozen NF4 Muse layer-26 image/instruction
features with causal SAM-derived milk and basket measurements. A
query-normalized prefix conditions a SmolVLA action expert; the policy
replans from camera and robot observations. The submitted experiment keeps
the action flow objective fixed and tests broader demonstration coverage.

## Results: frozen checkpoint 50, training initialization `episode_0000`

| Sampling seed base | Outcome | Actions |
| --- | --- | ---: |
| 7 | Success | 166 |
| 1007 | No success | 200 |
| 2007 | No success | 200 |

All three used checkpoint SHA-256
`fed8976c64c4b0842cb9158b675501c18d4ac8c8b2ae248e0aa1ff7947600bc2`;
initial-state, RGB and Muse-feature parity checks passed. The two unsuccessful
attempts are included in the results, not discarded. These trials share the
same demonstrated training initialization; they are not a held-out test.

The [18-second presentation video](../submission/object_dynamics_vla_demo.mp4)
is rendered from the actual successful rollout with
`bash scripts/render_hackathon_demo.sh` after downloading the source rollout.
It contains no synthesized robot footage or audio. The full
rollout, per-step trace, observations, noise arrays, and reports are saved in
the Modal volume under `/eval/muse_dataset50_episode0000_v1/` and
`/eval/muse_dataset50_episode0000_seed{1007,2007}_v1/`. See
[reproduction instructions](REPRODUCE_MUSE_SAM.md) for download commands and
the exact evaluation invocation. Those artifacts are not bundled in Git.

## Supporting experiment history

- Recorded-action replay of the demonstration succeeded, verifying the
  simulation and action replay path. This is not policy success.
- An input-alignment audit found and repaired a one-frame robot-observation
  timing mismatch across 42 development episodes.
- A fresh aligned single-episode model passed initial-input parity in a
  200-action closed-loop run, but did not secure or lift the carton. Its
  eight-frame in-sample action metric is not a full-frame evaluation.
- A dataset-level warm-started experiment used shuffled windows from 34
  training episodes, with eight reused development episodes excluded from
  optimization. Checkpoint 50 produced the result above. The warm start
  restarted AdamW from the earlier weights; it was not an exact optimizer
  continuation.

Native SmolVLA footage, if included, must be labeled as a separate reference;
it is not evidence that the Muse/SAM-conditioned policy succeeded. The
step-aligned gripper-sign comparison is confounded by the robot's divergent
physical position and does not establish an isolated gripper defect.

## Demo and evidence to assemble

1. Show the task, the tracked milk and basket, and the Muse/SAM-to-action
   architecture.
2. Show the actual learned policy's saved rollout, with a success/failure label
   and the relevant checkpoint and initialization identified.
3. Include training and development metrics only for the checkpoint actually
   evaluated; distinguish teacher-forced action prediction from simulator
   control and repeated development observations from untouched tests.
4. Link the repository, [alignment analysis](MUSE_CANDIDATE_INPUT_ALIGNMENT.md),
   [dataset training instructions](ALIGNED_DATASET_TRAINING.md), and saved
   video and trace artifacts when available.

**Current status:** component integration, alignment and one successful
Muse/SAM-conditioned rollout on a training initialization are demonstrated;
the 1/3 repeatability result does not establish reliable manipulation or
generalization, and does not isolate SAM's individual contribution.

The first successful rollout and report are available on the Modal data volume
under `/eval/muse_dataset50_episode0000_v1/`. Its checkpoint SHA-256 is
`fed8976c64c4b0842cb9158b675501c18d4ac8c8b2ae248e0aa1ff7947600bc2`.
