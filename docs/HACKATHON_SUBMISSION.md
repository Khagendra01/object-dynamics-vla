# Hackathon submission draft: Object Dynamics VLA

## Project

Object Dynamics VLA is a research prototype for using vision-language-model
features and tracked object measurements to condition a robot action policy.
Its target demonstration is a simulated LIBERO task: pick up a milk carton and
place it in a basket.

The experimental policy combines frozen NF4 Muse layer-26 image/instruction
features with causal SAM-derived milk and basket measurements. A
query-normalized prefix conditions a SmolVLA action expert; the policy
replans from camera and robot observations. The submitted experiment keeps
the action flow objective fixed and tests broader demonstration coverage.

## What has actually been demonstrated

- Recorded-action replay of the demonstration succeeded, verifying the
  simulation and action replay path. This is not policy success.
- An input-alignment audit found and repaired a one-frame robot-observation
  timing mismatch across 42 development episodes.
- A fresh aligned single-episode model passed initial-input parity in a
  200-action closed-loop run, but did not secure or lift the carton. Its
  eight-frame in-sample action metric is not a full-frame evaluation.
- A dataset-level warm-started experiment used shuffled windows from 34
  training episodes, with eight reused development episodes excluded from
  optimization. At checkpoint 50, the actual Muse/SAM-conditioned policy
  achieved the simulator's task-success condition after 166 actions on
  `episode_0000`, a training initialization. Initial RGB, Muse features,
  robot state, and SAM measurement parity checks passed. Two additional
  sampling seeds are being evaluated; this single success does not establish
  repeatability or generalization to new initializations.

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
reliable manipulation and generalization have not yet been established.

The first successful rollout and report are available on the Modal data volume
under `/eval/muse_dataset50_episode0000_v1/`. Its checkpoint SHA-256 is
`fed8976c64c4b0842cb9158b675501c18d4ac8c8b2ae248e0aa1ff7947600bc2`.
