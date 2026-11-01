# Frozen Muse + SAM VLA: implementation status

This is an experimental **SmolVLA-inspired adaptation**, not the official
SmolVLA model or a trained motor policy. The full specification is the owner's
`MUSE_SAM_VLA_END_TO_END_PLAN.txt`; implementation and experimental results
must be reported separately. The reference LeRobot source is pinned at
`552777b7c384e8e71a02303da39583f9a2653760`. Its flow interpolation is noise at
time one, velocity `noise - action`, with Euler sampling from one to zero.
The official SmolVLA one-update adapter reference has since been performed;
see `SMOLVLA_REFERENCE.md`. It is not a trained reference benchmark.

`configs/muse_sam_v1.json` fixes the existing milk development fingerprint,
model revisions, 20 Hz data convention, 4-frame history, 50-step actions,
network dimensions, optimization and three seeds. The current trainable
architecture has a Muse projection, robot and two-slot SAM history tokens,
two context blocks, and eight alternating causal self/cross-attention expert
blocks. A separate query uses **only** `a[t]` in the same expert blocks for a
one-step 2×5 dynamics target. Muse tokens are detached; SAM is an externally
cached frozen measurement. There is no action intervention or recovery gate.

`odvla/vla/dataset.py` constructs start/end-padded windows and checks each token
cache file's hash, timestamp, episode, and instruction. Statistics come from
training episodes only. Startup repeats the first observation and masks padded
history; end padding is masked from action loss. Observation encoders receive
no future labels. `odvla/vla/action_expert.py` shares expert weights across the
two objectives and uses valid-element mean losses. The implementation uses
PyTorch attention blocks, not LeRobot's unchanged action-expert implementation.

Muse block 26 of 52 is read with a forward hook, retaining every valid token.
The cache stores unprojected BF16 bit patterns, token IDs, masks, and positions;
no trainable projector is cached. Transformers 5.21.0 / torchvision 0.25.0
logs a **LANCZOS-to-BICUBIC fallback**; the exact same processor/backend must be
used at rollout. It is not equivalent to the configured original LANCZOS.
The SAM cache remains explicitly `causal: false`, with bounded prefix auditing.

The main trainer uses paired seeds, independent batch/noise/time RNG streams,
the same fixed-schedule validation flow selection, and snapshots every 250
updates. A seed-7 suite runner is experimental and subject to a runtime cap; do
not report partial arms as a matched outcome. It preloads the frozen token cache
in host RAM, so host memory and volume I/O are material resource costs.

## Motor checkpoint gate (2026-10-30)

`scripts/train_muse_vla.py` now accepts `--variant motor` to train without SAM
inputs or a dynamics loss. M0/M1/M2 require `--motor-checkpoint` pointing to a
selected `weights_best.pt`; the loader checks the motor configuration, exact
dataset fingerprint, train-only statistics, and state-dict shapes and logs the
checkpoint SHA-256. `--allow-scratch` is only for explicitly labelled diagnostics,
not a controlled comparison. At the time of the smoke this required the same
dataset fingerprint; the subsequent audited cross-dataset path and its remaining
limitations are described below.

On Modal T4, `~/modal-venv/bin/modal run scripts/modal_motor_smoke.py` performed
one effective-64 update from the existing milk development cache, seed 7. It
reported last microbatch flow MSE 2.8568447, validation flow MSE 2.9326054,
gradient norm 28.6879, and 102.33 seconds inside the training loop including
validation. The result and selected weights are at
`/data/runs/muse_motor_smoke_v1` on `object-dynamics-vla-data`. This is a
one-update smoke, not a learned motor checkpoint and not multi-task pretraining.
The first attempt (Modal app `ap-wkTFW8ritmBa6sYYAV7hJ6`) failed to start due
to a missing helper-module mount and was stopped; the corrected app was
`ap-dEu8cvGAyLZapzMImnB6Vr`. No matched M0/M1/M2 result follows from this smoke.

CPU-only Modal tests can be reproduced with
`~/modal-venv/bin/modal run scripts/modal_vla_tests.py`. Do not use the old
scratch seed-7 suite as the requested controlled comparison.

## Transfer and resume contracts

The trainer accepts `--dataset-audit` JSON for independently audited source
and destination datasets. It records source-demo identities for each split and
requires equal declared embodiment, controller, action/robot channels, frames,
units, gripper convention, camera/orientation, frequency, metric/slot roles and
Muse model/layer/processor, plus no motor-training overlap with destination
validation/test source identities. It checks metric and Muse declarations against
the actual manifests. Other semantics and source identities must be supported by
the audit evidence: these declarations cannot automatically detect paraphrased
or altered duplicate episodes or prove controller details. Unknown provenance
fails closed. Previous motor checkpoints have no such audit, so they cannot be
cross-dataset transferred without reconstructing and verifying their provenance;
do not add guessed metadata. Same-fingerprint transfers remain supported.

Cross-dataset transfer reuses the motor checkpoint's training-only normalizer
unchanged for observations, targets and output decoding. It does not fit a new
normalizer and silently change the policy's coordinate system. A supplied
preloaded dataset must already use those exact statistics. The current loader
still handles only a single dataset/cache at a time; this gate alone does not
implement multi-task joint training.

New runs atomically save `training_state.pt` containing model, optimizer,
update, best validation score, independent NumPy streams and PyTorch CPU/CUDA
RNG states at each `--save-every` checkpoint. `--resume` requires the original
schedule length, seed, dataset/cache identities, batch sizes and save interval;
it restores the saved normalizer. It cannot turn a completed 500-step schedule
into a 5,000-step schedule. Older weight-only runs are not resumable: a new
training phase from their weights requires an explicit new schedule and must not
be described as exact continuation. The CPU test interrupts a two-step run and
verifies bit-exact weights after resumption.
