# Robot-pretrained SmolVLA adapted to LIBERO

This experiment reuses the **complete official robot-trained SmolVLA policy**,
not an expert attached to Muse. It is a development experiment on previously
used milk validation episodes; it is not an independent evaluation or evidence
of generalization to a physical robot. It does not use SAM measurements.

## Checkpoint and compatibility

`lerobot/smolvla_base` revision `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6`
was downloaded to `/data/checkpoints/smolvla-base/` on the Modal volume. Its
`model.safetensors` SHA-256 is
`7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb`.
LeRobot source is pinned at `552777b7c384e8e71a02303da39583f9a2653760`.
The checkpoint's config describes 6 state/action channels and three RGB
cameras. The audited target dataset has 15 robot-state channels, one agentview
camera and 7 normalized Panda OSC_POSE action channels. We **do not interpret
the original six-channel joint-target outputs as OSC_POSE commands**: the
policy was adapted with supervised LIBERO actions and train-only LIBERO mean/
standard-deviation normalization. Pretrained maximum state/action dimension 32
is retained; all checkpoint weights loaded with `strict=True` before training.
The frozen VLM and fully trainable expert/state/action/time connectors are the
official implementation. The new camera configuration is one agentview, so the
policy is not receiving the checkpoint's original three camera streams.

The checkpoint's expert width is 720; its action input projection is `[720,32]`
and its first query projection `[960,720]`. Our Muse expert width is 512, uses
PyTorch attention blocks and consumes projected 6656-dimensional Muse tokens.
These weights do **not** directly fit the Muse expert. A future Muse integration
would need a compatible expert/attention design and checked connectors; do not
silently reshape, truncate, or treat this SmolVLA result as a Muse result.

The model card describes a community-trained base policy but does not give a
complete checkpoint-level provenance audit in this repository. Whether its
pretraining overlaps the LIBERO source or related data was not independently
established. The 16 validation windows were used for checkpoint selection, and
all 1,211 validation windows below are development data. The previously used
eight test initializations were not reused here.

## Fine-tuning

`scripts/modal_smolvla_libero.py` trained seed 7 on the existing milk train
episodes, 512×512 agentview JPEGs and demonstration instructions, batch 4,
AdamW learning rate `1e-4`, official flow-matching objective and 50-action
chunks. Checkpoints were selected by minimum fixed-noise validation flow MSE on
the first and middle frame of each of eight validation episodes. This is a very
small selection set and was also used for reporting. The second run was started
from the original pretrained checkpoint, not silently resumed from weight-only
files; it used a constant LR and 1,000 declared updates. The optimizer/RNG
snapshot is recorded, but this new driver does not yet implement resume.

| Update | validation flow MSE on 16 windows |
|---:|---:|
| 0 | 2.152840 |
| 250 | 0.844470 |
| 500 | 0.672295 |
| 750 | 0.690842 |
| 1000 | 0.634523 |

The 500-update run selected step 500; the 1,000-update run selected step 1000.
At step 1,000 the final sampled minibatch training loss was 0.4593. It took
655 seconds including validations and checkpoint writes on Modal T4, with peak
allocated GPU memory 2.08 GiB. A separate 10-update smoke had worse sampled
first-action metrics after training; it is retained at
`/data/runs/smolvla_libero_smoke10`. This experiment did not use 500/1,000
independent demonstrations or the effective batch 64 used in the Muse run.

## Offline first-action evaluation

On **all 1,211 reused validation frames**, with 10-step sampling and matched
seed 921, gripper sign agreement was 57.80% for the unadapted checkpoint,
86.13% after 500 updates, and 90.83% after 1,000 updates. Always-open/closed
baselines are 42.20%/57.80%; at 1,000 updates class recalls are 83.56% for
negative/open and 96.14% for positive/close. The 100% gripper agreement seen on
the 16 checkpoint-selection windows is not representative of the full split.
The unadapted model's native action semantics are incompatible with LIBERO, so
its zero-shot scores are an integration baseline, not a valid independent robot
performance estimate.

The 1,000-update native first-action MAE by channel `(x,y,z,rx,ry,rz,gripper)`
is `[.15782, .19242, .21071, .02695, .04800, .03823, .31675]`. Normalized MSE
by channel is `[.84921, .23664, .44436, 1.26471, 1.52815, 2.07088, .33040]`.
Full per-channel results, confusion matrix, and source statistics are saved at
`/data/eval/smolvla_libero_milk1000v1_offline_v1.json`. The 500-update and
unadapted results are at `/data/eval/smolvla_libero_milk500_offline_v1.json`.

## Closed-loop development rollouts

The 500- and 1,000-update selected checkpoints were tested on the same two
already-used milk validation initializations as the Muse development runs.
Each step renders current RGB, samples a 50-action chunk using `7+step`, executes
its first action clipped to `[-1,1]`, and replans. The policy uses no SAM or
simulator masks. The simulator runs pinned MuJoCo 2.3.7/LIBERO on Python 3.11;
the official policy runs on Python 3.12 in a separate Modal container. The
initial RGB and simulator-state hashes match the earlier Muse rollouts.

| Selected adaptation | Episode 0014 / demo_21 | Episode 0020 / demo_27 |
|---|---|---|
| 500 updates | success at step 151 | no success in 200 steps |
| 1,000 updates | success at step 135 | success at step 132 |

The 1,000-update runs each changed gripper-command sign 5 times, versus 2 in
each corresponding recorded demonstration; 47/135 and 53/132 steps respectively
had at least one clipped channel, mostly gripper. The two final RGB frames were
inspected and show the carton at the basket. Simulator success is the task
predicate, not proof of a stable grasp or transfer to a real robot. The median
policy-only sampling times were .329 and .325 seconds per step on T4, excluding
simulator RPC, rendering and preprocessing; peak allocated GPU memory was
2.26 GiB. Each run used one sampling schedule and one training seed.

Videos were copied to `/home/kgen/Desktop/smolvla_1000_episode14.mp4` and
`/home/kgen/Desktop/smolvla_1000_episode20.mp4`. Reports, complete videos and
raw/clipped per-step actions are under
`/data/eval/smolvla_libero_milk1000v1_episode_0014_v1/` and
`/data/eval/smolvla_libero_milk1000v1_episode_0020_v1/` on Modal. The first
500-update 0014 attempt produced a video but failed to save its trace because
the policy container's volume view was stale; the v2 rerun used explicit volume
reload and saved its trace. The failed attempt is retained.

The repository's 36-test Modal CPU suite passed after this work. A separate
one-step rollout using the final parameterized entry point also completed at
`/data/eval/smolvla_libero_milk1000v1_episode_0014_postcheck/`; it is a software
check, not another task trial. At the last billing check, the workspace summary
showed $7.40 in credits applied in November; the previous October cycle is
separate and the CLI summary does not expose the remaining credit balance.

## Reproduction and next steps

Existing artifacts are never overwritten; choose fresh suffixes/output paths.

```bash
~/modal-venv/bin/modal run scripts/modal_smolvla_checkpoint_audit.py
~/modal-venv/bin/modal run scripts/modal_smolvla_libero.py --steps 1000 --suffix NEWNAME
~/modal-venv/bin/modal run scripts/modal_smolvla_libero_eval.py --run-name milk1000v1
~/modal-venv/bin/modal run scripts/modal_smolvla_libero_rollout.py --episode episode_0014 --run-name milk1000v1 --suffix NEWNAME
~/modal-venv/bin/modal run scripts/modal_smolvla_libero_rollout.py --episode episode_0020 --run-name milk1000v1 --suffix NEWNAME
```

The 2/2 development outcomes are encouraging, but no independent success rate
or causal comparison against Muse can be inferred: training budgets, model
initialization, input architecture and observation conditioning differ. Before
generalizing, audit checkpoint/data overlap, freeze the protocol, evaluate truly
unused initializations and more tasks/seeds, and make the training driver
resumable. A SmolVLA-expert-to-Muse connector requires a separate compatibility
implementation and experiments; it is not implemented by this adaptation.
