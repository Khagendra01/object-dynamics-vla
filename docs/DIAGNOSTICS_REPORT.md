# Oracle-policy and SAM diagnostics

These are exploratory diagnostics on the milk task's **validation** episodes.
They do not replace the previous test results or establish a robust control
improvement. The simulator masks used by the policy and monitor are privileged.

## Action path and initial history

`scripts/modal_libero_policy_rollout.py` now writes per-step sampled and executed
commands, numerical forecasts and observations, and per-channel summaries to
`/data/eval/*_val_sample*_diagnostics.json`. Copies are in
`outputs/libero/oracle/`. Diagnostics compared training seed 7 checkpoints
for `objects_joint` and `objects_action_only`, on eight validation initializations
with action-sampling schedules 7 and 19. These are paired trajectories, not
independent scenes. Successes were 1/8 and 2/8 for joint, versus 1/8 and 0/8
for action-only, respectively.

The environment reports action bounds `[-1,1]` for all seven channels; its
`OperationalSpaceController` reports the same input bounds for the six arm
channels. The dataset is copied in the source controller convention. Per-channel
normalization is fitted on training episodes, and sampled actions are denormalized
using the saved training mean and standard deviation before clipping.

For joint seed 7 / sampling schedule 7, clipping frequencies by channel
`[x,y,z,rx,ry,rz,gripper]` were `[0, .0193, .0109, 0, 0, 0, .2588]`;
mean absolute clipping magnitudes were `[0, .00262, .00125, 0, 0, 0, .0634]`.
The corresponding action-only gripper clipping rate was .2727; for sampling
schedule 19 it was .3566 joint and .4256 action-only. These fractions are
per-step/per-channel, not per-episode. Detailed demonstration, pre-clip,
post-clip, and consecutive-command distributions are in the JSON traces.

The inspected robosuite `PandaGripper.format_action` uses `np.sign(action)` to
increment its internal gripper actuator state; its docstring says -1 opens and
1 closes. Clipping an out-of-range gripper value to [-1,1] preserves its sign,
so the high *gripper-channel* clipping rate alone does not mean the gripper
direction changed. Translation clipping can change the actual bounded command.
The recorded demonstration's gripper commands are +/-1; sampled gripper values
are continuous. Sign-switch rates on these validation trajectories were 7.2%
and 9.0% for joint schedules 7 and 19, versus 12.4% and 9.8% for action-only.
These are descriptive, not evidence that a gripper change would improve control.

The model integrates flow with 20 Euler steps. On every 25th joint/schedule-7
step, a 40-step integration with the same starting noise was also computed
without changing control. Mean absolute first-action differences in source
units were approximately `[.00067,.00301,.00229,.00017,.00030,.00019,.00537]`.
This is a sampled numerical comparison, not a proof of convergence. Rollout
history is bootstrapped by repeating the initial observation four times; normal
training windows use four actual consecutive observations. Per-episode/channel
differences against the first four recorded observations are in the traces.

## Shared encoder gradients and weight sweep

`scripts/modal_gradient_diagnostics.py` measured the unweighted individual loss
gradients of the **encoder parameters** on 32 randomly sampled training batches
at each frozen, validation-selected joint checkpoint. Mean action/dynamics norm
and mean cosine were: seed 7 `0.5193 / 0.00520`, `0.0133`; seed 19
`0.4981 / 0.00552`, `-0.0332`; seed 31 `0.4257 / 0.00551`, `0.0265`.
Cosines were negative on 37.5%, 62.5%, and 31.25% of sampled batches. These are
checkpoint snapshots before gradient clipping, not gradients throughout training,
and they do not include the selector parameters.

After that measurement, `scripts/modal_loss_weight_sweep.py` trained weights
1, 10, and 100 over seeds 7, 19, 31, evaluating **validation only**:

| Dynamics weight | Mean flow MSE | Mean sampled-action MSE | Mean dynamics MSE |
|---:|---:|---:|---:|
| 1 | 0.903345 | 0.712937 | 0.002208 |
| 10 | 0.902886 | 0.722525 | 0.002140 |
| 100 | 0.897588 | 0.747198 | 0.001901 |

Higher weight improved mean dynamics and flow metrics here but worsened sampled
action MSE, with substantial seed variation. Each weight's checkpoint was
selected by its own `flow + weight*dynamics` validation objective, so checkpoint
selection is not identical across weights. No new test evaluation or control
success claim was made. Raw batches and per-seed results are in
`milk_shared_gradient_diagnostics.json` and `milk_weight_sweep_val.json`.

## Passive numerical monitor

The historical monitor forecast the next numerical change before each step, conditioned on
the accepted clipped first action and a clipped *hypothetical* remainder of the
sampled action chunk, then records the observed post-step change and valid mask.
It does not alter control. This is a limitation of the current multi-action
dynamics head: the remainder is not yet executed and is not guaranteed to be
the actions actually taken. It has no trained forecast for action-only or
no-object variants. On joint seed 7 / schedule 7, the mean absolute error over
valid normalized-image-fraction features was 0.000991 over the single successful
episode and 0.001609 pooled over seven failures. Those are correlated time
steps, not independent trials; there are no grasp/miss/slip labels. This does
not establish event discrimination or a recovery trigger.

The corrected monitor is a separately trained `OneStepDynamicsHead`: it receives
only frozen encoded history and the accepted action at t, with target delta at
t+1. Training uses the parent policy's train/validation splits and normalization,
freezes that encoder, and ties the head to the parent weight hash and dataset
fingerprint. On milk it selected epoch 27 of 30, with one-step validation
normalized MSE 0.00106467; test was not accessed. The rollout enables it only
when its checkpoint is explicitly supplied and never silently reuses the old
chunk-conditioned head. A regression test changes later hypothetical actions
while keeping the accepted action fixed; the forecast is identical, and a full
chunk is rejected.

In the validation seed-7/schedule-7 rollout, the corrected passive monitor's
pooled native image-fraction MAE was 0.001675 versus 0.000427 for a zero-delta
persistence baseline, across 15,570 valid feature observations. This is an
on-policy diagnostic with oracle masks, not a matched event classifier: many
features are stationary, trajectories are correlated, and grasps/misses/slips
are not labeled. It does not establish that monitoring can trigger recovery.
Per-step predictions and observations are kept locally at
`outputs/libero/oracle/milk_joint_val_sample7_one_step_diagnostics.json`.

The follow-up compared the same frozen head with persistence separately on
validation demonstrations and on policy-generated validation trajectories
(sampling schedules 7 and 19). A predeclared exploratory motion proxy classifies
an object-step as moving if any valid image-fraction delta exceeds 0.002. Mean
absolute errors in image-fraction units:

| Source | All: head / persistence | Moving objects: head / persistence | Stationary objects: head / persistence |
|---|---:|---:|---:|
| Validation demonstrations | 0.001010 / 0.001211 | 0.002061 / 0.003470 | 0.000483 / 0.000078 |
| Policy schedule 7 | 0.001675 / 0.000427 | 0.002440 / 0.002941 | 0.001560 / 0.000050 |
| Policy schedule 19 | 0.002141 / 0.000454 | 0.002706 / 0.003845 | 0.002068 / 0.000020 |

The head beat persistence on this motion proxy, while its stationary predictions
were worse; stationary object-steps dominated the policy traces. The same eight
validation initializations recur across schedules, and validation also selected
the monitor checkpoint. These are not independent test trials, labeled grasps,
misses, or slips. `/data/eval/milk_one_step_distribution_val.json` contains
per-episode breakdowns and normalized MSE. No new held-out test was accessed.

## Separate SAM checks

- The official SAM `assets/images/test_image.jpg` with the notebook's `shoe`
  prompt returned 12 masks. The inspected overlay highlights the shoes; this is
  a positive control for this inference path, not a benchmark score.
- On the 128-pixel milk frame, agent-selected visual point guesses returned no
  masks. An agent-selected basket box returned one mask, with 0.96875 IoU against
  the simulator mask used **only for post-hoc scoring**. The box is not a human
  annotation and does not demonstrate autonomous text grounding.
- A forward tracking probe retained the initialized basket ID on only 3 of 150
  frames, with mean IoU 0.0194 when missing frames count as zero. A request for
  forced tracker propagation gave the same outcome because the wrapper did not
  forward that unsupported flag; it was not a distinct propagation mode.
  Prefix-versus-full checks at frames 50 and
  100 had zero differing pixels, but both masks were empty, so those checks are
  **vacuous** and do not verify causality. This is not a verified tracking
  episode.
- Actual **native 512×512** RGB was then rendered from the same simulator states,
  rather than upscaled from 128×128. The `carton` text prompt found the milk at
  frames 0/50/100, with best post-hoc IoU 0.973/0.953/0.984 against resized
  simulator masks; frames 50/100 each also contained a second detection. Other
  tested milk/basket phrases returned no instances. Visually agent-selected milk
  and basket boxes at frame 0 yielded IoU 0.973 and 0.986 respectively.
- Forward tracking through the full native-resolution episode still retained
  each initialized object ID on only 3/150 output frames. Mean IoU with missing
  frames counted as zero was 0.0197 for milk and 0.0197 for basket. At frames 50
  and 100, prefix and full-clip masks were identically empty; the comparison is
  again vacuous. This is a failed tracking probe, not a verified episode, and the
  box selection was agent-driven, not human annotation. Human-provided
  initialization and meaningful identity/occlusion and prefix checks remain open.

### Online human-annotated initialization: DAVIS 2017

**Follow-up:** the stage-by-stage investigation and a diagnostic point-path
workaround are in `docs/TRACKER_TRACE_REPORT.md`. The original box and point
failures arose at different stages. The workaround returned masks on all 69
frames of this one video, but the nonempty prefix/full comparison differed at
frame 10; it is not validated as a causal production adapter.

The [official DAVIS 2017 download page](https://davischallenge.org/davis2017/code.html)
provides human-annotated first-frame masks; the
[shijli/davis2017 repack](https://huggingface.co/datasets/shijli/davis2017)
provides the official 480p validation frames and masks under CC BY-NC 4.0,
with source-video terms also applicable. This is an independent video benchmark,
not a source of annotations for LIBERO.

`scripts/modal_davis_human_init.py` downloaded the pinned validation parquet,
verified its SHA-256, extracted the first validation video (`bike-packing`, 69
frames), and derived one box from the human-annotated first-frame object mask.
Later ground-truth masks were used only for post-hoc scoring. With this box,
SAM 3.1 returned the initialized object ID on only 3/69 propagated frames; mean
region IoU was 0.03187 with missing outputs counted as zero. Prefix-vs-full
outputs at frames 10 and 20 were equal but empty, so these comparisons are
vacuous. This reproduces the short-lived output pattern on a human-annotated
benchmark and motivates investigating predictor configuration/output handling;
it does not prove a model limitation. Detailed provenance and per-frame scores:
`outputs/libero/oracle/davis_human_init_sam31_probe.json`.
