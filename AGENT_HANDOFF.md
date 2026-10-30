# Coding-agent handoff

## 2026-10-30 continuation

The current Modal profile can access the existing `object-dynamics-vla-data`
volume. The present machine is Linux, lacks local PyTorch, and was reported by
the owner to have no GPU. `docs/MUSE_VLA_METHOD.md` records the new motor
checkpoint gate and the real one-update T4 smoke. This is not multi-task training
or a completed comparison. The new gate currently requires the same dataset
fingerprint and statistics across motor pretraining and adaptation; broader
pretraining needs explicit cross-dataset split and normalization provenance.
Existing scratch variants are diagnostics, not the requested common-checkpoint
experiment. The official SmolVLA adapter reference, broader LIBERO token caches,
full motor training, M0/M1/M2, development rollouts, and final evaluation remain
open. Check remaining credit and measured costs before expanding GPU jobs.

## Goal and constraints

Evaluate the user's idea: learn action-conditioned changes in numerical
object measurements alongside action prediction, with both losses updating shared
features. The eventual stack uses Muse Glimmer 30B and Meta SAM 3.1.

The user has an RTX 5090 32 GB, Isaac Sim, a robot of unspecified type, and
approximately $30 of optional Modal credit. **Start with existing datasets.**
Robot model, gripper, camera, OS, system RAM, Isaac version, and actual dataset
locations remain unknown. Discover these locally where possible. Do not guess
controller units, frame conventions, model APIs, or benchmark success.

## Status after 2026-10-29 execution

`docs/EXECUTION_REPORT.md` has the full evidence. Done on an Intel Mac (CPU) and
Modal T4/A100: synthetic smoke and ablations, CLI
audit/replay/predict/evaluate/report, HDF5 fixture import, GPU training, and two
real LIBERO-Object tasks (50 demos each) downloaded, imported, and audited.
The latest local automated suite passes 28 tests.

The latest validation-only action, gradient, passive-monitor, and independent
SAM probes are in `docs/DIAGNOSTICS_REPORT.md`. At frozen milk checkpoints the
mean unweighted dynamics gradient norm on encoder parameters was about 1% of
the action-gradient norm. A 1/10/100 weight sweep was validation-only and did
not establish better sampled-action error. Gripper clipping preserves direction
under the inspected PandaGripper sign-based interface. Native 512-pixel RGB
improved single-frame carton detection, but box-initialized full-episode tracking
retained each target ID on only 3/150 frames; prefix checks at 50/100 compared
empty masks and do not establish causality. A separate DAVIS video used a box
derived from a human-annotated first frame, but tracking was present on only 3/69
frames; its empty prefix checks also do not establish causality. This does not
provide LIBERO annotations or a verified human-initialized tracking episode. The
next untouched evaluation requires fresh initializations:
the original eight test demonstrations have already been evaluated.

The SAM stage trace and narrow missing-cache point-path workaround are documented
in `docs/TRACKER_TRACE_REPORT.md`. Fresh-container tests across three DAVIS clips
show identical repeated runs and identical results when same-length future frames
change, but prefix/full frame-10 masks still differ by 32, 7, and 4 pixels. It
is not validated as a strictly prefix-consistent production adapter. The passive monitor now
uses a separately trained one-action head; it does not receive hypothetical later
actions. Validation and on-policy limitations are in `docs/DIAGNOSTICS_REPORT.md`.

The later targeted bike-packing intervention identified sequence-length-dependent
pointer temporal normalization as the cause of that frame-10 discrepancy in
this point-path setup. Fixing only the encoding denominator to 15 made tested
lengths 11, 12, 15, 16, 17, and 69 identical at frame 10, with unchanged
available-past pointer selection. This is diagnostic, not yet a production or
general causality guarantee; `only_obj_ptrs_in_the_past_for_eval` was false.
Full details and effective TF32 flags are in `docs/TRACKER_TRACE_REPORT.md`.

The latest initialized RGB pilot and scoped bounded-memory audit are in
`docs/CAMERA_PILOT_REPORT.md`. The experimental point adapter uses the configured
pointer maximum, enables past-only pointer evaluation, and can disable known
single-GPU detector lookahead. Selected prefix checks matched on three DAVIS
clips and one 137-frame LIBERO milk validation episode; only the milk carton was
tracked in the LIBERO pilot. That pilot used an agent-selected first RGB point,
native 512-pixel simulator rendering, and simulator masks only for post-hoc
scoring (mean IoU 0.9638). Full videos are preloaded in the full-session tests,
not acquired live; no general online-causality guarantee is claimed. The next
policy comparison must train both variants on the same SAM measurements, and
this validation initialization is no longer untouched.

Subsequently `docs/TWO_OBJECT_VALIDATION.md` reports simultaneous milk and
basket point tracking on validation episodes 0014, 0020 and 0022. All outputs
were nonempty, milk mean IoU ranged 0.9265–0.9685, basket 0.9599–0.9788, and
the other target never had higher reference IoU. These nearly identical scenes
use visually selected points and full-video preload, not autonomous grounding or
validated live operation; all three validation initializations are now used.

The 42-episode dataset-level SAM experiment is in `docs/MILK_SAM_DEVELOPMENT.md`.
Both slots are cached for 6,098 frames; checkpoint selection and scoring use the
same development split, with test excluded. Action-conditioned one-step SAM
dynamics modestly beat action-free on validation normalized MSE, but not
persistence on stationary subsets. Matched policy training showed small offline
action-MSE improvements in three paired seeds. No SAM closed-loop rollout was
performed because the current adapter cannot append live frames.

`docs/SAM_CLOSED_LOOP_PILOT.md` supersedes that last rollout blocker for a slow
reference path: prefix replay matched the cache at nine sampled lengths. A
30-step paired seed-7 run on reused validation episode 0014 used only RGB-derived
SAM masks and paired sampling, with no missing IDs or task success. It is not a
full-horizon success comparison, is not real-time, and used repeated initial
history. First-action channelwise errors and clipping are documented there.

Text-only SAM 3.1 failed on this task: unique target/basket prompts returned zero
masks; only generic `can` matched 2–3 ambiguous instances. A full episode produced
296 missing and zero observed masks. Do not repeat that extraction or attach the
all-missing `/libero/masks_sam31_ep0` artifact.

The user selected oracle simulator masks to unblock the numeric mechanism test.
`scripts/extract_libero_oracle_masks.py` replayed recorded states and rendered
exact geom-name masks for `alphabet_soup_1` and `basket_1` on all 50 episodes:
15,605 observed / 15,616 slot-frames, 11 missing, 0 ambiguous, observed fraction
99.93%, RGB mean absolute difference 2.02/max 4.29. Provenance is explicitly
`oracle: true`; this is privileged, not camera-only. Dataset
`/libero/object_oracle`; split seed 17 is 34/8/8; fingerprint `99cf1697...`.

The real numeric experiment ran on a Modal A100-80GB. Joint dynamics training
(30 epochs, seed 7) selected epoch 5: validation flow `0.907371`, validation
dynamics `0.003332`; test dynamics `0.002951`, moving test `0.003572`. The
three-seed validation mean dynamics was `0.003249`. Full three-seed validation
and frozen seed-7 test ablations are in `docs/VALIDATION.md` and
`outputs/libero/oracle/`. Key finding:
the numerical loss improved measurement forecasting, but did **not** improve
open-loop action prediction over measurements-as-inputs-only; the robot-only
baseline was worse on validation but better on test, so that direction remains
inconclusive. These particular numbers are offline metrics; later closed-loop
rollouts are summarized below.

The matched oracle experiment was also repeated on the milk-in-basket task:
50 episodes and 14,562/14,562 observed slot-frames. Three-seed validation mean
joint/action-only flow MSE was `0.903345`/`0.904746`; frozen seed-7 test was
`0.909617`/`0.911965`. Joint/no-dynamics-actions test dynamics MSE was
`0.002289`/`0.002497`. The small action-metric difference reverses direction
between the two tasks; neither establishes a robust policy improvement.
See `docs/EXECUTION_REPORT.md` and `outputs/libero/oracle/milk_*`.

Muse Glimmer 30B revision `cec94fcfd6512f2cfe708138c32533b346ea4f69` has now
run on Modal A100: finite `[1,84,6656]` hidden states, peak 55.59 GiB, and a
stride-4 cache for all 50 milk episodes. With oracle masks plus Muse context,
three-seed validation joint/action-only flow MSE was `1.183339`/`1.176542`;
seed-7 test was `1.243515`/`1.213720`. This setup did not improve action
metrics. The image processor used a BICUBIC fallback; see the execution report.
Recorded milk-demo actions replayed in LIBERO reached success. Initial learned
policy closed-loop tests on eight held-out milk initial states used live oracle
masks: joint versus object-input/action-only successes were 1/8 vs 0/8 (seed 7),
0/8 vs 0/8 (seed 19), and 6/8 vs 6/8 (seed 31). Clipping was frequent; this is
one task, one sampling schedule, and the same eight initial states per seed, not
evidence of a robust improvement. SAM text prompts on the milk episode produced 300/300
missing slot-frames, so camera-only identity still needs work.

The next gate is perception realism and generalization, not another oracle run:
(a) choose a task with uniquely recognizable objects or build a verified causal
tracking adapter and (b) repeat across more tasks/splits/seeds. Isaac,
camera-only real-data control, broader policy rollouts, and robot work
remain blocked. Do not present the oracle policy as operating from camera
observations alone.

## First actions

1. Read README, `docs/VALIDATION.md`, and `docs/EXECUTION_REPORT.md`. Run
   `python -m unittest discover -s tests -v`. Do not repeat the completed LIBERO
   import, SAM failure, or oracle extraction unless you are intentionally testing
   a new perception path.
2. Run `python -m odvla doctor` in the user's CUDA environment. Confirm a real CUDA
   operation succeeds on the 5090. Keep Isaac and learning environments separate.
3. Inspect the oracle artifacts and results first. Treat them as a privileged
   mechanism check, not a camera-only policy.
4. For a new dataset, audit its schema before converting. Do not download every
   benchmark or create a new collection pipeline by default. Import one task and
   replay it; verify RGB orientation, action-before/after-state alignment, FPS,
   robot channels, and instruction/object identity.

Acceptance: an import audit and a correctly aligned replay, with no claimed SAM
measurements before masks have actually been extracted, and no camera-only claim
for oracle masks.

## Perception integration gate

5. SAM 3.1 has been executed on a Modal A100-80GB (commit `f1647e8`,
   checkpoint SHA256 `0567debe...`). The upstream `offload_state_to_cpu`
   mismatch is patched in `scripts/extract_sam31.py`. Its unique text-prompt
   result on this task was zero usable masks; do not repeat it unchanged.
6. The oracle mask path is complete for this task: 50 episodes, 99.93% observed,
   explicit `oracle: true` provenance. Do not describe it as camera-only.
7. The next perception gate is a task with uniquely recognizable objects or a
   verified causal streaming/tracking adapter. Multiple identical instances need
   explicit association; missing observations must remain missing.
8. For a streaming adapter, compare output at t when processing frames 0..t
   versus 0..t+k. The input representation for t must not change with future
   frames. Full-clip tracking can revise outputs: do not assume it is causal.

Acceptance: complete causal mask artifacts; observed/ambiguous/missing rates;
replay reviewed against RGB; pinned model provenance. A provenance declaration
alone is not proof of causality. Oracle provenance must travel with the results.

## First experiment gate

9. The real numeric-only experiment and matched comparisons have run with oracle
   masks. Joint dynamics improved measurement forecasting (validation 14.6% lower
   than the `objects_joint_no_dynamics_actions` head; test 8.4% lower) but did
   not improve open-loop action flow or sampled-action error over
   measurements-as-inputs-only. The robot-only baseline was
   worse on validation but better on test. Do not select on test or discard this
   negative result.
10. Repeat across more tasks, splits, and seeds to test whether that reversal is
    stable. Avoid a score dominated by static objects; inspect motion separately.
11. The pinned Muse BF16 hidden-state path was executed and cached for the milk
    oracle dataset; it worsened the reported action metrics. Validate a camera-only
    measurement pipeline and investigate generalization before claiming benefit.
    Do not replace hidden extraction with generated text as if it were equivalent.
12. The A100 smoke peaked at 55.59 GiB allocated. For other hardware or a
    quantized path, separately validate memory, numerical behavior, and output
    contracts. No full 30B fine-tuning is expected in v0.1.
13. When extending to new data or adding Muse features, run the four matched
    variants with identical data, splits, feature cache, training steps, seeds,
    history/horizon, and model width. All hyperparameter selection uses
    validation. Test evaluation occurs after choices are frozen.

Acceptance: reproducible logs and comparison.json for actual data; explain whether
actions improve dynamics prediction and whether auxiliary supervision improves
open-loop action prediction. Do not discard negative results.

## What needs implementation after the offline experiment

- Map the policy action convention to the chosen Isaac robot/controller. Add
  joint/workspace limits and collision checks in a robot-specific adapter.
- Implement closed-loop stepping: current observations → action chunk → validate
  → execute first action → refresh. Recompute dynamics if validation changes actions.
- Add RGB/depth calibration and distinguish measured visible surface points from
  object origins/centers of mass. The helper is not a grasp-pose estimator.
- Use simulator ground truth as supervision/evaluation; do not leak privileged
  current object poses into a policy claimed to work from real observations.
- Evaluate task success on held-out initializations and perturbations. Then compare
  perception-only versus numerical-supervision variants with repeated seeds.
- Add failures/recovery data only when the existing dataset cannot answer that question.
- Integrate the real robot after the simulator/control mapping has passed checks.

## Research extensions, not prerequisites

Learned regions within masks, relational effect goals, temporal attention,
contact/influence latent states, uncertainty heads/calibration, candidate action
scoring, active observation, and selective Muse adaptation. The present gate is
per-measurement and the temporal encoder is an MLP over a fixed history.
Implement one extension at a time with a corresponding ablation.

## Non-negotiable experiment hygiene

- No future inputs, bidirectional fill-in, or post-hoc trajectory labels as current features.
- Fit scalers only on training episodes. New features require new splits/fingerprints.
- Group related/augmented trajectories by source seed/demo before splitting.
- Missing measurements are not zero motion. IDs are categorical; no numeric ID regression.
- Keep physical units/coordinate frames explicit. Rotation requires a suitable representation.
- Record actual run status: syntax/contracts, CPU execution, GPU execution, offline
  metrics, simulator success, and real-robot success are different evidence levels.
- Do not enable robot control, spending, data upload, or model-code trust implicitly.
  Follow the user's current authorization and local environment rules for those actions.
