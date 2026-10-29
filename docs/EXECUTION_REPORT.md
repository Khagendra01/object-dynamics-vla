# Execution report — 2026-10-28–29

Scope: this report records the local CPU run, the subsequent Modal T4 run, a
real-data/SAM 3.1 pass on a Modal A100-80GB, and a simulator-oracle LIBERO
numeric experiment after text-only SAM failed. The local machine started with
only the bundled synthetic fixture; one official LIBERO-Object task and the SAM
3.1 checkpoint were then obtained through Modal. A later Modal pass downloaded
Muse and evaluated a cached-feature experiment. No local CUDA device, Isaac
install, or robot was available. Nothing below is evidence of VLA
quality, camera-only SAM tracking, a robust learned-policy simulator benefit, or
robot success. The SAM text-prompt result on the first task was negative; the
oracle numeric result is a privileged, offline mechanism check.

## Environment

- macOS 12.7.4, x86_64 (Intel), CPU only. `nvidia-smi` is unavailable.
- Python 3.12.0 in `.venv`.
- torch 2.2.2 (the last macOS x86_64 wheel), numpy 1.26.4, h5py 3.16.0,
  Pillow 13.0.0.
- numpy 2.5.4 was installed first by pip and broke `import torch` for torch 2.2.2
  (`_ARRAY_API not found`). Pinning `numpy<2` fixed it. This affects only this
  Intel-Mac environment; the package itself allows numpy 2.x.
- Modal CLI 1.6.3 installed in `.venv` (`1.5.5` exists on the system Python).
  Authentication was configured locally.

## Local checks (passed)

| Check | Result |
|---|---|
| `python -m unittest discover -s tests -v` | 19 passed, 0 skipped, 6.13 s (18 passed before the SAM shim regression test) |
| `python -m odvla doctor` | `{"cuda": false, ...}` — correct for this machine |
| `python -m odvla smoke --out outputs/my_smoke --epochs 30 --device cpu` | completed; 5.2 s total training time |
| `python -m scripts.run_ablations ... --epochs 30 --seeds 7` | completed; 4 variants |
| `odvla audit`, `replay`, `predict`, `evaluate`, `report` | all produced artifacts |
| HDF5 fixture import -> audit -> replay | 10 frames, robot `[10, 15]`, actions `[10, 7]`, 20 Hz timestamps |

### Smoke reproduction (held-out synthetic test)

Numbers match `docs/VALIDATION.md` exactly, confirming deterministic CPU
execution of the packaged fixture.

| Predictor | Normalized delta MSE |
|---|---:|
| dynamics | 0.00032929884 |
| persistence | 0.00487212045 |
| constant velocity | 0.00597673794 |

Best validation epoch: 29. Frozen `objects_joint` evaluation on test: dynamics
MSE `0.00032929884`, sampled-action normalized MSE `1.89332235`.

### Ablation reproduction (validation only, seed 7)

Matches the packaged table at its published precision, with identical
variant ordering and conclusions.

| Variant | Sampled action MSE | Dynamics MSE |
|---|---:|---:|
| `no_objects` | 2.1137 | head untrained |
| `objects_action_only` | 1.7515 | head untrained |
| `objects_joint` | 1.7523 | 0.00018425 |
| `objects_joint_no_dynamics_actions` | 1.7517 | 0.00397706 |

The packaged conclusion still holds on re-execution: the numerical objective
improved dynamics prediction and did not improve sampled action error over the
object-input/action-only variant, on this synthetic fixture and one seed.

## Modal execution (real GPU)

A new Volume `object-dynamics-vla-data` was created and the 164 KB synthetic
dataset plus matching `splits.json` were uploaded to `/smoke`.

- `L4` attempt: image built successfully in 41 s, then the function waited more
  than 30 minutes for L4 capacity. It was stopped; this was capacity
  unavailability, not a code failure.
- `T4` succeeded end-to-end twice:
  - `modal_smoke`, 2 epochs.
  - `modal_smoke30`, 30 epochs.
- Remote runtime: Python 3.12.10, torch 2.15.0+cu132, `device: cuda`,
  dataset fingerprint `3abab1158fae9962f9a37d5b0a4bda4ace16004a9ae6245e52a417a5bd9d52cb`.

GPU vs local CPU at 30 epochs:

| Run | Best epoch | Selection objective | Dynamics MSE | Flow MSE |
|---|---:|---:|---:|---:|
| local CPU | 29 | 1.413664 | 0.00018425 | 1.413481 |
| Modal T4 | 22 | 1.416968 | 0.00018170 | 1.416787 |

The small difference and different selected epoch are consistent with
nondeterministic GPU kernels; both runs trained successfully. Reproduction on
GPU is approximate, not bitwise.

Incremental cost for these runs, from `modal billing report --for today
--show-resources`: T4 compute `$0.00262234` and `$0.00196660`; CPU, memory and
egress together under `$0.005`. Total well under `$0.01`. The workspace billing
summary reports `$0.00` billed this cycle after credits and storage allowance;
that summary does not expose the remaining credit balance.

## Issues found and fixed or worked around

1. Packaged `odvla/*.py` files had modification times in the future relative to
   this machine's clock. Modal rejected the upload with
   `__main__.py was modified during build process`. Fixed by normalizing mtimes
   (`find odvla -name '*.py' -exec touch {} +`).
2. `modal volume get <volume> /runs/<dir> <dest>` in CLI 1.6.3 concatenated the
   directory's files into one local file instead of recreating the directory.
   Worked around by downloading each file individually.
3. `scripts/modal_train.py` hardcoded `A100-80GB` and a 30-minute timeout. Added
   `ODVLA_MODAL_GPU` and `ODVLA_MODAL_TIMEOUT` environment overrides; defaults
   are unchanged. T4 is sufficient for these small heads.
4. Intel-Mac torch/numpy incompatibility described above; `numpy<2` pinned in
   the local venv only, not in `pyproject.toml`.

## Real dataset gate: LIBERO-Object

The package does not name a user-specific dataset. `docs/SOURCES.md` points to
the official LIBERO project, and the importer supports LIBERO/robomimic-style
HDF5. One official task was downloaded rather than every suite, as the handoff
requires.

- Source: `https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets`,
  task file `libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5`.
- Local file: `data/libero/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5`
  (gitignored), 780,145,352 bytes,
  SHA256 `42189d4415d4c51aaaf0708300653fccc39239cd3f2709079a713cd8d1678a8d`.
  The Hugging Face dataset card declares Apache-2.0; verify licensing before
  redistributing.
- HDF5 audit (`outputs/libero/source_audit.json`): 50 demos, no file attributes.
  Each demo has `actions [T, 7]` float64, `obs/agentview_rgb [T, 128, 128, 3]`
  uint8, `obs/joint_states [T, 7]`, `obs/ee_states [T, 6]`,
  `obs/gripper_states [T, 2]`, plus `rewards`, `dones`, `robot_states`, `states`,
  and a wrist camera. Episode lengths are 136–196 frames (median 153); all 50
  end with reward 1 and done, consistent with successful demonstrations. The
  first demo has 148 frames.
- Official LIBERO code sets `controller="OSC_POSE"` and `control_freq=20`
  (`libero/libero/envs/env_wrapper.py`). The importer copies 7-D commands
  unchanged; they are not claimed to be meters or joint targets.
- Import ran on Modal and produced 50 canonical episodes at
  `/libero/object_raw`, with robot `[T, 15]`, actions `[T, 7]`, RGB
  `[T, 128, 128, 3]`, and 0.05 s timestamps. Instruction:
  `pick up the alphabet soup and place it in the basket`; object slots:
  `alphabet soup can, basket`.
- Canonical audit (`outputs/libero/object_raw_audit.json`): all 50 episodes have
  `observed_fraction = 0.0`; perception is `{"source": "not extracted",
  "causal": false}`. A 74-frame replay at `outputs/libero/object_replay.gif`
  (stride 2) was reviewed: RGB is upright, the robot and objects look correct,
  and both measurement slots are explicitly labeled MISSING/UNOBSERVED.
- Alignment sanity on `demo_0`: correlation between each action translation
  component and the next end-effector position delta is `0.925`, `0.996`, and
  `0.984` for x/y/z. This supports the obs/action pairing assumption but does not
  prove the controller convention for rollouts.
- Training on a Modal T4 refused as designed before touching the model:
  `ValueError: Perception causality must be audited and declared before training`.
  This is the intended gate, not a bug. No numeric result exists for this data.

### SAM 3.1 execution on Modal A100-80GB

The existing Modal secret `hf-token` has access to the manual-gated
`facebook/sam3.1` repository. The checkpoint was downloaded to the Volume at
`/checkpoints/sam3.1/sam3.1_multiplex.pt`: 3,502,755,717 bytes, SHA256
`0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`.
No token is stored in the repository or printed in logs.

The extraction image used Python 3.12, torch `2.10.0+cu128`, torchvision, and SAM
at commit `f1647e8d0c1d280991f4d1b7d974c2704537fe66`. `sam3`'s declared
dependencies were insufficient for the multiplex predictor; `einops`,
`pycocotools`, and `psutil` were required. An upstream API mismatch also appeared:
`Sam3BasePredictor.start_session` passes `offload_state_to_cpu` to
`Sam3MultiplexTrackingWithInteractivity.init_state`, which does not accept it.
`scripts/extract_sam31.py` now drops that unsupported keyword
(`_patch_multiplex_init_state`).

One-frame smoke, frame 0, prompt `alphabet soup can`:

- A100-SXM4-80GB; model build 19.7 s; peak allocated memory after the prompt
  5.53 GiB; zero masks.
- A prompt sweep at frames 0/50/100 and 128 vs 512 px tested `can`, `red can`,
  `soup can`, `alphabet soup can`, `basket`, and `wire basket` at probability
  threshold 0.15. 31 of 36 combinations returned zero instances.
- Only `can` detected anything: 3 instances at frame 0 (512 px), 3 at frame 50
  (both resolutions), and 2–3 at frame 100. Every unique target prompt
  (`alphabet soup can`, `soup can`, `red can`) and every basket prompt returned
  zero. Upscaling 128→512 did not resolve identity.
- Therefore the unique-instance text-prompt assumption fails on this task: the
  target can is never selected, and the only matching prompt is ambiguous among
  three cans.

A full single-episode run (`episode_0000`, 148 frames × 2 slots = 296 prompts)
completed in about 3 minutes wall-clock on the A100. Result:
`observed = 0`, `missing = 296`, `ambiguous = 0`; the mask array is
`[148, 2, 128, 128]` bool with zero foreground pixels. This is an artifact of
text-only prompting on 128 px LIBERO frames, not a training result.

Local artifacts: `outputs/libero/masks_sam31_ep0/provenance.json` and
`episode_0000.npz`. Incremental Modal cost for access checks, checkpoint download,
prompt sweeps, and the full episode was about `$0.27`.

Consequences:

- The checkpoint/token blocker is resolved, but text-prompt masks are not usable.
- The all-missing artifact is not a measured dataset and must not be attached.
- This result motivated the separately labeled oracle baseline below; it does not
  validate SAM perception.

### Oracle simulator masks and first real numeric experiment

After the SAM failure, the user selected **oracle simulator masks** to unblock the
numeric mechanism test. This is explicitly privileged: each frame is reconstructed
from the recorded simulator state, so the policy is *not* operating from camera
observations alone. It is an upper-bound/oracle result, not a deployable
perception result.

Method:

- LIBERO commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`; BDDL
  `libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl`;
  body/geom prefixes `alphabet_soup_1` and `basket_1` in canonical slot order.
- Modal CPU image with Python 3.10, MuJoCo 2.3.7, robosuite 1.4.0, bddl 1.0.1,
  gym 0.25.2, and OSMesa. For each frame, set the recorded `states[t]`, call
  `sim.forward()`, render `agentview` RGB (128×128) and instance segmentation,
  and select geoms by exact simulator name and `mjtObj.mjOBJ_GEOM`. No future
  state is read.
- 50 episodes, 15,616 slot-frames: **15,605 observed, 11 missing, 0 ambiguous**
  (99.93% observed). Only `episode_0022` and `episode_0027` had missing
  slot-frames; all other episodes were fully observed.
- RGB alignment: mean absolute difference `2.02` and maximum `4.29` over all
  frames, against a hard abort threshold of 20. Orientation was verified as
  upright; overlays show the can on the table, in the gripper, and in the basket.
- Provenance is explicit: `causal: true`, `oracle: true`, simulator-state inputs,
  exact geom-name identity, source SHA256, BDDL SHA256, and alignment statistics.
  The provenance file is `outputs/libero/oracle/mask_provenance.json`.
- Modal paths: masks `/libero/masks_oracle`, attached dataset
  `/libero/object_oracle`, audit `/libero/object_oracle_audit.json`. Local mirror:
  `outputs/libero/oracle/`.

Dataset and protocol:

- Canonical dataset is 50 episodes at 20 Hz; split seed 17 gives 34 train / 8 val
  / 8 test episodes, fingerprint
  `99cf1697d6e0d1a1b02206a03ca901b2a94527f5b40940175fa62eceba337edb`.
  History/horizon windows: 4,992 train, 1,307 validation, 1,159 test.
- Joint run: history 4, horizon 4, hidden 128, batch 64, lr 1e-3, dynamics weight
  1, seed 7, 30 epochs, A100-80GB. Best validation epoch 5; validation flow MSE
  `0.907371`, validation dynamics MSE `0.003332`.
- Frozen joint test: flow MSE `0.774518`, dynamics MSE `0.002951`, moving
  dynamics MSE `0.003572`, sampled-action normalized MSE `0.825362`.
- Validation comparison uses three seeds (7, 19, 31). Seed-7 test metrics for all
  four frozen variants are reported once after validation choices; no test metric
  was used to select the model.

Validation means:

| Variant | Flow MSE | Dynamics MSE | Moving dynamics MSE |
|---|---:|---:|---:|
| `no_objects` (robot state only) | 0.931857 | untrained head | untrained head |
| `objects_action_only` | 0.905614 | untrained head | untrained head |
| `objects_joint` | 0.905791 | 0.003249 | 0.003931 |
| `objects_joint_no_dynamics_actions` | 0.905476 | 0.003803 | 0.004580 |

Seed-7 test:

| Variant | Flow MSE | Dynamics MSE | Moving dynamics MSE | Sampled action MSE |
|---|---:|---:|---:|---:|
| `no_objects` | 0.762589 | untrained head | untrained head | 0.769867 |
| `objects_action_only` | 0.774000 | untrained head | untrained head | 0.823888 |
| `objects_joint` | 0.774518 | 0.002951 | 0.003572 | 0.825362 |
| `objects_joint_no_dynamics_actions` | 0.774308 | 0.003221 | 0.003896 | 0.824190 |

Interpretation:

- The action-conditioned numerical loss strongly improves measurement-delta
  prediction over the `objects_joint_no_dynamics_actions` head: three-seed
  validation mean `0.003249` vs `0.003803` (14.6% lower) and seed-7 test
  `0.002951` vs `0.003221` (8.4% lower). Actions carry
  predictive information about future measurements on this task.
- It does **not** improve open-loop action prediction over the
  measurements-as-inputs-only variant: validation-mean flow `0.905791` vs
  `0.905614` and seed-7 test `0.774518` vs `0.774000`. The differences are
  negligible and slightly favor the no-dynamics-loss variant; sampled action MSE
  also slightly favors it
  (`0.823888` vs `0.825362`).
- The robot-state-only `no_objects` baseline is worse than object-input variants
  on validation but better on test (`0.762589`). With eight test episodes and one
  frozen seed, that reversal is not enough to claim object measurements help or
  hurt action prediction. It does prevent a positive action-prediction claim.
- These are teacher-forced dynamics metrics using recorded future actions, with
  oracle measurements and no Muse/image context. They are not closed-loop task
  success. The observed benefit is in measurement forecasting, not demonstrated
  control improvement.
- Incremental Modal cost for oracle masks, attachment, ablations, joint training,
  and evaluation was about `$0.35`. Total tracked compute for 2026-10-28–29 was
  about `$0.63`, well below the `$30` credit.

### Second task: milk in basket

The same protocol was repeated on the public LIBERO-Object task
`pick_up_the_milk_and_place_it_in_the_basket`: 50 demonstrations, downloaded
directly to Modal (727,315,202 bytes, SHA256
`e56d954928e613f9db58085b7677ccf9c4824f1ab982265f71405d5bb14cfb4a`).
Oracle segmentation identifies `milk_1` and `basket_1` from privileged recorded
simulator states. All 14,562 slot-frames were observed; replay RGB mean absolute
difference was 2.120, maximum 4.744. Mask identity follows simulator geom names;
no separate visual overlay review was performed for this task. The measured
dataset is `/libero/milk_oracle` on Modal; split seed 17 gives 34/8/8 episodes
and fingerprint `df42c87826c6bb141f024001e1f5d8cba59466360ff9bd63f48ca56b2a6f547d`.

Validation uses the same 30-epoch four-variant suite and seeds 7, 19, 31:

| Variant | Mean flow MSE | Mean sampled action MSE | Mean dynamics MSE |
|---|---:|---:|---:|
| `no_objects` | 0.902524 | 0.729185 | untrained head |
| `objects_action_only` | 0.904746 | 0.716118 | untrained head |
| `objects_joint` | 0.903345 | 0.712937 | 0.002208 |
| `objects_joint_no_dynamics_actions` | 0.903416 | 0.712986 | 0.002346 |

The frozen seed-7 test gave joint vs object-input-only flow MSE
`0.909617` vs `0.911965`, and sampled action MSE `0.771005` vs `0.771712`.
Joint dynamics MSE was `0.002289` vs `0.002497` without dynamics action
conditioning. The no-objects test flow/sample MSE were `0.932286`/`0.952056`.
Thus the auxiliary objective shows a small action-metric improvement on milk,
opposite to the very small difference on alphabet soup; two related tasks and
one test seed are insufficient to establish a robust policy improvement. Both
experiments use privileged oracle masks, teacher-forced dynamics, and no
closed-loop success measurement. Artifacts are in
`outputs/libero/oracle/milk_*` and `/runs/libero_milk_oracle_ablations`.

### Muse context and controller replay follow-up

- SAM 3.1 single-frame text prompts were also tested on the first milk episode:
  all 300 slot-frame prompts were missing, with zero observed masks. This does
  not test point/box initialization or online tracking. The all-missing artifact
  at `/libero/masks_sam31_milk_ep0` is not a usable camera-only dataset.
- A further exploratory probe on frames 0, 50, and 100 of the milk training
  episode used six text prompts (`milk`, `milk carton`, `carton`, `basket`,
  `wire basket`, `container`), 512-pixel upscaling and threshold 0.15. All 18
  prompt/frame combinations returned zero instances. Oracle masks were used only
  for scoring, not prompting; this is not a tracking or point-prompt test.
- Muse Glimmer 30B is now publicly accessible, and pinned revision
  `cec94fcfd6512f2cfe708138c32533b346ea4f69` was downloaded to the Modal volume.
  A real A100-80GB image-plus-instruction forward pass produced finite hidden
  states of shape `[1,84,6656]`; peak allocated GPU memory was 55.59 GiB.
  Two repeated warmed-up passes were identical in this smoke test and took
  approximately 0.136–0.137 seconds each. Transformers 5.21.0 warned that
  torchvision 0.25 fell back from LANCZOS to BICUBIC; preprocessing is therefore
  not an exact match to the original model.
- Cached frozen Muse features for all 50 milk episodes at stride 4 in
  `/libero/milk_oracle_muse`; carried past context forward between cache frames.
  The dataset retains oracle simulator masks; it is not camera-only. Its new
  seed-17 split fingerprint is `3a2cdb2cefb382a2e6d5889eb3dd9f34a412826b7491ab145cff3f7b28318bed`.
- The 30-epoch, three-seed validation suite with Muse context yielded mean flow
  MSE `1.183339` joint versus `1.176542` object-input-only; sampled-action MSE
  was `1.097735` versus `1.088850`. Frozen seed-7 test flow was `1.243515`
  versus `1.213720`; sampled-action MSE was `1.272733` versus `1.155158`.
  This setup did not improve action metrics; it also performed worse than the
  previously reported numeric-only milk run. The comparison uses the same
  episode assignments but changes the input context and dataset fingerprint.
  The validation logs show training error falling while validation error rises,
  consistent with overfitting; they do not establish its cause.
- LIBERO/robosuite open-loop replay of recorded milk `demo_0` actions from its
  initial simulator state reached the success condition at step 138 of 150,
  final reward 1.0. Mean absolute flattened-state error against the next recorded
  state was 0.00160 (maximum 0.03931). This validates one recorded-action replay,
  not closed-loop evaluation or learned-policy success. Evidence is in
  `outputs/libero/oracle/milk_recorded_action_replay.json`.

### Initial learned-policy closed-loop rollouts

`scripts/modal_libero_policy_rollout.py` runs the frozen numeric-only milk
checkpoints in LIBERO using the eight held-out demonstration initial states.
At each 20 Hz step it samples an action chunk from the current observation
history, clips the first action to `[-1,1]`, executes it, and refreshes the
observation. Initial history repeats the first robot observation. Object-input
variants use live **privileged simulator-oracle masks**; `no_objects` uses robot
state alone. Maximum 200 steps, with the same sampling-seed schedule for each
variant. Results, paired by training seed:

| Training seed | Joint dynamics | Object-input/action-only |
|---|---:|---:|
| 7 | 1/8 | 0/8 |
| 19 | 0/8 | 0/8 |
| 31 | 6/8 | 6/8 |

For seed 7, `no_objects` and `objects_joint_no_dynamics_actions` each achieved
0/8. Action clipping occurred on many steps; particularly for `no_objects`,
152–159 of 200 steps per episode were clipped. These are eight repeated
demonstration-derived initial states per seed on one task, not 24 independent
scenes, and one sampling schedule. The result is highly seed-sensitive and does
not establish a causal benefit from the auxiliary dynamics objective, a
camera-only policy, or physical-robot success. The per-episode JSON reports are
in `outputs/libero/oracle/milk_*_closed_loop.json`.

## Static integration checks (API pre-checks)

These reduce API risk only; they do not validate weights, masks, or features.

- SAM official repository cloned at commit
  `f1647e8d0c1d280991f4d1b7d974c2704537fe66`.
  `build_sam3_multiplex_video_predictor` accepts exactly the arguments the adapter
  passes (`checkpoint_path`, `use_fa3`, `compile`, `warm_up`,
  `async_loading_frames`), and `handle_request` supports `start_session`,
  `add_prompt`, and `close_session` with the request keys used. Prompt outputs
  include `out_binary_masks` as `[N, H_original, W_original]` boolean arrays, which
  matches `unique_mask`. At static-check time no checkpoint or CUDA device was
  available; the later A100 run is reported above and found the upstream
  `offload_state_to_cpu` mismatch plus zero usable masks.
- The public Muse config at `meta-models/Muse-Glimmer-30B` returned HTTP 200 with
  `model_type: muse_glimmer` and architecture
  `MuseGlimmerForConditionalGeneration`. The current `transformers` wheel
  (`5.21.0`) contains `models/muse_glimmer`, registers `muse_glimmer` in the
  auto mappings, includes both `AutoModelForMultimodalLM` and
  `AutoModelForImageTextToText`, and its output class exposes `hidden_states`.
  At that earlier static-check stage no weights or forward pass were used; the
  subsequent real forward and cache are documented above.

## Not executed, and why

| Gate | Blocker |
|---|---|
| Full/user dataset audit/import/replay | Two LIBERO-Object tasks are done; no user-specific dataset was supplied |
| SAM 3.1 camera-only extraction | Executed on Modal A100-80GB; zero usable unique masks on the first task |
| Real numeric experiment | Executed on Modal A100 with oracle simulator masks; see results above |
| Muse hidden-feature cache | Executed on Modal A100 for milk, with oracle masks and BICUBIC preprocessing fallback |
| Camera-only real-data policy | Not established: oracle masks use privileged simulator state |
| Simulator / closed-loop / real robot | Initial LIBERO closed-loop policy rollouts executed; no Isaac or physical robot access |

No claim from the synthetic run or the oracle run transfers to those gates.

## Artifacts

Local (gitignored): `data/libero/libero_object/` (744 MB raw HDF5),
`outputs/libero/` (source/canonical audits, replay GIF, splits, SAM smoke
artifacts), `outputs/libero/oracle/` (mask provenance, oracle audit/manifest,
three-seed validation comparison, seed-7 test metrics, best-validation and
training logs), `outputs/my_smoke/`, `outputs/modal_run/`,
`outputs/modal_run30/`, `runs/ablations/`, `outputs/prediction_check.json`,
`outputs/report_check.html`, `outputs/fixture_*`, a shallow SAM clone at
`outputs/third_party/sam3` (commit `f1647e8`), and a LIBERO clone at
`outputs/third_party/LIBERO`.

Modal Volume `object-dynamics-vla-data` currently holds `/smoke`,
`/runs/modal_smoke`, `/runs/modal_smoke30`, the real-data work at `/libero/`
(source HDF5, canonical `object_raw`, splits, audit JSON, replay), the SAM
checkpoint at `/checkpoints/sam3.1/`, the all-missing
`/libero/masks_sam31_ep0`, oracle masks at `/libero/masks_oracle`, the attached
dataset at `/libero/object_oracle`, the suite at `/runs/libero_oracle_ablations`,
and evaluation JSON at `/eval/`. All synthetic runs are safe to delete; the
LIBERO source and canonical dataset are reproducible from the URL and import
command above.

## Next actions

The subsequent validation-only diagnostics and revised ordering are recorded in
`docs/DIAGNOSTICS_REPORT.md`; they supersede the older ordering below. Stabilize
the oracle-policy comparison and validate camera tracking separately before
combining them. The initial test demonstrations have already been evaluated, so
the next frozen comparison needs untouched initializations.

1. Treat the oracle result as a mechanism check, not a deployable policy. Any
   camera-only claim requires a task with uniquely identifiable objects or a
   verified causal streaming/tracking adapter; text-only unique prompts failed
   on this task.
2. Repeat the numeric experiment on more LIBERO tasks and repeated splits/seeds.
   The eight-episode test reversal between `no_objects` and object-input variants
   means the action-prediction question is still open; do not select on test.
3. Keep privileged simulator state out of any policy presented as camera-only.
   Oracle masks may supervise/evaluate, but the oracle provenance must travel
   with the dataset and results.
4. If you have a different existing dataset on the 5090, send its path/format;
   the same audit/import/replay gate applies.
5. Treat Muse as a separate, costed gate: 60 GB BF16 weights require a larger
   GPU; use Modal only after local profiling or a validated quantized hidden
   state path.
