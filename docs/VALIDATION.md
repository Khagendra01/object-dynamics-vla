# Validation performed in the packaging environment

Checked 2026-10-28–29. These are actual executions, not illustrative log values.

## Environment

CPU-only Linux, Python 3.12.14. Package versions: {"numpy": "2.3.5", "Pillow": "12.3.0", "h5py": "3.16.0", "torch": "2.15.0+cpu"}.
At packaging time, no CUDA GPU, SAM checkpoint, Muse checkpoint, user's robot,
or real dataset was available to these runs. HDF5 support and CPU PyTorch were installed for testing.

## Follow-up execution 2026-10-28 (Intel macOS + Modal T4)

`docs/EXECUTION_REPORT.md` records a later run on an Intel Mac (Python 3.12,
torch 2.2.2, CPU only) and on Modal. Summary:

- The initial follow-up run passed 18 tests; after adding the SAM compatibility regression test, 19 pass. `doctor` reported no CUDA, as expected.
- The 30-epoch CPU smoke and one-seed four-variant ablation reproduced the
  packaged numbers exactly (test dynamics MSE `0.00032929884`; ablation
  comparison unchanged).
- `audit`, `replay`, `predict`, `evaluate`, `report`, and a generated
  LIBERO-style HDF5 import ran successfully.
- One official LIBERO-Object task was downloaded (50 demos, 780,145,352 bytes),
  imported on Modal into 50 canonical episodes, audited, and replayed. All
  measurements are intentionally missing (`observed_fraction = 0.0`); a Modal T4
  training attempt correctly refused because perception causality was not
  declared. This is the expected gate, not a training result.
- Modal training ran end-to-end on a T4 with GPU torch `2.15.0+cu132`. The
  30-epoch validation dynamics MSE was `0.00018170` (GPU, best epoch 22) versus
  `0.00018425` (CPU, best epoch 29); differences are consistent with
  nondeterministic GPU kernels. Incremental Modal cost was under `$0.01`.
- SAM 3.1 ran on a Modal A100-80GB with checkpoint SHA256
  `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`.
  Model build took 19.7 s and peak allocated memory was 5.53 GiB for a one-frame
  prompt. The full first episode (148 frames × 2 prompts) produced
  `observed = 0`, `missing = 296`, `ambiguous = 0`; only the generic prompt
  `can` detected anything, and it returned 2–3 instances. Text-only unique
  prompts failed on this task, so SAM did not create usable measurements.
- Oracle simulator masks then unblocked the numeric mechanism test: 50 episodes,
  15,605/15,616 slot-frames observed (99.93%), RGB alignment mean 2.02/max 4.29,
  and explicit `oracle: true` provenance. The real numeric experiment ran on a
  Modal A100-80GB; results are below.
- L4 capacity was unavailable for over 30 minutes; the attempt was stopped.
  Muse, Isaac, and robot gates remain unexecuted.

## Automated tests

`python -m unittest discover -s tests -v`: **24 passed, zero skipped** after the
follow-up monitor, cache-merge, and temporal-probe tests. Historical local
`examples/` logs are not bundled in the public repository.

Tests cover group-disjoint splits, deterministic splitting, dataset-change
rejection, train-only normalization, future-input invariance, current action /
next-state alignment, missing-observation masking, path bounds, pixel and metric
geometry, HDF5-layout import, external mask attachment, ambiguous-mask rejection,
model-output contracts, shared gradients from both objectives, checkpoint reload,
evaluation, and prefix-only action sampling without future action labels.

## Actual synthetic run

30 generated episodes, 64 frames each, two objects, five image-space metrics,
20 Hz fixture clock. History=4, horizon=4, hidden=128, 30 epochs, seed=7.
Train/validation/test groups were separate. The checkpoint was chosen on
validation using the configured combined objective. The synthetic test split was
evaluated to verify evaluation plumbing, not to select settings for real data.

| Held-out synthetic test predictor | Normalized delta MSE |
|---|---:|
| dynamics | 0.000329299 |
| persistence | 0.004872120 |
| constant_velocity | 0.005976738 |

Best validation epoch: 29.

The fixture supplies randomized action blocks and action-driven square motion.
It deliberately makes action-conditioned dynamics learnable. The recorded actions
are not successful goal-directed robot demonstrations; the fixture is not an
imitation benchmark. Good dynamics performance on it does not establish physical
understanding or useful motor behavior.

## Actual synthetic ablations (validation only, one seed)

All four variants used the same dataset/splits, 30 epochs, hidden size and seed.
Without Muse context, the no-objects baseline has robot state only.

| Variant | Sampled action normalized MSE | Dynamics normalized MSE |
|---|---:|---:|
| no_objects | 2.113673 | untrained head |
| objects_action_only | 1.751501 | untrained head |
| objects_joint | 1.752309 | 0.000184255 |
| objects_joint_no_dynamics_actions | 1.751661 | 0.003977065 |

**The numerical objective improved numerical prediction here, but did not improve
sampled action error over the object-input/action-only variant.** That is the
observed result. One synthetic seed cannot answer whether this mechanism helps
real VLA control. The random action process also limits policy predictability.
The action-removed control tests predictive information from recorded actions;
it is not a causal-identification proof.

## Real LIBERO numeric experiment (oracle perception, 2026-10-29)

This is the first real-data numeric experiment. It is **not** camera-only
perception: masks were rendered from recorded privileged simulator states with
exact geom names `alphabet_soup_1` and `basket_1`. The 50 LIBERO-Object
demonstrations were replayed in MuJoCo; all 15,616 slot-frames were checked for
RGB alignment. The mask artifact contains 15,605 observed slot-frames, 11
missing, and 0 ambiguous; observed fraction is 99.93%, and only
`episode_0022`/`episode_0027` contain missing slot-frames.

Split seed 17: 34 train / 8 validation / 8 test episodes; fingerprint
`99cf1697d6e0d1a1b02206a03ca901b2a94527f5b40940175fa62eceba337edb`.
History/horizon windows: 4,992 train, 1,307 validation, 1,159 test. Joint run: history 4, horizon 4, hidden 128, batch 64, lr 1e-3, 30 epochs,
dynamics weight 1, seed 7. Best validation epoch 5.

Validation means over seeds 7, 19, 31:

| Variant | Flow MSE | Dynamics MSE | Moving dynamics MSE |
|---|---:|---:|---:|
| `no_objects` | 0.931857 | untrained head | untrained head |
| `objects_action_only` | 0.905614 | untrained head | untrained head |
| `objects_joint` | 0.905791 | 0.003249 | 0.003931 |
| `objects_joint_no_dynamics_actions` | 0.905476 | 0.003803 | 0.004580 |

Frozen seed-7 test:

| Variant | Flow MSE | Dynamics MSE | Moving dynamics MSE | Sampled action MSE |
|---|---:|---:|---:|---:|
| `no_objects` | 0.762589 | untrained head | untrained head | 0.769867 |
| `objects_action_only` | 0.774000 | untrained head | untrained head | 0.823888 |
| `objects_joint` | 0.774518 | 0.002951 | 0.003572 | 0.825362 |
| `objects_joint_no_dynamics_actions` | 0.774308 | 0.003221 | 0.003896 | 0.824190 |

The dynamic objective improves measurement-delta prediction over the
action-removed head (validation 14.6% lower; test 8.4% lower), so recorded
actions carry predictive information. It does **not** improve open-loop action
prediction over the measurements-as-inputs-only variant: validation-mean
`0.905791` vs `0.905614`, seed-7 test flow `0.774518` vs `0.774000`, sampled
action MSE `0.825362` vs `0.823888`. The robot-state-only baseline was
worse on validation but better on test, so the action-prediction direction is
inconclusive. Metrics are teacher-forced with recorded future actions and do not
measure closed-loop task success.

## Local inspection outputs

Later LIBERO closed-loop rollouts of frozen milk policies used eight held-out
demonstration initial states and a 200-step limit. Joint vs object-input-only
successes were 1/8 vs 0/8 for training seed 7, 0/8 vs 0/8 for seed 19, and
6/8 vs 6/8 for seed 31. These use simulator-oracle masks and one shared action
sampling schedule, with frequent action clipping; the same eight initial states
are reused across seeds. They do not establish a robust policy improvement or
camera-only control. See `docs/EXECUTION_REPORT.md` for protocol and artifacts.

- The original execution environment retained generated episodes and a manifest,
  not just aggregate scores. Local raw experiment artifacts are gitignored.
- Actual JSONL epoch logs, train-only stats, split/fingerprint and model config.
- Small trained reference-head checkpoints (not pretrained Muse/SAM weights).
- Held-out metrics, native-unit sample traces, prefix-only prediction JSON.
- Portable HTML report and animated measurement replay.
- Validation-only ablation logs/comparison and CPU environment information.

## Still not executed

- Camera-only SAM identity/tracking that works on this LIBERO task.
- Muse backbone adaptation (frozen hidden-state extraction and an oracle-mask
  milk-task cache were subsequently run on Modal; see `docs/EXECUTION_REPORT.md`).
- CUDA kernels on an RTX 5090, GPU performance, or VRAM profiling.
- Isaac closed-loop evaluation, task success or recovery.
- Real-robot commands or sim-to-real transfer.
- A camera-only real-data policy: the numeric experiment uses oracle masks.

The model-specific scripts have output-contract tests and source/API checks;
those are not end-to-end checkpoint validation. The coding-agent handoff calls
out their real execution as the next integration gates.
