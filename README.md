# Object Dynamics VLA — dataset-first experiment kit

Start here. This package turns our architecture discussion into a small, executable
research experiment that you or your coding agent can continue on your PC.

**Question:** does explicitly learning changes in SAM-derived object measurements
improve a policy compared with supplying those measurements only as inputs?

Your intended setup is an RTX 5090 with 32 GB VRAM, existing robot datasets,
Isaac Sim, a physical robot, and an optional $30 Modal credit budget.
Existing datasets come first. No new robot collection or cloud spend is required
to inspect and test this package.

## 1. See what was actually run here

This repository includes code and documentation. Third-party datasets,
pretrained checkpoints, credentials, and local experiment artifacts under
`examples/`, `outputs/`, and `runs/` are excluded. Findings and reproducibility
details are summarized in the reports below; paths under those directories or a
Modal volume refer to the original execution environment and are not bundled.

Open these files before installing anything:

- `docs/VALIDATION.md`: what passed, what was not run, and how to interpret results.
- `docs/DIAGNOSTICS_REPORT.md`: action clipping, shared gradients, validation-only
  weight sweep, passive prediction monitoring, and separate SAM probes.
- `docs/TRACKER_TRACE_REPORT.md`: DAVIS propagation-stage trace, availability
  timing, and the diagnostic point-path workaround.
- `docs/CAMERA_PILOT_REPORT.md`: bounded memory audit and one initialized
  LIBERO RGB tracking episode, including numerical measurement jitter.
- `docs/TWO_OBJECT_VALIDATION.md`: simultaneous milk/basket tracking and
  measurement checks across three initialized LIBERO validation episodes.
- `docs/MILK_SAM_DEVELOPMENT.md`: dataset-level shared SAM measurements,
  one-step dynamics and paired offline policy comparisons.
- `docs/SAM_CLOSED_LOOP_PILOT.md`: prefix-only replay benchmark, first-action
  diagnostics and a short paired simulator pilot.
- `docs/MUSE_VLA_METHOD.md` and `docs/MUSE_SAM_RESULTS.md`: experimental
  transformer architecture, frozen Muse cache, and measured implementation checks.
- `docs/MULTITASK_SAM_CLOSED_LOOP.md`: three-seed matched adaptations, paired
  development rollouts, the Muse chat-date confound and runtime parity checks.
- `docs/EXECUTION_REPORT.md`: later macOS + Modal T4/A100 execution, including a
  real LIBERO numeric experiment with oracle masks, costs, and remaining blockers.

**The synthetic smoke experiment uses generated moving squares, not SAM/Muse
outputs or real robot demonstrations. It verifies software and learning
mechanics, not VLA quality, grasp success, sim-to-real transfer, or the research
hypothesis.**

**Real-data follow-up (oracle perception):** on 50 LIBERO-Object episodes, the
joint action-conditioned measurement loss improved measurement forecasting over
the `objects_joint_no_dynamics_actions` head (three-seed validation mean
`0.003249` vs `0.003803`; seed-7 test `0.002951` vs `0.003221`), but did
**not** improve open-loop action prediction over measurements-as-inputs-only (validation mean `0.905791` vs
`0.905614`; seed-7 test flow `0.774518` vs `0.774000`; sampled action MSE
`0.825362` vs `0.823888`). A robot-state-only baseline was worse on validation but
better on the eight-episode test, so the action-prediction direction is
inconclusive. These masks use privileged simulator state, not camera-only
perception; these particular numbers are offline metrics. See
`docs/EXECUTION_REPORT.md`.

A second, similar milk-in-basket task showed a small improvement in action
metrics with the joint objective, opposite to the first task. Both results use
oracle masks; two tasks and one held-out test seed do not establish a robust
policy improvement. See the execution report for the matched comparison.

Initial closed-loop LIBERO rollouts on eight milk test initial states also show
strong seed sensitivity: joint vs object-input-only succeeded 1/8 vs 0/8, 0/8 vs
0/8, and 6/8 vs 6/8 across seeds 7, 19, and 31. These use simulator-oracle masks
and one sampling schedule; they do not establish a robust control benefit.

## 2. Install and reproduce the smoke run

Use Python 3.12 for an environment that will later host SAM. Keep Isaac Sim in its
own environment. The core also supports Python 3.10+.

Linux / WSL2:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CPU smoke only; skip this if compatible CUDA PyTorch is already installed.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[datasets]'
python -m odvla doctor
python -m unittest discover -s tests -v
python -m odvla smoke --out outputs/my_smoke --epochs 30 --device cpu
```

PowerShell: create the venv with `py -3.12 -m venv .venv`, activate with
`.venv\Scripts\Activate.ps1`, then use the same `python -m ...` commands.

For RTX 5090 training, install a current **CUDA-enabled** PyTorch build supporting
your card from <https://pytorch.org/get-started/locally/> instead of the CPU command.
Run `python -m odvla doctor`: it reports CUDA, compute capability, memory, compiled
architectures, and launches a tiny CUDA operation. GPU detection alone is not enough.
Do not install the CPU build over a working CUDA environment.

The optional model scripts need additional dependencies; see
[`docs/INTEGRATIONS.md`](docs/INTEGRATIONS.md). No model weights are bundled.

## 3. Start with your existing dataset

The follow-up execution downloaded and imported two official LIBERO-Object
tasks (50 demos each) on Modal; see
[`docs/EXECUTION_REPORT.md`](docs/EXECUTION_REPORT.md). Raw imports have missing
measurements until masks are attached; separately labeled simulator-oracle
datasets were created for both tasks. First inspect a local demonstration file:

```bash
python -m odvla audit /path/to/task_demo.hdf5 --out outputs/source_audit.json
```

The importer supports a conservative LIBERO/robomimic-style HDF5 layout:
`/data/demo_*/actions` and `/data/demo_*/obs/*`. It does not support every robotics
dataset automatically. If yours uses another format, implement a converter to the
documented canonical schema; do not silently rename fields and assume units.

Import **one task file** first. Replace every placeholder below with the correct
instruction and uniquely identifying object descriptions for that task.
`20` Hz is an example: verify the actual source control rate.

```bash
python -m odvla import-libero \
  --source /path/to/task_demo.hdf5 \
  --out data/task_raw \
  --objects 'red cup,white bowl' \
  --instruction 'Put the red cup inside the white bowl.' \
  --fps 20
python -m odvla audit data/task_raw --out outputs/import_audit.json
python -m odvla replay --dataset data/task_raw --episode episode_0000 --out outputs/raw_replay.gif
```

Default robot observation keys are `joint_states,ee_states,gripper_states` and the
camera key is `agentview_rgb`. Use `--robot-keys` / `--image-key` after inspecting
the source. Images are not flipped unless you explicitly pass `--flip-vertical`.
Check the replay and source action convention before training. The importer copies
commands without pretending they are meters or joint targets.

Imported numeric measurements are initially **missing**. That is intentional:
training refuses a dataset with no observed numerical targets.

## 4. Extract measurements, then optionally cache Muse features

### SAM 3.1

Install the official SAM repository and obtain authorized checkpoint access using
its instructions. See the integration guide for the exact adapter limitations.

```bash
python -m scripts.extract_sam31 \
  --dataset data/task_raw --out outputs/sam_one_episode \
  --checkpoint /path/to/sam3.1_multiplex.pt --episode episode_0000
```

Inspect this extraction before processing every episode:

```bash
python -m scripts.extract_sam31 \
  --dataset data/task_raw --out outputs/sam_all \
  --checkpoint /path/to/sam3.1_multiplex.pt
python -m odvla attach-masks \
  --dataset data/task_raw --masks outputs/sam_all --out data/task_measured
python -m odvla replay --dataset data/task_measured --episode episode_0000 --out outputs/measured_replay.gif
```

This correctness-first adapter uses **independent single-frame SAM 3.1 sessions**.
It has no future-frame leakage, but is slower than streaming and does not use
SAM's temporal tracking. It accepts a unique instance per prompt and marks
ambiguous detections missing. Do not describe this adapter as validated video
tracking. It has run on a Modal A100-80GB; on the first LIBERO-Object task all
unique prompts returned zero masks, so text-only prompting was not usable. See
[`docs/EXECUTION_REPORT.md`](docs/EXECUTION_REPORT.md).

Alternatively, attach externally generated, audited causal masks using the same
command. See `docs/DATA_SCHEMA.md` for mask layout and provenance.

### Muse Glimmer

First test numerical dynamics with `data/task_measured`. Then cache frozen
image+instruction hidden features for the controlled VLA comparison:

```bash
python -m scripts.cache_muse \
  --dataset data/task_measured --out data/task_with_muse \
  --revision YOUR_PINNED_HUGGING_FACE_MODEL_COMMIT --stride 4
```

The script is a BF16 hidden-feature integration, not a GGUF text server. It may
need a larger-memory GPU. Quantized text inference fitting 32 GB does not prove
that this extraction path fits. It fails if the backend does not expose expected
hidden states. A later Modal A100 smoke test and milk-task feature cache succeeded;
that experiment retained oracle masks and did not improve action metrics. See
`docs/EXECUTION_REPORT.md`.

For all comparisons below, use the same cached dataset and freeze it before
splitting. Regenerate splits when perception or context changes.

## 5. Train and compare

For the first numeric-only experiment, set `DATASET` to `data/task_measured`.
For the Muse experiment, use `data/task_with_muse`.
The commands below use the latter; substitute your actual directory.

```bash
python -m odvla split --dataset data/task_with_muse --out outputs/splits.json
python -m odvla train \
  --dataset data/task_with_muse --splits outputs/splits.json \
  --out runs/joint_seed7 --device cuda --epochs 30 --dynamics-weight 1
python -m odvla evaluate --dataset data/task_with_muse --run runs/joint_seed7 --split val --device cuda
python -m odvla report --run runs/joint_seed7 --out outputs/joint_report.html
```

The suite runs four matched variants and uses validation by default:

```bash
python -m scripts.run_ablations \
  --dataset data/task_with_muse --out runs/ablations \
  --device cuda --epochs 30 --seeds 7,19,31
```

| Variant | Object measurements | Numerical loss | Dynamics conditioned on actions |
|---|---|---|---|
| `no_objects` | Removed | Off | Head untrained |
| `objects_action_only` | Present | Off | Head untrained |
| `objects_joint` | Present | On | Yes |
| `objects_joint_no_dynamics_actions` | Present | On | No |

Without cached Muse context, `no_objects` is a **robot-state-only** baseline,
not an image/VLM baseline. The first numeric-only experiment tests the dynamics
mechanism; the Muse cache enables the fuller visual/language comparison.

Select settings on validation. After freezing choices, explicitly evaluate test:

```bash
python -m odvla evaluate --dataset data/task_with_muse --run runs/joint_seed7 --split test --device cuda
```

Measurements include normalized future-delta error, per-horizon error, native-unit
per-feature MAE, moving/stationary subsets, persistence/velocity baselines,
episode-level errors, and sampled open-loop action error.
**None of these is robot task success.** Closed-loop evaluation is a later gate.

## 6. Inspect a sampled action without executing it

```bash
python -m odvla predict --dataset data/task_with_muse --run runs/joint_seed7 \
  --episode episode_0000 --frame 12 --out outputs/prediction.json --device cuda
```

This uses only observations through the selected frame. It samples an action
chunk and forecasts numerical changes. It does not send commands to your robot.
See `odvla/inference.py` for the callable prefix-only API.

## 7. What your coding agent should do next

The LIBERO imports, text-only SAM probes, oracle experiments, Muse cache, and
initial simulator rollouts are already recorded. Inspect those results before
repeating them. The next gates are oracle-policy diagnostics and independently
validated camera tracking; see `docs/DIAGNOSTICS_REPORT.md`.

Give the agent this instruction:

> Read README.md and AGENT_HANDOFF.md. Run the tests and inspect the supplied
> synthetic and real-oracle results. Do not repeat the completed LIBERO import,
> failed text-only SAM extraction, or oracle pass unless testing new perception.
> Audit my existing dataset before collecting anything new. Preserve causal inputs
> and train-only normalization, then extend the matched experiments and report
> actual results and failures. Do not claim real robot success from offline losses.

[`AGENT_HANDOFF.md`](AGENT_HANDOFF.md) contains ordered tasks and acceptance checks.
The implementation status is explicit in [`docs/VALIDATION.md`](docs/VALIDATION.md).
The full research design and the smaller implemented model are distinguished in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Boundaries

- Core: executable offline experiments, not a pretrained generalist VLA.
- No robot-specific driver, Isaac rollout adapter, contact estimator, uncertainty
  calibration, active sensing policy, or numerical-goal planner is included.
- Existing HDF5 import is tested on a generated layout fixture and two downloaded
  LIBERO-Object tasks; other layouts still need converters.
- The real LIBERO masks are simulator-oracle, not camera-only SAM; do not present
  the resulting policy as deployable perception.
- No model/data licenses are transferred by this package's MIT code license.
- No cloud jobs launch automatically. Modal support is optional and was executed
  manually in the follow-up report.

Official references and version caveats: [`docs/SOURCES.md`](docs/SOURCES.md).
