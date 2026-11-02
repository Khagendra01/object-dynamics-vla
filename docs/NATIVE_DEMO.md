# Native-policy application

The complete native SmolVLA policy, including SmolVLM, owns motor control.
Muse routes the request once before execution. SAM supplies live overlays.
No learned bridge, numerical-dynamics loss or new training is involved.

Available skill: **pick up the milk and place it in the basket**. Episodes 0014
and 0020 are existing reused LIBERO development initializations with inspected
RGB initialization points. This is a bounded simulator application.

## Run on the authenticated machine

From the repository root, use the environment authenticated to the existing
`object-dynamics-vla-data` Modal volume and containing the prior checkpoints:

```bash
python -m pip install -e '.[modal]'
python -m scripts.native_demo --request "Put the milk carton in the basket" --output outputs/milk-demo
```

Open the downloaded `outputs/milk-demo/index.html`. It includes overlay and
original videos, final image, task status, planner selection, report and trace.
The default is a 200-action cap on episode 0014. Output folders and remote run
names must be fresh; existing artifacts are not overwritten.

Other development initialization:

```bash
python -m scripts.native_demo --episode episode_0020 --output outputs/milk-demo-20
```

Check existing remote artifacts without starting GPU jobs:

```bash
python -m scripts.native_demo --check-only
```

Explicitly bypass Muse to select the registered skill:

```bash
python -m scripts.native_demo --skill milk_in_basket --output outputs/explicit-skill
```

Add `--no-tracking` to explicitly disable SAM. These bypasses are recorded;
they are not presented as three-component runs. `--max-steps 1` performs only
one action and is not a full task attempt.

## Runtime

1. A CPU preflight checks existing paths. No models are downloaded or trained.
2. Muse runs once on A100-80GB and selects `milk_in_basket` or `unsupported`.
   Generation is constrained to registered JSON choices with fixed descriptive
   reasons. A rejected request executes no actions. Model output cannot select
   a checkpoint path, code or arbitrary motor instruction.
3. SmolVLA runs on T4, strictly loading the released checkpoint and its existing
   1,000-update LIBERO adaptation. The official pre/postprocessor and saved
   training normalization are retained. It receives current JPEG, raw robot
   state, canonical registered instruction and NumPy `7+step` action noise.
   It predicts a chunk, executes its first action clipped to `[-1,1]`, and replans.
4. A CPU simulator executes actions. A separate A100 SAM worker tracks milk and
   basket from the existing RGB points. Its output reaches only the overlay
   and report. Tracking failures are recorded; they cannot alter motor inputs
   or cause a hidden control-policy fallback.
5. Completion, the step cap, runtime limit or error ends execution. The final
   image is rendered after the final action. Reports contain raw robot state,
   sampled/executed actions, clipping, image/noise/checkpoint hashes and status.

Completion comes from the simulator task predicate. No simulator object masks
are policy inputs. SAM is passive tracking, not a grasp/slip detector. The
available skill and fixed development points do not provide general language
grounding or autonomous initialization. Video plays at simulated 20 Hz; the
simulator pauses during inference and this is not real-time.

The default launches paid functions using the existing allocation: Muse once,
native SmolVLA on T4 and SAM on A100. Check remaining credit first; preflight
does not retrieve the invoice. The control loop has a 900-second budget and
200-action cap. Function timeouts/startup are additional bounds, not a dollar
cap. Explicit `--skill milk_in_basket --no-tracking` is the cheaper native-only
check. Existing feature caches and bridge weights are not required.

## Local check without cloud access

```bash
python -m scripts.native_demo --dry-run --output outputs/demo-ui-check
```

This is an explicitly marked **synthetic local preview**. It checks recording,
overlays, MP4 encoding, trace/report and viewer creation. It runs no Muse, SAM,
robot policy, simulator physics or robot and does not claim task success.
It requires the core dependencies and `ffmpeg`.

## Verification and next execution

Implemented and checked in a CPU workspace without Modal credentials. The
new GPU application has **not** been executed here. Earlier native-policy
successes remain prior evidence, not results for this entry point.

Local checks: 44 tests passed; source compilation passed; Modal 1.6.3 imported
the app and displayed its CLI; a synthetic run produced videos and a viewer.
Remote preflight stopped at `Token missing. Could not authenticate client.`
No GPU job was launched.

Tests cover bounded routing, constrained tokens, unchanged action inputs when
SAM fails, original noise, clipping, invalid-action rejection and HTML escaping.
They do not certify model generation, cloud builds or new task completion.
The next action on the connected machine is the full command above.
