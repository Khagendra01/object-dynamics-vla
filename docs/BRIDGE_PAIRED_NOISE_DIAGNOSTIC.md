# Paired-noise diagnostic: native SmolVLA versus Muse/SAM bridge

## Scope

Offline diagnostic only, on **six frames (0, 30, 70, 120, 160, 199)** from the
previously used `episode_0014` combined-policy failed 200-step trajectory. It
does not evaluate independent task success, does not establish causality, and
does not train or test the numerical-dynamics auxiliary objective. The prior
native and combined closed-loop rollouts used NumPy and CUDA-Torch noise,
respectively, so equal seeds did not make their outcomes paired.

The original combined rollout saved frames and actions, not robot states or
SAM measurements. We replayed its exact recorded executed actions in LIBERO,
reconstructed robot state and replayed frozen SAM on the observed image prefix.
Each selected replay RGB JPEG matched its recorded original byte-for-byte;
independent SAM prefix replay matched at 0, 70 and 199. RGB equality does not
independently prove simulator hidden-state identity. Muse features were
re-extracted at the six frames; frame 0 was bit-identical to the old feature
cache. Later-frame features were not saved during the original rollout, so
their equality to the original live features cannot be checked. Actual Muse
cache, timestamp, source trace, model and checkpoint provenance are recorded.

One NumPy-generated `[1,50,32]` noise array per frame was saved and passed
unchanged to all four configurations. These are the native policy's seed rule,
not the CUDA-Torch noise originally used by the combined rollout. This is an
offline paired input comparison, not a retroactive paired closed-loop trial.

The four configurations:

1. **Native original:** SmolVLM + native 1,000-update LIBERO-adapted expert.
2. **Bridge frozen:** aligned Muse/SAM connector + that same unmodified expert.
3. **Native updated:** SmolVLM + expert/state/action/time projections from the
   later joint-adaptation checkpoint. This is not an expert-only intervention.
4. **Bridge updated:** joint-adaptation connector and expert; the combined
   checkpoint used in the failed rollout.

All configurations see the same reconstructed current RGB, state, instruction
and same noise within a frame. The bridge additionally sees causal SAM history.
We evaluate the 10-step denoising trajectory, including all 50 predicted action
positions, and compare velocity predictions at the *same native reference x_t*
at each time. We also save each configuration's own evolving denoising outputs,
the initial noise and sampled first action. Native original's complete
normalized 50×7 action output matched the official LeRobot policy exactly at
all six frames (max absolute error 0). At frame 0 its sampled first action also
exactly matched the recorded successful native rollout's first action. This
does not reproduce the rest of that successful rollout or prove transfer.

## Findings

| Frame | Native original first action `(y,z,gripper)` | Bridge frozen | Native updated | Bridge updated |
|---:|---|---|---|---|
| 0 | `(-.410, .005, -.696)` | `(-.316, -.001, -.705)` | `(-.391, .000, -.706)` | `(-.664, .046, -1.009)` |
| 30 | `(-.451, -.340, -1.213)` | `(-.920, -.392, -1.687)` | `(-.428, -.329, -1.217)` | `(-.596, -.518, -1.487)` |
| 70 | `(.052, .180, -.321)` | `(-.328, .778, -.026)` | `(.073, .121, -.362)` | `(-.146, .579, .377)` |
| 120 | `(.263, .186, .945)` | `(.315, .237, .916)` | `(.260, .203, 1.015)` | `(.417, .291, 1.024)` |
| 160 | `(.329, -.046, .994)` | `(.426, -.064, 1.023)` | `(.342, -.041, .999)` | `(.685, .195, .994)` |
| 199 | `(.011, -.024, -.716)` | `(.029, .070, -.435)` | `(.041, -.057, -.707)` | `(.419, -.002, .522)` |

Values are **pre-clipping** LIBERO OSC_POSE commands; x and rotation values,
full action vectors and all 10 velocity errors are in the machine-readable
results. Clipping can change the differences actually executed. For example,
both gripper predictions at step 30 are clipped to -1 for some configurations.
At frame 70 the native original/frozen-bridge gripper predictions are negative
while the updated bridge predicts positive; at 199 the native original predicts
negative and the updated bridge positive. These are matched-noise predictions
on the *combined policy's* observations, not demonstrated correct actions.

On the same native x_t at frame 0, the first denoising velocity MSE across 50×7
is 0.00361 for native updated, 0.22076 for bridge frozen, and 0.16550 for
bridge updated. At frame 120 the corresponding first-step errors are 0.00388,
0.63521 and 0.78480. The updated native model is much closer to the original
native than either bridged model on these sampled inputs. Joint adaptation did
not uniformly improve bridge fidelity: e.g. at frame 120 the updated bridge's
reference-path velocity MSE peaks at 1.45776, versus 0.67451 for the frozen
bridge. At other frames the updated bridge is closer, so no uniform ordering is
claimed. Expert adaptation may still matter in closed loop; these six frames do
not establish that it is harmless or prove which component caused the failure.

The earliest sampled divergence is already at frame 0 (approach). Inspection
of frame 70 shows the gripper close to the milk carton; by frames 120 and 160
the carton remains outside the basket and the gripper is away from it. The
samples do not establish exactly when contact/grasp failed, nor a successful
lift or placement phase; further phase labelling requires inspecting the full
video and/or contact data. Large bridge/native differences at later frames are
on states caused by the failed combined policy and are not comparable to the
states of the separate successful native rollout.

## Precision and runtime settings

The instrumented preparation run recorded settings immediately before each
Muse forward, after tracker operations. Before model construction matmul and
cuDNN TF32 were false and float32 matmul precision was `highest`; after model
construction both TF32 flags were **true**, precision was `high`, and CUDA
autocast was enabled. Immediately before each Muse forward, inside the explicit
autocast-disabled context, both TF32 flags were **true**, precision was `high`,
autocast was **false**, and attention implementation was `sdpa`. The before
and after snapshots do not isolate which constructor changed the settings.
The newly extracted first-frame features matched the historical cache exactly
despite those settings. This is evidence of a constructor-side change and a
candidate for controlled investigation, **not proof of the cause** of the
earlier intermittent mismatch. Settings were observed, not silently changed.
The live runner now writes `backend.json` at first forward.

## Artifacts and reproducibility

Artifacts are on Modal under `/data/eval/muse_smolvla_bridge_diagnostic_v1/`:
`manifest.json`, `backend.json`, six `frame_*.npz` observation/noise files,
`results_v2.json`, and `outputs_*_v2.npz` full denoising arrays. The initial
`results.json` incorrectly labels each entire observation-file hash as
`noise_sha256`; **use `results_v2.json`**, which separately records the actual
noise-byte SHA-256 and observation-file SHA-256. Both results are preserved.

```bash
~/modal-venv/bin/modal run scripts/modal_bridge_diagnostic.py --stage prepare
~/modal-venv/bin/modal run scripts/modal_bridge_diagnostic.py --stage evaluate
~/modal-venv/bin/modal run scripts/modal_bridge_diagnostic.py --stage inspect
~/modal-venv/bin/modal run scripts/modal_vla_tests.py
```

The first two stages refuse to overwrite their existing output paths; choose
fresh paths/suffixes in the script when reproducing. The current 38-test CPU
suite passed. Suggested next step is a controlled frozen-expert alignment
experiment with more diverse training examples/noise levels, matched
validation, and an explicit fidelity gate before any expert updates, followed
by paired SAM/dynamics ablations. No extra training is implied by this report.
