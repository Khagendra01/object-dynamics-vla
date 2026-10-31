# Provisional milk-only Muse motor development

This is a single-task, single-training-seed development experiment, not the
requested multitask motor pretraining or a final evaluation. The 1,211
validation windows were used both to select checkpoints and to report scores.
The old test episodes remain used data. Frozen Muse and SAM cached inputs and
their provenance are described in `MUSE_VLA_METHOD.md` and `MUSE_SAM_RESULTS.md`.

## Runs and budget

The owner reported $15 remaining Modal credit. Modal's published rates checked
at execution were $0.000164/s T4, $0.0000131/core/s CPU and
$0.00000222/GiB/s memory. A T4 with 4 requested CPU cores and 64 GiB is about
$1.30/hour at those rates, excluding storage, extra usage and other charges.
Actual billing and remaining credit were not independently retrieved. The
following two training functions spent 1,004 and 2,199 seconds respectively
inside the functions, roughly $1.16 at requested-resource rates, excluding
startup, other jobs, storage and potential higher actual usage. The evaluation
used 16 GiB, 4 CPU cores and a T4. These are estimates, not invoices.

Reproduce in a workspace containing the existing volume and with unused output
paths; the commands intentionally refuse to overwrite existing runs:

```bash
~/modal-venv/bin/modal run scripts/modal_motor_development.py
~/modal-venv/bin/modal run scripts/modal_motor_comparison.py
~/modal-venv/bin/modal run scripts/modal_motor_offline_eval.py
```

The motor checkpoint trained 500 updates, effective batch 64, seed 7, on 4,887
training windows, with robot and frozen Muse inputs, no SAM inputs and no
dynamics loss. Validation flow MSE was 0.981211 at step 250 and 0.791921 at
step 500; it decreased, but this does not demonstrate closed-loop skill. The
selected checkpoint is `/data/runs/muse_motor_milk_seed7_500_v1/weights_best.pt`.
M0/M1/M2 each started from that same checkpoint and ran 500 more updates with
matched seed, batches, action noise, timestep draws and validation selection.
Their outputs are under `/data/runs/muse_motor_milk_comparison_seed7_v1/`.

| Variant | step 250 validation flow MSE | selected step 500 validation flow MSE |
| --- | ---: | ---: |
| M0 | 0.776046 | 0.745410 |
| M1 | 0.776073 | 0.745364 |
| M2 | 0.777706 | 0.743971 |

The small differences in these selected validation losses cannot establish an
improvement in control. M2 uses a fixed 0.1 dynamics weight. At its first
minibatch the unweighted shared action and dynamics gradient norms were 1.2351
and 6.1630, respectively; weighted dynamics norm is about 0.6163. These values
are initial diagnostics, not evidence of steady-state gradient balance.

## Offline action and dynamics results

`/data/eval/muse_motor_milk_offline_seed7_v1.json` contains paired seed-921
10-step samples, channelwise errors, masks, example targets and forecasts.
M0/M1/M2 first-action normalized MSE by channel:

| channel | M0 | M1 | M2 |
| --- | ---: | ---: | ---: |
| x | .63553 | .63422 | .63477 |
| y | .12582 | .12587 | .12568 |
| z | .20280 | .20210 | .20126 |
| rx | .83744 | .83612 | .83590 |
| ry | 1.23555 | 1.23745 | 1.23824 |
| rz | 1.70205 | 1.70417 | 1.70125 |
| gripper | .21767 | .21688 | .21617 |

Gripper sign agreement is 0.93724 for each variant. M2 one-step normalized
dynamics MSE by metric (centroid u/v, bbox width/height, visible area) is
`[.01180, .01353, .01191, .01589, .01158]`, compared with persistence
`[.000395, .001637, .001278, .004373, .000237]`: persistence is better on every
metric overall. Each metric has 2,406 valid object transitions. M0 and M1
dynamics heads were not trained; their forecast errors are not a meaningful
forecasting baseline. No simulator success result follows from these offline
metrics.

One real validation example, episode 0014 frame 0: demonstrated action is
`[.05893, -.04018, 0, 0, .01821, 0, -1]`; M2's sampled first action before
clipping is `[.19599, -.13561, .07395, -.01148, .04161, -.00862, -1.16977]`.
The cached milk change target normalized by training standard deviations is
`[-.000246, .01439, 0, 0, .01986]`; the M2 forecast is
`[.08493, .05255, .07649, -.01169, -.17749]`. Both object slots were valid.
These are recorded samples, not executed actions.

### Class imbalance and forecast diagnostic

The separately executed `scripts/modal_motor_diagnostics.py` produced the same
validation confusion matrix for all three variants: target minus predicted
minus/plus = 458/53, target plus predicted minus/plus = 23/677. Always-minus
accuracy is 42.20%, always-plus is 57.80%; minus and plus recall are 89.63%
and 96.71%. No target or prediction was exactly zero. These are offline,
reused-validation sampled commands; they are not independent control outcomes.

M2's train-set forecasting also loses to persistence on every metric. For a
declared moving subset, an object is moving when **any** valid normalized
metric has absolute one-step change greater than 0.1; this is an image-space
measurement heuristic, not physical ground-truth motion. Per-object counts are
1,873 moving / 7,833 stationary for train and 432 / 1,974 for validation.
For moving train objects M2 MSE by metric is
`[.00810, .00901, .01409, .04300, .00786]` versus persistence
`[.000759, .00782, .00673, .04141, .000841]`. For moving validation objects it
is `[.01347, .01556, .01628, .03441, .01015]` versus
`[.000714, .00746, .00337, .02436, .000678]`. Persistence is also better on
all stationary metrics. Full per-metric results are stored on the Modal volume
at `/data/eval/muse_motor_milk_diagnostics_seed7_v1.json`. This diagnostic
uses demonstrated `a[t]` only and does not evaluate a newly trained head.

The official SmolVLA reference, wider LIBERO compatibility and token caches,
multitask motor checkpoint, three-seed matched comparisons, and new Muse/SAM
closed-loop rollout remain unfinished. Cross-dataset transfer requires the
explicit audits described in `MUSE_VLA_METHOD.md`.
