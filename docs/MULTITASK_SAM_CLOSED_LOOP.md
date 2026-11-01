# Two-task motor checkpoint: matched SAM development comparison

## Scope and important cache confound

These are **development** results on two already-used milk validation
initializations, not fresh tests. The motor checkpoint is the seed-7 1,000-step
milk/alphabet-soup run in `MULTITASK_MOTOR_DEVELOPMENT.md`. All M0/M1/M2
adaptations start from that same checkpoint; seeds 7, 19 and 31 are adaptation
seeds, not independent motor-pretraining seeds. Each arm has 500 updates,
effective batch 64, paired batches/noise/time draws and the same fixed validation
flow checkpoint rule. Validation is also used for reporting below.

**The existing two-task motor run has a task-correlated prompt confound.** Muse's
chat template silently injects the current date. Decoding the first and last
cached frame of each of 42 milk and 42 soup episodes found 84/84 milk prefixes
dated `2026-10-30` and 84/84 soup prefixes dated `2026-10-31`. The model could
use this as a task cue, so these results cannot establish language grounding or
unconfounded multitask transfer. All historical data and results are preserved.
The rollout fixes the date to the milk cache date and verifies the first live
image, input IDs, positions, attention mask and BF16 features against the cache.
The first pilot used the runtime date `2026-11-01`, differed at two input IDs and
in hidden features, and is only a diagnostic. New `FrozenMuseTokens.encode`
calls require an explicit date; new cache scripts record it. The two-task
runner now refuses caches with missing or unequal date metadata. Corrected
multi-task pretraining requires uniformly dated regenerated caches, not merely
altering the old manifests.

## Offline action prediction and forecasting

Selected validation flow MSE / sampled first-action normalized MSE, averaged
equally over seven channels:

| Adaptation seed | M0 | M1 | M2 |
|---|---:|---:|---:|
| 7 | .762111 / .857468 | .761709 / .857114 | .760926 / .856847 |
| 19 | .768160 / .741668 | .767995 / .741202 | .767317 / .741934 |
| 31 | .767885 / .743054 | .769224 / .742484 | .767149 / .743560 |

The selected-validation flow differences are small. Sampled M2 is slightly
better than M1 at seed 7 but worse at seeds 19 and 31. These measurements do
not establish a SAM or dynamics-policy benefit. First-action normalized MSE by
channel `(x,y,z,rx,ry,rz,gripper)`:

| Seed | M0 | M1 | M2 |
|---|---|---|---|
| 7 | .7713, .1328, .2117, 1.5112, 1.3860, 1.7436, .2458 | .7711, .1328, .2102, 1.5110, 1.3875, 1.7462, .2411 | .7687, .1329, .2113, 1.5047, 1.3955, 1.7426, .2423 |
| 19 | .6333, .1220, .1536, 1.1792, 1.2387, 1.6675, .1973 | .6334, .1216, .1536, 1.1785, 1.2378, 1.6666, .1969 | .6337, .1212, .1527, 1.1799, 1.2409, 1.6694, .1958 |
| 31 | .6278, .1172, .1460, 1.1697, 1.2864, 1.6704, .1838 | .6276, .1172, .1465, 1.1702, 1.2838, 1.6686, .1835 | .6271, .1178, .1459, 1.1720, 1.2873, 1.6719, .1829 |

All variants' gripper sign agreement was approximately 93.6%, 94.5% and 95.0%
for seeds 7, 19 and 31 respectively. Always-minus/always-plus baselines are
42.2%/57.8%. Per-class recalls and native-unit channel MAE are in
`/data/eval/multitask_milk_offline_3seed_v1.json` on the Modal volume.

M2's one-step normalized numerical MSE by metric `(centroid_u, centroid_v,
bbox_width, bbox_height, visible_area)` was:

- seed 7: `[.02320, .01724, .02677, .02973, .01591]`
- seed 19: `[.01134, .01183, .01215, .01301, .01082]`
- seed 31: `[.01094, .01102, .01116, .01255, .01099]`
- persistence: `[.000395, .001637, .001278, .004373, .000237]`

Persistence wins every aggregate metric for every seed and every metric on the
declared moving/stationary subsets. Moving means any valid normalized metric
changes by more than 0.1 for an object; this is not physical ground-truth
motion. Seed-7 initial shared gradient norms were 0.9944 for action and 5.7512
for unweighted dynamics; the declared dynamics weight is 0.1. The forecast uses
demonstrated `a[t]` only. M0/M1 forecast heads are untrained and are not
forecasting baselines.

## Closed-loop development control

The motor-only pilot executed 10 steps, no success, on reused `episode_0014`.
After fixing the date, its first-frame frozen Muse tokens matched the cache
bit-for-bit. M1/M2 used fresh RGB observations, fixed agent-selected initial
points, frozen SAM masks, first of 50 sampled actions, clipping to `[-1,1]`, and
replanning; the monitor was disabled. No simulator masks were used as inputs.

| Reused state | Adaptation seed | M1 success / steps / clipped steps | M2 success / steps / clipped steps |
|---|---:|---|---|
| 0014 / demo_21 | 7 | no / 200 / 15 | no / 200 / 33 |
| 0014 / demo_21 | 19 | no / 200 / 11 | no / 200 / 11 |
| 0014 / demo_21 | 31 | no / 200 / 46 | no / 200 / 61 |
| 0020 / demo_27 | 7 | no / 200 / 12 | no / 200 / 13 |

Clipping was exclusively in the gripper channel. All eight runs had matching
paired initial RGB hashes, no reported missing/duplicate/empty tracking IDs,
and no task success. Nonempty IDs alone do not prove correct object tracking.
Each was capped at 200 steps and used sampling seed `7 + step`. These are four
pairs, two similar states, and one task—not a success-rate benchmark. No fresh
test initializations were evaluated. A sampled inspection of RGB frame 160 in
the two seed-7 runs on 0014 and M1 on 0020 showed the carton outside the basket;
this is not exhaustive visual scoring.

The incremental tracker is experimental and depends on pinned SAM internals.
It reserves 200 image slots using only copies of the first observed RGB; each
slot is replaced only when that observation arrives. Available sequence length
is advanced one frame at a time, past-only/prefetch guards remain active, and
the monitor is off. Incremental outputs matched saved prefix replay for all 70
frames of both seed-7 trajectories. During every 200-step rollout, independent
prefix replay matched exactly for both objects at steps 0, 9, 29, 69, 99,
149 and 199. This is bounded evidence on these trajectories, not a general
online-causality guarantee. Initial live Muse/cache parity passed on every run.

Peak allocated memory was 66.14 GiB on A100-80GB with Muse, SAM and expert
co-resident, so this does not fit an unverified 32 GiB RTX 5090 BF16 runtime.
For seed 7 on 0014, median per-step Muse / incremental SAM / policy times were
approximately .217 / .162 / .042 seconds for M1 and .220 / .163 / .043 for M2,
excluding rendering, simulation and occasional prefix cross-checks. It is not
real-time. The 70-step prefix-replay pilot took about 12.5 seconds for its
70-frame final prefix. Complete videos, raw/clipped actions, masks, per-step
tracking IDs, timing, checkpoint hashes and reports are under:

- `/data/eval/multitask_muse_sam_paired_incremental200/` (seed 7, 0014)
- `/data/eval/multitask_muse_sam_paired_seed19cap200/` (seed 19, 0014)
- `/data/eval/multitask_muse_sam_paired_seed31cap200/` (seed 31, 0014)
- `/data/eval/multitask_muse_sam_paired_seed7ep20cap200/` (seed 7, 0020)

The corresponding eight videos were also copied to the current machine's
Desktop. The first seed uses names `muse_m1_development.mp4` and
`muse_m2_development.mp4`; the other six use seed/episode suffixes.

## Reproduction and remaining work

```bash
~/modal-venv/bin/modal run scripts/modal_multitask_matched.py --seed 7
~/modal-venv/bin/modal run scripts/modal_multitask_matched.py --seed 19
~/modal-venv/bin/modal run scripts/modal_multitask_matched.py --seed 31
~/modal-venv/bin/modal run scripts/modal_multitask_offline.py
~/modal-venv/bin/modal run scripts/modal_sam_incremental_probe.py --variant M1
~/modal-venv/bin/modal run scripts/modal_sam_incremental_probe.py --variant M2
~/modal-venv/bin/modal run scripts/modal_muse_sam_paired_rollout.py --max-steps 200 --max-seconds 1800 --suffix NEWNAME --tracking-mode incremental --train-seed 7 --episode episode_0014
~/modal-venv/bin/modal run scripts/modal_vla_tests.py
```

Runners refuse to overwrite existing artifacts. The frozen historical caches
and motor checkpoint are necessary for these exact development reproductions.
The new pooled training guard deliberately refuses those date-unannotated,
date-confounded caches for a **new** motor-training run. The reproducible tests
passed 36/36 on Modal CPU. Billing summary for the current November cycle showed
$6.73 credits applied at the last check; reporting can lag, and prior-cycle costs and storage are
separate, and the actual remaining balance was not exposed by this summary.
No additional credit was purchased.

Next research gates are date-consistent cache extraction and motor retraining,
stronger motor-learning/rollout diagnostics on more tasks, and a frozen protocol
before truly unused initializations. Isaac and the physical robot have not been
mapped or evaluated.
