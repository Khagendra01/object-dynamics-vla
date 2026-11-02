# Frozen-motor bridge repair: matched-noise development evaluation

## Scope and training

This repair retains the existing Muse/SAM-to-native-SmolVLA connector architecture.
It starts from the 250-update alignment checkpoint and keeps the entire adapted
SmolVLA policy frozen, including the expert and state/action/time projections.
No numerical-dynamics objective is used. The frozen native policy provides the
teacher for all comparisons. Optimizer and RNG snapshots are saved for resume;
exact interruption/resumption parity has not been independently tested for this
new training driver.

Training used **1,250 updates × batch 4 = 5,000 presentations**, shuffled without
replacement within an epoch: all **4,887 distinct milk training windows** were
presented. For each batch, teacher and student were queried at exactly the same
noisy action and time for both (1) a Beta-sampled demonstration interpolant
and (2) a detached state produced after 1–9 steps of the student's own denoising
trajectory. The two velocity losses were equally weighted; the gripper channel
had weight 3 versus 1 for each other channel. AdamW LR was `3e-5`, seed 7.
No frozen-policy parameters received gradients. Peak reported allocated T4
memory was 2.26 GiB; training took 3,531 seconds.

The checkpoint was selected on the same 16 reused validation windows used in
earlier development. Weighted teacher-velocity errors on those windows:

| Updates | interpolant | student state | equal-weight mean |
|---:|---:|---:|---:|
| 0 | .22067 | .41296 | .31682 |
| 250 | .03658 | .03751 | .03705 |
| 500 | .03006 | .02726 | .02866 |
| 750 | .02730 | .02552 | .02641 |
| 1000 | .02998 | .03077 | .03038 |
| 1250 | .02079 | .01950 | .02014 |

The selected step-1250 bridge is
`/data/runs/muse_smolvla_bridge_repair1250v1/weights_best.pt`, SHA-256
`bd707fbef138be46fe057bc21c93b7db3b073db07a12911e26131cc0cff59a5c`.
It must be loaded with the original frozen adapted SmolVLA checkpoint, not the
subsequently updated expert.

## All validation frames: teacher fidelity

On all **1,211 frames of eight reused validation episodes**, fixed per-frame
NumPy noise was shared among the native teacher, the original aligned bridge
with frozen expert, and the repaired bridge with that same frozen expert.
Each uses its own 10-step sampling trajectory; velocity errors are additionally
evaluated on the same native teacher denoising states. The 16 selection windows
are included in these 1,211 frames; this is development, not an independent
generalization estimate. No held-out test initializations were used.

| Measurement | Original aligned bridge | Repaired bridge |
|---|---:|---:|
| Teacher gripper-sign agreement | 81.50% | 96.45% |
| Recall when teacher sign negative | 90.17% | 96.94% |
| Recall when teacher sign positive | 76.23% | 96.15% |
| First-action MAE, x | .08675 | .02516 |
| First-action MAE, y | .33622 | .05394 |
| First-action MAE, z | .19278 | .06447 |
| First-action MAE, rx | .01398 | .00648 |
| First-action MAE, ry | .02171 | .00929 |
| First-action MAE, rz | .01891 | .00715 |
| First-action MAE, gripper | .47368 | .09526 |

Actions are in native LIBERO command units before clipping; the metric is
agreement with the teacher, not correctness against an expert demonstration.
For the gripper channel, the teacher-path first-denoising-step velocity MSE fell
from .45991 to .03521; at the final denoising step it fell from .24139 to
.02453. Per-channel, per-denoising-step errors, first-action RMSE, and per-episode
metrics are in
`/data/eval/muse_smolvla_bridge_repair1250v1_fullval_v2.json`. Repaired
teacher-relative gripper agreement ranged from 90.82% to 99.29% by episode.
`fullval_v1` is preserved; v2 corrected the repaired bridge to use the exact
LeRobot state preprocessing used in training and live rollout. The aggregate
sign agreement happened to remain identical.

## Failed-trajectory inspection

We reran the earlier matched-noise diagnostic on its six reconstructed frames
using the repaired bridge as an additional configuration, with the native
adapted expert frozen. At frame 70, the repaired gripper command was -.398
versus native -.321; the earlier updated bridge had produced +.377. At frame
199, however, the repaired bridge predicted **+.849** while native predicted
**-.716**. At frame 160, repaired y/z were .744/.213 versus native .329/-.046.
These frames are observations from the earlier failed combined-policy trajectory,
not demonstrations that the teacher action is correct there. The 6-frame
results and full denoising arrays are under
`/data/eval/muse_smolvla_bridge_diagnostic_v1/results_v3.json` and
`outputs_*_v3.npz`; see `BRIDGE_PAIRED_NOISE_DIAGNOSTIC.md` for reconstruction
limitations. The `results_v2` prior diagnostic remains preserved.

## Paired live rollout

Both runs started from the same reused `episode_0014` initial state and JPEG,
used the same pinned adapted motor checkpoint and the same saved per-step
`[1,50,32]` NumPy noise tensors. The native policy used native SmolVLM; the
other used live frozen Muse/SAM and the repaired frozen-expert bridge. A
separate audit verified the actual SHA-256 of the noise tensor on all **135
overlapping steps**. Subsequent observations differ after the actions diverge.

| Policy | Result | Steps | Gripper sign changes | Steps with any clipping |
|---|---|---:|---:|---:|
| Native adapted SmolVLA | simulator success | 135 | 5 | 47 |
| Repaired Muse/SAM bridge + frozen expert | no success at cap | 200 | 22 | 75 |

This is **one paired reused development state**, not a success rate. The native
run ended when its success predicate became true; the repaired run reached the
200-step cap. Its final RGB shows the milk carton outside the basket. Initial
Muse token/cache parity passed exactly; incremental SAM matched observed-prefix
replay at steps 0, 9, 29, 69, 99, 149 and 199. No oracle masks were used.
Measured peak allocated Muse/SAM GPU memory was 65.97 GiB, with the frozen
expert on a separate T4. The nominal 20-fps videos are not a real-time speed
claim. The native first action reproduced the earlier successful reference.

Videos: `/home/kgen/Desktop/smolvla_pair_native.mp4` and
`/home/kgen/Desktop/muse_sam_bridge_repaired_paired.mp4`. The raw bank and
audit are `/data/eval/muse_smolvla_noise_matched7v1.npy` (SHA-256
`e3b34e03c05977f8e23baa3ab9e9e062dc49acc8514c46359b93ddba18d7a1e0`)
and `/data/eval/muse_smolvla_repair1250v1_paired_audit_v1.json`; full traces and
videos are under `/data/eval/muse_smolvla_live_episode_0014_pairnativev1/`
and `/data/eval/muse_smolvla_live_episode_0014_pairrepairedv1/`.

An initial actor smoke check found that using NumPy-normalized robot state in
the live bridge changed the frame-0 action by up to .00062 versus the offline
LeRobot-normalized reference. The repaired actor now uses the same LeRobot
preprocessing as the teacher and training; a rerun reproduced both offline
first actions within `1e-5` and matched the noise hash. This correction does
not rewrite the older updated-bridge rollout. Muse's recorded effective backend
settings and strict first-frame cache parity guard remain in place; the cause
of the earlier intermittent feature mismatch is still unproven.

## Interpretation and reproduction

More bridge training improved teacher-relative fidelity on the reused
demonstration observations, but **did not preserve the teacher's successful
closed-loop behavior in this paired run**. The remaining gripper reversals and
out-of-distribution discrepancies make further error analysis useful before
claiming that motor behavior is preserved. Teacher agreement is not proof the
teacher is correct on a failed-trajectory state. SAM/dynamics comparisons remain
separate future work; this repair did not include a dynamics loss.

```bash
~/modal-venv/bin/modal run scripts/modal_bridge_repair.py --steps 1250 --batch-size 4 --suffix NEWNAME
~/modal-venv/bin/modal run scripts/modal_bridge_repair.py --stage evaluate --suffix NEWNAME --batch-size 4
~/modal-venv/bin/modal run scripts/modal_muse_smolvla_live.py --stage noise --noise-bank NEWBANK
~/modal-venv/bin/modal run scripts/modal_muse_smolvla_live.py --max-steps 200 --suffix NEWNATIVE --episode episode_0014 --mode native --noise-bank NEWBANK
~/modal-venv/bin/modal run scripts/modal_muse_smolvla_live.py --max-steps 200 --suffix NEWREPAIR --episode episode_0014 --mode repaired --noise-bank NEWBANK
~/modal-venv/bin/modal run scripts/modal_paired_rollout_audit.py
~/modal-venv/bin/modal run scripts/modal_vla_tests.py
```

Live `repaired` mode and the audit currently name the recorded checkpoint/run
paths; change them to evaluate a different new run. Existing outputs are never
overwritten. The unit suite passed 39 tests, including a test that teacher and
student see identical states/times and only the student branch receives
gradients. The paired smoke and audit are separate execution checks.
