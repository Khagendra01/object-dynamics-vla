# Native SmolVLA scratch-expert reference

## Scope and setup

This is an in-sample comparison on the 150-frame `episode_0000` training
demonstration. It does not measure generalization, isolate quantization, or
establish a SAM benefit. The Muse variant uses its custom query-normalized
pooled prefix and causal cached SAM history; the native variant uses the
official SmolVLA conditioning path and does not use SAM.

For the native reference, the pinned official SmolVLM2-500M-Video-Instruct
backbone is loaded from its pretrained vision-language checkpoint and frozen.
The official SmolVLA expert and action/state projections are initialized from
scratch; no robot-policy weights or teacher are loaded. It receives the same
150 RGB frames (all hashes verified), robot states, 50-action windows, saved
train-only normalization statistics, and 3,000 sequential batch-one updates
as the query-normalized Muse fit. Both use AdamW lr 1e-4, betas (0.9, 0.95),
default weight decay 0.01, gradient clipping 10, CUDA BF16 autocast, the same
flow convention, and ten-step sampling. Evaluation uses the same eight frames
and fixed CUDA noise seeds. The backbone and input interface differ by design;
the native reference uses Transformers 5.5.4, while Muse uses 5.21.0. Both use
Torch 2.10.0 and the same pinned LeRobot commit. Equal random seeds do not
establish identical initial expert tensors across the different constructors.

## Fixed observations and rollout

| Measurement | Native scratch | Query-normalized Muse/SAM |
|---|---:|---:|
| Initial normalized first-action MAE, 8 seeds | 1.015 | 1.002 |
| Final normalized first-action MAE, 8 seeds | 0.615 | 0.768 |
| Final flow loss, last 100 training updates | 0.277 | 0.423 |
| Gripper sign agreement, 8 frames × 4 seeds | 23/32 | 24/32 |
| Gripper raw-action MAE, same 32 samples | 0.859 | 0.713 |
| Training-initialization rollout, 200 actions | no success | no success |
| Actions with any clipping | 37 | 46 |
| Executed gripper sign changes | 40 | 74 |

Training losses are stochastic and the native and adapter paths have different
conditioning and frozen visual backbones. The eight-frame metrics and one
rollout are small development diagnostics, not statistical performance
estimates. The native model fits the chosen first-action metric somewhat better,
but both leave substantial error and both fail the same in-sample rollout;
this does not identify the cause as Muse, quantization, or the adapter.

The two rollouts used the same simulator state and initial rendered JPEG hashes,
the same action clipping and first-action replanning, and the same `7 + step`
CUDA sampling seeds. The saved SHA-256 hash of every generated `[1,50,32]`
noise tensor matched the query-normalized Muse rollout for all 200 actions.
They used different observations after their trajectories diverged. In native
rollout, 37 gripper values and one other channel value were clipped, occurring
across 37 actions; sign changes are not necessarily grasp events.

## Artifacts

Modal volume:

* `/eval/native_scratch_episode0000_v1/` — fit report, trainable weights, stats.
* `/eval/native_scratch_episode0000_diagnostic_v1/report.json` — 32 sampled predictions.
* `/eval/native_scratch_episode0000_rollout_v1/` — report, trace, frames and video.
* `/eval/quantized_muse_full_prenorm_v1/` and
  `/eval/muse_smolvla_live_episode_0000_prenormmatchedv1/` — comparison fit and rollout.

Local reports and video are in `outputs/`. The original Muse checkpoints were
not overwritten. `scripts/modal_native_scratch_reference.py` and
`scripts/modal_native_scratch_rollout.py` execute the native training,
diagnostic and rollout; use fresh output names to avoid overwriting artifacts.
