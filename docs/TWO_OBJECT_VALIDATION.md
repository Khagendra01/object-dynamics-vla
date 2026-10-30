# Two-object initialized LIBERO validation

On Modal A100-80GB, `scripts/modal_sam31_two_object_validation.py` tracked milk
and basket simultaneously in three preselected milk-task validation episodes.
The episodes were the first three in the existing validation split, not a random
sample. The images show nearly identical initial layouts. All three validation
initializations have now been used for development, not held out for a final
evaluation. The previously evaluated test initializations remain used as well.

The input was 512×512 simulator-rendered agentview RGB at 20 Hz, JPEG quality
98. After visually inspecting each first RGB frame, the agent chose the same
points in each: milk `(128,277)` and basket `(410,210)` in 512-pixel xy
coordinates. These are explicit agent-selected points, not autonomous language
grounding or human clicks. Two IDs were initialized in one SAM session. The
experimental adapter uses the missing-cache merge, fixed configured pointer
scale, past-only pointers, and single-GPU detector prefetch guard described in
`CAMERA_PILOT_REPORT.md`; the full sessions preload video and are not live.

Simulator instance masks were opened only after the tracking and prefix checks,
then used to score 128×128 nearest-neighbor-downsampled predictions. They are
privileged simulator references, not perception inputs. Mean, fifth-percentile,
and minimum IoU include all frames. For each output we also compared its IoU to
the other object's reference; this is a limited cross-object confusion check,
not a complete identity-switch metric against every scene object.

| Validation episode / source | Frames | Milk mean / p05 / min IoU | Basket mean / p05 / min IoU | Nonempty per object |
|---|---:|---:|---:|---:|
| 0014 / demo_21 | 137 | 0.9639 / 0.9204 / 0.8609 | 0.9788 / 0.9668 / 0.9626 | 137 / 137 |
| 0020 / demo_27 | 135 | 0.9685 / 0.9355 / 0.9053 | 0.9599 / 0.9578 / 0.9564 | 135 / 135 |
| 0022 / demo_29 | 140 | 0.9265 / 0.7628 / 0.7451 | 0.9754 / 0.9555 / 0.9430 | 140 / 140 |

Both IDs appeared in all outputs; in no frame did either output have higher IoU
with the other target than with its own reference. This does not exclude partial
contamination, confusion with distractors, or hidden ID problems outside these
episodes. Four regularly spaced overlay frames per episode were visually
inspected for both objects; the smaller minimum IoU in episode 0022 is not
necessarily at one of those sampled frames.

Follow-up inspection of the saved masks found the first worst milk frame in
episode 0022 at frame 28 (IoU 0.745098; frame 29 is tied). Neighboring IoUs at
frames 26–30 are 0.789474, 0.763158, 0.745098, 0.745098 and 0.745098. In the
RGB overlays at frames 27–29, the prediction remains on the carton but has a
persistent narrow disagreement inside its visible silhouette; it is not an
obvious switch to the basket. These observations do not establish the cause of
the disagreement, and use the 128-pixel scoring raster.

Mean absolute geometry errors use image-fraction units and only valid pairs;
all frames were valid and all consecutive pairs were valid here. The five
components are centroid u/v, bbox width/height, and visible area. Frame-delta
error is prediction minus reference change and mixes segmentation jitter and
tracking error; it is not physical velocity.

| Episode / object | Per-frame geometry MAE | Frame-delta error MAE |
|---|---|---|
| 0014 milk | 0.000959, 0.000971, 0.000969, 0.000627, 0.000203 | 0.000519, 0.000784, 0.001608, 0.001149, 0.000136 |
| 0014 basket | 0.001638, 0.000122, 0.007813, 0.000228, 0.000421 | 0.000070, 0.000101, 0.000000, 0.000345, 0.000050 |
| 0020 milk | 0.000859, 0.000600, 0.000810, 0.000637, 0.000092 | 0.000448, 0.000619, 0.001516, 0.000700, 0.000106 |
| 0020 basket | 0.000755, 0.002998, 0.000000, 0.007292, 0.001133 | 0.000058, 0.000091, 0.000000, 0.000466, 0.000042 |
| 0022 milk | 0.000749, 0.002165, 0.001060, 0.001786, 0.000243 | 0.000460, 0.000828, 0.001574, 0.001574, 0.000133 |
| 0022 basket | 0.000845, 0.001293, 0.000000, 0.007533, 0.000426 | 0.000075, 0.000167, 0.000000, 0.000112, 0.000077 |

The basket exhibits roughly one 128-grid-pixel systematic bbox width or height
error in some episodes despite high IoU. Reference rasterization and downsampling
affect this comparison. A 17-frame prefix replay matched full-session masks
pixel-for-pixel for both objects at frames 0, 1, 5, 14, 15 and 16 in each
episode (36 nonempty comparisons). This is only one prefix length and was not a
repeat of the prior detailed memory instrumentation; it is not an online
causality guarantee.

Raw predictions, JSON, and inspection montages are on the Modal volume at
`/data/eval/sam31_two_object_validation_v1/`. Source data and checkpoints are
identified in `CAMERA_PILOT_REPORT.md` and `AGENT_HANDOFF.md`. To reproduce on a
fresh volume, render the three demos with `scripts/modal_libero_native_render.py`
using `--full`, then run `modal run scripts/modal_sam31_two_object_validation.py
--episode episode_0014` (also 0020 and 0022). The script refuses to overwrite an
existing JSON report. No training dataset or camera-based policy was produced;
further variety, initialization robustness, and online frame handling remain
unvalidated. Both eventual policy variants must use the same SAM measurements.
