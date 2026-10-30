# Initialized SAM point-tracking pilot

This is an **experimental, initialized camera-RGB tracking** diagnostic. It is
not autonomous language grounding, a trained camera-based policy, physical robot
performance, or proof that every upstream SAM code path is causal.

## Adapter and bounded audit

`odvla.experimental_sam31.experimental_point_adapter` packages the missing-cache
merge correction and fixes pointer temporal encoding using the tracker's
configured `max_obj_ptrs_in_encoder` (16 here; denominator 15). It enables
`only_obj_ptrs_in_the_past_for_eval` and preserves the real session length and
upstream pointer-selection bounds. It changes methods on a predictor instance,
so it must not be shared concurrently across sessions. The adapter is labeled
experimental and `validated_online: false`.

The upstream detector also attempted to build a *next-frame* backbone buffer.
For this single-GPU point-path audit, an explicit optional guard skipped those
future prefetch requests and rejected any current chunk spanning a future frame.
No fake images or modified `num_frames` were used. The guard supports forward
tracking only; it is not a general multi-GPU implementation.

`scripts/modal_sam31_point_pilot.py` compared nonempty output masks from
independent prefix-only sessions against a full-session replay at frames
0, 1, 5, 14, 15, 16, 20, and the last frame. The last-frame comparison uses
the same length as the full run and is therefore a repeatability check, not a
different-length check. All other selected prefix/full comparisons matched
pixel-for-pixel on each of three DAVIS clips and the LIBERO episode below.

During each full run the probe recorded selected object pointers and spatial
memory source identities at every noninitial frame, tracker image-feature cache
requests/keys, detector buffer keys and requested chunks, and output availability.
On these runs no recorded selected memory or requested cached image feature was
from a future frame; future detector prefetch requests were blocked. The scoped
instrumentation observed every noninitial frame's temporal-memory hook, and
every output was emitted with zero **frame-index** delay in the point path.
These observations apply to the instrumented paths and tested sessions only.
Full-video sessions still preload the image sequence, while prefix sessions use
only the actual prefix. No appendable live-camera implementation was tested.

With the checkpoint, DAVIS data, and native LIBERO render already on the Modal
volume, repeat individual audits under a fresh output name:

```bash
modal run scripts/modal_sam31_point_pilot.py --dataset davis --clip-index 0 --run-name repeat1
modal run scripts/modal_sam31_point_pilot.py --dataset libero --run-name repeat1
```

DAVIS clip indices 0, 1, and 2 select `bike-packing`, `blackswan`, and
`bmx-trees`. The pilot refuses to overwrite an existing report with the same
dataset, clip, and run name. Render the LIBERO RGB beforehand with
`modal run scripts/modal_libero_native_render.py --demo demo_21 --full`; the
renderer also refuses to overwrite an existing render.

| Clip | Frames | Nonempty masks | Mean IoU | Nonempty sampled prefix checks |
|---|---:|---:|---:|---:|
| DAVIS bike-packing | 69 | 69 | 0.6710 | 8/8 matched |
| DAVIS blackswan | 50 | 50 | 0.9578 | 8/8 matched |
| DAVIS bmx-trees | 80 | 80 | 0.4463 | 8/8 matched |
| LIBERO milk validation episode | 137 | 137 | 0.9638 | 8/8 matched |

DAVIS points were derived from human-annotated first-frame masks; later masks
were used for scoring only. DAVIS void pixels were excluded. These are three
videos and one object per video, not a benchmark-wide score.

## LIBERO numerical-quality report

- Task: `pick_up_the_milk_and_place_it_in_the_basket`, validation
  `episode_0014` / source `demo_21`, 137 frames, 20 Hz. This validation
  initialization has now been used for development and is not untouched.
- Input: native 512×512 MuJoCo `agentview` RGB re-rendered from the recorded
  simulator states, saved as JPEG. This is simulated camera RGB, not a physical
  camera recording. The agent selected a single point at normalized xy
  `(0.25, 0.541016)` by visually inspecting the first RGB frame. This is not a
  human click and not text grounding. Only the milk object was tracked; the
  basket was not independently tracked in this pilot.
- Simulator instance masks were opened **after tracking and prefix checks** and
  used only for post-hoc scoring. The 512-pixel predictions were downsampled with
  nearest-neighbor to the 128×128 simulator-mask grid before all reported
  metrics. This introduces resolution/aliasing limitations.
- Mean mask IoU was **0.963754**, fifth percentile 0.920424, minimum 0.860927;
  all 137 predicted masks and all 136 consecutive prediction/reference pairs
  were nonempty/valid. Four overlays (frames 0, 45, 91, 136) were inspected and
  show the initialized milk carton, gripper interaction, and basket placement.

All geometry errors below are mean absolute errors in image-fraction units,
using valid paired frames. Frame-delta error is the absolute difference between
the predicted and reference *frame-to-frame* changes; it measures a combination
of segmentation jitter and tracking error, not physical object velocity.

| Measurement | Per-frame MAE | Frame-delta error MAE | Frame-delta error standard deviation |
|---|---:|---:|---:|
| centroid u | 0.000964 | 0.000532 | 0.000905 |
| centroid v | 0.000986 | 0.000802 | 0.001270 |
| bbox width | 0.000969 | 0.001608 | 0.003669 |
| bbox height | 0.000627 | 0.001149 | 0.003142 |
| visible area | 0.000203 | 0.000134 | 0.000203 |

The width and height delta errors are larger than their per-frame errors in
this episode, so segmentation variation can appear as object change. This is one
visually initialized validation episode, not an estimate across environments or
occlusions. The local report and inspection image are
`outputs/libero/oracle/sam31_point_pilot_milk_val_v3.json` and
`outputs/libero/oracle/sam31_point_pilot_milk_val_v3_review.png`; raw masks and
full audits remain on the Modal volume under `/data/eval/sam31_point_pilot_v3/`.

For the eventual policy comparison, both variants must be trained on the **same
SAM-derived measurements** with the same splits. Feeding these measurements to
an oracle-trained policy would instead measure distribution shift. Neither
comparison nor a camera-based closed-loop policy is reported here. The
one-action monitor remains passive.
