# Canonical data contract

## Directory

```text
dataset/
  manifest.json
  episodes/
    episode_0000.npz
    episode_0001.npz
    ...
```

NPZ arrays are loaded with `allow_pickle=False`. Use numeric arrays only.
Paths in the manifest must stay inside the dataset root.

Minimal manifest:

```json
{
  "schema_version": 1,
  "synthetic": false,
  "metric_names": ["centroid_u", "centroid_v", "bbox_width", "bbox_height", "visible_area"],
  "metric_units": ["image_fraction", "image_fraction", "image_fraction", "image_fraction", "image_fraction"],
  "object_slots": ["red cup", "white bowl"],
  "action_convention": "Document the actual source controller, units and reference frame here",
  "perception": {"source": "your audited extractor", "causal": true},
  "episodes": [
    {"id": "episode_0000", "file": "episodes/episode_0000.npz", "group": "source_demo_0", "instruction": "Put the red cup in the white bowl."}
  ]
}
```

There must be at least five independent groups for the default three-way split.
The example shows one episode only for schema readability.

## Arrays and time alignment

| Array | Shape | Meaning |
|---|---|---|
| `features` | `[T,O,F]` float | Measured values at observation time t |
| `valid` | `[T,O,F]` bool | Whether each measurement was actually observed |
| `robot` | `[T,R]` finite float | Robot observation at t, same channel order across episodes |
| `actions` | `[T,A]` finite float | Command executed after observation t, affecting observation t+1 |
| `timestamps` | `[T]` float64 | Strictly increasing observation times in seconds |
| `context` | `[T,C]` optional float | Cached frozen image+instruction representation available by t |
| `rgb` | `[T,H,W,3]` optional uint8 | Original RGB observations; required by model-extraction/replay scripts |

Object and feature dimensions must match the manifest. Every episode must use the
same robot/action/context dimensions and slot semantics for a run. Put entirely
different robots/tasks into separate datasets until an embodiment adapter exists.

`context` is an empty `[T,0]` array when absent. That experiment is numerical/state
learning, not a language-conditioned VLA. Simulator privileged state must not be
inserted into context for a deployment claim based on images.

For history L and prediction horizon H at time t:

- Input observations: `t-L+1 ... t`.
- Action target / dynamics condition: `actions[t:t+H]`.
- Numerical targets: `features[t+1:t+H+1] - features[t]`.
- Valid target: observed at both the anchor t and the corresponding future time.
- Valid starts: `L-1 <= t < T-H`.
- Number of windows per episode: `max(0, T-H-L+1)`.

The final action has no subsequent observed state and is not used as a complete
dynamics target. Do not shift an imported dataset until its convention is checked.
An existing dataset with post-action images needs an explicit alignment conversion.

Current input is built by `encode_history`, which does not require any future
frames or action labels. The tests mutate future data and verify input invariance.

## Numerical features

The five implemented mask measurements are:

- Centroid: `((mean(x)+0.5)/W, (mean(y)+0.5)/H)` in image pixel-center convention.
- Bounding width/height: occupied pixel extent divided by image width/height.
- Visible area: number of foreground pixels divided by image area.

These describe the visible silhouette; they are not true surface area or volume.
Empty masks produce invalid measurements, not a zero-sized physical object.
Occlusion, viewpoint, lighting, segmentation error and physical motion can all
change the measurements. The initial fixed-camera experiment limits one source
of ambiguity but does not eliminate the others.

Feature values, masks, pairwise changes, and capped log observation ages enter
the learned encoder. Missing values are zeroed *after normalization* with explicit
validity channels; they are not backfilled from future frames.

Train-only means/stds are fitted per measurement channel across training objects.
Robot, action and context statistics are also train-only. Minimum std is 1e-3.
Numerical deltas are divided by feature std; output MAE is converted back to
manifest units. Never compare unscaled pixel error with meter error as one loss.

The v0.1 loss assumes ordinary scalar channels. Exclude angle wraparound and
quaternion channels until an appropriate rotation representation/loss is added.

## Optional depth geometry

`backproject_visible_point` implements pixel+optical-axis-depth backprojection and
camera-to-robot-base transformation. It returns a median visible surface point.
It is tested numerically, but is not wired automatically into the five-feature
extractor. A calibrated converter can append xyz to the schema and appropriate
validity channels before splitting.

Depth must be in meters and represent optical-axis z. Ray distance needs conversion.
Camera intrinsics must correspond to the actual image size and pixel convention.
The helper uses the supplied pixel indices; confirm calibration's center convention.
`T_base_camera` maps camera coordinates into the robot base frame.

Do not compare a visible-surface centroid with a simulator object-origin position
as if they were identical. Choose consistent targets or a separately trained pose head.

## External SAM / simulator masks

```text
masks/
  provenance.json
  episode_0000.npz
  ...
```

Each file contains `masks`: boolean `[T,O,H,W]`, with the same timestamps,
original image resolution, and ordered object slots as the corresponding episode.
All frames are required. Missing objects use an empty mask.

Example provenance:

```json
{
  "source": "SAM 3.1 adapter, pinned repository commit and checkpoint hash",
  "causal": true,
  "identity_assumption": "Document instance initialization and association",
  "uses_video_tracking": true
}
```

`causal=true` is a declaration, not automatic certification. Audit prefix consistency
before using offline video tracks as policy inputs. Simulator masks may be used for
an explicitly labeled oracle-perception comparison, not presented as SAM performance.

## Splits and reproducibility

The default splitter shuffles groups deterministically and assigns approximately
70%/15%/15% to train/validation/test. The same `group` must be used for related
augmentations or source demonstrations. If all episodes are independent, use the
episode ID as group. Episode disjointness alone does not create an unseen-scene,
unseen-object or unseen-task test: those require deliberate group assignments.

Manifest and episode file hashes form the dataset fingerprint. A changed dataset
invalidates an existing split/run. Identical files crossing splits are rejected.
Semantic duplicates that differ in bytes still require a dataset-specific audit.

The trainer does not construct test windows. Evaluation explicitly loads test only
when requested. Loading file hashes to verify integrity is not test-set fitting.
