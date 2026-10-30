# Dataset-level SAM measurement experiment

Development-only offline results; no closed-loop camera policy was evaluated.
The shared dataset contains 34 training and eight validation episodes (6,098
frames total), and excludes all eight previously used test episodes. Validation
is used for checkpoint selection and reporting, so its numbers are optimistic;
none of these initializations is an untouched final evaluation.

## Frozen perception and cache

`sam31_milk_development_config.json` freezes the SAM revision and checkpoint
hash, source dataset hash, adapter settings, slot order, fixed points and frame
rate. `scripts/modal_milk_render_development.py` re-renders 512×512 RGB from
recorded simulator states at the user-supplied 20 Hz; the agent inspected a
contact sheet of all 42 first RGB frames with the two fixed points marked.
`scripts/modal_milk_sam_cache.py` tracks both IDs together using those points,
stores 128×128 nearest-neighbor masks, metrics, validity, ID status and original
timestamps. All 12,196 object-frames had unique IDs and nonempty masks. The
original extraction was interrupted after 29 episodes and resumed in a fresh
container with seed 2468 reset; the saved provenance records this boundary.
No reference masks were opened by extraction. Image sequence and cache hashes
are stored on the Modal volume.

This is initialized simulated RGB on preloaded videos, with only bounded prefix
checks, not a verified live or generally causal adapter. The dataset explicitly
sets `causal: false`; training requires an explicit experimental opt-in rather
than relabeling it causal. No filtering or future-defined labels affect inputs.
All variants use the same cached measurements, and simulator masks are opened
only in separate post-hoc scoring after extraction/training.

The dataset fingerprint is
`b7bae449415728262307c262677def270cd50380406183d2455540351a1c72c4`.
Perception cache, rendered frames, dataset and scoring artifacts are respectively
`/data/libero/milk_sam_masks_v1`, `/data/libero/milk_sam_native_v1`,
`/data/libero/milk_sam_development_v1`, and
`/data/eval/milk_sam_measurement_audit_v1.json`. Raw data stay off Git.

Mean validation IoU is 0.9433 milk and 0.9751 basket. Yet milk width-change
MAE against reference changes is 0.001903 overall and 0.004207 on moving width
changes; height is 0.001273 overall and 0.002866 moving. Moving is defined
separately for each object/metric as absolute reference change > 0.002 image
fraction, for scoring only. Tiny moving subsets, especially basket area, cannot
support strong conclusions.

## One-step dynamics

`scripts/modal_milk_sam_dynamics.py` trains same-architecture predictors on
SAM-derived one-step deltas, with four-observation history and either executed
action `a[t]` or a zeroed action input. The selection rule for both is minimum
validation SAM normalized one-step MSE across 25 epochs. Train-only statistics,
paired seeds 7/19/31, and 1,179 validation windows were used. Reference masks
were first loaded after all six model checkpoints were trained. The monitor is
passive. No action after `a[t]` enters this comparison.

Validation SAM normalized MSE (no-action → action) was 0.0009880 → 0.0009522,
0.0009859 → 0.0009548, and 0.0009711 → 0.0009373 for seeds 7, 19 and 31.
This modest improvement on the selection data does not imply uniform improvement
against simulator changes or over persistence.

The table reports **reference-change MAE** in image-fraction units for
persistence / no-action / action. Learned numbers are means across three seeds;
moving and stationary subsets are defined by the reference for scoring only.
The complete per-seed, SAM-target and reference-target results and counts are in
`/data/runs/milk_sam_one_step_v1/comparison.json`.

| Object / metric | Moving count | All | Moving | Stationary |
|---|---:|---|---|---|
| milk centroid u | 523 | .004444 / .001687 / .001276 | .009478 / .002466 / .001785 | .000430 / .001066 / .000871 |
| milk centroid v | 471 | .002691 / .001393 / .001321 | .005948 / .002042 / .001778 | .000523 / .000962 / .001018 |
| milk bbox width | 312 | .002081 / .002652 / .002613 | .007863 / .006835 / .006824 | .000000 / .001147 / .001098 |
| milk bbox height | 307 | .002167 / .002546 / .002558 | .008321 / .006182 / .006142 | .000000 / .001266 / .001297 |
| milk area | 1 | .000203 / .000196 / .000193 | .002502 / .002125 / .002178 | .000201 / .000194 / .000191 |
| basket centroid u | 2 | .000109 / .000340 / .000363 | .002178 / .001998 / .002204 | .000106 / .000337 / .000360 |
| basket centroid v | 14 | .000191 / .000348 / .000328 | .002349 / .001575 / .001390 | .000165 / .000333 / .000315 |
| basket bbox width | 4 | .000027 / .000190 / .000187 | .007812 / .007804 / .007838 | .000000 / .000165 / .000161 |
| basket bbox height | 21 | .000146 / .000386 / .000408 | .008185 / .007234 / .007315 | .000000 / .000262 / .000283 |
| basket area | 0 | .000092 / .000103 / .000107 | n/a | .000092 / .000103 / .000107 |

For milk centroid motion the learned models beat persistence, and action
conditioning improves the moving-centroid MAE. Persistence remains better on
stationary centroid/box subsets and on overall width/height. Moving-label
thresholds are post-hoc reference labels, never a runtime gate.

## Matched policy training

`scripts/modal_milk_sam_policy.py` uses the same cached SAM dataset for both
variants, paired seeds, 25 epochs and the same minimum validation **flow MSE**
checkpoint criterion. Offline action sampling uses the same fixed seed-921 noise
schedule and 20 flow steps for every variant. Dynamics-loss weight is 0 versus
1. Validation sampled-action normalized MSE (action-only → joint):

| Seed | Selected epoch action-only / joint | Action-only | Joint |
|---:|---:|---:|---:|
| 7 | 21 / 21 | 0.731224 | 0.727839 |
| 19 | 25 / 25 | 0.656592 | 0.652896 |
| 31 | 18 / 18 | 0.758274 | 0.755678 |

These small, consistent offline differences are development-set action prediction,
not control success. Results and checkpoints are under
`/data/runs/milk_sam_matched_policy_v1`. A matched SAM-based closed-loop rollout
remains outstanding: the current tracker preloads video and does not implement
appendable live observations. Do not substitute oracle masks into these policies
and call that a SAM control result; reserve genuinely unused simulator
initializations for a later frozen evaluation.
