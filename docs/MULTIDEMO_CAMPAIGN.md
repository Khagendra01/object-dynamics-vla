# Multi-demonstration ten-task campaign

This campaign expands the ten-task integration data while keeping frozen NF4
Muse, frozen SAM, the query-normalized trainable prefix and action expert, and
the flow-only objective. It does not establish mastery of the ten tasks.

## Data and tracking

The prepared fixture uses three training demonstrations for each of nine new
LIBERO-Object tasks and retains all 34 original milk training demonstrations.
`ten_task_multidemo_v1/renders.json` records the reused and new renders:
28 selected episodes and 4,156 frames, including one milk episode; the other
33 milk episodes are loaded from the original aligned training split. The
metadata registry and splits are in
[`configs/libero_object_10_prepared_registry.json`](../configs/libero_object_10_prepared_registry.json).
Development and evaluation demonstrations are excluded from optimizer batches.

First-frame SAM points are fixed, task-specific points proposed by the
assistant after inspecting RGB previews. They are not autonomous object
selection and are not independently verified annotations. The chocolate
pudding target remains heavily occluded by another object at initialization.
Its target measurements are masked as invalid throughout this campaign rather
than presented as reliable target tracking; the basket and image/instruction
inputs remain available. The three selected demonstrations for a given task
look visually similar at initialization; more demonstrations here should not
be mistaken for demonstrated scene diversity.

`scripts/modal_libero_multidemo.py` renders only the preassigned training
episodes and produces prefix-checked incremental SAM measurements. Completed
episodes are committed individually and can be reused on restart. The
`sam_preview.png` visualization and per-episode metadata live in the Modal
volume under `/libero/ten_task_multidemo_v1/`.

## Sampling, storage and continuation

The sampler chooses task → episode → window. It records presentations per task
and per episode, and unique windows per task. The loader rejects prepared
records assigned to a non-training split. Its registry and prepared-directory
arguments support arbitrary task counts; a synthetic 100-task loader test does
not mean 100 real tasks have been prepared or trained. The current preparation
script and pinned registry cover ten tasks.

The original cache was already a 2 GiB **disk** LRU. This campaign adds a
separate 512 MiB RAM LRU above a 20 GiB disk LRU. These bounds count array
payload bytes and cache-file bytes respectively, not total process RSS or total
Modal volume usage. Features are recomputed on a miss; Muse stays frozen and
the trainable prefix is never cached. The cache key includes image, instruction,
task, episode, frame, model, processor and backend identity. The RAM and disk
caches are single-worker implementations, not shared distributed caches.

Both prior checkpoints remain frozen: the successful milk reference
`fed8976c…` and the ten-task integration checkpoint `7c86adc0…`. The expanded
campaign initializes model and optimizer from the latter but **restarts its
sampler** because the dataset changed. It is a new phase, not exact resumption
of the old data schedule. Continuations within the new phase save and restore
model, optimizer, sampler, CPU/CUDA RNG and per-task seen-window accounting.
The original milk normalization remains fixed and is verified by SHA-256.

The first bounded H200 block is configured for ten new-phase updates at
effective batch 64, with checkpoints every five updates. The report records
RAM hits, disk hits, misses, extraction time, cache lookup time, disk/RAM
occupancy and task/episode/window coverage. Its fixed-noise offline diagnostics
use three windows from each **training** task and are not a task-success rate.
Actual time, hit rate, checkpoint and billing must be read from the completed
report before deciding on any extension. Runtime bounds are not invoice-level
spending caps.

### Measured first block

The first block completed **10 new-phase updates / 640 presentations**, with
all three prepared episodes seen for each new task and 34 milk episodes seen.
It continued the integration checkpoint's optimizer but started a new sampler
for the expanded dataset. The frozen checkpoint-10 SHA-256 is
`c8091d2108416af3b3d21268b7b08a901ce5bbf52d66ab764f71631d25ae6993`.

During training, the RAM cache served 16 requests, disk served 90, and 534
required Muse recomputation: **106/640 hits (16.6%)**. This early rate is not a
steady-state estimate; the run expands the window population while filling a
larger cache. At the end, disk entries used 5,202,014,639 bytes and RAM array
payloads used 532,720,175 bytes, within the configured limits. The run reported
about 74 seconds extracting features and 53 seconds looking up cache entries;
its total elapsed time including setup and offline diagnostics was about 561
seconds. H200 function startup and other charges are not a hard billing cap.

After measuring that pace, a continuation toward 30 new-phase updates was
started with its own runtime bound. Full-horizon seed-7 rollouts of frozen
checkpoint 10 on milk and alphabet soup were also started; both are training
initializations. Their final results must be read from the saved reports before
making a control claim.

Key volume paths:

```text
/libero/ten_task_multidemo_v1/renders.json
/libero/ten_task_multidemo_v1/sam_extract.json
/eval/muse_ten_task_multidemo_audit_v1.json
/eval/muse_ten_task_multidemo_v1/last.pt
/eval/muse_ten_task_multidemo_v1/report.json
/eval/muse_multidemo_frozen10_v1/checkpoint_10.pt
```

## Evaluation

Freeze a selected new-phase checkpoint before evaluation. Full-horizon
rollouts on milk and representative new objects should preserve their seeds,
initial input parity, all attempts and video/trace artifacts. The currently
prepared fixed prompts cover training initializations only: held-out
initializations need their own honestly documented initialization and tracking
validation. Do not describe a failed or truncated run as a task success, and
do not infer held-out reliability from training-window action metrics.
