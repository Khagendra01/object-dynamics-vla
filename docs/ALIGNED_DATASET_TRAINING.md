# Aligned dataset training

This replaces the single-episode, chronological batch-one experiment with
shuffled training windows across the aligned training split. It retains the
existing NF4 Muse -> query-normalized prefix -> official action expert
architecture, causal SAM measurements, 50-action horizon and demonstrated-action
flow objective. The dynamics objective remains off. This is an architecture
adaptation, not a literal reproduction of native SmolVLA conditioning.

## Two commands

Run from the repository root using the authorized Modal profile that owns the
existing aligned dataset and checkpoint volume. A friend's separate workspace
must have those assets before these commands can work.

```bash
python -m modal run -m scripts.modal_muse_dataset_train --stage cache --max-seconds 600 --profile khagendrakhatri365
```

The H200 extractor writes actual NF4 feature tensors, hashes and frame manifests
under `/libero/muse_aligned_nf4_v1`. If it returns `complete: false`, the same
command resumes from committed frames. It checks model/runtime identity and
RGB hashes rather than regenerating completed frames. A cache must be complete
before training; extraction is real paid compute, not a free prerequisite.
The default 600-second budget includes model loading, with a separate
3,600-second function timeout. No weights are downloaded by this job.

```bash
python -m modal run -m scripts.modal_muse_dataset_train --stage train --max-seconds 1800 --profile khagendrakhatri365
```

Training loads all train episodes and keeps the reused validation episodes out
of optimization. Defaults: A100 80 GB, batch 8, accumulation 8, effective batch
64, 1,000 updates in the new phase, checkpoint/evaluation every 250 updates.
It warm-starts from `/eval/quantized_muse_aligned7500_v1/preflight_weights.pt`
only when the aligned dataset fingerprint and normalization match exactly.
The old checkpoint has no optimizer state, so this initial warm start explicitly
restarts AdamW; it does not claim uninterrupted continuation from step 7,500.
Use `--init-fit ''` to train a fresh expert instead. New-phase update numbers
exclude source single-episode updates; logs include actual window presentations.

If training stops at its time budget, resume with the **same** training contract:

```bash
python -m modal run -m scripts.modal_muse_dataset_train --stage train --resume --max-seconds 1800 --profile khagendrakhatri365
```

For a detached chunk that does not require the local CLI to stay connected, use
`modal run --detach` and add `--background` to spawn the training call. Inspect
the Modal app and the saved report before launching another chunk; a spawned
chunk does not automatically resume itself.

`last.pt` restores prefix/expert/projection weights, AdamW state, shuffled epoch
order/cursor and NumPy generator, plus Torch CPU/CUDA RNG. Changing batch size,
accumulation, data/cache or initialization source rejects resume. The target
update count may be increased explicitly with `--updates`; it may not decrease.
A numerical GPU next-update equivalence check is still required before
claiming exact numerical reproducibility; CPU tests cover the shuffled schedule.

## Outputs and evaluation

`/eval/muse_aligned_dataset_v1/` contains stats, report, atomic `last.pt` and
numbered checkpoints. The report includes train/validation episode IDs,
effective batch, window presentations, cache and checkpoint hashes. Periodic
ten-step sampled-action evaluation uses four fixed windows per reused validation
episode, reports per-channel raw/normalized error and both gripper classes,
and does not consume the training RNG. These observations are development data.
Checkpointing happens before evaluation. Evaluation is deferred when fewer than
120 seconds remain in the requested training-loop budget.

The metadata/weights follow the existing quantized rollout loader's contract:
the new-phase checkpoint number can be supplied to the existing candidate runner
alongside `--fit-name muse_aligned_dataset_v1` and the aligned dataset name.
An actual cache/train/rollout integration has not yet been executed on GPU.
No rollout or further paid job is automatically launched by this script.

The displayed training-loop cost uses the requested A100/CPU/RAM rates and
excludes setup, startup/builds, writeback, extraction and storage; it is not an
invoice or a total spending cap. Choose runtimes against actual available credit.

Local verification: Python compilation, Modal CLI import/help and CPU tests.
No successful manipulation or performance improvement is claimed by this change.
