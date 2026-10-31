# Two-task frozen-Muse motor development

This is a two-task, seed-7 development run, not a demonstrated competent robot
policy, an independent benchmark or the complete multi-task/SmolVLA reference
requested by the owner. It preserves the earlier milk-only baseline.

## Data and compatibility audit

The existing LIBERO-Object milk and alphabet-soup data share 15 state and 7
action channels and the same imported robot-key order, 20 Hz frequency,
agentview identity, orientation and observation-before-action assumption.
`scripts/modal_multitask_controller_audit.py` instantiated both tasks in the
pinned LIBERO/robosuite environment. Both used SingleArm,
OperationalSpaceController, action bounds `[-1,1]`, identical controller
input/output limits, and PandaGripper (negative opens, positive closes). Recorded
`demo_0` replay reached success on both; mean state MAE was 0.001397 milk and
0.002295 soup. This checks two demonstrations and actual simulator configuration;
it does not prove every source demo's controller metadata.

Only pre-existing train/validation demonstrations were rendered at native
512-pixel agentview with JPEG quality 98. The soup pilot produced 148 frames,
with 128-pixel replay RGB MAE 2.068. The full render contains 42 train/validation
episodes, excludes the eight test episodes, and is under
`/data/libero/object_native_512_v1`. The first full render was preempted after
26 episodes; the resumed script verified each existing JPEG against a fresh
same-backend render byte-for-byte before reuse. Full image and episode hashes,
split membership and RGB differences are in that directory's `provenance.json`.

The pinned frozen Muse model, decoder block 26, and processor were applied to
6,593 soup frames. A 148-frame pilot took 62.0 s after model loading and peaked
at 55.57 GiB. Full extraction took 2,719.6 s after loading, produced 37,783,217,144
bytes, and peaked at 55.57 GiB. The processor hash matches the milk cache;
Transformers again reported LANCZOS-to-BICUBIC fallback. The local return of
the pilot failed to deserialize `torch.__version__` because local torch is absent;
the pilot manifest was committed and the return value was fixed. Full cache:
`/data/libero/object_muse_middle_tokens_v1`.

The shared normalizer was fitted from **only** the two tasks' training episodes.
The soup dataset has missing object measurements, which are not treated as
observations. Source identities use the SHA-256 of the original HDF5 file plus
source demonstration identifier. The runner checks that training sources do not
overlap either task's validation/test sources, including the eight old milk test
episodes, and writes audited contracts at
`/data/eval/multitask_source_contracts_v1.json`. Source hashes and split checks
cannot rule out semantic duplicates from differently encoded datasets.

## Motor training and transfer smoke

The action-only expert was trained from scratch using 10,117 pooled train
windows and 2,574 pooled validation windows, uniform over frames, effective
batch 64, seed 7. No SAM inputs or numerical objective were used. Validation is
used both for selection and scoring.

| Update | pooled validation flow MSE |
| ---: | ---: |
| 250 | 0.962155 |
| 500 | 0.780076 |
| 750 | 0.735857 |
| 1000 | 0.712241 |

Initial preload took 2,925 s; the run stopped at step 500 after its runtime cap,
with model, optimizer and RNG state saved. Ordered eight-thread preload on the
resume took 132 s and restored the original 1,000-step schedule, completing in
754 s including preload. The selected weights, optimizer/RNG state, stats,
configuration, audit and logs are under
`/data/runs/muse_multitask_milk_soup_motor_seed7_v1`. The source-only motor
checkpoint is not a policy-success result.

One M1 milk update from that two-task checkpoint passed the cross-dataset audit
and retained the source normalizer. Its untrained-adaptation smoke validation
flow MSE was 0.719425; it is not an M1/M2 comparison. Artifacts are under
`/data/runs/multitask_to_milk_M1_smoke_v1`. The output includes the motor
checkpoint SHA-256 and overlap audit.

## Reproduction, budget and outstanding work

The commands below refuse to overwrite completed outputs. A resume requires
the existing partially completed run and its identical source contracts.

```bash
~/modal-venv/bin/modal run scripts/modal_object_render.py --full
~/modal-venv/bin/modal run scripts/modal_object_muse_cache.py --full
~/modal-venv/bin/modal run scripts/modal_multitask_controller_audit.py
~/modal-venv/bin/modal run scripts/modal_multitask_motor.py
~/modal-venv/bin/modal run scripts/modal_multitask_motor.py --resume
~/modal-venv/bin/modal run scripts/modal_multitask_transfer_smoke.py
~/modal-venv/bin/modal run scripts/modal_vla_tests.py
```

At the previously recorded Modal published rates, 2,719.6 s of A100-80GB plus
4 CPU cores and 128 GiB is approximately $2.80, excluding model load and extra
usage. The initial/resumed motor jobs used T4, 4 CPU cores and 112 GiB for about
3,543 and 754 seconds respectively: approximately $2.00 at requested-resource
rates, excluding startup and other charges. Rendering, pilot, audit, smoke and
storage are additional. Actual billing and remaining credit were not retrieved.

The official pinned SmolVLA reference through the adapter, independent test
states, skill rollouts and matched three-seed M0/M1/M2 adaptation remain open.
The two-task checkpoint uses one seed, only 1,000 updates and pooled selection;
neither data scale nor validation loss establishes motor competence. The train
runner currently relies on the specific audited Modal driver to construct the
pooled dataset; `prepared_provenance` must not be used with arbitrary unaudited
caller-supplied samples. Neither test episodes nor simulator masks were inputs
to this motor training.
