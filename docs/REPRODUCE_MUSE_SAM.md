# Reproduce the Muse/SAM LIBERO demonstration

The documented result is **1 success in 3 sampling seeds** for the same frozen
checkpoint and demonstrated training initialization. It is not a held-out
generalization result. The evaluator performs real inference and simulation on
Modal and consumes GPU credits.

## Frozen checkpoint and downloads

The authoritative checkpoint is `checkpoint_50.pt` on the Modal volume
`object-dynamics-vla-data` in the `tobiwarwager` workspace. It includes optimizer
state and weighs about 720 MiB. The following commands require an authorized
profile for that workspace; replace `newaccount` if your local profile differs.
From the repository root, with `outputs/` present:

```bash
python -m modal volume get object-dynamics-vla-data /eval/muse_aligned_dataset_v1/checkpoint_50.pt outputs/muse_dataset_checkpoint_50.pt --profile newaccount
python -m modal volume get object-dynamics-vla-data /eval/muse_aligned_dataset_v1/stats.npz outputs/muse_dataset_stats.npz --profile newaccount
sha256sum outputs/muse_dataset_checkpoint_50.pt outputs/muse_dataset_stats.npz
```

Expected SHA-256:

```text
fed8976c64c4b0842cb9158b675501c18d4ac8c8b2ae248e0aa1ff7947600bc2  outputs/muse_dataset_checkpoint_50.pt
e32acff474d12f798459df2ca80ac65f8fa2c7251f26d4f32d66de5e73926b54  outputs/muse_dataset_stats.npz
```

These are workspace-private Modal downloads, not public anonymous links. The
checkpoint and normalization statistics are not committed to this repository.
To reproduce outside this workspace, the required assets below must first be
obtained and placed in a volume with the same paths.

## Pinned inputs and environment

- Muse: `meta-models/Muse-Glimmer-30B`, revision
  `cec94fcfd6512f2cfe708138c32533b346ea4f69`, layer 26, NF4 double
  quantization, BF16 compute/storage, pinned chat date `2026-10-30`.
- SmolVLM reference configuration: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`,
  revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`. The action expert and
  prefix weights come from checkpoint 50; VLM weights are not loaded into the
  action worker.
- SAM 3.1 source revision `f1647e8d0c1d280991f4d1b7d974c2704537fe66`;
  `sam3.1_multiplex.pt` SHA-256
  `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`.
- LIBERO revision `8f1084e3132a39270c3a13ebe37270a43ece2a01`; the milk demo
  HDF5 SHA-256 is `e56d954928e613f9db58085b7677ccf9c4824f1ab982265f71405d5bb14cfb4a`.
  It can be fetched with `python -m modal run -m scripts.modal_download_libero_milk`.
- Aligned dataset fingerprint
  `86d9fdf00329e8f18df23bab67d70dbfae914bc68c6884cdd8938f8518386bed`.
  `episode_0000` is a **training** episode; normalization is the saved
  train-split `stats.npz`, not recomputed from the rollout.
- Python packages and image construction are specified in
  `scripts/modal_muse_motor_rollout.py`, `scripts/modal_muse_sam_paired_rollout.py`,
  and `scripts/modal_quantized_muse_preflight.py`; key pins include Torch
  2.10.0, Transformers 5.21.0, bitsandbytes 0.48.2, MuJoCo 2.3.7, and
  robosuite 1.4.0. The live Muse/SAM worker uses H200 and the action worker
  uses A100-80GB.

The initialized simulator state hash is
`02ffa3feb4de735d2c90d5c756eadef0d07eaa86611e0ddf596599ca65c39a88`.
SAM starts from two fixed, RGB-inspected image points, not simulator masks;
their coordinates and tracking settings are in `sam31_milk_development_config.json`.
The rollout restores TF32-off, autocast-off settings for Muse and checks
initial state, RGB, feature, robot, and SAM parity before issuing actions.

## One evaluation command

With the assets already present in the authorized workspace, run a new
seed-7 evaluation using a **fresh output name**:

```bash
python -m modal run --detach -m scripts.modal_muse_smolvla_live --stage candidate --fit-name muse_aligned_dataset_v1 --checkpoint-step 50 --dataset muse_aligned_development_v1 --name muse_dataset50_reproduction_seed7_v1 --max-steps 200 --seed-base 7 --baseline-trace muse_aligned7500_episode0000_v1 --background --profile newaccount
```

The command requires the checkpoint, `stats.npz` and `report.json` under
`/eval/muse_aligned_dataset_v1/`, the aligned dataset and milk HDF5 under
`/libero/`, both model directories and the SAM checkpoint under
`/checkpoints/`, and the baseline report under
`/eval/muse_aligned7500_episode0000_v1/report.json`. It checks the same
initial-state and per-step noise hashes as the baseline. GPU/backend differences
can cause parity checks to stop the run before any action. The pinned inputs and
seed alone do not guarantee bit-exact reproduction on different hardware.

The job writes `report.json`, `trace.json`, `rollout.mp4`, per-step observations
and `noise.npy` to `/eval/muse_dataset50_reproduction_seed7_v1/`. Use
`python -m modal volume get object-dynamics-vla-data /eval/muse_dataset50_reproduction_seed7_v1 <existing-local-directory> --profile newaccount`
to download the finished directory. The report's simulator success flag, not an
offline action metric, determines whether that attempt succeeded.
