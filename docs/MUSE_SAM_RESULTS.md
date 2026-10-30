# Muse/SAM VLA development results

## Environment and resource boundaries

The workstation is Intel macOS, torch 2.2.2 CPU; CUDA and MPS are unavailable,
so no local RTX 5090 operation could be verified. Modal reused the existing
SAM and Muse checkpoints, with no new model download. The user authorized at
most $10 of additional Modal usage for this work. Modal billing showed $11.48
credits applied for the month at the start, but did not expose remaining credit.

## First interface and token checks

On a real training `episode_0000` window at t=0, the 4×15 robot history had
validity `[false, false, false, true]`, and all 50 action targets were valid.
The 35,271,697-parameter experimental expert completed one CPU optimizer step
and sampled a first action from the matching **real cached** 429×6656 Muse
token tensor. One local run took 3.42 seconds; loss was 2.5482 (flow 2.5305,
dynamics 0.1773). The action was untrained and is not a robot result. Parameter
names and counts are printed by `scripts/vla_smoke_update.py`.

Five real validation RGB frames through frozen Muse peaked at 55.57 GiB GPU
allocation. Layer 26 of 52 produced 429 BF16 tokens × 6656 values per frame,
approximately 5.71 MB uncompressed. The first forward took 1.47 seconds;
the other four took 0.185–0.194 seconds, excluding processor/load overhead.
The unprojected cache for all 42 existing train/validation episodes was then
completed in 2,175.7 seconds after start on A100; raw features and manifests
are under `/data/libero/muse_middle_tokens_development_v1`. A separately cached
137-frame validation episode completed in 62.9 seconds after model load. Its
first cached frame and a fresh same-backend extraction matched bit-for-bit for
features, input IDs, attention mask and positions. This is a one-frame
same-backend check, not a broad rollout parity guarantee.

The frozen processor is Transformers 5.21.0 with torchvision 0.25.0. It logs a
LANCZOS-to-BICUBIC fallback, which is explicitly recorded and must be reproduced
in both caching and online inference. No simulator masks were used in feature
extraction. Source checkpoints, raw features and frame data remain outside Git.

## Expert checks

An eight-episode overfit diagnostic used only the first frame of each of the
first eight training episodes, real frozen Muse tokens, 100 updates, batch 8,
and a fixed training-only scoring schedule. Its flow MSE decreased from 2.1401
to 0.4023. It is a mapping/optimization check, not a validation outcome.

An effective-64 real-data trainer smoke (one M2 update) ran to completion and
reported validation flow MSE 2.9331. This is untrained smoke evidence, not a
comparison. End-to-end time including uncached token file reads and validation
was 720 seconds. A T4 resident-token batch-64 microbenchmark measured ~0.996
seconds/update and 5.94 GiB peak allocation, excluding loading, validation and
checkpointing; this is not a complete job cost. Cache size and host-memory
charges must be considered when projecting full 5,000-update paired runs.

The initial resident-cache seed-7 suite attempt was canceled when its local
runner timed out during preload, before any result or checkpoint. Later suite
outputs must be distinguished from this attempt. No official SmolVLA reference,
completed matched Muse training matrix, trained Muse/SAM rollout, or fresh
evaluation is claimed by these diagnostics.
