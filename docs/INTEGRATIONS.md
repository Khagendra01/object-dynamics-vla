# Model, simulator and cloud integration notes

## SAM 3.1

Install the official repository in a separate Python 3.12+ perception environment,
using its current supported PyTorch/CUDA instructions. Request/check checkpoint
access through the official repository. Record repository commit and checkpoint
SHA256. Do not put tokens or model weights into this project or its ZIP.

The official SAM 3.1 entry point checked during package authoring is
`build_sam3_multiplex_video_predictor`. The default `build_sam3_image_model`
downloads SAM 3 by default; using that API without checking the checkpoint is
not sufficient to claim SAM 3.1.

The supplied script intentionally opens independent single-frame sessions and
issues text prompts for unique instances. This structurally limits inputs to
the current frame. It uses the actual 3.1 multiplex builder, but does not exploit
tracking and may be slow.

Execution on 2026-10-28 (Modal A100-80GB, SAM commit `f1647e8`, checkpoint
SHA256 `0567debe...`) exposed two practical issues. The declared package
dependencies were insufficient: `einops`, `pycocotools`, and `psutil` were also
needed. `Sam3BasePredictor.start_session` also passes `offload_state_to_cpu` to
the multiplex tracker's `init_state`, which rejects it; the adapter now drops
that unsupported keyword. Model build took 19.7 s and peak allocated memory for
a one-frame prompt was 5.53 GiB.

Mask quality was the real blocker: on LIBERO-Object at 128×128, unique phrases
(`alphabet soup can`, `soup can`, `red can`, `basket`, `wire basket`) returned
zero masks. Only `can` matched, as 2–3 ambiguous instances. A full episode
produced 296 missing and zero observed masks. Upscaling to 512 did not fix
identity. Do not treat this as a usable perception adapter for that task.

What was tested here: output adaptation, multiple-instance rejection, shape
validation, measurement extraction, real checkpoint loading, and single-frame
GPU execution. What remains untested or failed: unique identity on LIBERO,
temporal tracking, and RTX 5090 inference.

For efficient video tracking, implement a verified forward/online adapter and
compare prefix outputs with full-stream outputs. Never promote a smoothed
full-clip result to a current input without proving it does not use future frames.

## LIBERO oracle masks (explicitly privileged)

Text-only SAM could not identify the target on this task, so the numerical
mechanism test uses simulator-oracle masks. This is **not** a camera-only
perception result. Each frame is rebuilt from the recorded simulator state;
masks are selected by exact geom name (`alphabet_soup_1`, `basket_1`) from
`sim.render(..., segmentation=True)`.

- `scripts/extract_libero_oracle_masks.py` performs state replay and mask
  rendering. It writes `[T, O, 128, 128]` boolean masks in canonical slot order
  and a provenance file.
- `scripts/modal_libero_oracle.py` runs the extractor on a Modal CPU image with
  MuJoCo OSMesa and the pinned LIBERO commit.
- Provenance includes `causal: true`, `oracle: true`, simulator-state inputs,
  source/BDDL SHA256, exact identity assumptions, RGB alignment statistics, and
  observed/missing/ambiguous counts. The training gate accepts it because it is
  declared causal, but the result must always be labeled oracle.

The full 50-episode pass produced 15,605 observed and 11 missing slot-frames
(99.93% observed); RGB mean absolute difference was 2.02 and maximum 4.29.
Attached dataset `/data/libero/object_oracle`, split fingerprint
`99cf1697...`, 34/8/8 train/val/test. The real mechanism test and matched
ablations are reported in `docs/EXECUTION_REPORT.md`: the numerical loss improved
measurement forecasting but did not improve open-loop action prediction.

## Muse Glimmer 30B

The pinned public revision `cec94fcfd6512f2cfe708138c32533b346ea4f69` has now
been tested on Modal A100-80GB. A real image-plus-instruction forward produced
finite `[1,84,6656]` hidden states with 55.59 GiB peak allocated memory; a
stride-4 cache was produced for the 50-episode milk oracle dataset. The
matched experiment did not improve action metrics. Transformers fell back from
LANCZOS to BICUBIC with the installed torchvision. The approximately 60 GB BF16
checkpoint remains on the Modal volume; see `docs/EXECUTION_REPORT.md`.

The script uses a model class exposed by the installed Transformers auto-model
API, a pinned checkpoint revision, frozen BF16 weights, and the last language
input token's hidden representation of current image+instruction. It does not
generate descriptions, train the 30B backbone, or use private activations from
a text-only API.

Run a one-episode tiny extraction before processing a dataset. Check:

1. The installed Transformers version recognizes `muse_glimmer` and the model's
   architecture. The checkpoint config inspected during authoring references a
   development version; a generic lower-bound dependency is not a verified lock.
2. The processor's multimodal chat template accepts the image object and creates
   the expected image tokens. Inspect output shapes and token lengths.
3. The model forward call exposes language hidden states. The helper fails if not.
4. GPU memory for weights, vision processing, hidden states and intermediate
   activations. Published GGUF text inference memory is a different workload.
5. Feature determinism in eval mode and prefix-only use. Context is carried forward
   between processed frames, so a stride introduces explicit representation lag.

Possible extraction machine: a profiled A100/H100 80 GB job. BF16 30B weights alone
are approximately 60 GB in decimal bytes; activations and vision processing need
additional memory. An 80 GB device is a candidate, not a fit guarantee.

Quantized hidden extraction could reduce local memory, but needs separate API,
compatibility, numerical and performance validation. It is not silently enabled.

## RTX 5090 scheduling

Use sequential stages: simulator/export → SAM preprocessing → Muse cache → train
small heads → closed-loop evaluation. The cache permits head training without
keeping large frozen models or the simulator resident at the same time.

Before a long run, use `doctor`, a few extraction frames, and a tiny training job.
The code does not contain custom CUDA kernels or depend on FlashAttention.

## Modal: optional, never launched automatically

Base GPU rates checked against the official pricing page during authoring:

| GPU | Approximate GPU-only hourly price | $30 GPU-only ceiling |
|---|---:|---:|
| A100 80 GB | $2.50 | 12 h |
| H100 | $3.95 | 7.6 h |
| H200 | $4.54 | 6.6 h |

CPU, memory, container loading and pricing modifiers affect total spend. Treat
these as dated estimates and recheck the official page. Use the local GPU first.

The optional template trains the small heads on a user-populated Modal Volume:

```bash
python -m pip install -e '.[modal]'
modal setup
# Upload only the selected dataset after deciding what you want to send.
modal volume create object-dynamics-vla-data
modal volume put object-dynamics-vla-data data/task_with_muse /task_with_muse
modal volume put object-dynamics-vla-data outputs/splits.json /task_with_muse/splits.json
modal run scripts/modal_train.py --dataset-name task_with_muse --run-name joint_cloud_seed7 --epochs 30
```

The template has one container and a 30-minute timeout. Override the GPU with
`ODVLA_MODAL_GPU` (for example `T4` or `L4`) and the timeout in seconds with
`ODVLA_MODAL_TIMEOUT`. A T4 was sufficient for the small reference heads in the
2026-10-28 execution report; an L4 was requested first, the image built in 41 s,
but capacity was unavailable for over 30 minutes. Check availability before
blaming the code.

It is not a dollar-denominated spending limit, does not implement
resume-on-preemption, and was not deployed during package authoring. It has since
been executed end-to-end on a T4 with the synthetic fixture. Save checkpoints more
frequently for longer cloud runs. Consult current Modal CLI help if transfer
syntax differs. A Volume is billable persistent storage subject to your plan; the
template does not remove uploaded data automatically.

Operational notes from that execution:

- For CPU data prep on the Volume (audit/import/split/replay), use
  `modal run scripts/modal_odvla.py --args "audit /data/path --out /data/out.json"`.
  The runner mounts the Volume at `/data`, commits changes, and does not train.
- `scripts/modal_sam31.py` runs the single-frame SAM adapter on an A100-80GB
  against the checkpoint on the Volume. The `hf-token` secret is only needed to
  download the gated checkpoint, not for extraction.
- `scripts/modal_libero_oracle.py` renders oracle masks (`--episode ""` for all
  50); `scripts/modal_odvla.py --args "attach-masks ..."` attaches them;
  `scripts/modal_real_ablations.py` runs the matched suite; and
  `scripts/modal_real_eval.py` evaluates a frozen run.
- If Modal reports a local file `was modified during build process`, check for
  future-dated mtimes and normalize them (`find odvla -name '*.py' -exec touch {} +`).
- Modal CLI 1.6.3 concatenated a remote directory's files into one local file for
  `modal volume get <volume> /runs/<dir> <dest>`. Download files individually
  (`modal volume get <volume> /runs/<dir>/<file> <dest>`) and verify the result.

The template is for training heads, not a ready Muse extraction service. Adapt a
small, locally validated Muse forward pass to a separate cloud function only if
profiling justifies it. No API credentials are bundled or read by default.

## Isaac and real robot

Isaac is for later closed-loop evaluation once the offline experiment is valid.
Use your installed version's compatible Isaac Lab release and robot asset.
A100/H100 devices without RT cores are not supported Isaac rendering hardware;
keep rendering on the local RTX machine.

A future adapter must map source commands to the actual controller and obtain
current camera/proprioceptive observations. The package intentionally does not
guess the unknown robot model, gripper, control rate or calibration.
