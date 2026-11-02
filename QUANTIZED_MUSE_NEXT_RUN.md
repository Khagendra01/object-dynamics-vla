# Next run: NF4 Muse + SAM + randomly initialized official action expert

This is the next bounded execution, not a new policy-performance claim. It uses
the existing pinned Muse checkpoint, milk training episodes, RGB renders and
SAM measurement cache on `object-dynamics-vla-data`.

## Run

From the repository root in the PC environment already authenticated to Modal:

```bash
python -m modal run -m scripts.modal_quantized_muse_preflight --name quantized_muse_preflight_v1
```

Defaults: one H200, four distinct real training frames, at most 20 optimizer
updates, batch one, 120-second training-loop cap, 900-second function timeout.
No GPU time is spent downloading model weights. Missing existing assets cause a
failure. Existing output directories are never overwritten; use a new `--name`
on a rerun. The small SmolVLM configuration is required on the existing volume;
its pretrained weights are not loaded and its vision/language forward is not
used. No robot-trained policy checkpoint or teacher is loaded.

The H200 function's maximum GPU allocation at the 900-second timeout is about
$1.14 at $4.54/hour, **not** a complete invoice or overall spending cap. Requested
CPU/RAM, startup, builds and any other resources are separate. There is no
automatic larger training job, GPU fallback, or multi-GPU launch.

## What runs

1. Load Muse with Transformers/bitsandbytes NF4, BF16 compute/storage and a fixed
   revision/date. Keep the backbone frozen. Reject missing actual 4-bit modules
   or unexpected quantized modules outside the text decoder.
2. Extract fresh image-and-instruction features from four training frames. Do
   not reuse the historical BF16 cache as if it were an NF4 cache.
3. Feed these features, normalized robot state and causal SAM history through
   the existing trainable `MuseSamPrefix` adapter.
4. Train a newly initialized official SmolVLA action expert and its projections
   with demonstrated-action flow loss. Verify expert, prefix and SAM-encoder
   gradients and optimizer changes; require zero frozen-backbone gradients.
5. Sample a finite 50-by-7 action chunk with ten denoising steps. Execute no
   action and access no held-out test episode.

The existing learned-prefix adapter is an architectural adaptation. This probe
does not claim to reproduce the original layer-wise SmolVLM conditioning or
establish that its pooled prefix retains enough information for control. It
tests quantized visual feature access and training a random expert, rather
than transferring an expert already trained for another backbone.

## Retrieve and interpret

```bash
python -m modal volume get object-dynamics-vla-data /eval/quantized_muse_preflight_v1 ./outputs/quantized_muse_preflight_v1
```

Artifacts: `report.json` and `preflight_weights.pt`. The report contains input
hashes, actual quantized module names, package versions, GPU/backend settings,
losses, gradient/update checks, peak GPU memory, separate feature-forward and
expert-update timings, and requested-resource cost arithmetic. Failure reports
are committed when the function can handle the exception; a hard timeout or
process kill may prevent that final write.

Passing means **the training interface works**, not that the random policy can
manipulate objects. It triggers the next step: a tiny-subset fitting experiment
and a simulator rollout using the same NF4 feature path. It does not trigger
an expensive full run. For cost estimates at effective batch 64, first measure
that batch/accumulation configuration and include Muse extraction and rollout
costs; multiplying a batch-one warm update by 100,000 is not valid.

Local validation covers loading arguments and backward compatibility, Python
compilation and Modal CLI import. The real NF4 image path, H200 memory footprint
and official-expert integration require the authenticated GPU run above.
