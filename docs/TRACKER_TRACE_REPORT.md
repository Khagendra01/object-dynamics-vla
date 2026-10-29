# DAVIS SAM 3.1 propagation trace

This is an integration diagnostic on the 69-frame DAVIS `bike-packing`
validation video, not a LIBERO result or a benchmark comparison. The pinned
SAM commit is `f1647e8d0c1d280991f4d1b7d974c2704537fe66`. The human-annotated
first-frame object-1 mask supplies a box and an interior point; later masks are
used only for evaluation. The image is **910×480**. The normalized box is
`[0.284615, 0.343750, 0.249451, 0.656250]` in **xywh** order; the point is
`[0.385165, 0.705208]` in **xy** order. Coordinates were divided by width for
x and height for y. Original box initialization returned ID 0 and first-frame
IoU 0.72867; the point initialization response was empty, but the subsequent
first propagated ID 1 mask had first-frame IoU 0.73336.

`scripts/modal_sam31_trace.py` records the full machine-readable stage trace at
`/data/eval/davis_sam31_stage_trace_v2.json` on the Modal volume, copied locally
to `outputs/libero/oracle/davis_sam31_stage_trace_v2.json`. Raw outputs and DAVIS
images remain outside the public repository. The probe uses the unmodified
upstream propagation implementation with the existing single-session
`offload_state_to_cpu` compatibility shim. It compares the official request API
to calling the underlying model directly in independent sessions, then separately
tests a narrow diagnostic point-path workaround. Both baseline paths agreed on
the first 11 emitted frames.

## Request and availability

The request was forward propagation from frame 0, with no frame limit and an
output threshold argument of 0.0. The wrapper accepts
`force_tracker_propagation`, but the interactivity model's method signature does
not explicitly declare it; the pinned wrapper omits it. **The earlier forced
and unforced experiments did not establish different propagation modes.** Box
initialization selected `propagation_full`; point initialization selected
`propagation_partial` for ID 1. The full path's hot-start delay is 15 and its
postprocess batch size is 16. In this run, public outputs describing frames
0–15 were emitted only after **30 frames had been processed** (through frame
29). A consumer must not backdate those delayed results as if they were
available at the described frame.

An earlier attempt with `max_frame_num_to_track=31` raised an upstream
`pos_pred_mask.shape[0] == 1` assertion near its final frame. The recorded trace
uses the full 69-frame limit and completed. The bounded-run assertion has not
been isolated, so this trace does not establish that limited propagation works.

## Box path: frames 0–10

The table shows the raw mask area in full video pixels for target ID 0 and the
actual postprocessing/collector result. `unconfirmed` is the status supplied to
the filter, which can use a later frame's status; removed and suppressed ID
lists passed to this filter were empty for these rows. Tracker confidence was
approximately 1.0 for the nonempty raw masks. Detection score remained 0.9416
through frame 6 and was the sentinel -10000 at frame 7.

| Frame | Raw area | Active ID after frame | Filter / public response | Stored area |
|---:|---:|---|---|---:|
| 0 | 38057 | 0 | ID 0 unconfirmed; empty | 0 |
| 1 | 37996 | 0 | ID 0 unconfirmed; empty | 0 |
| 2 | 37875 | 0 | ID 0 unconfirmed; empty | 0 |
| 3 | 37631 | 0 | ID 0 unconfirmed; empty | 0 |
| 4 | 37033 | 0 | ID 0 unconfirmed; empty | 0 |
| 5 | 36044 | 0 | ID 0 returned | 36044 |
| 6 | 34145 | 0 | ID 0 returned | 34145 |
| 7 | 34176 | none | ID 0 returned, score -10000 | 34176 |
| 8 | none | none | empty | 0 |
| 9 | none | none | empty | 0 |
| 10 | none | none | empty | 0 |

Thus the early missing public outputs were filtered despite nonempty internal
masks. By frame 7 the active tracker ID was gone, and from frame 8 there was no
internal mask to return. The logged removed/suppressed lists do not identify the
reason for that transition; the trace does not prove why the full grounding path
removed the active object. Our collector did not drop a returned ID in these rows.

## Point path: frames 0–10

ID 1's internal tracker produced nonempty 288×288 mask logits in every frame
0–10. At frame 1 the positive-logit area was 7511 and raw tracker score was
12.25. The frame had **no cached full-grounding output**; the pinned
`_build_sam2_output` immediately returned `{}` before merging the available
refined ID 1 mask. The public response and collector were therefore empty from
frame 1 onward, even though internal masks existed. This behavior was identical
through the official request wrapper and the direct underlying model call.

A diagnostic-only workaround returns the refined masks when that frame has no
cache, while preserving the original merge when a cache exists. It restored
nonempty public ID 1 masks on all 69 frames. On this one video the mean region
IoU against later annotations was **0.6710**; no future annotation supplied a
prompt. This is a targeted workaround for this failure path, not a generally
validated production adapter or autonomous identity grounding.

The workaround's point-path outputs were emitted after each corresponding frame
was processed in this instrumented run. For the separate 11-frame-prefix versus
full-video test, the frame-10 masks differed by 32 pixels; frame 20 matched.
Both were nonempty. This is not a successful strict prefix-consistency test;
the observed difference could involve numerical or length-dependent behavior
and requires controlled repeatability analysis before any causal claim.

## Fresh-process repeatability and future-content controls

`scripts/modal_sam31_repeatability.py` subsequently ran the requested matrix
on `bike-packing` and two additional DAVIS validation clips, `blackswan` and
`bmx-trees`, selected in dataset order before viewing their tracking outcomes.
Each case ran in a **single-use container** with a fresh predictor, fixed seed,
checkpoint, CUDA/bfloat16 precision and configuration. The second repetition
ran cases in reverse order. The human-annotation-derived initial point was held
fixed within each clip; later labels were used only for post-hoc scoring.
The v1 script set TF32 flags before predictor construction, but the pinned
constructor enables them again. Effective flags were not logged, so the v1
results must not be described as verified TF32-disabled measurements. The script
now sets and records those flags after construction for future runs.

For each clip, the first 11 JPEGs were byte-identical across cases. At the same
full video length, the future was either original, reversed after frame 10, or
replaced by repeated frame 10; input hashes confirm that future bytes changed.
The 11-frame prefix was also compared with the full-length input. Frame-10
288×288 pre-threshold tracker logits and output masks are saved on the Modal
volume under `/data/eval/sam31_repeatability_v1*`; logits were cast to float32
for storage, and raw arrays are not public.

| Clip | Frames | Full-video mean IoU | Same-input repeat | Same-length altered future | Prefix vs full: mask pixels / raw-logit sign flips |
|---|---:|---:|---|---|---:|
| bike-packing | 69 | 0.6710 | exact | exact | 32 / 4 |
| blackswan | 50 | 0.9578 | exact | exact | 7 / 3 |
| bmx-trees | 80 | 0.4463 | exact | exact | 4 / 1 |

The bike-packing IoU is from the earlier full-video workaround run; its matrix
cases stop after frame 10. The other two IoUs come from their matrix full runs.
On all three clips, both repeated fresh runs of each case were bit-identical
for frame-10 logits and binary masks; both changed-future cases also exactly
matched the original full-length case. Prefix-vs-full raw-logit max/mean absolute
differences were 0.625/0.0325, 0.375/0.0521, and 1.0/0.0625 respectively.
Thus the observed differences are repeatable and associated with sequence
length or a related processing condition in this setup, not the tested future
image substitutions. This does not identify the underlying cause or prove
independence from every possible future input.

The partial point path emitted frame 10 after processing frames 0–10 in every
case, within its normal loop, not a postprocess-buffer flush. In the prefix case
frame 10 was also the final input frame; in the full and altered-future cases it
was nonterminal. The prefix/full lengths still produced nonidentical logits and
masks, so this is **not yet strict prefix consistency** or proof of a live
appendable online adapter. The scores above are one object per video and do not
establish LIBERO performance, robust occlusion handling, or autonomous grounding.

## Temporal pointer normalization: targeted causal intervention

`scripts/modal_sam31_temporal_probe.py` tested the sequence-length hypothesis on
the same `bike-packing` clip and annotation-derived initial point. Each case ran
in a fresh single-use A100 container with the same first 11 image bytes, seed,
checkpoint and bfloat16 autocast. The model's active tracker was
`sam3.model.video_tracking_multiplex_demo.Sam3VideoTrackingMultiplexDemo`; the
memory-attention wrapper was `sam3.model.model_misc.TransformerWrapper`, with a
four-layer `sam3.model.decoder.TransformerEncoderDecoupledCrossAttention`,
`DecoupledTransformerDecoderLayerv2` layers, and `SimpleRoPEAttention` cross
attention. Object pointers and their temporal encoding were enabled, with a
configured maximum of 16. `only_obj_ptrs_in_the_past_for_eval` was **false**, so
past-only behavior must not be assumed generally. In this experiment the actual selected pointer
frames were inspected and all were already available.

At frame 10, actual selected pointers were frames `0,9,8,7,6,5,4,3,2,1` in
every case. The baseline's observed pointer budget was `min(num_frames,16)`;
the active `_get_tpos_enc` call divided relative offsets by budget minus one.
Against the 16-frame baseline:

| Declared length | Pointer budget | Temporal denominator | Max raw-logit difference | Different output pixels |
|---:|---:|---:|---:|---:|
| 11 | 11 | 10 | 0.6250 | 32 |
| 12 | 12 | 11 | 0.4375 | 43 |
| 15 | 15 | 14 | 0.2500 | 28 |
| 16 | 16 | 15 | 0 | 0 |
| 17 | 16 | 15 | 0 | 0 |
| 69 | 16 | 15 | 0 | 0 |

The diagnostic intervention changed **only** the `max_abs_pos` argument to
`_get_tpos_enc`, fixing the denominator to 15. It did not change `num_frames`,
pad the input, or alter upstream pointer-selection bounds. Pointer selections
through frame 10 matched the 16-frame baseline, and the instrumentation required
every selected pointer in the intervention to resolve to a frame no later than
the current frame. With the fixed denominator, frame-10 logits and masks were
bit-identical to the 16-frame result for **all six lengths**. Fresh-container
repeats at lengths 11 and 16 were also exact in both modes.

This confirms temporal pointer normalization as the cause of the observed
length-dependent frame-10 discrepancy **for this point-path setup**. It does not
prove a general streaming or causality guarantee, address the separate box-path
removal, or validate a production adapter. The complete JSON and pre-threshold
arrays are retained under `/data/eval/sam31_temporal_probe_v1/` on Modal; the JSON
summary is mirrored at `outputs/libero/oracle/sam31_temporal_probe_v1_summary.json`.

The instrumented run observed both TF32 flags as `true` immediately after the
pinned predictor constructor. It then set both to `false`; both were confirmed
false at frame-10 emission. The earlier v1 repeatability matrix did not record
effective post-construction TF32 flags, so it remains historical evidence of
repeatability rather than a verified TF32-disabled run.
