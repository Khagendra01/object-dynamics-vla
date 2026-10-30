# Frozen Muse + SAM VLA: implementation status

This is an experimental **SmolVLA-inspired adaptation**, not the official
SmolVLA model or a trained motor policy. The full specification is the owner's
`MUSE_SAM_VLA_END_TO_END_PLAN.txt`; implementation and experimental results
must be reported separately. The reference LeRobot source is pinned at
`552777b7c384e8e71a02303da39583f9a2653760`. Its flow interpolation is noise at
time one, velocity `noise - action`, with Euler sampling from one to zero.
The official SmolVLA reference training run has **not** yet been performed.

`configs/muse_sam_v1.json` fixes the existing milk development fingerprint,
model revisions, 20 Hz data convention, 4-frame history, 50-step actions,
network dimensions, optimization and three seeds. The current trainable
architecture has a Muse projection, robot and two-slot SAM history tokens,
two context blocks, and eight alternating causal self/cross-attention expert
blocks. A separate query uses **only** `a[t]` in the same expert blocks for a
one-step 2×5 dynamics target. Muse tokens are detached; SAM is an externally
cached frozen measurement. There is no action intervention or recovery gate.

`odvla/vla/dataset.py` constructs start/end-padded windows and checks each token
cache file's hash, timestamp, episode, and instruction. Statistics come from
training episodes only. Startup repeats the first observation and masks padded
history; end padding is masked from action loss. Observation encoders receive
no future labels. `odvla/vla/action_expert.py` shares expert weights across the
two objectives and uses valid-element mean losses. The implementation uses
PyTorch attention blocks, not LeRobot's unchanged action-expert implementation.

Muse block 26 of 52 is read with a forward hook, retaining every valid token.
The cache stores unprojected BF16 bit patterns, token IDs, masks, and positions;
no trainable projector is cached. Transformers 5.21.0 / torchvision 0.25.0
logs a **LANCZOS-to-BICUBIC fallback**; the exact same processor/backend must be
used at rollout. It is not equivalent to the configured original LANCZOS.
The SAM cache remains explicitly `causal: false`, with bounded prefix auditing.

The main trainer uses paired seeds, independent batch/noise/time RNG streams,
the same fixed-schedule validation flow selection, and snapshots every 250
updates. A seed-7 suite runner is experimental and subject to a runtime cap; do
not report partial arms as a matched outcome. It preloads the frozen token cache
in host RAM, so host memory and volume I/O are material resource costs.
