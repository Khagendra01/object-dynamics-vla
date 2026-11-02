# Research and implementation plan: Muse + SAM + pretrained SmolVLA expert

**Status: research/implementation plan, not an implemented or validated model.**
The requested runtime policy is frozen Muse Glimmer 30B + frozen SAM 3.1 + the
native robot-pretrained SmolVLA action expert, connected by learned modules.
SmolVLM may be used as a frozen teacher during training and as a control; it must
not silently remain the perception source of the final Muse-conditioned policy.

## Evidence and relevant prior work

- [SmolVLA paper](https://arxiv.org/html/2506.01844v1) and the
  [pinned official expert implementation](https://github.com/huggingface/lerobot/blob/552777b7c384e8e71a02303da39583f9a2653760/src/lerobot/policies/smolvla/smolvlm_with_expert.py)
  motivate interleaved self/cross attention and flow-matching actions. The exact
  pinned source, not a diagram or the paper's inconsistent flow notation, is
  authoritative for our cache, masking and sampler contract.
- [Flamingo](https://arxiv.org/abs/2204.14198),
  [Perceiver IO](https://arxiv.org/abs/2107.14795), and
  [BLIP-2](https://arxiv.org/abs/2301.12597) establish useful precedents for
  trainable attention-based bottlenecks connecting frozen pretrained models.
  They do not establish that their connectors preserve a robot expert's skills.
- [Prefix-tuning](https://arxiv.org/abs/2101.00190) motivates learned per-layer
  attention prefixes, but its NLP results do not prove a generated multimodal
  robot prefix will work. This proposal uses observation-dependent prefixes.
- [Distillation](https://arxiv.org/abs/1503.02531) motivates retaining the
  existing policy as a teacher. Here, matching a continuous flow velocity at the
  same noisy actions and time is an adaptation; it is not a published result for
  this exact bridge.
- [π0](https://arxiv.org/abs/2410.24164) and
  [RoboFlamingo](https://arxiv.org/abs/2311.01378) demonstrate pretrained
  VLM/robot-policy combinations, not interchangeability of arbitrary VLM and
  action-expert weights. [DAgger](https://arxiv.org/abs/1011.0686) motivates
  measuring rollout distribution shift; it does not authorize assuming an
  on-policy expert or collecting new robot data automatically.

Our evidence: the native SmolVLA policy adapted on milk succeeded on two reused
development states; the independently initialized Muse expert did not. These
are not a controlled head-to-head architecture study. The historical milk and
soup Muse caches contain different automatically injected dates; this is a
task-correlated confound, documented in `MULTITASK_SAM_CLOSED_LOOP.md`.

## Actual interface at the pinned LeRobot commit

The checkpoint `lerobot/smolvla_base` revision
`5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6` has a **720-wide** expert and
16 retained VLM/expert layers. SmolVLM's hidden width is 960, with 15 query
heads, 5 key/value heads and head dimension 64. `action_in_proj` has shape
`[720,32]`; the first expert query projection has shape `[960,720]`. Our old
512-wide expert is not a weight-compatible target.

The expert does not consume one final VLM vector. At inference, the official
SmolVLM prefix prefill builds **one key/value cache per layer**, logically
`[batch, 5, prefix_length, 64]` for each K and V. In this implementation
prefix keys have already had RoPE applied; values have not. Even-numbered
layers use joint prefix/suffix self-attention; odd-numbered layers use the
expert's pretrained cross-attention projections on flattened VLM K/V. The
prefix mask, positions, GQA head layout, and the suffix offset are part of the
contract. During Euler denoising the official code appends suffix K/V on
self-attention layers and then crops the cache back to the prefix length.

Training without caching runs prefix and suffix together. It cannot simply be
replaced by `inputs_embeds=[None, suffix]` with an arbitrary cache without
checking autograd, layer behavior and masks. Build a differentiable, tested
expert-only adapter that produces the **same outputs as the pinned official
code when handed the official prefix K/V**, including gradients into supplied
K/V. Do not assume inference `DynamicCache` mutation is automatically safe for
training. Prove forward/gradient parity using actual official teacher prefixes
before putting Muse behind it. Preserve the same 50-action chunk, padding,
`noise_at_one`, Beta time sampling and 10-step integration from the pinned code.

## Proposed bridge: observation-conditioned per-layer K/V

```
RGB + instruction -- frozen Muse block 26 --> token features [B,N,6656] + mask/positions
                                                   |
robot state/history --> pretrained state_proj -----+--> trainable observation bridge
SAM history ---------> trainable SAM projection ---+    (explicit validity and time)
                                                        |
                                             [B,L,2,5,P,64] K/V
                                                        |
                                    native pretrained SmolVLA expert + action head
                                                        |
                                               50 actions; execute first
```

Starting engineering configuration, **not a claimed optimum**:

1. Mask invalid Muse tokens, project 6656→512 with a trainable norm/projection,
   preserve token positions and image/text type metadata. Start with Muse block
   26 as already recorded. Do not use generated descriptions or a single vector.
2. Use a small learned-query cross-attention resampler, initially 64 latents of
   width 512, with mask support and positional/type information. Check that
   language and small objects survive this bottleneck; 64 is a starting budget,
   not proof of sufficiency. Record/query ablations rather than hiding loss of
   spatial or language information.
3. Normalize robot observations with the exact training-only statistics saved
   with the chosen LIBERO-adapted SmolVLA checkpoint; pad to 32 and initially
   retain its native `state_proj`. Encode SAM's per-object measurement history,
   backward changes, validity, observation age, object identity and timestamps
   with a separate trainable projection. Missing data is masked, not filled
   from the future. Choose a common, deterministic startup history rule.
4. Fuse the resampled Muse features, robot token(s), and SAM tokens; generate
   **separate K and V for each of the 16 layers**, each `[B,5,P,64]`. A shared
   512-wide trunk with layer-specific heads avoids a full 6656→all-layer
   projection. Choose one explicit convention: output post-RoPE prefix keys in
   the same coordinate system as the teacher cache; values remain unrotated.
   Do not rotate cached keys again in the expert. Apply prefix masks and suffix
   positional offsets exactly as in the pinned implementation.
5. Strictly load the native 720-wide expert **and** action input/output/time
   projections. Primary starting point is the existing 1,000-update
   LIBERO-adapted checkpoint, because it already maps the robot's 15/7 state and
   action convention. Also retain the original released checkpoint as a
   provenance/ablation control. The former is pretrained and then adapted; it
   must not be described as untouched original weights. Verify keys, shapes,
   tensor hashes and normalizer before any optimization. No weight reshaping or
   transfer into the old 512-wide expert.

The bridge outputs synthetic cache values: they are not automatically equivalent
to actual SmolVLM K/V. The expert's pretrained cross projections are retained,
so a connector that emits values in the wrong key/value or positional basis can
pass shape tests and still destroy motor behavior. A simple 6656→720 projection
is therefore not a sufficient compatibility strategy.

## Training sequence and comparison

**Phase 0 — verification.** Lock data, source identities, normalization,
checkpoint hashes and fixed Muse chat date. The existing milk cache is a
development starting point with date `2026-10-30`; for any new multitask
training regenerate or otherwise recompute date-consistent actual features,
not just manifest labels. Preserve old caches. Exclude all validation and
previously used test episodes from training and teacher-target generation.
Confirm pretrained checkpoint/data overlap before calling any test independent.

**Phase 1 — connector alignment.** Use the frozen, LIBERO-adapted SmolVLA as a
teacher on training frames only. Feed the teacher the same RGB, instruction and
robot state; feed the student actual cached Muse tokens and robot state. Freeze
the native student expert and teacher; train only the observation bridge. For
each example, draw one `x_t` and `t`, evaluate teacher/student using the same
noisy action chunk, and minimize masked velocity MSE; pair seeds/noise streams.
Optionally add a declared supervised flow loss on demonstration actions. The
teacher is only a training scaffold: it can transfer its mistakes and does not
prove Muse grounding. Direct elementwise K/V MSE is valid only if teacher and
student have deliberately aligned prefix positions/lengths and masks; with a
64-latent resampler they generally do not. Do not invent token correspondences.

**Phase 2 — joint motor adaptation.** Initialize the native expert, action/time
heads and trained connector from Phase 1; unfreeze and train the full expert and
connectors, keeping Muse and SAM frozen. Save a common checkpoint before SAM
comparisons. Use independent RNG streams for batches, action noise and any
auxiliary loss. Save optimizer, schedule, RNG, cache/processor identities and
training data provenance for exact resume; never extend a completed zero-LR
schedule as though it had been trained for a longer horizon.

**Phase 3 — controlled SAM/dynamics experiment.** From the same Phase-2
checkpoint and with matching initialization, data, noise and budget:

- M0: Muse + robot, action objective; SAM inputs masked.
- M1: same bridge/expert plus valid SAM observations, action objective only.
- M2: same inputs/architecture as M1, action objective plus masked one-step
  `object_features[t+1]-object_features[t]`, conditioned on history ≤t and
  demonstrated `a[t]` only. The dynamics head must share trainable features
  with the policy. Declare weight 0.1 initially and report action/dynamics
  shared gradient norms and per-metric errors against persistence, including
  moving/stationary subsets. Do not tune silently on test results.

M1–M0 estimates the input effect; M2–M1 estimates the auxiliary objective.
Teacher/distillation objectives and schedules must be matched across arms and
logged. Select all checkpoints using one fixed validation action criterion;
state explicitly that validation is also used for selection. Paired seeds 7,
19 and 31 share the same motor checkpoint and separately controlled RNG.

## Acceptance tests and controls

1. **Native parity:** strict-load all expert/action-head tensors; compare the
   expert-only wrapper to the official model's output and gradients under real
   official K/V for multiple masks, chunk lengths and noise/times. Verify exact
   layer count, `[B,5,P,64]` layouts, dtype/device, position, mask and cache
   crop; gradients to bridge K/V must be nonzero.
2. **Information path:** on held-out *development* windows, permute/replace
   RGB and instruction separately, randomize Muse tokens and mask SAM while
   holding state/noise fixed. Report action sensitivity, not merely nonzero
   gradients. Show padding and invalid SAM features cannot alter valid results.
3. **Overfit and transfer:** first overfit a tiny training subset; report teacher
   versus student velocity and sampled-action errors, per-channel first-action
   error and gripper per-class recall against majority baselines. Demonstrate
   that the connector preserves meaningful use of pretrained motor weights,
   compared with a random-expert control under the same pipeline and budget.
4. **Causality:** future frames/actions cannot alter current inputs or one-step
   dynamics; SAM prefix replay is the correctness reference. The incremental
   SAM implementation has only bounded parity evidence, not a general guarantee.
5. **Control:** run M0/M1/M2 on the same reused development states, paired
   sampling seeds, execute the first action and reobserve, up to 200 steps.
   Save videos, sampled/executed actions, clipping, latency and tracking status.
   The existing SmolVLA-only result is an informative baseline, not the final
   proposed model. Do not use its prior validation successes as untouched tests.

## Hardware, cost and stop conditions

The existing BF16 Muse extraction peaked around 55.6 GiB on A100; the combined
Muse/SAM/old expert runtime peaked around 66.1 GiB. It is **not established**
that the target model fits a 32 GiB 5090. Train bridge/expert from frozen cached
Muse features and cached SAM where possible, retain prefix-only rollout parity,
and measure actual memory/throughput before choosing a GPU. A quantized or
offloaded Muse runtime must separately demonstrate hidden-state availability,
feature agreement and control behavior; it cannot be assumed equivalent to
BF16-cache training. SmolVLM teacher features may be cached separately and must
not be used by the final Muse inference path.

Modal's observed billing summaries do not expose the exact remaining allocated
credit. Before paid extraction/training, check current balance and prices;
bound each job and preserve partial results. No GPU costs or success claims are
implied by this plan. If official K/V wrapper parity, source-feature alignment,
or budget checks fail, report that failure and stop rather than presenting the
standalone SmolVLA baseline as the requested architecture.
