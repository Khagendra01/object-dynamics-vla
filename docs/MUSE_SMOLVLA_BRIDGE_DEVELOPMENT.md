# Muse/SAM to pretrained SmolVLA: development integration

This is a **development result, not a successful task-completion result**.
The implemented path feeds cached frozen Muse block-26 tokens, normalized robot
state and cached frozen SAM measurements into trainable connectors, which
generate per-layer key/value prefixes for the native SmolVLA action expert.
The old 512-wide expert is not used. The native 720-wide expert, state and
action/time projections were initialized from the existing 1,000-update
LIBERO-adapted SmolVLA checkpoint, itself strictly initialized from the pinned
released robot checkpoint. SmolVLM was a teacher during connector alignment;
the combined student action predictions reported below use Muse/SAM inputs, not
SmolVLM perception. A 30-action live Muse/SAM rollout is described below.

## Native-interface verification

`scripts/modal_smolvla_bridge_parity.py` strictly loaded the original released
checkpoint and compared the official full prefix+suffix forward with the new
expert-only path. The prefix was synthetic embeddings processed by the official
prefix prefill, with a padded mask and one fixed noisy action/time. On Modal CPU
the maximum absolute difference was **0.0**; all 16 K and V inputs had nonzero
gradients, minimum norm 0.0016632. Each K/V had shape `[1,5,7,64]`. This is a
single parity case, not broad coverage of all masks, positions or cache types.
`PrefixCache` appends suffix K/V functionally, preserving gradients and avoiding
mutation of the supplied prefix.

## Training

The connector resamples Muse into 64 learned-query tokens and fuses the native
state projection and four-frame, two-object SAM observations before generating
16 K/V pairs. It uses valid masks and train-only normalization. Muse token
features already contain Muse's contextual/positional representation; this
prototype does not separately add explicit image/text type or position features.

1. **Connector alignment:** seed 7, 250 updates, batch 1, AdamW `1e-4`, uniform
   noise time, frozen adapted expert. The teacher received the corresponding
   RGB/instruction/state while the student received Muse tokens/state/SAM.
   Teacher-velocity MSE on the same 16 reused development windows fell from
   **1.46219 to 0.20821**. Results and the final snapshot are at
   `/data/runs/muse_smolvla_bridge_alignment250v1/`. This is teacher matching,
   not measured task success or proof of grounding.
2. **Action adaptation:** initialize the connector above, unfreeze the native
   expert, state projection and action/time heads, and optimize demonstration
   action flow, with Muse/SAM frozen. Seed 7, 250 updates, batch 1, AdamW
   `1e-5`; the pinned Beta time distribution was used. On 16 reused validation
   windows selected and reported with fixed seeds, flow MSE was 0.74276 before
   training, 0.69010 at step 50, 0.65550 at 100, 0.66367 at 150, **0.64215 at
   200** and 0.64834 at 250. Selected step 200 is
   `/data/runs/muse_smolvla_bridge_action250v1/weights_best.pt`. Its expert
   tensors are stored as named trainable parameters and must be applied over
   the exact adapted checkpoint, along with the separately stored bridge.

The 16 windows are first/middle frames from eight validation episodes, used for
selection, and the teacher checkpoint was also selected on validation data.
Historical Muse milk cache uses the fixed `2026-10-30` template; the two-task
milk/soup cache date confound remains unresolved. Only milk train episodes were
used for these updates. The training driver saves optimizer/RNG snapshots but
does **not** yet implement resume. There is no M0/M1/M2 matched experiment or
dynamics objective in these runs.

## Offline sampled-action diagnostic

On the same 16 checkpoint-selection windows, fixed noise and the native 10-step
sampler gave per-channel first-action MAE `(x,y,z,rx,ry,rz,gripper)` of
`[.0922,.4323,.1322,.0258,.0520,.0321,.1785]`; gripper sign agreed on all 16.
This very small, reused selection set cannot establish task performance; the
earlier standalone policy's 100% on these windows also fell on the full split.

With identical action noise, masking SAM changed the seven outputs by mean
absolute `[.0151,.0594,.0545,.0026,.0062,.0030,.0357]`; zeroing Muse features
changed them by `[.1304,.4196,.4007,.0226,.0177,.0141,.4075]`. These are input
sensitivity checks, not proof that the signals are correct or beneficial. The
full diagnostic is `/data/eval/muse_smolvla_bridge_action250v1_offline16_v1.json`.
No held-out test initializations were used for this model.

## Live Muse/SAM closed-loop pilot

`scripts/modal_muse_smolvla_live.py` connects the simulator, frozen Muse and
SAM in a Python 3.11 A100-80GB container to the actual pretrained-expert
checkpoint in a separate Python 3.12 T4 policy service. The policy receives the
current Muse layer-26 features, causal SAM history and state, samples a 50-action
chunk and executes the first action before taking a new observation. The
simulator provides no oracle masks. The prompt, object points and image/state
normalization are the same development setup used previously.

On previously used validation `episode_0014` / `demo_21`, the combined model
executed **30 steps, with no task success by the 30-step cap**. This is a short
pilot, not a failed 200-step trial or a success-rate estimate. The final RGB
shows the milk outside the basket; the trace includes clipped robot actions.
The step-200 checkpoint SHA-256 is
`3791c0672bf9ff2f31f395a9d7cc50e7fc3ef3558f7d71a6d4e1430e52834f1f`.
The first live Muse tokens matched the historical cache bit-for-bit; incremental
SAM matched independent observed-prefix replay at steps 0, 9 and 29. Peak
allocated memory in the Muse/SAM container was 62.20 GiB; total duration was
193.9 seconds including model startup and prefix checks. The native expert ran
on a separate T4, so that number is not total system memory. The initial image
hash matched the demonstration source and earlier development rollouts.

The first one-step attempt stopped before acting because its Muse features did
not exactly match, although its input IDs, attention mask, positions and RGB
hash matched. A second one-step attempt and the 30-step run passed exact parity;
the first discrepancy remains unexplained, and its failure artifact is kept at
`/data/eval/muse_smolvla_live_episode_0014_smoke1/`. Do not drop the strict
parity check or assume that every run is reproducible. The successful one-step
smoke and 30-step pilot are under the same prefix with suffixes `smoke2` and
`pilot30`; the latter contains `report.json`, `trace.json`, RGB frames and a
video. The video was also copied to
`/home/kgen/Desktop/muse_sam_smolvla_pilot30.mp4`.

## Verification, limitations and next steps

- The Modal CPU suite passed 38 tests, including invalid-input masking and
  functional cache-gradient tests. The native parity test is separate.
- Connector and action training peaked at 2.26 GiB reported allocated T4 GPU
  memory; this excludes live Muse and SAM. Co-resident BF16 Muse/SAM previously
  needed about 66 GiB. A 32 GiB inference deployment remains unverified.
- Diagnose the intermittent first-frame feature mismatch and run full 200-step
  development rollouts, then paired states and seeds before claiming success.
- Run full-validation and image/instruction counterfactuals; test random-expert
  versus pretrained expert, M0/M1/M2 and dynamics-vs-persistence with matched
  seeds/budgets. The proposed model may still ignore important information or
  fail despite lower offline MSE.
- The credit balance was not exposed by the Modal CLI. Current November billing
  was approximately $8.18 credits applied, with the prior cycle separate; paid
  live rollouts require checking the remaining allocation first.

Reproduction uses fresh suffixes; existing artifacts are not overwritten:

```bash
~/modal-venv/bin/modal run scripts/modal_smolvla_bridge_parity.py
~/modal-venv/bin/modal run scripts/modal_muse_smolvla_bridge.py --steps 250 --suffix NEWALIGN
~/modal-venv/bin/modal run scripts/modal_muse_smolvla_bridge.py --steps 250 --suffix NEWACTION --phase action --init-bridge NEWALIGN
~/modal-venv/bin/modal run scripts/modal_muse_smolvla_eval.py --suffix NEWSUFFIX
~/modal-venv/bin/modal run scripts/modal_muse_smolvla_live.py --max-steps 30 --suffix NEWPILOT
~/modal-venv/bin/modal run scripts/modal_vla_tests.py
```

The offline evaluator currently names the recorded action run explicitly; the
suffix selects a fresh output and does not change which checkpoint is tested.
