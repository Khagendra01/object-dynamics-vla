# Official SmolVLA adapter reference

The official LeRobot SmolVLA implementation at commit
`552777b7c384e8e71a02303da39583f9a2653760` was run on Modal T4 on a real
milk training observation/action window. This is a one-update integration
check, not a trained control benchmark, and is separate from the Muse model.

The reference uses frozen pretrained
`HuggingFaceTB/SmolVLM2-500M-Video-Instruct` revision
`7b375e1b73b11138ff12fe22c8f2822d8fe03467` and a randomly initialized official
SmolVLA action expert. It does **not** load the robot-trained `smolvla_base`
checkpoint; therefore no claim of pretrained robot-data independence or policy
competence is made. Potential overlap of SmolVLM2's visual/language pretraining
with LIBERO imagery was not audited. No test episode was accessed.

The run uses the official pre/postprocessor and official flow training and
10-step sampling. Inputs came from development `episode_0000`, frame 0, its
512×512 agentview JPEG (SHA-256
`6cffc8a3845b5c7eaf772fdd6b895c822c40b20a66b5c3a07ef731d7e22a9109`), and
the instruction `pick up the milk and place it in the basket`. The normalizer
was fitted on training episodes only. The processor's state and action
normalization matched the existing adapter; shapes were `[1,15]` and
`[1,50,7]`, and all 50 action steps were valid. A separate end-of-episode
processor check verified 2 valid and 48 padded steps. The instruction produced
11 valid tokens. The VLM was frozen and had zero parameter gradients.

Actual first demonstrated raw action:
`[.05357, -.18482, 0, 0, .05679, -.04929, -1]`.
Its normalized target was
`[-.09345, -.64514, .13738, -.09385, .87416, -1.06861, -1.17146]`.
One optimizer update had flow loss 2.452990, expert gradient norm 35.0784 and
nonzero inspected parameter update norm 0.08917. Sampled first action after
unnormalization was `[-.23708, -3.60907, -.65859, .07272, .13964, -.04907,
-2.19347]`; it is untrained and **was not clipped or executed**. The 10-step
sampler output was finite. Peak GPU allocation was 1.671 GiB; elapsed function
time including model load, processing, update and sampling was 12.01 s. Python
3.12, torch 2.10.0+cu128, torchvision 0.25.0 and Transformers 5.5.4 were used.

The first attempt failed before an update because LeRobot's add-batch processor
does not add a batch axis to `[50,7]` action chunks. The adapter was corrected
to pass explicit `[1,50,7]` actions and `[1,50]` padding. A subsequent v1
reference completed; v2 additionally fixed the expert initialization seed and
checked episode-end padding and an actual parameter update. Both results remain
on the volume at `/data/eval/official_smolvla_reference_v1.json` and
`/data/eval/official_smolvla_reference_v2.json`.

Reproduce with a fresh output name and the existing dataset volume:

```bash
~/modal-venv/bin/modal run scripts/modal_smolvla_reference.py --output-name official_smolvla_reference_reproduction.json
```

This validates the integration path, not whether SmolVLA or the Muse expert
learns or succeeds in closed-loop control. The reference's weights were not
attached to Muse and no action-expert weight compatibility was assumed.
